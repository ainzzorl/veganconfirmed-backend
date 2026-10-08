import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from services.gemini_service import GeminiService
from services.desktop_service import (
    DesktopAnswerCutShort,
    DesktopService,
    DesktopUnavailable,
)
from services.firestore_service import FirestoreService
from services.privacy import IpPrivacy, strip_url
from services.page_scope import (
    AUTOMATIC_TRIGGER,
    ScopeDecision,
    decide_scope,
    describe_scope,
)
from models.analysis_model import PageAnalysis
from models.content_model import (
    AnalysisFeedbackRequest,
    PageAnalysisRequest,
    PageAnalysisResponse,
)
from models.database_model import (
    APICallRecord,
    MenuRecord,
    ProviderCall,
    ShoppingItemRecord,
    STATUS_ANALYSIS_FAILED,
    STATUS_INVALID_REQUEST,
)
from utils.logger import setup_logger

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)


# The two providers, by the name their ``ProviderCall`` is stored under.
DESKTOP = "desktop"
GEMINI = "gemini"

# Which provider serves which flow. Hard-coded: it is a decision about what each
# flow needs, not about how a deployment is wired.
#
# The automatic trigger fires from the extension's add-to-cart handler, which
# interrupts someone mid-purchase and only ever needs the item branch, so it
# runs on the local machine for free. Everything else is a manual check with a
# user waiting on it, and goes to Gemini.
#
# This picks who goes *first*; the other provider stays the fallback either way.
# Where only one is configured — the eval harness pins one per combo — there is
# nothing to route.
PROVIDER_BY_TRIGGER = {AUTOMATIC_TRIGGER: DESKTOP}
DEFAULT_PROVIDER = GEMINI


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _as_str(value: Any) -> str:
    """Coerce a value from a raw payload to a string; missing becomes empty."""
    return "" if value is None else str(value)


def _as_optional_str(value: Any) -> Optional[str]:
    """Same, but keeping "absent" distinct from "empty"."""
    return None if value is None else str(value)


def _as_str_list(value: Any) -> Optional[list[str]]:
    """Coerce a value from a raw payload to a list of strings, or ``None``."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


class AnalysisCore:
    """Core analysis functionality that can be used by both Flask and Cloud Functions"""

    def __init__(
        self,
        enable_database: bool = True,
        collection_name: str = "api_calls",
    ):
        """
        Initialize the analysis core

        Args:
            enable_database: Whether to enable database operations (useful for Cloud Functions)
            collection_name: Firestore collection name for storing API calls
        """
        # Which providers exist to route between; ``PROVIDER_BY_TRIGGER``
        # decides who gets what.
        use_desktop = _env_flag("USE_DESKTOP_SERVER", True)
        disable_gemini_fallback = _env_flag("DISABLE_GEMINI_FALLBACK", False)

        self.desktop_service = DesktopService() if use_desktop else None
        self.gemini_service = None if disable_gemini_fallback else GeminiService()

        # Whether a page's own declaration of its kind may narrow the scope.
        # On by default. The signals are recorded on every request either way,
        # next to the scope they produced, so the rules stay measurable against
        # real traffic and this can be turned off again without a deploy.
        self.use_page_signals = _env_flag("PAGE_SIGNAL_SCOPE", True)

        if self.desktop_service is None and self.gemini_service is None:
            raise ValueError(
                "No analysis provider configured: enable USE_DESKTOP_SERVER "
                "and/or allow the Gemini fallback (DISABLE_GEMINI_FALLBACK=false)"
            )

        self.enable_database = enable_database
        if enable_database:
            self.database_service = FirestoreService(collection_name)
            self.ip_privacy = IpPrivacy()
        else:
            self.database_service = None
            logger.info("Database operations disabled")

    def _providers_for(self, trigger_type: Optional[str]) -> list[tuple[str, Any]]:
        """The configured providers, the one this flow routes to first."""
        preferred = PROVIDER_BY_TRIGGER.get(trigger_type, DEFAULT_PROVIDER)
        services = {DESKTOP: self.desktop_service, GEMINI: self.gemini_service}
        order = [preferred] + [name for name in services if name != preferred]
        return [(name, services[name]) for name in order if services[name]]

    def _dispatch(
        self, method: str, trigger_type: Optional[str] = None, **kwargs
    ) -> tuple[Dict[str, Any], ProviderCall]:
        """Run ``method`` on each provider ``trigger_type`` routes to, in turn.

        Both providers expose the same analysis methods, so this is shared
        across analysis kinds. Once none is left the error propagates — except
        for an answer a local model ran out of room to finish, which is worth
        keeping when there is nothing better to replace it with.

        Returns:
            A tuple of (analysis_result, provider_call), where the
            ``ProviderCall`` says which provider and model actually served the
            request, what it charged, and — for the desktop-server — the
            ``lms_jobs`` document it ran as. Only the provider that answered
            knows all of that, so each one describes its own call.
        """
        providers = self._providers_for(trigger_type)
        for index, (name, service) in enumerate(providers):
            nxt = providers[index + 1][0] if index + 1 < len(providers) else None
            try:
                return getattr(service, method)(**kwargs)
            except DesktopUnavailable as e:
                if nxt is None:
                    raise
                logger.warning(f"{name} unavailable ({e}); falling back to {nxt}")
            except DesktopAnswerCutShort as e:
                # A provider that can answer for the whole page beats a partial
                # one; the part the local model finished is only better than
                # nothing.
                if nxt is None:
                    logger.warning(
                        f"{name} answer cut short ({e}) and no fallback is "
                        f"left; answering with the part it finished"
                    )
                    return e.analysis, e.provider_call
                logger.warning(f"{name} answer cut short ({e}); falling back to {nxt}")
            except Exception as e:
                if nxt is None:
                    raise
                logger.error(f"{name} analysis failed ({e}); falling back to {nxt}")

        raise ValueError("No analysis provider configured")

    def scope_for(self, analysis_request: PageAnalysisRequest) -> ScopeDecision:
        """The page kinds this request may be, and the rule that said so.

        Public because it answers a question about the request alone — no model
        is involved — which the eval harness asks so its report can say what
        each case was actually asked. ``PAGE_SIGNAL_SCOPE`` is applied here, so
        every caller sees the scope this deployment really uses.
        """
        return decide_scope(
            url=analysis_request.url,
            trigger_type=analysis_request.trigger_type,
            source=analysis_request.source,
            page_signals=(
                analysis_request.page_signals if self.use_page_signals else None
            ),
        )

    def _run_analysis(
        self, analysis_request: PageAnalysisRequest
    ) -> tuple[Dict[str, Any], ProviderCall, ScopeDecision]:
        """Run page analysis on the provider this request routes to.

        ``title`` carries the restaurant name when the Maps extractor resolved
        one: it is the best label the page has, and the prompt reads it as a
        hint rather than as an instruction about the page's kind.

        The scope, by contrast, *is* an instruction about the page's kind — but
        only ever a negative one. ``page_scope`` reads the request's own
        metadata (which flow triggered it, which extractor produced it, what
        host it came from, what the page declares itself to be) and drops a
        branch the request cannot need, leaving the model a single-branch
        question. It is decided from the request alone (``scope_for``); the
        providers only carry it.

        Returns (analysis_result, provider_call, scope) — see ``_dispatch`` for
        the first two. The scope is carried out so the record can say both what
        the model was asked and which exclusion narrowed it.
        """
        decision = self.scope_for(analysis_request)
        page_kinds = decision.page_kinds
        logger.info(
            f"Page-kind scope: {describe_scope(page_kinds)} "
            f"(rule={decision.rule}, "
            f"trigger={analysis_request.trigger_type or 'unknown'}, "
            f"source={analysis_request.source or 'unknown'})"
        )

        analysis_result, provider_call = self._dispatch(
            "analyze_page",
            trigger_type=analysis_request.trigger_type,
            url=analysis_request.url,
            title=analysis_request.restaurant_name or analysis_request.title,
            content=analysis_request.content,
            user_avoided_ingredients=analysis_request.user_avoided_ingredients,
            page_kinds=page_kinds,
        )
        return analysis_result, provider_call, decision

    def _save_record(
        self,
        analysis_request: PageAnalysisRequest,
        analysis: PageAnalysis,
        client_ip: str,
        user_agent: Optional[str],
        provider_call: ProviderCall,
        started_at: Optional[datetime] = None,
        scope: Optional[ScopeDecision] = None,
    ) -> Optional[str]:
        """Persist the analysis to the api_calls collection.

        ``started_at`` is when the request arrived; the record is stamped as
        finished here, so the pair spans everything the server did with it.

        Returns the document ID, which the response passes to the extension as
        ``analysis_id`` so feedback has something to point at. ``None`` when
        there is no database or the write failed — a save that goes wrong is
        still only a logging concern, and the client drops its feedback control
        rather than the answer.
        """
        try:
            if not self.database_service:
                return None

            # The stored branches carry the analysis's own field names, so
            # these are a straight copy rather than a mapping.
            item = analysis.shopping_item
            menu = analysis.menu
            client = self.ip_privacy.describe(client_ip)

            record = APICallRecord(
                ip_hash=client.ip_hash,
                country=client.country,
                user_agent=user_agent,
                content=analysis_request.content,
                item_url=analysis_request.url,
                origin_url=analysis_request.url,  # Using same URL as origin for now
                title=analysis_request.title,
                page_kind=analysis.page_kind,
                shopping_item=(
                    ShoppingItemRecord(**item.model_dump()) if item else None
                ),
                menu=(MenuRecord(**menu.model_dump()) if menu else None),
                language=analysis_request.language,
                summary=analysis.summary,
                service=provider_call.service,
                model=provider_call.model,
                token_usage=provider_call.token_usage,
                # Points at the desktop-server job behind this analysis, so the
                # request and response it actually exchanged can be read back
                # from ``lms_jobs``. Null for Gemini.
                lms_job_id=provider_call.lms_job_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                extension_version=analysis_request.extension_version,
                installation_id=analysis_request.installation_id,
                user_avoided_ingredients=analysis.user_avoided_ingredients,
                trigger_type=analysis_request.trigger_type,
                trigger_element_text=analysis_request.trigger_element_text,
                trigger_element_selector=analysis_request.trigger_element_selector,
                # Stored whether or not they were allowed to narrow the scope,
                # so what a rule *would* have done can be re-derived from real
                # traffic and checked against the verdict the model gave.
                page_signals=analysis_request.page_signals,
                page_scope_rule=scope.rule if scope else None,
                page_scope_kinds=list(scope.page_kinds) if scope else None,
            )
            record_id = self.database_service.save_api_call(record)
            logger.info(f"Saved API call to Firestore with ID: {record_id}")
            return str(record_id)

        except Exception as db_error:
            # Continue with the response even if the database save fails.
            logger.error(f"Failed to save analysis to Firestore: {db_error}")
            return None

    def _save_failure(
        self,
        request_data: Dict[str, Any],
        status: str,
        error: Exception,
        client_ip: str,
        user_agent: Optional[str],
        started_at: Optional[datetime] = None,
    ) -> None:
        """Persist a request that produced no analysis.

        Failures go into the same collection as successes, with the analysis
        fields left empty, ``status`` saying how the request fell over and
        ``error`` saying why — so a page the extension could not get an answer
        for is still there to be found, next to the ones that worked.

        The request is taken as a plain dict rather than a
        ``PageAnalysisRequest``, because the commonest thing to record is a
        payload that would not validate as one; every value is coerced for the
        same reason. After validation the caller passes the validated
        request's own fields, which carry the same names.
        """
        if not self.database_service:
            return

        data = request_data if isinstance(request_data, dict) else {}
        url = strip_url(_as_str(data.get("url")))

        try:
            client = self.ip_privacy.describe(client_ip)
            record = APICallRecord(
                ip_hash=client.ip_hash,
                country=client.country,
                user_agent=user_agent,
                content=_as_str(data.get("content")),
                item_url=url,
                origin_url=url,
                title=_as_str(data.get("title")),
                language=_as_optional_str(data.get("language")),
                status=status,
                error=f"{type(error).__name__}: {error}",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                extension_version=_as_optional_str(data.get("extension_version")),
                installation_id=_as_optional_str(data.get("installation_id")),
                user_avoided_ingredients=_as_str_list(
                    data.get("user_avoided_ingredients")
                ),
                trigger_type=_as_optional_str(data.get("trigger_type")),
                trigger_element_text=_as_optional_str(data.get("trigger_element_text")),
                trigger_element_selector=_as_optional_str(
                    data.get("trigger_element_selector")
                ),
            )
            record_id = self.database_service.save_api_call(record)
            logger.info(f"Saved failed API call to Firestore with ID: {record_id}")

        except Exception as db_error:
            # A failure that cannot be stored is still only a logging concern:
            # the caller is on its way to answering with the original error.
            logger.error(f"Failed to save failed request to Firestore: {db_error}")

    def analyze_page(
        self,
        request_data: Dict[str, Any],
        client_ip: str = "unknown",
        user_agent: str = None,
    ) -> tuple[Dict[str, Any], int]:
        """
        Analyze any webpage: classify it, then answer the question that kind poses

        Args:
            request_data: Dictionary containing the request data
            client_ip: IP address of the client
            user_agent: User agent string from the client

        Returns:
            Tuple of (response dictionary, HTTP status code)
        """
        # Every record is stamped with this, so a stored failure is timed the
        # same way an answer is.
        started_at = datetime.now(timezone.utc)
        try:
            # Validate request data
            try:
                analysis_request = PageAnalysisRequest(**request_data)
            except Exception as e:
                logger.error(f"Validation error: {e}")
                self._save_failure(
                    request_data,
                    STATUS_INVALID_REQUEST,
                    e,
                    client_ip,
                    user_agent,
                    started_at,
                )
                return {"error": f"Invalid request data: {str(e)}"}, 400

            # Stripped before the URL is analyzed, logged or stored.
            requested_url = analysis_request.url
            analysis_request = analysis_request.model_copy(
                update={"url": strip_url(requested_url)}
            )

            logger.info(
                f"Analyzing page from URL: {analysis_request.url} "
                f"(extension version: {analysis_request.extension_version or 'unknown'}, "
                f"installation: {analysis_request.installation_id or 'unknown'})"
            )
            logger.info(
                f"User avoided ingredients: {analysis_request.user_avoided_ingredients}"
            )

            analysis_result, provider_call, scope = self._run_analysis(analysis_request)

            # The providers answer a flat dict, whose field order is what keeps
            # each verdict conditioned on the reasoning above it. Grouping it by
            # page_kind happens here, once, and everything downstream — the
            # response, the record, the log line — reads the nested form.
            analysis = PageAnalysis.from_flat(analysis_result)

            # Saved before the response is built, because the response
            # carries the ID the record was stored under: it is what
            # POST /api/feedback later refers to.
            analysis_id = (
                self._save_record(
                    analysis_request,
                    analysis,
                    client_ip,
                    user_agent,
                    provider_call,
                    started_at,
                    scope,
                )
                if self.enable_database
                else None
            )

            response = PageAnalysisResponse(
                url=requested_url,
                analysis=analysis.to_response_dict(),
                timestamp=analysis_request.timestamp,
                analysis_id=analysis_id,
            )

            usage = provider_call.token_usage
            logger.info(
                f"Analysis completed for {analysis_request.url}: "
                f"page_kind={analysis.page_kind}, "
                f"{len(analysis.menu.items) if analysis.menu else 0} items, "
                f"tokens={usage.total_tokens if usage else 'unknown'}, "
                f"took {(datetime.now(timezone.utc) - started_at).total_seconds():.1f}s"
            )
            return response.model_dump(), 200

        except Exception as e:
            # Everything after validation lands here: both providers failing,
            # or anything else on the way to a response.
            logger.error(f"Error analyzing page: {e}")
            self._save_failure(
                request_data,
                STATUS_ANALYSIS_FAILED,
                e,
                client_ip,
                user_agent,
                started_at,
            )
            return {"error": "Internal server error"}, 500

    def record_feedback(
        self, request_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], int]:
        """Store what a user made of an analysis, on the analysis's own record.

        Nothing about the rater is recorded: the analysis being rated already
        says which installation asked for it, from which IP hash and on which
        extension version, and the rater is that same user.

        Args:
            request_data: Dictionary containing the feedback data

        Returns:
            Tuple of (response dictionary, HTTP status code). 404 means no such
            analysis, which is the only real check standing between this
            endpoint and anyone at all — the document IDs are unguessable, and
            like every other endpoint here this one is unauthenticated.
        """
        try:
            try:
                feedback = AnalysisFeedbackRequest(**request_data)
            except Exception as e:
                logger.error(f"Invalid feedback: {e}")
                return {"error": f"Invalid request data: {str(e)}"}, 400

            if not self.database_service:
                logger.warning("Feedback received with no database configured")
                return {"error": "Feedback is not available"}, 503

            saved = self.database_service.save_feedback(
                feedback.analysis_id, feedback.rating, feedback.comment
            )
            if not saved:
                logger.warning(f"Feedback for unknown analysis {feedback.analysis_id}")
                return {"error": "No such analysis"}, 404

            logger.info(
                f"Feedback '{feedback.rating}' on {feedback.analysis_id} "
                f"({'with' if feedback.comment else 'no'} comment)"
            )
            return {"status": "ok"}, 200

        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return {"error": "Internal server error"}, 500
