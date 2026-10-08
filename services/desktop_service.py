"""Analysis via the local desktop-server (LM Studio) over Firestore.

The desktop-server (https://github.com/ainzzorl/desktop-server) runs LLMs
locally and communicates over Firestore: a client writes a job document to the
``lms_jobs`` collection with ``status: "pending"`` and an opaque OpenAI-format
``request``; a worker on the desktop machine claims it (``pending`` ->
``in_progress``), calls LM Studio, and writes back ``status: "success"`` with an
OpenAI-format ``response`` (or ``status: "failed"`` with ``error``).

This service submits an analysis job and waits for completion using a Firestore
real-time listener, falling back to reading the job document whenever the
listener goes quiet. Availability is detected up front via a heartbeat document
the desktop-server writes periodically, with a job-pickup timeout as a safety
net. ``DesktopUnavailable`` signals the caller to fall back
to Gemini.
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, NamedTuple, Optional, Tuple

from google.cloud import firestore

from models.database_model import ProviderCall, TokenUsage
from services.menu_prompt import includes_user_avoided
from services.page_scope import ALL_PAGE_KINDS, describe_scope
from services.page_prompt import (
    PAGE_CONTENT_TOKEN_LIMIT,
    add_incomplete_answer_warning,
    add_truncation_warning,
    build_page_instructions,
    build_page_content,
    build_page_json_output_instruction,
    estimate_tokens,
    normalize_page_analysis,
)

logger = logging.getLogger(__name__)


# Name this provider is recorded under on an api_calls record.
SERVICE_NAME = "desktop"

# The collection the desktop-server watches for jobs. Read through
# ``jobs_collection_name`` so that everything pointing at it — this service
# writing jobs, the database viewer resolving the job a record links to — agrees
# on which collection that is.
DEFAULT_JOBS_COLLECTION = "lms_jobs"

# Job statuses, mirroring desktop-server's worker.
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"

# System prompt steering the local model to produce real, complete field
# content. Any OpenAI ``response_format`` json_schema (even ``strict: false``)
# suppresses gpt-oss's reasoning channel in LM Studio, after which it
# degenerates the trailing string fields into placeholder/whitespace junk. We
# therefore request the JSON via the prompt (see
# ``build_page_json_output_instruction``) and run in text mode, keeping the
# reasoning step — and the field content — intact.
#
# The second sentence guards the failure that dominates the menu branch:
# padding the output with dishes that were never on the page.
PAGE_SYSTEM_PROMPT = (
    "You are an analyst advising a vegan reader. Reason carefully about the "
    "page, then output the requested JSON object. Describe only what is "
    "actually in the content — never invent a product or a dish, and never "
    "guess an ingredient that is neither stated nor inherent. Every string "
    'field must contain complete, real content — never placeholders like "...", '
    '"analysis", whitespace, or empty strings.'
)

# Context window (in tokens) the desktop-server loads the model with; prompt and
# completion together have to fit inside it. openai/gpt-oss-20b runs with 16384
# there — see LMS_CONTEXT_LENGTH in tests/eval/run.sh, which is the same knob.
#
# It is a small window for a prompt this size, so what is left for the page is
# derived from it per request (see ``_page_content_budget``) rather than written
# down as a second constant that could drift out of agreement with it.
DEFAULT_CONTEXT_LENGTH = 16384

# Completion reserve for a request that cannot be a menu, in place of
# LMS_MAX_TOKENS. That cap is sized for a long menu, while an item answer is a
# handful of fields: 106–405 tokens in the eval, reasoning included, at low
# effort. The difference goes to the page. A menu still reaches this service as Gemini's fallback,
# and keeps the full reserve.
ITEM_COMPLETION_RESERVE = 2048

# Slack left for what the character-based estimate cannot see: the chat
# template's role markers and control tokens, and the drift of CHARS_PER_TOKEN
# on any single page.
PROMPT_OVERHEAD_TOKENS = 256

# Reasoning-effort levels LMS_REASONING_EFFORT accepts. The value is passed
# through to LM Studio as the OpenAI ``reasoning_effort`` field: a model whose
# chat template reads it (gpt-oss) spends more of the completion budget thinking
# the higher it is, buying accuracy with latency; other models ignore it.
REASONING_EFFORTS = ("low", "medium", "high")

# What an unset LMS_REASONING_EFFORT means. Low is what the pages we send want:
# the analysis is a read-off-the-page judgement rather than a puzzle, and
# reasoning tokens come out of the same completion cap as the JSON, so thinking
# longer mostly costs latency and risks the cap.
DEFAULT_REASONING_EFFORT = "low"


def jobs_collection_name() -> str:
    """The Firestore collection desktop-server jobs live in."""
    return os.getenv("LMS_JOBS_COLLECTION", DEFAULT_JOBS_COLLECTION)


def configured_reasoning_effort() -> str:
    """The reasoning effort to request, defaulting to DEFAULT_REASONING_EFFORT.

    Rejects an unrecognized value here rather than letting LM Studio refuse the
    request one page into a run.
    """
    value = os.getenv("LMS_REASONING_EFFORT", "").strip().lower()
    if not value:
        return DEFAULT_REASONING_EFFORT
    if value not in REASONING_EFFORTS:
        raise ValueError(
            f"LMS_REASONING_EFFORT={value!r} is not one of "
            f"{', '.join(REASONING_EFFORTS)}"
        )
    return value


class JobResult(NamedTuple):
    """What one desktop-server job answered.

    ``cut_short`` says the completion ran into the token cap mid-answer, so
    ``payload`` holds only the part the model finished writing.
    """

    payload: Dict[str, Any]
    token_usage: Optional[TokenUsage]
    job_id: str
    cut_short: bool = False


def _close_truncated_json(text: str) -> Optional[str]:
    """Close a JSON object the model stopped writing part-way through.

    Rewinds to the last value it finished — an array element, an object member —
    and closes every container still open there, dropping the half-written tail.
    A menu cut off after 40 of 57 dishes then reads back as those 40 dishes.

    Returns None if nothing was finished (or the text was never JSON), which is
    the caller's signal that there is nothing to keep.
    """
    stack: list[list] = []  # [container char, awaiting a key] per open container
    cut: Optional[Tuple[int, str]] = None  # where to cut, and what to close there

    def mark(index: int) -> None:
        """Record that a complete value ends at ``index``."""
        nonlocal cut
        closers = "".join("}" if frame[0] == "{" else "]" for frame in reversed(stack))
        cut = (index, closers)

    i, n = 0, len(text)
    while i < n:
        if cut is not None and not stack:
            break  # the top-level value closed; anything after it is prose
        char = text[i]
        if char == '"':
            is_key = bool(stack) and stack[-1][0] == "{" and stack[-1][1]
            i = _skip_string(text, i)
            if i is None:  # unterminated: the string itself is what got cut off
                break
            if stack and not is_key:
                mark(i)
        elif char in "{[":
            stack.append([char, char == "{"])
            i += 1
        elif char in "}]":
            if not stack:
                break
            stack.pop()
            i += 1
            mark(i)
        elif char == ":":
            if stack and stack[-1][0] == "{":
                stack[-1][1] = False
            i += 1
        elif char == ",":
            if stack and stack[-1][0] == "{":
                stack[-1][1] = True
            i += 1
        elif char.isspace():
            i += 1
        else:
            # A bare literal (number, true, false, null). Only complete once
            # something delimits it — at the end of the text it may itself be
            # the value that got cut in half.
            end = i
            while end < n and text[end] not in ',{}[]" \t\r\n':
                end += 1
            if end >= n:
                break
            if stack:
                mark(end)
            i = end

    if cut is None:
        return None
    index, closers = cut
    return text[:index] + closers


def _skip_string(text: str, start: int) -> Optional[int]:
    """Return the index just past the JSON string starting at ``start``, or None
    if it is never closed."""
    i = start + 1
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == '"':
            return i + 1
        i += 1
    return None


class DesktopUnavailable(Exception):
    """Raised when the desktop-server is not reachable/processing jobs.

    Signals the caller to fall back to another analysis provider.
    """


class DesktopAnswerCutShort(Exception):
    """Raised when the model ran into its completion cap mid-answer.

    Signals the caller to fall back to another analysis provider, which can
    answer for the whole page rather than the part this one got through. It
    carries that part — normalized and flagged for the reader, exactly as
    ``analyze_page`` would have returned it — so a caller with nowhere to fall
    back to can still answer with it instead of failing the page outright.
    """

    def __init__(
        self,
        message: str,
        analysis: Dict[str, Any],
        provider_call: ProviderCall,
    ):
        super().__init__(message)
        self.analysis = analysis
        self.provider_call = provider_call


class DesktopService:
    def __init__(self):
        """Initialize the desktop-server analysis client."""
        self.model_name = os.getenv("LMS_MODEL", "openai/gpt-oss-20b")
        self.jobs_collection = jobs_collection_name()
        self.status_collection = os.getenv(
            "DESKTOP_STATUS_COLLECTION", "desktop_server_status"
        )
        self.status_doc = os.getenv("DESKTOP_STATUS_DOC", "worker")
        # The model's loaded context window, and the output-token cap carved out
        # of it. Whatever is left over after the cap is what the prompt may use;
        # the JSON analysis itself needs far less than the cap.
        self.context_length = int(
            os.getenv("LMS_CONTEXT_LENGTH", str(DEFAULT_CONTEXT_LENGTH))
        )
        self.max_tokens = int(os.getenv("LMS_MAX_TOKENS", "8192"))
        self.temperature = float(os.getenv("LMS_TEMPERATURE", "0.4"))
        # Reasoning tokens come out of the same completion cap as the JSON, so
        # raising this spends part of ``max_tokens`` on thinking.
        self.reasoning_effort = configured_reasoning_effort()
        self.pickup_timeout = float(os.getenv("DESKTOP_PICKUP_TIMEOUT_SECONDS", "10"))
        self.completion_timeout = float(
            os.getenv("DESKTOP_COMPLETION_TIMEOUT_SECONDS", "180")
        )
        # How long the job listener may stay silent before the document is read
        # directly instead — see ``_wait_for_completion``.
        self.listener_check_interval = float(
            os.getenv("DESKTOP_LISTENER_CHECK_SECONDS", "10")
        )
        self.heartbeat_max_age = float(
            os.getenv("DESKTOP_HEARTBEAT_MAX_AGE_SECONDS", "30")
        )

        # google.cloud.firestore picks up the emulator automatically when
        # FIRESTORE_EMULATOR_HOST is set, and Application Default Credentials in
        # production. Project comes from GOOGLE_CLOUD_PROJECT (or the emulator).
        self.db = firestore.Client()

        logger.info(
            "Desktop service initialized (model=%s, jobs=%s, reasoning_effort=%s)",
            self.model_name,
            self.jobs_collection,
            self.reasoning_effort,
        )

    def is_available(self) -> bool:
        """Return True if the desktop-server heartbeat is fresh."""
        try:
            snap = (
                self.db.collection(self.status_collection)
                .document(self.status_doc)
                .get()
            )
            if not snap.exists:
                return False
            last_heartbeat = (snap.to_dict() or {}).get("last_heartbeat")
            if last_heartbeat is None:
                return False
            age = (datetime.now(timezone.utc) - last_heartbeat).total_seconds()
            return age <= self.heartbeat_max_age
        except Exception as e:
            logger.warning(f"Failed to read desktop-server heartbeat: {e}")
            return False

    def _build_chat_request(
        self, system_content: str, user_content: str, reserve: int
    ) -> Dict[str, Any]:
        """Assemble the OpenAI-format request the desktop-server forwards on.

        Static instructions (persona + task + output format) go in the system
        role; the variable page data goes in the user role.

        Text mode (no ``response_format``): the JSON shape is steered via the
        prompt instead, which keeps gpt-oss's reasoning active and prevents the
        trailing fields from degenerating into placeholder junk. See
        ANALYSIS_SYSTEM_PROMPT.
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        request: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self._completion_cap(messages, reserve),
            "temperature": self.temperature,
        }
        request["reasoning_effort"] = self.reasoning_effort
        return request

    def _completion_reserve(self, page_kinds: tuple[str, ...]) -> int:
        """Completion tokens the page budget is planned around for this scope."""
        if "restaurant_menu" in page_kinds:
            return self.max_tokens
        return min(self.max_tokens, ITEM_COMPLETION_RESERVE)

    def _completion_cap(self, messages: list[dict], reserve: int) -> int:
        """How many tokens this request's completion may use.

        ``reserve`` is the floor the page budget is planned against (see
        ``_page_content_budget``), not the whole story: a page that came in
        under that budget — most of them — leaves the rest of the context window
        unused, and the completion is the only thing that can spend it. That
        matters because gpt-oss reasons its way through most of the budget
        before writing any JSON, so on a long menu the slack is dishes.

        Measured against the same conservative estimate the page budget uses, so
        the two cannot disagree about what fits: the floor is therefore only a
        guard, never the binding value.
        """
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        window_left = self.context_length - prompt_tokens - PROMPT_OVERHEAD_TOKENS
        return max(reserve, window_left)

    def build_page_request(
        self,
        url: str,
        title: str,
        content: str,
        user_avoided_ingredients: Optional[list[str]] = None,
        page_kinds: tuple[str, ...] = ALL_PAGE_KINDS,
    ) -> Dict[str, Any]:
        """Build the OpenAI-format request the desktop-server forwards to LM Studio.

        This is the single source of truth for the payload, so a direct LM Studio
        call (e.g. the curl emitted on test failure) matches the real request.
        """
        request, _ = self._build_page_request(
            url=url,
            title=title,
            content=content,
            user_avoided_ingredients=user_avoided_ingredients,
            page_kinds=page_kinds,
        )
        return request

    def _page_content_budget(
        self, system_content: str, frame: str, reserve: int
    ) -> int:
        """Tokens this model can spend on the page text itself.

        LM Studio silently drops whatever does not fit the context the model was
        loaded with, so a prompt that overruns it would come back analyzed in
        part and presented as whole. The page is the only part that can give:
        the instructions are fixed, and the completion cap has to stay — a menu
        of 57 dishes has come back as ~4k tokens of JSON, before the reasoning
        the model emits ahead of it.

        So the page gets what is left after ``reserve``, and never more than
        the budget the rest of the app works to. At the defaults (16384-token
        window) that is around 1800 tokens with both branches and the 8192 menu
        reserve, and the full PAGE_CONTENT_TOKEN_LIMIT for an item-or-other
        request.
        """
        budget = (
            self.context_length
            - reserve
            - PROMPT_OVERHEAD_TOKENS
            - estimate_tokens(system_content)
            - estimate_tokens(frame)
        )
        if budget <= 0:
            raise RuntimeError(
                f"the desktop-server instructions (~{estimate_tokens(system_content)} "
                f"tokens) plus the {reserve}-token completion cap already "
                f"fill the model's {self.context_length}-token context window, "
                f"leaving no room for the page; raise LMS_CONTEXT_LENGTH or lower "
                f"LMS_MAX_TOKENS"
            )
        return min(PAGE_CONTENT_TOKEN_LIMIT, budget)

    def _build_page_request(
        self,
        url: str,
        title: str,
        content: str,
        user_avoided_ingredients: Optional[list[str]] = None,
        page_kinds: tuple[str, ...] = ALL_PAGE_KINDS,
    ) -> Tuple[Dict[str, Any], bool]:
        """Build the page request, also reporting whether the page had to be cut.

        The page gets PAGE_CONTENT_TOKEN_LIMIT or whatever this model's context
        window leaves over, whichever is less — see ``_page_content_budget``.

        This is where a narrowed ``page_kinds`` pays off most: the instructions
        and the JSON field list are both charged against the same context window
        as the page, so dropping the unused branch hands its tokens straight to
        ``_page_content_budget`` — roughly half again as much page text on the
        default 16384-token window.
        """
        instructions, has_user_avoided = build_page_instructions(
            user_avoided_ingredients=user_avoided_ingredients,
            page_kinds=page_kinds,
        )
        instructions += build_page_json_output_instruction(
            include_user_avoided=has_user_avoided, page_kinds=page_kinds
        )
        system_content = PAGE_SYSTEM_PROMPT + "\n\n" + instructions

        reserve = self._completion_reserve(page_kinds)
        budget = self._page_content_budget(
            system_content,
            build_page_content(url=url, title=title, content="")[0],
            reserve,
        )
        user_content, truncated = build_page_content(
            url=url, title=title, content=content, token_limit=budget
        )

        if truncated:
            logger.warning(
                f"Truncated {url} to the {budget}-token page budget; the "
                f"analysis will not cover all of it"
            )

        request = self._build_chat_request(
            system_content=system_content,
            user_content=user_content,
            reserve=reserve,
        )
        return request, truncated

    def analyze_page(
        self,
        url: str,
        title: str,
        content: str,
        user_avoided_ingredients: Optional[list[str]] = None,
        page_kinds: tuple[str, ...] = ALL_PAGE_KINDS,
    ) -> Tuple[Dict[str, Any], ProviderCall]:
        """Analyze any webpage via the desktop-server.

        A page past PAGE_CONTENT_TOKEN_LIMIT is analyzed in part, and the result
        says so (see ``add_truncation_warning``).

        ``page_kinds`` narrows the question to the branches this request could
        need — see ``page_scope``.

        Returns:
            Tuple of (analysis results, how the call was served). The usage on
            the ``ProviderCall`` is ``None`` when the response did not report
            any; its ``lms_job_id`` points at the job this ran as, so the stored
            record can be read back against the desktop-server's own document.

        Raises:
            DesktopUnavailable: if the desktop-server is offline or doesn't pick
                up the job in time (the caller should fall back to Gemini).
            DesktopAnswerCutShort: if the model ran out of completion budget
                part-way through its answer. What it did finish rides along on
                the exception, for a caller with no other provider to fall back
                to.
        """
        request, truncated = self._build_page_request(
            url=url,
            title=title,
            content=content,
            user_avoided_ingredients=user_avoided_ingredients,
            page_kinds=page_kinds,
        )

        result = self._run_job(request, url)
        analysis = normalize_page_analysis(
            result.payload,
            include_user_avoided=includes_user_avoided(user_avoided_ingredients),
            page_kinds=page_kinds,
        )
        if truncated:
            add_truncation_warning(analysis)

        provider_call = ProviderCall(
            service=SERVICE_NAME,
            model=self.model_name,
            token_usage=result.token_usage,
            lms_job_id=result.job_id,
        )

        if result.cut_short:
            add_incomplete_answer_warning(analysis)
            raise DesktopAnswerCutShort(
                f"job {result.job_id} ran out of completion budget after "
                f"{len(analysis['items'])} item(s)",
                analysis=analysis,
                provider_call=provider_call,
            )

        logger.info(
            f"Successfully analyzed {url} via desktop-server "
            f"(scope={describe_scope(page_kinds)}, "
            f"page_kind={analysis['page_kind']}, {len(analysis['items'])} items"
            f"{', truncated' if truncated else ''})"
        )
        return analysis, provider_call

    def _run_job(self, request: Dict[str, Any], url: str) -> JobResult:
        """Submit a job to the desktop-server and return what it answered.

        Raises:
            DesktopUnavailable: if the desktop-server is offline or doesn't pick
                up the job in time (the caller should fall back to Gemini).
        """
        if not self.is_available():
            raise DesktopUnavailable("desktop-server heartbeat is stale or missing")

        doc_ref = self.db.collection(self.jobs_collection).document()
        doc_ref.set(
            {
                "model": self.model_name,
                "status": STATUS_PENDING,
                "created_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP,
                "request": request,
            }
        )
        logger.info(f"Submitted desktop-server job {doc_ref.id} for {url}")

        job = self._wait_for_completion(doc_ref)

        if job.get("status") == STATUS_FAILED:
            raise RuntimeError(
                f"desktop-server job {doc_ref.id} failed: {job.get('error')}"
            )

        payload, cut_short = self._parse_response(job)
        return JobResult(
            payload=payload,
            token_usage=self._token_usage(job),
            job_id=doc_ref.id,
            cut_short=cut_short,
        )

    def _wait_for_completion(self, doc_ref) -> Dict[str, Any]:
        """Block on a Firestore listener until the job reaches a terminal state.

        Distinguishes "never picked up" (-> DesktopUnavailable) from "picked up
        but never finished" (-> timeout error).

        The listener is the fast path, not the only one: whenever it has been
        silent for ``listener_check_interval`` the document is read directly.
        A stream that stops delivering — which several listeners opened at once
        have been seen to do, the client logging a ``ListenResponse`` decode
        error — is otherwise indistinguishable from a job nobody picked up, and
        a job that finished seconds in would be given up on, or waited out for
        the whole completion timeout.
        """
        state: Dict[str, Any] = {
            "job": None,
            "seen_in_progress": False,
            "terminal": False,
        }
        done = threading.Event()

        def callback(snapshots, changes, read_time):
            for snap in snapshots:
                if not snap.exists:
                    continue
                data = snap.to_dict() or {}
                state["job"] = data
                status = data.get("status")
                if status in (STATUS_IN_PROGRESS, STATUS_SUCCESS, STATUS_FAILED):
                    state["seen_in_progress"] = True
                if status in (STATUS_SUCCESS, STATUS_FAILED):
                    state["terminal"] = True
                    done.set()

        watch = doc_ref.on_snapshot(callback)
        try:
            started = time.monotonic()
            # Never wait past the point a decision is due.
            interval = min(
                self.listener_check_interval,
                self.pickup_timeout,
                self.completion_timeout,
            )
            while True:
                if done.wait(interval):
                    return state["job"]

                # Nothing from the listener this round: ask the document itself
                # rather than reading silence as an answer.
                job = self._read_job(doc_ref)
                status = job.get("status")
                if status in (STATUS_SUCCESS, STATUS_FAILED):
                    return job
                if status == STATUS_IN_PROGRESS:
                    state["seen_in_progress"] = True

                waited = time.monotonic() - started
                if not state["seen_in_progress"] and waited >= self.pickup_timeout:
                    raise DesktopUnavailable(
                        f"job {doc_ref.id} not picked up within {self.pickup_timeout}s"
                    )
                if waited >= self.completion_timeout:
                    raise RuntimeError(
                        f"desktop-server job {doc_ref.id} timed out after "
                        f"{self.completion_timeout}s"
                    )
        finally:
            watch.unsubscribe()

    def _read_job(self, doc_ref) -> Dict[str, Any]:
        """Read the job document directly. Never raises: this is the fallback
        for a quiet listener, and a failed read just means waiting a bit more."""
        try:
            snap = doc_ref.get()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to read desktop-server job {doc_ref.id}: {e}")
            return {}
        return (snap.to_dict() or {}) if snap.exists else {}

    def _parse_response(self, job: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Extract and parse the structured JSON payload from an OpenAI response.

        Returns ``(payload, cut_short)``: ``cut_short`` says the model ran into
        the completion cap mid-answer, so what came back is only as far as it
        got — a long menu is the case that reaches it, since gpt-oss spends much
        of the cap reasoning before writing any JSON at all.

        Defaulting is left to the caller (``normalize_page_analysis``), which
        knows which branch of the response the page_kind called for.
        """
        response = job.get("response") or {}
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("desktop-server response missing choices")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        cut_short = choice.get("finish_reason") == "length"
        cap = (job.get("request") or {}).get("max_tokens", self.max_tokens)

        if not content:
            if cut_short:
                raise RuntimeError(
                    f"desktop-server response hit the {cap}-token completion "
                    f"cap before writing any answer (all of it went to "
                    f"reasoning); lower LMS_REASONING_EFFORT or raise "
                    f"LMS_MAX_TOKENS"
                )
            raise RuntimeError("desktop-server response missing message content")

        try:
            # A whole answer — including one the cap stopped only afterwards,
            # on trailing prose.
            return json.loads(self._extract_json(content)), False
        except (json.JSONDecodeError, RuntimeError):
            if not cut_short:
                raise

        # Cut off mid-JSON: close what the model did finish writing rather than
        # losing the page over the part it did not. The caller marks the result
        # as incomplete for the reader.
        start = content.find("{")
        repaired = _close_truncated_json(content[start:]) if start != -1 else None
        payload = None
        if repaired is not None:
            try:
                payload = json.loads(repaired)
            except json.JSONDecodeError:
                payload = None
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"desktop-server response hit the {cap}-token completion cap "
                f"before any of its answer could be salvaged; lower "
                f"LMS_REASONING_EFFORT or raise LMS_MAX_TOKENS"
            )
        logger.warning(
            "desktop-server response hit the %s-token completion cap; keeping "
            "the %s field(s) it finished writing",
            cap,
            len(payload),
        )
        return payload, True

    @staticmethod
    def _token_usage(job: Dict[str, Any]) -> Optional[TokenUsage]:
        """Normalize the OpenAI-format ``usage`` block of a completed job.

        Read defensively and never raise: LM Studio may omit the block (and
        reasoning tokens are only broken out by some models), and a missing
        token count must not cost us an otherwise good analysis.
        """
        usage = (job.get("response") or {}).get("usage")
        if not isinstance(usage, dict) or not usage:
            return None

        def count(source: Dict[str, Any], name: str) -> Optional[int]:
            try:
                return int(source[name])
            except (KeyError, TypeError, ValueError):
                return None

        details = usage.get("completion_tokens_details")
        return TokenUsage(
            prompt_tokens=count(usage, "prompt_tokens"),
            completion_tokens=count(usage, "completion_tokens"),
            total_tokens=count(usage, "total_tokens"),
            reasoning_tokens=(
                count(details, "reasoning_tokens")
                if isinstance(details, dict)
                else None
            ),
        )

    @staticmethod
    def _extract_json(content: str) -> str:
        """Extract the JSON object from a text-mode response.

        Without an OpenAI ``response_format`` schema the model returns plain
        text, which is normally just the JSON object but may occasionally be
        wrapped in markdown fences or surrounded by stray prose. Pull out the
        outermost ``{...}`` so ``json.loads`` sees clean JSON.
        """
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise RuntimeError("desktop-server response did not contain a JSON object")
        return content[start : end + 1]
