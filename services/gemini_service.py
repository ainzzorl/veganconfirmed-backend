import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import os
import logging
from typing import Dict, Any, Optional, Tuple
import json

from models.database_model import ProviderCall, TokenUsage
from services.page_scope import ALL_PAGE_KINDS, describe_scope
from services.page_prompt import (
    add_truncation_warning,
    build_page_instructions,
    build_page_content,
    build_gemini_page_schema,
    normalize_page_analysis,
    error_analysis,
)

logger = logging.getLogger(__name__)

# Name this provider is recorded under on an api_calls record.
SERVICE_NAME = "gemini"


def _token_usage(response: Any) -> Optional[TokenUsage]:
    """Normalize Gemini's ``usage_metadata`` into a ``TokenUsage``.

    Read defensively: the field is absent on responses that did not come from a
    live API call (the eval's on-disk cache replays a stand-in that carries only
    the text), and usage must never be the reason an analysis fails.
    """
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None

    def count(name: str) -> Optional[int]:
        try:
            return int(getattr(metadata, name, None))
        except (TypeError, ValueError):
            return None

    return TokenUsage(
        prompt_tokens=count("prompt_token_count"),
        completion_tokens=count("candidates_token_count"),
        total_tokens=count("total_token_count"),
        reasoning_tokens=count("thoughts_token_count"),
    )


class GeminiService:
    def __init__(self):
        """Initialize Gemini service with API key"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=api_key)
        # The model is built per request so the analysis instructions can be
        # passed as a system_instruction (it varies with user_avoided_ingredients).
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

        logger.info(
            f"Gemini service initialized successfully (model: {self.model_name})"
        )

    def analyze_page(
        self,
        url: str,
        title: str,
        content: str,
        user_avoided_ingredients: list[str] | None = None,
        page_kinds: tuple[str, ...] = ALL_PAGE_KINDS,
    ) -> Tuple[Dict[str, Any], ProviderCall]:
        """
        Analyze any webpage: classify it, then answer the question that kind poses

        Args:
            url: The webpage URL
            title: The page title (or restaurant name, when the caller knows one)
            content: The extracted text/markdown content of the page
            user_avoided_ingredients: List of ingredients and materials the user considers non-vegan, in addition to the common animal-derived materials.
            page_kinds: The kinds this request could be — see ``page_scope``.
                Narrowing it prunes the other branch from both the instructions
                and the response schema.

        Returns:
            Tuple of (analysis results, how the call was served). The usage on
            the ``ProviderCall`` is ``None`` when the response did not report
            any.
        """
        try:
            instructions, has_user_avoided = build_page_instructions(
                user_avoided_ingredients=user_avoided_ingredients,
                page_kinds=page_kinds,
            )
            page_content, truncated = build_page_content(
                url=url, title=title, content=content
            )

            # Build schema based on whether user_avoided_ingredients is needed
            response_schema = build_gemini_page_schema(
                include_user_avoided=has_user_avoided, page_kinds=page_kinds
            )

            # Static instructions go in the system_instruction; the variable page
            # data is the request content.
            model = genai.GenerativeModel(
                self.model_name, system_instruction=instructions
            )

            # Generate response with structured output
            response = model.generate_content(
                page_content,
                generation_config=GenerationConfig(
                    # Extraction and classification, not writing: the same page
                    # should not swing between vegan and not-vegan across
                    # requests. Gemini's default of 1.0 left the eval's
                    # borderline cases behaving like a coin flip.
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

            analysis = normalize_page_analysis(
                json.loads(response.text),
                include_user_avoided=has_user_avoided,
                page_kinds=page_kinds,
            )
            if truncated:
                add_truncation_warning(analysis)

            logger.info(
                f"Successfully analyzed {url} "
                f"(scope={describe_scope(page_kinds)}, "
                f"page_kind={analysis['page_kind']}, {len(analysis['items'])} items)"
            )
            return analysis, self._provider_call(_token_usage(response))

        except Exception as e:
            logger.error(f"Error analyzing page with Gemini: {e}")
            return (
                error_analysis("Unable to analyze this page due to a technical error"),
                self._provider_call(None),
            )

    def _provider_call(self, usage: Optional[TokenUsage]) -> ProviderCall:
        """Describe the call this service just served.

        There is no job document to point at: Gemini is called directly, so
        ``lms_job_id`` stays unset (unlike the desktop-server, whose analyses go
        through ``lms_jobs``).
        """
        return ProviderCall(
            service=SERVICE_NAME, model=self.model_name, token_usage=usage
        )
