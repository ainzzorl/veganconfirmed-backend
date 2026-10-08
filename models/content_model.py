from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional

from models.analysis_model import PageAnalysis

# The two ratings a user can give. Kept a plain set of strings rather than an
# enum, matching how verdicts are typed elsewhere.
FEEDBACK_RATINGS = {"up", "down"}

# Cap on a feedback comment. The endpoint is unauthenticated like every other
# one here, so the only thing standing between it and an arbitrarily large
# write is this number.
COMMENT_CHAR_LIMIT = 2000


class PageAnalysisRequest(BaseModel):
    """Request model for analyzing any page.

    One envelope for every kind of page, because the caller no longer decides
    what kind it is — the model does. The optional fields are hints the
    extension supplies when it happens to know them: ``trigger_*`` come from
    the add-to-cart path, ``place_id``/``restaurant_name`` from the Google Maps
    extractor. Everything but the page itself is optional so that a payload
    from either extractor validates.
    """

    url: str = Field(..., description="URL of the webpage")
    content: str = Field(..., description="Extracted text/markdown content of the page")
    timestamp: str = Field(
        ..., description="ISO timestamp of when content was extracted"
    )
    title: str = Field(
        default="",
        description="Page title, or the restaurant name when the extractor resolved one",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language of the web page (from the <html lang> attribute)",
    )
    user_avoided_ingredients: Optional[list[str]] = Field(
        default=[], description="List of ingredients the user wants to avoid"
    )
    trigger_type: Optional[str] = Field(
        default=None, description="Type of trigger: 'manual' or 'automatic'"
    )
    trigger_element_text: Optional[str] = Field(
        default=None,
        description="Text of the element that triggered analysis (if automatic)",
    )
    trigger_element_selector: Optional[str] = Field(
        default=None,
        description="CSS selector of the element that triggered analysis (if automatic)",
    )
    place_id: Optional[str] = Field(
        default=None,
        description="Stable identifier for the place (e.g. the Google Maps feature ID), used as the client-side cache key",
    )
    source: Optional[str] = Field(
        default=None,
        description=(
            "Which extractor produced the content: 'page', "
            "'google_maps_menu_tab', 'google_maps_panel', or 'pdf' (a PDF "
            "menu's text layer; the extension never sends one without text)"
        ),
    )
    page_signals: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "What the page declares about its own kind: an 'og_type', the "
            "schema.org 'schema_types' of its JSON-LD and microdata, and the "
            "third-party 'asset_hosts' it loads from. Read by "
            "``page_scope``, which owns every rule written against it — the "
            "extension only reports, so the rules can change without an "
            "extension release."
        ),
    )

    @field_validator("page_signals", mode="before")
    @classmethod
    def _ignore_unusable_signals(cls, value: Any) -> Optional[Dict[str, Any]]:
        """A hint in the wrong shape means "the page declared nothing".

        It is advisory and client-supplied, so rejecting it would turn a page we
        could still have analyzed into a 400. What is inside the dict is left
        exactly as it arrived; ``page_scope`` is equally forgiving about reading
        it.
        """
        return value if isinstance(value, dict) else None

    restaurant_name: Optional[str] = Field(
        default=None,
        description="Restaurant name, when the extractor resolved one from the page",
    )
    extension_version: Optional[str] = Field(
        default=None,
        description="Version of the extension that sent the request, from its manifest. Absent on requests from versions predating this field.",
    )
    installation_id: Optional[str] = Field(
        default=None,
        description=(
            "Random ID the extension mints on first use and keeps in local storage, "
            "grouping calls from one installation. Client-supplied and reset by a "
            "reinstall, so it is not an identity and must not be trusted as one."
        ),
    )


class PageAnalysisResponse(BaseModel):
    """Response model for page analysis.

    ``analysis`` is validated rather than passed through as a bare dict, so the
    contract the extension reads is enforced here and not only described in the
    README. It is built by ``PageAnalysis.to_response_dict``, which adds the
    flat aliases the released extension still reads on top of the model's own
    fields — hence ``PageAnalysis``'s ``extra="allow"``.
    """

    url: str = Field(..., description="URL of the analyzed webpage")
    analysis: PageAnalysis = Field(
        ...,
        description="Analysis results: a page_kind plus the fields that kind calls for",
    )
    timestamp: str = Field(..., description="Timestamp of the analysis")
    analysis_id: Optional[str] = Field(
        default=None,
        description=(
            "The api_calls document this analysis was stored as, and what "
            "POST /api/feedback refers to. Null when the database is off or "
            "the write failed; the extension hides its feedback control then."
        ),
    )


class AnalysisFeedbackRequest(BaseModel):
    """Request model for feedback on an analysis.

    Carries no identifiers of its own: the analysis it names already records
    who asked for it, and the rater is that same user. Sending them again
    would only duplicate the record it is attached to.

    ``rating`` arrives on its own the moment the user clicks a thumb; a
    comment, if one is written at all, arrives in a second call with the same
    rating and is merged onto the same record.
    """

    analysis_id: str = Field(
        ..., description="The api_calls document the feedback is about"
    )
    rating: str = Field(..., description="Verdict on the analysis: 'up' or 'down'")
    comment: Optional[str] = Field(
        default=None,
        description=(
            "What the user wrote, if anything. Capped at COMMENT_CHAR_LIMIT "
            "characters — free text from an unauthenticated endpoint."
        ),
    )

    @field_validator("rating")
    @classmethod
    def check_rating(cls, v: str) -> str:
        if v not in FEEDBACK_RATINGS:
            raise ValueError(f"rating must be one of {sorted(FEEDBACK_RATINGS)}")
        return v

    @field_validator("comment")
    @classmethod
    def check_comment(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if len(v) > COMMENT_CHAR_LIMIT:
            raise ValueError(f"comment must be at most {COMMENT_CHAR_LIMIT} characters")
        return v or None
