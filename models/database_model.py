from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Tokens an analysis provider reported for a single LLM call.

    Both providers report usage, in their own shape — Gemini's
    ``usage_metadata`` and the desktop-server's OpenAI-format ``usage`` — so it
    is normalized here into one set of names. Fields are optional because a
    provider may omit them (older desktop-server responses, cached eval
    replays); ``None`` means "not reported", which is not the same as zero.
    """

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # Includes reasoning/thinking tokens, which the providers count towards the
    # total but do not always break out per field.
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None


class ProviderCall(BaseModel):
    """Which provider served one analysis, what it cost, and where it ran.

    Assembled by the provider that answered — it is the only place that knows
    the model it used and, for the desktop-server, the job it went through — and
    unpacked onto the record in ``AnalysisCore._save_record``.
    """

    service: str  # "desktop" or "gemini"
    model: str
    token_usage: Optional[TokenUsage] = None
    # Document ID of the desktop-server job this analysis ran as, in the
    # ``lms_jobs`` collection (see DesktopService). None for Gemini, which has
    # no job to point at.
    lms_job_id: Optional[str] = None


# The two per-kind branches, as stored. Field for field these mirror
# ``analysis_model``'s ``ShoppingItemAnalysis`` and ``MenuAnalysis`` — a stored
# verdict and a returned one are the same thing, and giving them two vocabularies
# only bought a rename on the way in.
#
# They are separate classes because they are read under different rules. The
# analysis models are strict: they describe what this code is generating right
# now, so an enum is an enum. These describe data at rest, which some older or
# newer version of the prompt may have written, so the enum-valued fields are
# plain strings — a record naming a verdict this build has never heard of is
# still a record worth reading back, not an exception. Same reason
# ``APICallRecord`` allows extra fields.


class ShoppingItemRecord(BaseModel):
    """The product verdict, as stored — null on a record of any other kind."""

    explicit_ingredients: list[str] = []
    inferred_ingredients: list[str] = []
    animal_derived_ingredients: list[str] = []
    is_vegan: Optional[bool] = None
    is_cruelty_free: Optional[bool] = None
    cruelty_free_explanation: Optional[str] = None
    confidence_level: Optional[str] = None


class MenuRecord(BaseModel):
    """The menu verdict, as stored — null on a record of any other kind.

    ``items`` holds the per-dish verdicts as the analysis returned them (name,
    section, verdict, reason, explicit_ingredients, inferred_ingredients, and
    user_avoided_ingredients when the request carried an avoid-list), already
    bounded by ``menu_prompt.MAX_MENU_ITEMS`` — which keeps the list far inside
    Firestore's 1MB document limit.
    """

    restaurant_name: Optional[str] = None
    vegan_friendliness: Optional[str] = None
    items: list[dict[str, Any]] = []


# How a request ended, stored on every record. A failure is kept alongside the
# successes rather than dropped: what the extension asked for is worth having
# whether or not an answer came back.
STATUS_OK = "ok"  # Analyzed and answered.
STATUS_INVALID_REQUEST = "invalid_request"  # Payload did not validate.
STATUS_ANALYSIS_FAILED = "analysis_failed"  # Every provider errored.


class FeedbackRecord(BaseModel):
    """What a user said about an analysis, stored as a map on its record.

    It hangs off the analysis rather than living in its own collection because
    an analysis belongs to exactly one request from one installation, so there
    is only ever one party in a position to rate it — and no identity is
    repeated here that the record it is attached to does not already hold.

    ``created_at`` is when the thumb was clicked and ``updated_at`` when the
    record last changed, so a comment written afterwards (or a rating switched
    from up to down) is visible as such.
    """

    rating: str  # "up" or "down"
    comment: Optional[str] = None  # None until the user writes one
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class APICallRecord(BaseModel):
    """Database model for storing API call records"""

    id: Optional[str] = None  # Firestore document ID
    # Kept instead of the IP; see services/privacy.py.
    ip_hash: Optional[str] = None
    country: Optional[str] = None
    user_agent: Optional[str] = None
    content: str  # Truncated on write; see FirestoreService
    item_url: str
    origin_url: str
    title: str
    page_kind: Optional[str] = None  # "shopping_item", "restaurant_menu" or "other"
    # The kind-specific halves of the analysis. Exactly one is set, chosen by
    # page_kind; both are None for "other" and for a request that failed. There
    # is no separate is_shopping column: page_kind already says so. Query them by
    # path (where("shopping_item.is_vegan", "==", True)); Firestore indexes
    # subfields itself, and the service deliberately ships no index config.
    shopping_item: Optional[ShoppingItemRecord] = None
    menu: Optional[MenuRecord] = None
    language: Optional[str] = None  # Language of the web page
    # Empty on a failed request, which has no analysis to describe.
    summary: str = ""
    status: str = STATUS_OK
    # Why a non-"ok" record has no analysis, as "ExceptionType: message".
    error: Optional[str] = None
    service: Optional[
        str
    ] = None  # Service that processed the request: "desktop" or "gemini"
    model: Optional[str] = None  # Model that processed the request
    token_usage: Optional[TokenUsage] = None  # Tokens the provider reported
    # Desktop-server job the analysis ran as; see ProviderCall.lms_job_id.
    lms_job_id: Optional[str] = None
    # When the server took the request in hand and when it was done with it,
    # answered or failed; their difference is how long the call took. Null on
    # records written before requests were timed.
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    extension_version: Optional[str] = None  # Extension version that sent the request
    # Groups calls from one extension installation; see PageAnalysisRequest.
    installation_id: Optional[str] = None
    created_at: Optional[datetime] = None
    user_avoided_ingredients: Optional[list[str]] = None
    trigger_type: Optional[str] = None  # "manual" or "automatic"
    trigger_element_text: Optional[
        str
    ] = None  # Text of the element that triggered analysis (if automatic)
    trigger_element_selector: Optional[
        str
    ] = None  # CSS selector of the element that triggered analysis (if automatic)
    # What the page declared about its own kind (see PageAnalysisRequest), the
    # page_scope rule that chose this request's scope, and the scope itself —
    # the page kinds the model was actually offered. Stored together so a rule
    # can be re-derived offline from the raw signals and checked against the
    # verdict the model gave, which is the only way to measure a pruning rule's
    # precision: once a rule fires, the branch it dropped is not one the model
    # could have answered with. The kinds are stored alongside the rule rather
    # than derived from it, so a record still says what the model was asked
    # after the rules behind that rule name have changed.
    page_signals: Optional[dict[str, Any]] = None
    page_scope_rule: Optional[str] = None
    page_scope_kinds: Optional[list[str]] = None
    # What the user made of the answer, if they said. Written long after the
    # rest of the record, by POST /api/feedback.
    feedback: Optional[FeedbackRecord] = None

    class Config:
        # Allow extra fields for Firestore compatibility
        extra = "allow"
