"""The page-analysis prompt: one question asked of any page.

The extension can be pointed at anything — an Amazon listing, a restaurant's
own site, a Google Maps place panel, a news article — and has no reliable way
to tell them apart before asking. So there is one prompt and one call: the
model first classifies the page into a ``page_kind``, then fills in only the
branch that kind calls for.

What the request *can* sometimes say is which branch it could not possibly
need. ``page_scope`` works that out and passes it here as ``page_kinds``; every
builder below then describes only the branches in that scope. The question the
model is asked narrows, but it is never answered for it — "other" survives
every scope, so the classification step remains real.

The two branches live in their own modules (``analysis_prompt`` for shopping
items, ``menu_prompt`` for restaurant menus) and are spliced together here.
This module owns what is common to every page: the classification rules, the
one field answered for all three kinds (``summary``), the two schema flavors,
and normalization.

What the model answers here is **flat**, and stays that way: the field order in
``_required_fields`` is what keeps each verdict conditioned on the reasoning
above it. The shape the extension and the database see — ``shopping_item`` and
``menu`` maps, plus the flat aliases the released extension still reads — is
built from this one by ``models.analysis_model.PageAnalysis``.
"""

import math
import re

from services import analysis_prompt, menu_prompt
from services.page_scope import ALL_PAGE_KINDS, ITEM_OR_OTHER, MENU_OR_OTHER


# What a page can be. "other" is a first-class outcome, not a failure: most of
# the web is neither a product nor a menu, and saying so plainly is the useful
# answer.
#
# A single request may be asked a narrower question than this — see
# ``page_scope``, which rules out a branch when the request itself makes it
# impossible or pointless. Every builder here takes that scope as ``page_kinds``
# and describes only the branches it admits, so the model is never asked to
# hold a question the caller has already answered.
PAGE_KINDS = list(ALL_PAGE_KINDS)

# Budget for the page text handed to the model — the single limit on how much
# of a page gets analyzed, by either provider.
#
# It applies before the page's kind is known, since that is now the model's
# decision rather than the caller's. The menu case sets the size: a menu
# truncated mid-list silently loses dishes, whereas a product page carries its
# ingredients near the top and loses nothing by having room to spare.
#
# It is counted in tokens, not characters, because tokens are the unit of the
# thing that actually binds — the context window of the model reading it. The
# same character count is worth wildly different token counts across scripts
# (a Chinese or Japanese menu runs several times denser than an English one),
# so a character cap that fits comfortably in English can overrun a window
# outright in CJK. 6144 tokens is roughly 18k characters of English, close to
# the 20k-character cap this replaced.
#
# It is a ceiling, not a promise: a model whose context window cannot seat it
# passes a smaller ``token_limit`` to ``build_page_content``, which is what the
# local model does (see ``DesktopService._page_content_budget``). Gemini's
# window is large enough that this ceiling is the only thing binding there.
PAGE_CONTENT_TOKEN_LIMIT = 6144

# How text is converted to tokens without adding a tokenizer dependency.
#
# The o200k-family tokenizers behind both providers average ~4 characters per
# token on English prose; 3.0 is deliberately pessimistic, since scraped page
# text (markup leftovers, prices, non-English words) tokenizes denser than prose
# and the cost of over-estimating is only a shorter page.
#
# CJK is the case a flat ratio gets badly wrong, and menus are exactly where it
# shows up: those characters cost close to a token each, so a page of them is
# three to four times the tokens a flat ratio would predict. They are counted
# separately rather than averaged in, because averaging would under-count a CJK
# menu and over-count everything else.
CHARS_PER_TOKEN = 3.0
CJK_TOKENS_PER_CHAR = 1.0

# Hiragana, katakana, CJK ideographs (incl. extension A and compatibility), and
# Hangul syllables.
_CJK_PATTERN = re.compile(
    "[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
)

# Appended to the summary when the page did not fit the budget, so a verdict
# formed on partial data never reads as a verdict on all of it.
TRUNCATION_WARNING = (
    "Note: this page was too long to analyze in full, so only the beginning of "
    "it was read and some of the content was not seen."
)

# Appended when the model ran out of completion budget part-way through writing
# its answer, so a half-written list never reads as the whole page's verdict.
INCOMPLETE_ANSWER_WARNING = (
    "Note: the analysis ran out of room before it was finished, so some of the "
    "page is missing from it."
)



_SHARED_FIELD_DESCRIPTIONS = {
    # Rendered per scope by ``_page_kind_description`` — a description naming a
    # value the enum does not offer is a contradiction the model has to resolve.
    "page_kind": {
        "shopping_item": '"shopping_item" for a product that can be purchased',
        "restaurant_menu": '"restaurant_menu" for a list of dishes that can be ordered',
        "other": '"other" for anything else',
    },
    "summary": "A complete one-to-two sentence plain-language summary of what this page holds and what it means for a vegan, and the only prose the reader is shown. For a shopping item, open directly with the vegan (and cruelty-free, if applicable) verdict and its reason, without any preamble about the page itself. For a restaurant menu, say roughly how many vegan options there are and where they are. For any other kind of page, say briefly what the page is. Never empty or a placeholder.",
    "user_avoided_ingredients": analysis_prompt.FIELD_DESCRIPTIONS[
        "user_avoided_ingredients"
    ],
}

# The classification section of the prompt, per scope.
#
# Written out per scope rather than generated, because this is prompt text: what
# the small models downstream key on is the exact wording, and a sentence
# assembled from fragments reads like one. The pruned variants say explicitly
# where the excluded branch's pages go — a menu under ``ITEM_OR_OTHER`` has to
# land on "other", and the model will only do that reliably if it is told so.
_KIND_DESCRIPTIONS = {
    "shopping_item": '- "shopping_item": the page presents a specific product, good, or service that can be purchased — food, beverages, clothing, electronics, household items, beauty products, and so on, new or secondhand',
    "restaurant_menu": '- "restaurant_menu": the page lists dishes, drinks, or other orderable items from a restaurant, cafe, bar, or takeaway, usually with names and often prices or descriptions',
}

_OTHER_DESCRIPTIONS = {
    ALL_PAGE_KINDS: '- "other": anything else — news articles, blog posts, informational pages, social media, search results, and restaurant listings that show only address, hours, ratings, or reviews with no dish list',
    ITEM_OR_OTHER: '- "other": anything else — news articles, blog posts, informational pages, social media, search results, cart and checkout pages, and any restaurant\'s menu or listing',
    MENU_OR_OTHER: '- "other": anything else — news articles, blog posts, informational pages, social media, search results, product and shopping pages, and restaurant listings that show only address, hours, ratings, or reviews with no dish list',
}

_TIEBREAKS = {
    ALL_PAGE_KINDS: [
        '- A used, secondhand or resale item offered for sale is "shopping_item", the same as a new one — what it is made of matters just as much',
        '- A restaurant page with an actual list of dishes is "restaurant_menu", not "shopping_item", even when the dishes have prices and can be ordered',
        '- A shop selling a single food product is "shopping_item", not "restaurant_menu"',
        '- A restaurant\'s homepage with no dishes on it is "other". Do not invent a menu that is not in the content.',
        '- A page with no clear product and no dish list is "other". This is a normal, common answer — prefer it over a forced guess.',
    ],
    ITEM_OR_OTHER: [
        '- A used, secondhand or resale item offered for sale is "shopping_item", the same as a new one — what it is made of matters just as much',
        '- A restaurant menu — a list of dishes rather than of products — is "other" here. Do not report its dishes as a shopping item, and do not pick one dish to answer about.',
        '- A cart, checkout, order-confirmation or account page is "other", even on a shop, and even when it lists things that were bought.',
        '- A page with no clear product is "other". This is a normal, common answer — prefer it over a forced guess.',
    ],
    MENU_OR_OTHER: [
        '- A restaurant\'s homepage with no dishes on it is "other". Do not invent a menu that is not in the content.',
        '- A listing that shows only address, hours, ratings or reviews is "other".',
        '- A page with no dish list is "other". This is a normal, common answer — prefer it over a forced guess.',
    ],
}

_THEN_LINES = {
    ALL_PAGE_KINDS: "THEN: fill in only the fields belonging to the kind you chose, and leave the other branch's fields null or empty. The fields for each kind are listed below.",
    ITEM_OR_OTHER: "THEN: fill in the fields belonging to the kind you chose. They are listed below.",
    MENU_OR_OTHER: "THEN: fill in the fields belonging to the kind you chose. They are listed below.",
}

# What "other" means for the fields that exist in this scope's schema. Naming a
# field the schema does not carry is worse than saying nothing: it invites the
# model to invent the key.
_OTHER_BRANCHES = {
    ALL_PAGE_KINDS: """- Set is_vegan, is_cruelty_free, cruelty_free_explanation, confidence_level and restaurant_name to null
- Set materials_found, inferred_materials, animal_derived_materials and items to empty lists and vegan_friendliness to "none\"""",
    ITEM_OR_OTHER: """- Set is_vegan, is_cruelty_free, cruelty_free_explanation and confidence_level to null
- Set materials_found, inferred_materials and animal_derived_materials to empty lists""",
    MENU_OR_OTHER: """- Set restaurant_name to null
- Set items to an empty list and vegan_friendliness to "none\"""",
}


def _scope(page_kinds) -> tuple[str, ...]:
    """Normalize a caller's scope to one of the three known tuples."""
    scope = tuple(page_kinds)
    if scope not in _OTHER_DESCRIPTIONS:
        raise ValueError(f"unknown page-kind scope: {scope!r}")
    return scope


def _page_kind_description(scope: tuple[str, ...]) -> str:
    """The ``page_kind`` schema description, naming only the kinds in scope."""
    parts = [_SHARED_FIELD_DESCRIPTIONS["page_kind"][kind] for kind in scope]
    return "What kind of page this is: " + ", ".join(parts)


# Field order in the response, and the reason this module cares about it.
#
# A model answers these in the order they are listed, so the order decides what
# each answer is conditioned on. With ``is_vegan`` second, the verdict was a
# single token emitted before the model had written a word of reasoning, and
# Gemini's flash-lite tier does no thinking of its own to fall back on: it
# returned `is_vegan: true` next to an explanation concluding "100% wool, which
# is an animal product. Therefore, the item is not vegan."
#
# So every verdict now comes after the prose that argues it. The evidence the
# product verdict rests on leads (materials read off the page), then the menu
# payload, then the shared prose, and only then the booleans. ``page_kind``
# stays first because both branches are conditioned on it; the item branch's
# ``confidence_level`` stays last within it, because it judges everything above.
#
# Note for Gemini specifically: this list is the *only* ordering signal we can
# send. The pinned google-ai-generativelanguage Schema has no
# ``property_ordering`` field, and its ``properties`` is an unordered proto map
# — the server follows ``required``.
def _required_fields(page_kinds: tuple[str, ...]) -> list[str]:
    """The response's field order, restricted to the branches in scope."""
    item = "shopping_item" in page_kinds
    menu = "restaurant_menu" in page_kinds
    return (
        ["page_kind"]
        + (analysis_prompt.ITEM_EVIDENCE_FIELDS if item else [])
        + (menu_prompt.MENU_FIELDS if menu else [])
        + ["summary"]
        + (analysis_prompt.ITEM_VERDICT_FIELDS if item else [])
    )


def build_page_instructions(
    user_avoided_ingredients: list[str] | None = None,
    page_kinds: tuple[str, ...] = ALL_PAGE_KINDS,
) -> tuple[str, bool]:
    """Build the static page-analysis instructions (the task, not the page data).

    These describe *how* to analyze and are identical across requests (modulo
    the optional user-avoided-ingredients clause and the scope), so callers
    place them in the system role and pass the per-request page data separately
    via ``build_page_content``.

    ``page_kinds`` narrows the question — see ``page_scope``. A narrowed scope
    describes one branch instead of two, which is most of this text: it is what
    frees the local model's context for the page itself, and it removes the
    "leave the other branch null" bookkeeping that a small model has to hold
    while it works.

    Returns ``(instructions, has_user_avoided)``, where ``has_user_avoided``
    indicates whether the user_avoided_ingredients fields belong in the
    response schema.
    """
    scope = _scope(page_kinds)
    has_user_avoided = menu_prompt.includes_user_avoided(user_avoided_ingredients)

    kind_lines = [_KIND_DESCRIPTIONS[kind] for kind in scope if kind != "other"]
    kind_lines.append(_OTHER_DESCRIPTIONS[scope])

    branches = []
    if "shopping_item" in scope:
        branches.append(analysis_prompt.build_item_branch(user_avoided_ingredients))
    if "restaurant_menu" in scope:
        branches.append(menu_prompt.build_menu_branch(user_avoided_ingredients))

    kinds = "\n".join(kind_lines)
    tiebreaks = "\n".join(_TIEBREAKS[scope])
    branch_text = "\n\n".join(branches)

    instructions = f"""Analyze the provided webpage content for a vegan reader.

FIRST: Classify the page by setting page_kind to exactly one of:
{kinds}

Choosing between them:
{tiebreaks}

{_THEN_LINES[scope]}

{branch_text}

IF page_kind IS "other":
{_OTHER_BRANCHES[scope]}
- Say briefly in the summary what the page actually is

ALWAYS, for every page_kind:
- Answer the fields in the order they are listed in the response format. The evidence and the summary come before the verdicts they support, and each verdict must agree with the reasoning above it — never contradict your own summary.
- summary is answered for every page_kind. Never leave it empty or fill it with placeholder text."""

    return instructions, has_user_avoided


def estimate_tokens(text: str) -> int:
    """Estimate how many tokens ``text`` occupies.

    Deliberately an over-estimate — see ``CHARS_PER_TOKEN`` for the ratios and
    why CJK characters are counted apart from the rest.
    """
    cjk = len(_CJK_PATTERN.findall(text))
    return math.ceil(cjk * CJK_TOKENS_PER_CHAR + (len(text) - cjk) / CHARS_PER_TOKEN)


def _cut_to_token_budget(text: str, budget_tokens: int) -> str:
    """Return the longest prefix of ``text`` estimated to fit ``budget_tokens``.

    Binary search over the prefix length: ``estimate_tokens`` only grows as the
    prefix does, and searching it keeps the cut honest for mixed-script pages,
    where no single characters-per-token ratio would place it correctly.
    """
    if estimate_tokens(text) <= budget_tokens:
        return text

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(text[:middle]) <= budget_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def build_page_content(
    url: str, title: str, content: str, token_limit: int = PAGE_CONTENT_TOKEN_LIMIT
) -> tuple[str, bool]:
    """Build the per-request page-data message that accompanies the instructions.

    This is the variable part of the prompt (the actual page being analyzed)
    and belongs in the user role, separate from the static instructions
    returned by ``build_page_instructions``.

    Page text past ``token_limit`` is cut. Returns ``(message, truncated)`` —
    callers pass ``truncated`` to ``add_truncation_warning`` so the reader is
    told the verdict was formed on part of the page.
    """
    page_text = _cut_to_token_budget(content, token_limit)

    return (
        f"""Page URL: {url}
Page Title: {title}
Content:
{page_text}""",
        len(page_text) < len(content),
    )


def _add_warning(analysis: dict, warning: str) -> dict:
    """Append a warning to the prose the extension actually shows."""
    existing = (analysis.get("summary") or "").rstrip()
    analysis["summary"] = f"{existing} {warning}".strip()
    return analysis


def add_truncation_warning(analysis: dict) -> dict:
    """Flag an analysis formed on only the beginning of the page."""
    return _add_warning(analysis, TRUNCATION_WARNING)


def add_incomplete_answer_warning(analysis: dict) -> dict:
    """Flag an analysis the model did not finish writing."""
    return _add_warning(analysis, INCOMPLETE_ANSWER_WARNING)


def _describe_type(prop: dict) -> str:
    """Render a schema property's type as prose for the text-mode instruction."""
    if "enum" in prop:
        return " or ".join(f'"{e}"' for e in prop["enum"])
    prop_type = prop["type"]
    if prop_type == "array":
        item_type = prop["items"].get("type")
        return "array of strings" if item_type == "string" else "array of objects"
    if isinstance(prop_type, list):
        return " or ".join(prop_type)  # e.g. "boolean or null"
    return prop_type


def build_page_json_output_instruction(
    include_user_avoided: bool, page_kinds: tuple[str, ...] = ALL_PAGE_KINDS
) -> str:
    """Build a prompt suffix instructing the model to emit the JSON directly.

    Used for the local desktop-server (LM Studio) path, which does NOT use an
    OpenAI ``response_format`` schema: under token-level constrained decoding
    gpt-oss skips its reasoning step and degenerates the trailing string fields
    into placeholder/whitespace junk. Steering the JSON shape via the prompt
    instead keeps the reasoning channel active and the fields complete.

    Field names, order, and types are derived from the OpenAI response schema so
    this stays in sync with the single source of truth — including the scope,
    which decides which branch's fields the schema carries at all.
    """
    schema = build_openai_page_schema(include_user_avoided, page_kinds)
    lines = []
    for field in schema["required"]:
        prop = schema["properties"][field]
        lines.append(f"- {field} ({_describe_type(prop)}): {prop['description']}")
        if field != "items":
            continue
        # Expand the nested dish object, which carries most of the payload.
        item_schema = prop["items"]
        for item_field in item_schema["required"]:
            item_prop = item_schema["properties"][item_field]
            lines.append(
                f"    - {item_field} ({_describe_type(item_prop)}): "
                f"{item_prop['description']}"
            )
    fields = "\n".join(lines)
    return (
        "\n\nRespond with ONLY a single JSON object (no markdown, no code "
        "fences, no text before or after) containing exactly these keys:\n"
        f"{fields}\n\n"
        "Every string field must contain complete, real content. Never use "
        'placeholders such as "...", "analysis", whitespace, or empty strings.'
    )


def _build_schema(
    include_user_avoided: bool,
    nullable_flavor: bool,
    page_kinds: tuple[str, ...] = ALL_PAGE_KINDS,
) -> dict:
    """Build the response schema in one of the two flavors.

    ``nullable_flavor`` selects Gemini's ``nullable: True`` over the OpenAI
    ``type``-union spelling; the OpenAI-compatible endpoint used by LM Studio
    does not understand Gemini's ``nullable`` keyword.

    ``page_kinds`` restricts both the ``page_kind`` enum and which branch's
    properties are carried. The excluded branch is absent from the schema
    entirely rather than present-and-nullable: on Gemini it would otherwise
    still be a required field the model has to emit, and on the local path its
    field list is a third of the JSON instruction.
    """
    scope = _scope(page_kinds)
    properties = {
        "page_kind": {
            "type": "string",
            "enum": list(scope),
            "description": _page_kind_description(scope),
        },
    }
    if "shopping_item" in scope:
        properties.update(analysis_prompt.item_properties(nullable_flavor))
    if "restaurant_menu" in scope:
        properties.update(
            menu_prompt.menu_properties(include_user_avoided, nullable_flavor)
        )
    properties.update(
        {
            "summary": {
                "type": "string",
                "description": _SHARED_FIELD_DESCRIPTIONS["summary"],
            },
        }
    )

    required = _required_fields(scope)

    if include_user_avoided:
        properties["user_avoided_ingredients"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": _SHARED_FIELD_DESCRIPTIONS["user_avoided_ingredients"],
        }
        required.append("user_avoided_ingredients")

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if not nullable_flavor:
        schema["additionalProperties"] = False
    return schema


def build_gemini_page_schema(
    include_user_avoided: bool, page_kinds: tuple[str, ...] = ALL_PAGE_KINDS
) -> dict:
    """Build the Gemini-flavored response schema (nullable via ``nullable``)."""
    return _build_schema(include_user_avoided, True, page_kinds)


def build_openai_page_schema(
    include_user_avoided: bool, page_kinds: tuple[str, ...] = ALL_PAGE_KINDS
) -> dict:
    """Build the OpenAI/JSON-Schema-flavored response schema (``type`` unions)."""
    return _build_schema(include_user_avoided, False, page_kinds)


def normalize_page_analysis(
    analysis: dict,
    include_user_avoided: bool,
    page_kinds: tuple[str, ...] = ALL_PAGE_KINDS,
) -> dict:
    """Fill in optional fields so both providers return the same shape.

    Forces the losing branch's fields to null/empty, so a result can never look
    like two kinds of page at once. The nesting the extension sees, and the
    legacy flat aliases that go with it, are added later by
    ``models.analysis_model.PageAnalysis``.

    A ``page_kind`` outside ``page_kinds`` falls back to "other" rather than
    being honored. The local path steers its JSON by prompt rather than by
    constrained decoding, so a scoped-out kind can still come back in the
    response — and answering "restaurant_menu" from a schema carrying no menu
    fields would reach the extension as a menu with no dishes.
    """
    page_kind = analysis.get("page_kind")
    if page_kind not in page_kinds:
        page_kind = "other"
    analysis["page_kind"] = page_kind

    is_item = page_kind == "shopping_item"
    is_menu = page_kind == "restaurant_menu"

    analysis_prompt.normalize_item_fields(analysis, applies=is_item)
    menu_prompt.normalize_menu_fields(
        analysis, applies=is_menu, include_user_avoided=include_user_avoided
    )

    analysis.setdefault("summary", "")

    # A menu carries its avoid-list hits per dish; the top-level field is the
    # product answer, and stays present either way for a consistent shape.
    analysis.setdefault("user_avoided_ingredients", [])
    if not isinstance(analysis["user_avoided_ingredients"], list):
        analysis["user_avoided_ingredients"] = []

    return analysis


def error_analysis(message: str) -> dict:
    """The shape returned when analysis could not be run at all."""
    return normalize_page_analysis(
        {
            "page_kind": "other",
            "summary": message,
        },
        include_user_avoided=False,
    )
