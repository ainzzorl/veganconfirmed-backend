"""Offline tests for the page budget: how much of a page is analyzed, and what
the reader is told when it was not all of it.

Pure functions over strings, so they run anywhere:

    uv run pytest tests/test_page_prompt.py

What they protect: a page over the budget is analyzed in part, and a verdict on
part of a page must never be presented as a verdict on all of it.
"""

from __future__ import annotations

from services.page_prompt import (
    PAGE_CONTENT_TOKEN_LIMIT,
    TRUNCATION_WARNING,
    add_truncation_warning,
    build_page_content,
    estimate_tokens,
)

_LATIN_DISH = "Falafel Plate - chickpeas, tahini, parsley, lemon. $12.00\n"
# Same idea, in the script where characters and tokens part ways.
_CJK_DISH = "牛肉麺 豚骨ラーメン 餃子 チャーハン 麻婆豆腐 焼き鳥 天ぷら 味噌汁\n"


def _page(content: str) -> tuple[str, bool]:
    return build_page_content(
        url="https://example.com/menu", title="The Olive Branch", content=content
    )


def test_short_page_is_sent_whole():
    content = "Barista Oat Milk. Ingredients: water, oats, rapeseed oil, salt."
    message, truncated = _page(content)

    assert truncated is False
    assert content in message


def test_long_page_is_cut_to_the_budget():
    message, truncated = _page(_LATIN_DISH * 2000)

    assert truncated is True
    assert estimate_tokens(message) <= PAGE_CONTENT_TOKEN_LIMIT + estimate_tokens(
        _page("")[0]
    )


def test_budget_is_counted_in_tokens_not_characters():
    # The point of a token budget. These two pages are the same length in
    # characters, but the CJK one is worth several times the tokens, so it has
    # to be cut where the Latin one is not.
    target_chars = 12000
    latin = _LATIN_DISH * (target_chars // len(_LATIN_DISH))
    cjk = _CJK_DISH * (target_chars // len(_CJK_DISH))

    assert abs(len(latin) - len(cjk)) < max(len(_LATIN_DISH), len(_CJK_DISH))
    assert _page(latin)[1] is False
    assert _page(cjk)[1] is True


def test_cut_page_stays_within_the_budget_in_any_script():
    for content in (
        _LATIN_DISH * 2000,
        _CJK_DISH * 2000,
        (_LATIN_DISH + _CJK_DISH) * 1000,
    ):
        message, truncated = _page(content)

        assert truncated is True
        assert estimate_tokens(message) <= PAGE_CONTENT_TOKEN_LIMIT + estimate_tokens(
            _page("")[0]
        )


def test_warning_is_appended_to_the_shown_field():
    analysis = add_truncation_warning({"summary": "Vegan."})

    assert analysis["summary"] == f"Vegan. {TRUNCATION_WARNING}"


def test_warning_survives_an_empty_field():
    analysis = add_truncation_warning({"summary": None})

    assert analysis["summary"] == TRUNCATION_WARNING
