"""End-to-end analysis tests against the desktop-server worker.

Each test submits a real analysis request through ``AnalysisCore`` (which writes
a job to the Firestore emulator), the desktop-server worker picks it up and runs
it through LM Studio (openai/gpt-oss-20b), and we validate the result.
"""

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.integration


def _lms_chat_completions_url() -> str:
    """The LM Studio chat-completions endpoint (mirrors DesktopService config)."""
    base = os.getenv("LMS_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    return base + "/chat/completions"


def _curl_for(request_data: dict) -> str:
    """A copy-pasteable curl that replays this case against LM Studio directly.

    Bypasses Firestore and the desktop-server worker: it sends the exact same
    OpenAI-format payload DesktopService would forward, straight to LM Studio.
    """
    from services.desktop_service import DesktopService

    payload = DesktopService().build_page_request(
        url=request_data["url"],
        title=request_data["title"],
        content=request_data["content"],
        user_avoided_ingredients=request_data.get("user_avoided_ingredients"),
    )
    body = json.dumps(payload, indent=2, ensure_ascii=False)

    # Single-quoted heredoc keeps the JSON verbatim (no shell expansion, no
    # quote-escaping headaches even when the prompt contains quotes/newlines).
    return (
        f"curl -sS {_lms_chat_completions_url()} \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  --data-binary @- <<'LMS_EOF'\n"
        f"{body}\n"
        f"LMS_EOF"
    )


@contextmanager
def _curl_on_failure(request_data: dict):
    """On any failure inside the block, print a direct-to-LM-Studio curl."""
    try:
        yield
    except BaseException:
        print(
            f"\n##### Replay this case directly against LM Studio "
            f"({request_data['url']}) #####"
        )
        print(_curl_for(request_data))
        raise


# The nested branches, plus the flat aliases the released extension (1.0.4)
# reads — including ``explanation``, which is no longer asked for or stored and
# comes back as a copy of ``summary``. Both halves are part of the response
# contract, so both are asserted end to end.
REQUIRED_KEYS = {
    "page_kind",
    "shopping_item",
    "menu",
    "is_shopping_item",
    "is_vegan",
    "is_cruelty_free",
    "cruelty_free_explanation",
    "confidence_level",
    "explanation",
    "summary",
    "user_avoided_ingredients",
}


def _request(url: str, title: str, content: str, **extra) -> dict:
    return {
        "url": url,
        "title": title,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


# (id, request, expectation, validator(analysis)) for clearly-decidable pages.
CASES = [
    (
        "vegan_food",
        _request(
            url="https://example.com/oat-milk",
            title="Barista Oat Milk 1L",
            content=(
                "Barista Oat Milk. Ingredients: water, oats (10%), rapeseed oil, "
                "salt, calcium carbonate, vitamins. 100% plant-based. No dairy. "
                "Add to cart - $3.99."
            ),
        ),
        {"page_kind": "shopping_item", "is_vegan": True},
        lambda a: a["page_kind"] == "shopping_item" and a["is_vegan"] is True,
    ),
    (
        "leather_product",
        _request(
            url="https://example.com/leather-wallet",
            title="Genuine Leather Bifold Wallet",
            content=(
                "Handcrafted bifold wallet made from 100% genuine full-grain "
                "cowhide leather. Premium animal leather, ages beautifully. "
                "Buy now - $49.99."
            ),
        ),
        {"page_kind": "shopping_item", "is_vegan": False},
        lambda a: a["page_kind"] == "shopping_item" and a["is_vegan"] is False,
    ),
    (
        "non_shopping_article",
        _request(
            url="https://news.example.com/article",
            title="Local council debates new park hours",
            content=(
                "The city council met on Tuesday to discuss extending opening "
                "hours for public parks over the summer. Residents shared mixed "
                "opinions during the public comment period."
            ),
        ),
        {"page_kind": "other", "is_vegan": None},
        lambda a: a["page_kind"] == "other" and a["is_vegan"] is None,
    ),
    (
        "restaurant_menu",
        _request(
            url="https://example.com/olive-branch/menu",
            title="The Olive Branch",
            content=(
                "The Olive Branch — Menu.\n"
                "Starters: Hummus & Flatbread (chickpeas, tahini, lemon) £6.50. "
                "Halloumi Fries with honey drizzle £7.00.\n"
                "Mains: Falafel Plate (VG) — chickpea falafel, tabbouleh, "
                "tahini £13.00. Grilled Lamb Kofta with garlic yoghurt £17.50."
            ),
        ),
        {"page_kind": "restaurant_menu", "is_vegan": None},
        # The dish list is what makes this a menu, so an empty `items` is a
        # failure even when the classification is right.
        lambda a: (
            a["page_kind"] == "restaurant_menu"
            and a["is_vegan"] is None
            and len(a["menu"]["items"]) >= 2
        ),
    ),
]


@pytest.mark.parametrize(
    "case_id, request_data, expectation, validate", CASES, ids=[c[0] for c in CASES]
)
def test_analysis_end_to_end(
    analysis_core, case_id, request_data, expectation, validate
):
    with _curl_on_failure(request_data):
        response, status = analysis_core.analyze_page(request_data)

        assert status == 200, f"unexpected status {status}: {response}"
        analysis = response["analysis"]

        # Structure: all required keys present with the right basic types.
        assert REQUIRED_KEYS.issubset(
            analysis.keys()
        ), f"missing keys: {REQUIRED_KEYS - set(analysis.keys())}"
        assert analysis["page_kind"] in ("shopping_item", "restaurant_menu", "other")
        assert isinstance(analysis["is_shopping_item"], bool)
        assert analysis["is_vegan"] is None or isinstance(analysis["is_vegan"], bool)

        # Exactly one branch is filled, chosen by page_kind; the other is null.
        if analysis["page_kind"] == "shopping_item":
            assert analysis["menu"] is None
            assert analysis["confidence_level"] in ("high", "medium", "low")
        elif analysis["page_kind"] == "restaurant_menu":
            assert analysis["shopping_item"] is None
            assert isinstance(analysis["menu"]["items"], list)
            # Only a product carries a single confidence; a menu answers per dish.
            assert analysis["confidence_level"] is None
        else:
            assert analysis["shopping_item"] is None and analysis["menu"] is None
        # summary must be real prose, not placeholder/whitespace junk — the
        # failure mode that text-mode (vs constrained json_schema) fixed.
        assert isinstance(analysis["summary"], str)
        assert (
            len(analysis["summary"].strip()) >= 10
        ), f"summary too short/degenerate: {analysis['summary']!r}"
        assert analysis["explanation"] == analysis["summary"]
        assert isinstance(analysis["user_avoided_ingredients"], list)

        # Semantics: clearly-decidable cases must come out right.
        actual = {key: analysis.get(key) for key in expectation}
        assert validate(analysis), (
            f"semantic check failed for {case_id}\n"
            f"  expected: {expectation}\n"
            f"  actual:   {actual}\n"
            f"  full analysis: {analysis}"
        )


def test_user_avoided_ingredients(analysis_core):
    """A user-avoided ingredient present on the page is reported back."""
    request_data = _request(
        url="https://example.com/protein-bar",
        title="Chocolate Protein Bar",
        content=(
            "Chocolate protein bar. Ingredients: oats, cocoa, almonds, honey, "
            "sea salt. Buy now - $2.49."
        ),
        user_avoided_ingredients=["honey"],
    )

    with _curl_on_failure(request_data):
        response, status = analysis_core.analyze_page(request_data)
        assert status == 200, f"unexpected status {status}: {response}"

        analysis = response["analysis"]
        found = [i.lower() for i in analysis.get("user_avoided_ingredients", [])]
        assert any(
            "honey" in i for i in found
        ), f"expected 'honey' in user_avoided_ingredients: {analysis}"
