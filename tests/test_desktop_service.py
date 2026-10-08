"""Offline tests for the desktop-server prompt's fit in the context window.

The rest of DesktopService talks to Firestore and LM Studio (see
tests/integration), but sizing the prompt is arithmetic over strings, so these
run anywhere:

    uv run pytest tests/test_desktop_service.py

What they protect: LM Studio silently drops whatever does not fit the context
the model was loaded with, so the page budget has to be derived from that window
rather than assumed — and when the window cannot seat the prompt at all, that
has to fail loudly instead of returning an analysis of the part that fit.
"""

from __future__ import annotations

import pytest

from services import desktop_service
from services.desktop_service import (
    DEFAULT_CONTEXT_LENGTH,
    DesktopAnswerCutShort,
    DesktopService,
    JobResult,
)
from services.page_prompt import (
    INCOMPLETE_ANSWER_WARNING,
    PAGE_CONTENT_TOKEN_LIMIT,
    TRUNCATION_WARNING,
    estimate_tokens,
)
from services.page_scope import ITEM_OR_OTHER, MENU_OR_OTHER

_MENU = "Falafel Plate - chickpeas, tahini, parsley. $12.00\n" * 2000


@pytest.fixture
def make_service(monkeypatch):
    """Build a DesktopService with no Firestore client and a pinned context."""
    monkeypatch.setattr(desktop_service.firestore, "Client", lambda *a, **k: None)

    def _make(
        context_length: int = DEFAULT_CONTEXT_LENGTH, max_tokens: int = 8192
    ) -> DesktopService:
        monkeypatch.setenv("LMS_CONTEXT_LENGTH", str(context_length))
        monkeypatch.setenv("LMS_MAX_TOKENS", str(max_tokens))
        return DesktopService()

    return _make


def _prompt_tokens(request: dict) -> int:
    return sum(estimate_tokens(m["content"]) for m in request["messages"])


def _page_request(service: DesktopService) -> dict:
    return service.build_page_request(
        url="https://example.com/oat-milk",
        title="Barista Oat Milk 1L",
        content="Oat milk. Ingredients: water, oats, salt.",
    )


def test_reasoning_effort_defaults_to_low(make_service, monkeypatch):
    # Reasoning tokens share the completion cap with the JSON, so an unset
    # environment asks for the cheap level rather than the model's own default.
    monkeypatch.delenv("LMS_REASONING_EFFORT", raising=False)
    assert _page_request(make_service())["reasoning_effort"] == "low"

    monkeypatch.setenv("LMS_REASONING_EFFORT", "High")
    assert _page_request(make_service())["reasoning_effort"] == "high"


def test_an_unrecognized_reasoning_effort_is_rejected(make_service, monkeypatch):
    # LM Studio would otherwise refuse the request one page into a run.
    monkeypatch.setenv("LMS_REASONING_EFFORT", "maximum")

    with pytest.raises(ValueError, match="LMS_REASONING_EFFORT"):
        make_service()


def test_the_prompt_fits_the_window_it_is_sent_to(make_service):
    service = make_service()
    request, truncated = service._build_page_request(
        url="https://example.com/menu",
        title="The Olive Branch",
        content=_MENU,
        user_avoided_ingredients=["palm oil"],
    )

    assert truncated is True
    assert (
        _prompt_tokens(request) + service.max_tokens
        <= service.context_length - desktop_service.PROMPT_OVERHEAD_TOKENS
    )


def test_a_wider_window_reads_more_of_the_page(make_service):
    narrow, _ = make_service(context_length=16384)._build_page_request(
        url="https://example.com/menu", title="The Olive Branch", content=_MENU
    )
    wide, _ = make_service(context_length=32768)._build_page_request(
        url="https://example.com/menu", title="The Olive Branch", content=_MENU
    )

    assert len(wide["messages"][1]["content"]) > len(narrow["messages"][1]["content"])


def test_a_narrowed_scope_reads_more_of_the_page(make_service):
    """Why the scope exists at all, on the path where the window actually binds.

    The instructions and the JSON field list are charged against the same
    window as the page, so the branch a scoped request does not need is page
    text it could have read instead.
    """
    service = make_service(context_length=DEFAULT_CONTEXT_LENGTH)
    page = dict(url="https://example.com/menu", title="The Olive Branch", content=_MENU)

    both, _ = service._build_page_request(**page)
    one_branch, _ = service._build_page_request(**page, page_kinds=ITEM_OR_OTHER)

    assert len(one_branch["messages"][1]["content"]) > len(
        both["messages"][1]["content"]
    )
    # The prompt still has to fit the window it is sent to.
    assert (
        _prompt_tokens(one_branch) + one_branch["max_tokens"]
        <= service.context_length - desktop_service.PROMPT_OVERHEAD_TOKENS
    )


def test_an_item_request_reserves_an_item_sized_completion(make_service):
    # An item answer is short, so the menu-sized LMS_MAX_TOKENS reserve would
    # only cost page text; a scope that admits a menu keeps it.
    service = make_service()
    page = dict(url="https://example.com/menu", title="The Olive Branch", content=_MENU)

    item, _ = service._build_page_request(**page, page_kinds=ITEM_OR_OTHER)
    menu, _ = service._build_page_request(**page, page_kinds=MENU_OR_OTHER)

    assert item["max_tokens"] >= desktop_service.ITEM_COMPLETION_RESERVE
    assert menu["max_tokens"] == service.max_tokens
    # On the default window the item page is held only by the shared ceiling.
    assert item["messages"][1]["content"] == desktop_service.build_page_content(
        **page, token_limit=PAGE_CONTENT_TOKEN_LIMIT
    )[0]


def test_the_shared_ceiling_still_caps_a_very_wide_window(make_service):
    # A window with room to spare does not mean the whole page: the budget the
    # rest of the app works to is still the ceiling.
    service = make_service(context_length=131072)
    frame = desktop_service.build_page_content(
        url="https://example.com/menu", title="The Olive Branch", content=""
    )[0]

    assert service._page_content_budget(
        "instructions", frame, service.max_tokens
    ) == (PAGE_CONTENT_TOKEN_LIMIT)


def test_a_window_too_small_for_the_prompt_is_rejected(make_service):
    # Falls back to Gemini via AnalysisCore._dispatch rather than analyzing
    # whatever survived the runtime's own truncation.
    service = make_service(context_length=8192, max_tokens=4096)

    with pytest.raises(RuntimeError, match="no room for the page"):
        service._build_page_request(
            url="https://example.com/oat-milk", title="Oat Milk", content="oats"
        )


def test_a_truncated_page_is_flagged_in_the_analysis(make_service, monkeypatch):
    service = make_service()
    monkeypatch.setattr(
        service,
        "_run_job",
        lambda request, url: JobResult(
            {
                "page_kind": "shopping_item",
                "summary": "Vegan. Only plant ingredients.",
                "is_vegan": True,
            },
            None,
            "job-1",
        ),
    )

    analysis, _ = service.analyze_page(
        url="https://example.com/oat-milk",
        title="Barista Oat Milk 1L",
        content="Oat milk. Ingredients: water, oats, salt.\n" * 2000,
    )

    assert analysis["summary"].endswith(TRUNCATION_WARNING)


def test_the_analysis_reports_the_job_it_ran_as(make_service, monkeypatch):
    # The job ID is the record's link back to what the desktop-server actually
    # exchanged with LM Studio, and only this service ever sees it.
    service = make_service()
    monkeypatch.setattr(
        service,
        "_run_job",
        lambda request, url: JobResult(
            {"page_kind": "shopping_item", "summary": "Vegan.", "is_vegan": True},
            None,
            "job-1",
        ),
    )

    _, call = service.analyze_page(
        url="https://example.com/oat-milk",
        title="Barista Oat Milk 1L",
        content="Oat milk. Ingredients: water, oats, salt.",
    )

    assert call.service == "desktop"
    assert call.model == service.model_name
    assert call.lms_job_id == "job-1"


def test_an_untruncated_page_is_not_flagged(make_service, monkeypatch):
    service = make_service()
    monkeypatch.setattr(
        service,
        "_run_job",
        lambda request, url: JobResult(
            {
                "page_kind": "shopping_item",
                "summary": "Vegan. Only plant ingredients.",
                "is_vegan": True,
            },
            None,
            "job-1",
        ),
    )

    analysis, _ = service.analyze_page(
        url="https://example.com/oat-milk",
        title="Barista Oat Milk 1L",
        content="Oat milk. Ingredients: water, oats, salt.",
    )

    assert analysis["summary"] == "Vegan. Only plant ingredients."


def _length_capped_job(content: str) -> dict:
    """A completed job whose answer ran into the model's completion cap."""
    return {
        "status": "success",
        "response": {
            "choices": [
                {"message": {"content": content}, "finish_reason": "length"},
            ]
        },
    }


def test_an_answer_cut_off_mid_json_keeps_the_dishes_it_finished(make_service):
    # gpt-oss spends much of the completion cap reasoning, so a long menu can
    # run out of budget part-way through the JSON. The dishes it did write are
    # worth more than the parse error that unterminated JSON would raise.
    service = make_service()
    job = _length_capped_job(
        '{"page_kind":"restaurant_menu","restaurant_name":"Agave",'
        '"items":[{"name":"Guacamole","verdict":"vegan"},'
        '{"name":"Calamari","verdict":"not_vegan"},'
        '{"name":"Mole Poblano","explicit_ingredients":["chicken"],"reason":"Inc'
    )

    payload, cut_short = service._parse_response(job)

    assert cut_short
    assert payload["restaurant_name"] == "Agave"
    assert [item["name"] for item in payload["items"]] == [
        "Guacamole",
        "Calamari",
        "Mole Poblano",
    ]


def test_a_cut_off_answer_is_raised_with_the_part_that_was_written(
    make_service, monkeypatch
):
    # Raised rather than returned so a caller with a second provider gets a
    # whole-page answer; the part written rides along for one without.
    service = make_service()
    monkeypatch.setattr(
        service,
        "_run_job",
        lambda request, url: JobResult(
            {"page_kind": "shopping_item", "summary": "Vegan.", "is_vegan": True},
            None,
            "job-1",
            True,
        ),
    )

    with pytest.raises(DesktopAnswerCutShort) as raised:
        service.analyze_page(
            url="https://example.com/oat-milk",
            title="Barista Oat Milk 1L",
            content="Oat milk. Ingredients: water, oats, salt.",
        )

    analysis = raised.value.analysis
    assert analysis["summary"].endswith(INCOMPLETE_ANSWER_WARNING)
    assert raised.value.provider_call.lms_job_id == "job-1"


def test_an_answer_cut_off_before_it_started_is_rejected(make_service):
    # Nothing to keep: the cap went entirely to reasoning, which has to fail
    # loudly (and name the cap) rather than pass an empty analysis on.
    service = make_service()

    with pytest.raises(RuntimeError, match="completion cap"):
        service._parse_response(_length_capped_job("Here is the menu:"))


def test_a_short_page_hands_its_slack_to_the_completion(make_service):
    # What the page does not use is the completion's to spend: a long menu
    # needs every token it can get, and the window is the only real limit.
    service = make_service()

    short = _page_request(service)
    long = service.build_page_request(
        url="https://example.com/menu",
        title="The Olive Branch",
        content=_MENU,
    )

    assert short["max_tokens"] > long["max_tokens"]
    assert short["max_tokens"] + _prompt_tokens(short) <= service.context_length


def test_the_completion_never_drops_below_the_reserved_cap(make_service):
    # The page budget is planned against LMS_MAX_TOKENS, so the completion can
    # never be handed less than that however the prompt comes out.
    service = make_service()

    request = service.build_page_request(
        url="https://example.com/menu",
        title="The Olive Branch",
        content=_MENU,
    )

    assert request["max_tokens"] == service.max_tokens
