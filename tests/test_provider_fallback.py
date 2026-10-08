"""Which provider answers a request, and who answers when it cannot finish.

The routing itself is ``PROVIDER_BY_TRIGGER``; what is checked here is that a
request reaches it and that the other provider backs it up either way. An answer
the local model ran out of completion budget to finish is the awkward case: it
is real, but it only covers part of the page, so it must lose to a provider that
can answer for all of it — and win over nothing at all.
"""

from __future__ import annotations

from analysis_core import AnalysisCore
from models.database_model import ProviderCall
from services.desktop_service import DesktopAnswerCutShort, DesktopUnavailable

PARTIAL = {"page_kind": "restaurant_menu", "items": [{"name": "Guacamole"}]}
WHOLE = {"page_kind": "restaurant_menu", "items": [{"name": "Guacamole"}] * 40}

DESKTOP_CALL = ProviderCall(service="desktop", model="openai/gpt-oss-20b")
GEMINI_CALL = ProviderCall(service="gemini", model="gemini-2.5-flash-lite")


class CutShortDesktop:
    def analyze_page(self, **kwargs):
        raise DesktopAnswerCutShort(
            "ran out of completion budget",
            analysis=PARTIAL,
            provider_call=DESKTOP_CALL,
        )


class DownDesktop:
    def analyze_page(self, **kwargs):
        raise DesktopUnavailable("heartbeat is stale")


class FakeDesktop:
    def analyze_page(self, **kwargs):
        return WHOLE, DESKTOP_CALL


class FakeGemini:
    def analyze_page(self, **kwargs):
        return WHOLE, GEMINI_CALL


class DownGemini:
    def analyze_page(self, **kwargs):
        raise RuntimeError("Gemini said no")


def _core(desktop=None, gemini=None) -> AnalysisCore:
    core = AnalysisCore.__new__(AnalysisCore)
    core.desktop_service = desktop
    core.gemini_service = gemini
    return core


def _dispatch(core: AnalysisCore, trigger_type=None):
    return core._dispatch("analyze_page", trigger_type=trigger_type)


def test_the_automatic_trigger_goes_to_the_desktop_server():
    _, call = _dispatch(_core(FakeDesktop(), FakeGemini()), "automatic")

    assert call.service == "desktop"


def test_a_manual_trigger_goes_to_gemini():
    _, call = _dispatch(_core(FakeDesktop(), FakeGemini()), "manual")

    assert call.service == "gemini"


def test_a_request_with_no_trigger_goes_to_gemini():
    _, call = _dispatch(_core(FakeDesktop(), FakeGemini()))

    assert call.service == "gemini"


def test_gemini_falls_back_to_the_desktop_server():
    _, call = _dispatch(_core(FakeDesktop(), DownGemini()), "manual")

    assert call.service == "desktop"


def test_the_desktop_server_falls_back_to_gemini():
    _, call = _dispatch(_core(DownDesktop(), FakeGemini()), "automatic")

    assert call.service == "gemini"


def test_the_only_configured_provider_serves_every_trigger():
    """What the eval harness relies on: one provider pinned, no routing."""
    for trigger in ("automatic", "manual", None):
        _, call = _dispatch(_core(FakeDesktop(), None), trigger)
        assert call.service == "desktop"


def test_a_cut_short_answer_gives_way_to_the_fallback():
    analysis, call = _dispatch(_core(CutShortDesktop(), FakeGemini()), "automatic")

    assert analysis == WHOLE
    assert call.service == "gemini"


def test_a_cut_short_answer_is_kept_when_there_is_no_fallback():
    analysis, call = _dispatch(_core(CutShortDesktop(), None), "automatic")

    assert analysis == PARTIAL
    assert call.service == "desktop"
