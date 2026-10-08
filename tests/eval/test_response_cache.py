"""Offline tests for the eval's on-disk response cache.

    uv run pytest tests/eval/test_response_cache.py

What they protect: a cached answer stands in for a call that is never made, so
the key has to cover everything that decides the answer. If it missed a
parameter — the model, the reasoning effort, the prompt — a run would silently
report another configuration's numbers as this one's.
"""

from __future__ import annotations

from typing import Any

import pytest

from models.database_model import TokenUsage
from services.desktop_service import DesktopService, JobResult
from tests.eval import desktop_cache


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    """Install the cache over a stand-in for the real (Firestore) job run.

    Yields the list of requests that actually reached it, so a hit is visible as
    a request that never arrives.
    """
    submitted: list[dict[str, Any]] = []

    def fake_run_job(self, request, url):
        submitted.append(request)
        return JobResult(
            payload={"page_kind": "food", "items": []},
            token_usage=TokenUsage(prompt_tokens=11, completion_tokens=22),
            job_id=f"job{len(submitted)}",
            cut_short=True,
        )

    # Recorded before install wraps it, so teardown restores the real method
    # and takes the cache wrapper with it.
    monkeypatch.setattr(DesktopService, "_run_job", fake_run_job)
    return submitted, desktop_cache.install(tmp_path, refresh=False)


def _service(model: str = "openai/gpt-oss-20b") -> DesktopService:
    """A DesktopService with only the field the cache reads (no Firestore)."""
    service = DesktopService.__new__(DesktopService)
    service.model_name = model
    return service


def _request(**overrides: Any) -> dict[str, Any]:
    return {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "page text"}],
        "max_tokens": 8192,
        "temperature": 0.4,
        "reasoning_effort": "medium",
        **overrides,
    }


def test_an_identical_job_is_served_from_disk(jobs):
    submitted, stats = jobs
    service = _service()

    first = DesktopService._run_job(service, _request(), "https://example.com")
    second = DesktopService._run_job(service, _request(), "https://example.com")

    assert len(submitted) == 1
    assert (stats.hits, stats.misses) == (1, 1)
    # Everything analyze_page goes on to read comes back, including the cut-short
    # flag it turns into a DesktopAnswerCutShort.
    assert second == first


def test_reasoning_effort_and_model_are_part_of_the_key(jobs):
    submitted, stats = jobs
    service = _service()

    DesktopService._run_job(service, _request(), "https://example.com")
    DesktopService._run_job(service, _request(reasoning_effort="high"), "u")
    DesktopService._run_job(service, _request(temperature=0.9), "u")
    DesktopService._run_job(_service("qwen/qwen3-30b"), _request(), "u")

    assert len(submitted) == 4
    assert (stats.hits, stats.misses) == (0, 4)


def test_a_hit_replays_the_latency_the_job_originally_had(jobs, monkeypatch):
    submitted, stats = jobs
    service = _service()

    clock = iter([100.0, 107.5])
    monkeypatch.setattr(desktop_cache.time, "monotonic", lambda: next(clock))
    DesktopService._run_job(service, _request(), "https://example.com")
    assert stats.replayed_s == 0.0

    DesktopService._run_job(service, _request(), "https://example.com")
    # Otherwise a replayed case would report ~0s and flatter the model.
    assert stats.replayed_s == pytest.approx(7.5)
