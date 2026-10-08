"""On-disk cache for the desktop-server (LM Studio) calls an eval run makes.

A local model costs no money but plenty of time — a menu case runs for minutes
— and the eval replays the same corpus over and over (tweaking a prompt, adding
a case, re-rendering a report). This caches what a job answered so an identical
job is only run once, the same way Gemini responses are cached.

The cache is keyed on the *exact* OpenAI-format request the desktop-server is
handed, which is everything that decides the answer: the model, the messages
(system prompt, page content, avoided ingredients), ``reasoning_effort``,
``temperature`` and ``max_tokens`` — the last of which moves with
``LMS_CONTEXT_LENGTH``/``LMS_MAX_TOKENS`` and with how much of the page fits.
Hashing the whole request means no parameter can be forgotten: change any of
them and the key changes, so a stale answer can never be served. What is *not*
in the key is the machine behind it — LM Studio's version, the quantization it
loaded, the sampling seed — so entries are only as comparable as that setup is
stable (the same caveat Gemini's server-side model version has).

It is installed by wrapping ``DesktopService._run_job``, the boundary where a
job is written to Firestore and waited on. A hit therefore submits nothing: the
worker, LM Studio and even the heartbeat check are skipped, while prompt
building and ``normalize_page_analysis`` still run for real on every case.
Failures (an unavailable worker, a job that failed or came back unparseable)
raise as usual and are never cached.

Because the wrap sits just above ``_parse_response``, the *parsed* payload is
what gets stored — including the repair of a JSON answer the completion cap cut
short. A change to that parsing needs ``--refresh-cache`` to take effect.

Each entry also records how long the real job took, and a hit adds that to
``CacheStats.replayed_s`` so the runner reports the latency the job had when it
actually ran, instead of the ~0s a cached case would otherwise show.

Eval-only: nothing here is imported by the app, and installing patches
``services.desktop_service`` in this process alone.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from tests.eval.response_cache import CacheStats, DiskCache, jsonable

# Bumped when the entry layout or key material changes, so old entries are
# ignored rather than misread.
CACHE_VERSION = 1


def _key(model: str, request: Dict[str, Any]) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "model": model,
        "request": jsonable(request),
    }


def install(cache_dir: Path, refresh: bool = False) -> CacheStats:
    """Route desktop-server jobs through the on-disk cache for this process."""
    from models.database_model import TokenUsage
    from services import desktop_service

    cls = desktop_service.DesktopService
    # Re-installing (e.g. in tests) rewraps the real method, never a wrapper.
    real_run_job = getattr(cls._run_job, "_uncached", cls._run_job)
    cache = DiskCache(
        cache_dir,
        refresh=refresh,
        label="desktop cache",
        miss_label="real desktop-server job(s)",
    )

    def _cached_run_job(
        self: Any, request: Dict[str, Any], url: str
    ) -> desktop_service.JobResult:
        key = _key(self.model_name, request)
        digest, cached = cache.lookup(self.model_name, key)
        if cached is not None and isinstance(cached.response.get("payload"), dict):
            cache.stats.hits += 1
            cache.stats.record_replay(cached.elapsed_s)
            usage = cached.response.get("token_usage")
            return desktop_service.JobResult(
                payload=cached.response["payload"],
                token_usage=TokenUsage(**usage) if isinstance(usage, dict) else None,
                # The job the answer originally came from, so a replayed record
                # still points at a document that really exists.
                job_id=str(cached.response.get("job_id", "")),
                cut_short=bool(cached.response.get("cut_short")),
            )

        cache.stats.misses += 1
        started = time.monotonic()
        result = real_run_job(self, request, url)
        elapsed = time.monotonic() - started
        cache.store(
            self.model_name,
            digest,
            key,
            {
                "payload": result.payload,
                "token_usage": (
                    result.token_usage.model_dump() if result.token_usage else None
                ),
                "job_id": result.job_id,
                "cut_short": result.cut_short,
            },
            elapsed,
        )
        return result

    _cached_run_job._uncached = real_run_job  # type: ignore[attr-defined]
    cls._run_job = _cached_run_job  # type: ignore[method-assign]
    return cache.stats
