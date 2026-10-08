"""On-disk cache for the Gemini calls an eval run makes.

Gemini calls cost real money, and the eval replays the same corpus over and
over (tweaking a prompt, adding a case, re-rendering a report). This caches the
API response so an identical call is only paid for once.

The cache is keyed on the *exact* request: the model name, the system
instruction, the page content, and the generation config (including the
response schema). Change any of those — a different model, an edited prompt, a
re-extracted page — and the key changes, so a stale answer can never be served.

It is installed by wrapping ``genai.GenerativeModel`` (the boundary where the
paid call actually happens), so only the network round-trip is cached: the
prompt building and response normalisation still run for real on every case.
Failures raise as usual and are never cached.

Each entry also records how long the real call took, and a hit adds that to
``CacheStats.replayed_s`` so the runner can report the latency the call had when
it was actually made, instead of the ~0s a cached case would otherwise show.

The entry layout and key digest are shared with the desktop cache — see
response_cache.py.

Eval-only: nothing here is imported by the app, and installing patches the
``google.generativeai`` module in this process alone.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tests.eval.response_cache import CacheStats, DiskCache, jsonable

# Bumped when the entry layout or key material changes, so old entries are
# ignored rather than misread.
CACHE_VERSION = 1


class _CachedResponse:
    """Stand-in for a Gemini response; carries the one field callers read."""

    def __init__(self, text: str) -> None:
        self.text = text


def _key(
    model: str, init_kwargs: dict[str, Any], call_kwargs: dict[str, Any]
) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "model": model,
        "init": jsonable(init_kwargs),
        "call": jsonable(call_kwargs),
    }


def install(cache_dir: Path, refresh: bool = False) -> CacheStats:
    """Route Gemini generation through the on-disk cache for this process.

    Replaces ``genai.GenerativeModel`` with a proxy that consults the cache
    before constructing the real model, so a hit costs nothing at all.
    """
    from services import gemini_service

    genai = gemini_service.genai
    # Re-installing (e.g. in tests) rewraps the real class, never a proxy.
    real_model_cls = getattr(
        genai.GenerativeModel, "_uncached_cls", genai.GenerativeModel
    )
    cache = DiskCache(
        cache_dir,
        refresh=refresh,
        label="gemini cache",
        miss_label="paid API call(s)",
    )

    class _CachingGenerativeModel:
        _uncached_cls = real_model_cls

        def __init__(self, model_name: str, **init_kwargs: Any) -> None:
            self._model_name = model_name
            self._init_kwargs = init_kwargs
            self._real: Any = None

        def __getattr__(self, name: str) -> Any:
            # Anything beyond generate_content goes straight to the real model.
            if self._real is None:
                self._real = real_model_cls(self._model_name, **self._init_kwargs)
            return getattr(self._real, name)

        def generate_content(self, contents: Any, **call_kwargs: Any) -> Any:
            call = dict(call_kwargs, contents=contents)
            key = _key(self._model_name, self._init_kwargs, call)
            digest, cached = cache.lookup(self._model_name, key)
            if cached is not None and isinstance(cached.response.get("text"), str):
                cache.stats.hits += 1
                cache.stats.record_replay(cached.elapsed_s)
                return _CachedResponse(cached.response["text"])

            cache.stats.misses += 1
            if self._real is None:
                self._real = real_model_cls(self._model_name, **self._init_kwargs)
            started = time.monotonic()
            response = self._real.generate_content(contents, **call_kwargs)
            elapsed = time.monotonic() - started
            cache.store(
                self._model_name,
                digest,
                key,
                {"text": response.text},
                elapsed,
            )
            return response

    genai.GenerativeModel = _CachingGenerativeModel
    return cache.stats
