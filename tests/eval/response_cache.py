"""On-disk cache of provider responses, shared by the eval's provider caches.

Both providers replay the same corpus over and over (tweaking a prompt, adding
a case, re-rendering a report) and both are expensive to re-run: Gemini charges
per call, the desktop-server spends minutes of local GPU time. This holds the
machinery both caches use — the key digest, the entry layout, the hit/miss
tally — while each provider module owns the boundary it patches and what it
stores there (see gemini_cache.py and desktop_cache.py).

An entry lives at ``<cache_dir>/<provider>/<model>/<digest>.json`` and records
the full request it answers, so the key can be validated on read and a cached
answer inspected by hand.

Eval-only: nothing here is imported by the app.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class CacheStats:
    """Tally of what one cache did during a run (reported at the end)."""

    # How the summary line names this cache and the calls it did not avoid.
    label: str = "cache"
    miss_label: str = "real call(s)"
    hits: int = 0
    misses: int = 0
    # Wall-clock the served hits originally took, so a caller can attribute the
    # time a cached call is standing in for. Monotonically increasing; read the
    # delta across a section to get that section's replayed time.
    replayed_s: float = 0.0
    # Per-thread copy of the same total: with cases analyzed in batches, the
    # run-wide delta would credit one case with another thread's replay.
    _local: threading.local = field(default_factory=threading.local, repr=False)

    @property
    def thread_replayed_s(self) -> float:
        """The calling thread's share of ``replayed_s``."""
        return getattr(self._local, "replayed_s", 0.0)

    def record_replay(self, elapsed_s: float) -> None:
        """Count a cache hit standing in for a call that took ``elapsed_s``."""
        self.replayed_s += elapsed_s
        self._local.replayed_s = self.thread_replayed_s + elapsed_s

    def summary(self) -> str:
        calls = self.hits + self.misses
        return (
            f"{self.label}: {self.hits}/{calls} served from disk, "
            f"{self.misses} {self.miss_label}"
        )


@dataclass
class CachedCall:
    """A cache hit: the stored response plus how long the original call took."""

    response: dict[str, Any]
    elapsed_s: float


def jsonable(value: Any) -> Any:
    """Best-effort, stable JSON view of a request argument.

    Schemas, configs and message lists arrive as dicts or small dataclasses;
    anything exotic falls back to ``repr``, which is still stable for equal
    inputs.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return repr(value)


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _slug(model: str) -> str:
    """Filesystem-safe directory name for a model (e.g. ``openai/x`` → ``openai_x``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model) or "unknown"


class DiskCache:
    """Reads/writes cache entries under ``<cache_dir>/<model>/<digest>.json``.

    ``key`` is the provider's own description of the request (it must carry a
    ``version``, bumped when its layout changes). It is hashed into the entry
    name and stored alongside the response, so a stale layout or a changed
    request can never be served as a hit.
    """

    def __init__(
        self,
        cache_dir: Path,
        refresh: bool = False,
        label: str = "cache",
        miss_label: str = "real call(s)",
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.stats = CacheStats(label=label, miss_label=miss_label)

    def _entry_path(self, model: str, digest: str) -> Path:
        return self.cache_dir / _slug(model) / f"{digest}.json"

    def lookup(
        self, model: str, key: dict[str, Any]
    ) -> tuple[str, Optional[CachedCall]]:
        """Return (digest, cached call or None)."""
        digest = hashlib.sha256(_canonical(key).encode("utf-8")).hexdigest()
        if self.refresh:
            return digest, None

        path = self._entry_path(model, digest)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return digest, None

        # Guard against a stale layout or (astronomically unlikely) collision:
        # the entry has to describe the very request we are about to make.
        if entry.get("version") != key.get("version") or entry.get("request") != key:
            return digest, None
        response = entry.get("response")
        if not isinstance(response, dict):
            return digest, None
        # Entries written before timings were recorded have no elapsed_s; they
        # replay as 0s (as they always did) until they are refreshed.
        elapsed = response.get("elapsed_s")
        elapsed_s = float(elapsed) if isinstance(elapsed, (int, float)) else 0.0
        return digest, CachedCall(response=response, elapsed_s=elapsed_s)

    def store(
        self,
        model: str,
        digest: str,
        key: dict[str, Any],
        response: dict[str, Any],
        elapsed_s: float,
    ) -> None:
        path = self._entry_path(model, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": key.get("version"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    # The full request is kept both to validate the key on read
                    # and so a cached answer can be inspected by hand.
                    "request": key,
                    "response": {**response, "elapsed_s": round(elapsed_s, 3)},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
