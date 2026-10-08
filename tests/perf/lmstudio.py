"""HTTP client for LM Studio's native REST API (``/api/v0``).

The OpenAI-compatible ``/v1`` endpoint the rest of the stack uses reports no
timings. ``/api/v0`` returns a ``stats`` block on every non-streaming
completion, which is where every number in this benchmark comes from::

    "stats": {"tokens_per_second": 22.046, "time_to_first_token": 0.0699,
              "generation_time": 13.678, "stop_reason": "maxPredictedTokensReached"}

Two things about that block, both measured rather than assumed:

- ``generation_time`` is the server's *total* time and includes prefill, while
  ``tokens_per_second`` is the *pure decode* rate. The identity holds exactly:
  ``tokens_per_second == completion_tokens / (generation_time - ttft)``. So
  ``completion_tokens / generation_time`` is not the decode rate — on a long
  prompt it is roughly half of it.
- A *streaming* request emits no stats chunk at all (the last chunk is a bare
  ``finish_reason``, then ``[DONE]``), so requests here are non-streaming.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import requests

# The types /api/v0/models reports for text models; anything else (today only
# "embeddings") cannot serve a chat completion and is never benchmarked.
LLM_TYPES = ("llm", "vlm")

# gpt-oss emits `<|channel|>final <|constrain|>json<|message|>` on a small
# fraction of requests and llama.cpp's grammar rejects it, discarding an
# otherwise valid answer. Measured at 5-8% of requests, serially. It is a flake,
# not a measurement, so a request that hits it is retried rather than recorded.
HARMONY_MARKER = "does not match the expected"

# How far the recomputed decode rate may drift from LM Studio's own before the
# sample is flagged: the two agreed to 14 significant figures when measured, so
# any real divergence means the server changed its accounting.
STATS_TOLERANCE = 0.02


def api_base() -> str:
    """The ``/api/v0`` root, derived from the same env var the rest of the repo reads."""
    base = os.getenv("LMS_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base + "/api/v0"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    type: str
    arch: str
    publisher: str
    quantization: Optional[str]
    state: str
    max_context_length: int
    loaded_context_length: Optional[int]

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ModelInfo":
        return cls(
            id=d["id"],
            type=d.get("type", ""),
            arch=d.get("arch", ""),
            publisher=d.get("publisher", ""),
            quantization=d.get("quantization"),
            state=d.get("state", ""),
            max_context_length=int(d.get("max_context_length") or 0),
            loaded_context_length=d.get("loaded_context_length"),
        )


@dataclass(frozen=True)
class Sample:
    """One measured request. Never raises: a failure comes back with ``ok=False``."""

    ok: bool
    wall_s: float
    ttft_s: Optional[float] = None
    generation_s: Optional[float] = None
    server_tok_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    finish_reason: Optional[str] = None
    model_info: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_class: Optional[str] = None

    @property
    def decode_s(self) -> Optional[float]:
        """Server time spent decoding, i.e. total minus prefill."""
        if self.generation_s is None or self.ttft_s is None:
            return None
        return self.generation_s - self.ttft_s

    @property
    def decode_tok_s(self) -> Optional[float]:
        decode = self.decode_s
        if not decode or not self.completion_tokens:
            return None
        return self.completion_tokens / decode

    @property
    def prefill_tok_s(self) -> Optional[float]:
        if not self.ttft_s or not self.prompt_tokens:
            return None
        return self.prompt_tokens / self.ttft_s

    @property
    def overhead_s(self) -> Optional[float]:
        """Client-side round trip minus the server's own accounting (HTTP + JSON)."""
        if self.generation_s is None:
            return None
        return self.wall_s - self.generation_s

    @property
    def stats_mismatch(self) -> bool:
        """True if LM Studio's reported rate no longer matches the recomputed one."""
        mine, theirs = self.decode_tok_s, self.server_tok_s
        if mine is None or not theirs:
            return False
        return abs(mine - theirs) / theirs > STATS_TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            decode_s=self.decode_s,
            decode_tok_s=self.decode_tok_s,
            prefill_tok_s=self.prefill_tok_s,
            overhead_s=self.overhead_s,
            stats_mismatch=self.stats_mismatch,
        )
        return d

    @classmethod
    def from_response(cls, body: dict[str, Any], wall_s: float) -> "Sample":
        stats = body.get("stats") or {}
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        choices = body.get("choices") or [{}]
        return cls(
            ok=True,
            wall_s=wall_s,
            ttft_s=stats.get("time_to_first_token"),
            generation_s=stats.get("generation_time"),
            server_tok_s=stats.get("tokens_per_second"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            stop_reason=stats.get("stop_reason"),
            finish_reason=choices[0].get("finish_reason"),
            model_info=body.get("model_info") or {},
            runtime=body.get("runtime") or {},
        )


def list_models(timeout: float = 10) -> list[ModelInfo]:
    resp = requests.get(api_base() + "/models", timeout=timeout)
    resp.raise_for_status()
    return [ModelInfo.from_json(d) for d in resp.json().get("data", [])]


def llm_models(timeout: float = 10) -> list[ModelInfo]:
    """Every model that can serve a chat completion, embeddings excluded."""
    return [m for m in list_models(timeout) if m.type in LLM_TYPES]


def reachable(timeout: float = 2) -> bool:
    try:
        return requests.get(api_base() + "/models", timeout=timeout).status_code == 200
    except Exception:
        return False


def chat(
    request: dict[str, Any],
    *,
    timeout: float,
    session: Optional[requests.Session] = None,
) -> Sample:
    """POST one completion and return what it cost. Failures come back as samples."""
    http = session or requests
    started = time.perf_counter()
    try:
        resp = http.post(
            api_base() + "/chat/completions",
            json=request,
            timeout=(5, timeout),
        )
    except requests.Timeout as e:
        return Sample(
            ok=False,
            wall_s=time.perf_counter() - started,
            error=str(e),
            error_class="timeout",
        )
    except requests.RequestException as e:
        return Sample(
            ok=False,
            wall_s=time.perf_counter() - started,
            error=str(e),
            error_class="transport",
        )
    wall_s = time.perf_counter() - started

    if resp.status_code != 200:
        detail = resp.text[:400]
        return Sample(
            ok=False,
            wall_s=wall_s,
            error=f"HTTP {resp.status_code}: {detail}",
            error_class="harmony" if HARMONY_MARKER in detail else "http",
        )
    try:
        body = resp.json()
    except ValueError as e:
        return Sample(ok=False, wall_s=wall_s, error=str(e), error_class="malformed")

    if not (body.get("stats") or {}).get("generation_time"):
        return Sample(
            ok=False,
            wall_s=wall_s,
            error="response carried no stats block",
            error_class="malformed",
        )
    return Sample.from_response(body, wall_s)
