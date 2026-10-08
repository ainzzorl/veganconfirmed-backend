"""Aggregating samples and rendering them.

Reps are summarised by the **median**, not the mean: N is small (3 by default)
and the noise is one-sided — a background process or a thermal blip lengthens a
run and nothing shortens it — which is exactly where a mean misleads. Every raw
rep goes into the results JSON, so a mean or a percentile can be recomputed
later without re-running anything.

The per-model rollup is the one figure that is *not* a median. It is
token-weighted (``sum(completion_tokens) / sum(decode_s)``) because averaging
per-cell rates would weight a 128-token cell the same as a 512-token one.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tests.perf.lmstudio import Sample

# Metrics summarised across reps, in report order.
MEDIAN_FIELDS = (
    "ttft_s",
    "decode_tok_s",
    "prefill_tok_s",
    "generation_s",
    "wall_s",
    "overhead_s",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "server_tok_s",
)


@dataclass
class CellResult:
    id: str
    family: str
    status: str = "ok"
    error: Optional[str] = None
    retries: int = 0
    nominal_prompt_tokens: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)
    samples: list[Sample] = field(default_factory=list)

    @property
    def ok_samples(self) -> list[Sample]:
        return [s for s in self.samples if s.ok]

    def median(self) -> dict[str, Optional[float]]:
        return _median_of(self.ok_samples)

    def stop_reason(self) -> Optional[str]:
        ok = self.ok_samples
        return ok[-1].stop_reason if ok else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "status": self.status,
            "error": self.error,
            "retries": self.retries,
            "nominal_prompt_tokens": self.nominal_prompt_tokens,
            "meta": self.meta,
            "median": self.median(),
            "spread": _spread_of(self.ok_samples),
            "samples": [s.as_dict() for s in self.samples],
        }


@dataclass
class ModelResult:
    model: str
    status: str = "ok"
    error: Optional[str] = None
    load_s: Optional[float] = None
    warmup_s: Optional[float] = None
    info: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    cells: list[CellResult] = field(default_factory=list)

    def rollup(self) -> dict[str, Optional[float]]:
        samples = [s for c in self.cells for s in c.ok_samples]
        return {
            "decode_tok_s": _weighted(samples, "completion_tokens", "decode_s"),
            "prefill_tok_s": _weighted(samples, "prompt_tokens", "ttft_s"),
            "cells_ok": sum(1 for c in self.cells if c.status == "ok"),
            "cells": len(self.cells),
            "retries": sum(c.retries for c in self.cells),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "error": self.error,
            "load_s": self.load_s,
            "warmup_s": self.warmup_s,
            "info": self.info,
            "tokenizer": self.tokenizer,
            "rollup": self.rollup(),
            "cells": [c.as_dict() for c in self.cells],
        }


def _median_of(samples: list[Sample]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for name in MEDIAN_FIELDS:
        values = [v for v in (getattr(s, name) for s in samples) if v is not None]
        out[name] = statistics.median(values) if values else None
    return out


def _spread_of(samples: list[Sample]) -> dict[str, Any]:
    """Min/max of the two headline metrics, so a noisy cell is visible in the JSON."""
    out: dict[str, Any] = {}
    for name in ("ttft_s", "decode_tok_s"):
        values = [v for v in (getattr(s, name) for s in samples) if v is not None]
        out[name] = {"min": min(values), "max": max(values)} if values else None
    return out


def _weighted(
    samples: list[Sample], numerator: str, denominator: str
) -> Optional[float]:
    total_n = sum(getattr(s, numerator) or 0 for s in samples)
    total_d = sum(getattr(s, denominator) or 0 for s in samples)
    return total_n / total_d if total_d else None


# ------------------------------------------------------------------ rendering

_STOP_ABBR = {
    "maxPredictedTokensReached": "cap",
    "eosFound": "eos",
    "stopStringFound": "stop",
    "contextLengthReached": "ctx",
}


def _num(value: Optional[float], width: int, places: int = 1) -> str:
    return f"{value:>{width}.{places}f}" if value is not None else f"{'·':>{width}}"


def _int(value: Optional[float], width: int) -> str:
    return f"{int(value):>{width}}" if value is not None else f"{'·':>{width}}"


def render(models: list[ModelResult], config: dict[str, Any], environment: dict) -> str:
    lines: list[str] = []
    runtime = environment.get("runtime") or {}
    lines.append("=" * 100)
    lines.append("LM Studio performance benchmark")
    cap = (
        f"  real_cap={config['max_tokens'] or 'production'}"
        if config["workload"] != "synthetic"
        else ""
    )
    lines.append(
        f"  workload={config['workload']}  reps={config['reps']}  "
        f"ctx={config['context_length']}  parallel={config['parallel']}"
        + cap
        + f"  cache={'bust' if config['nonce'] else 'allow'}"
    )
    if runtime:
        lines.append(f"  runtime={runtime.get('name')} {runtime.get('version')}")
    lines.append("=" * 100)

    header = (
        f"{'model':<30} {'cell':<34} {'ptok':>6} {'ctok':>6} {'rtok':>6} "
        f"{'ttft_s':>7} {'prefill/s':>10} {'decode/s':>9} {'gen_s':>7} {'stop':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for m in models:
        if m.status != "ok" and not m.cells:
            lines.append(f"{m.model:<30} {m.status.upper()}: {m.error}")
            continue
        for cell in m.cells:
            if cell.status != "ok":
                lines.append(
                    f"{m.model:<30} {cell.id:<34} "
                    f"{cell.status.upper()}{': ' + cell.error if cell.error else ''}"
                )
                continue
            med = cell.median()
            lines.append(
                f"{m.model:<30} {cell.id:<34} "
                f"{_int(med['prompt_tokens'], 6)} {_int(med['completion_tokens'], 6)} "
                f"{_int(med['reasoning_tokens'], 6)} {_num(med['ttft_s'], 7, 3)} "
                f"{_num(med['prefill_tok_s'], 10)} {_num(med['decode_tok_s'], 9)} "
                f"{_num(med['generation_s'], 7)} "
                f"{_STOP_ABBR.get(cell.stop_reason() or '', '·'):>5}"
            )

    lines.append("")
    roll_header = (
        f"{'model':<30} {'arch':<12} {'quant':<8} {'load_s':>7} "
        f"{'prefill/s':>10} {'decode/s':>9} {'cells':>7} {'flakes':>7}  status"
    )
    lines.append(roll_header)
    lines.append("-" * len(roll_header))
    for m in models:
        r = m.rollup()
        status = m.status if m.status == "ok" else f"{m.status}: {m.error}"
        cells = f"{r['cells_ok']}/{r['cells']}"
        lines.append(
            f"{m.model:<30} {str(m.info.get('arch') or '·'):<12} "
            f"{str(m.info.get('quant') or '·'):<8} {_num(m.load_s, 7)} "
            f"{_num(r['prefill_tok_s'], 10)} {_num(r['decode_tok_s'], 9)} "
            f"{cells:>7} {r['retries']:>7}  {status}"
        )

    mismatched = [
        f"{m.model} {c.id}"
        for m in models
        for c in m.cells
        for s in c.ok_samples
        if s.stats_mismatch
    ]
    if mismatched:
        lines.append("")
        lines.append(
            "WARNING: LM Studio's tokens_per_second no longer matches "
            "completion_tokens/(generation_time-ttft) for: "
            + ", ".join(sorted(set(mismatched)))
        )
    return "\n".join(lines)


def write_json(
    path: Path, models: list[ModelResult], config: dict[str, Any], environment: dict
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": config,
                "environment": environment,
                "aggregate": {
                    "models": len(models),
                    "ok": sum(1 for m in models if m.status == "ok"),
                    "failed": sum(1 for m in models if m.status != "ok"),
                },
                "models": [m.as_dict() for m in models],
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return path
