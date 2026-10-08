"""CLI and measurement loop for the LM Studio performance benchmark.

Model-major, one request at a time: loading is by far the most expensive
operation, so a model is loaded once, warmed up once, and then every cell and
rep runs against that instance before the next model is loaded.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from services.page_prompt import CHARS_PER_TOKEN
from tests.perf import corpus, lmscli, lmstudio, report, workloads
from tests.perf.report import CellResult, ModelResult

DEFAULT_CONTEXT_LENGTH = 16384
DEFAULT_PROMPT_SIZES = (512, 2048, 8192)
DEFAULT_OUTPUT_SIZES = (128, 512)
# Production's LMS_MAX_TOKENS: what the page budget is planned against, so the
# real prompts come out the size the app would send.
PAGE_MAX_TOKENS = 8192
# Three timeouts in a row means the model cannot serve this workload at all;
# continuing just spends the rest of the budget discovering that again.
MAX_CONSECUTIVE_TIMEOUTS = 3
RESULTS_DIR = Path(__file__).parent / "results"


def _log(message: str, indent: int = 0, when: bool = True) -> None:
    """Progress goes to stderr so stdout stays the report alone (and pipeable).

    ``when`` is how the -v-only lines opt in: without it a run prints one line
    per model plus anything that went wrong, which is all an unattended sweep
    needs.
    """
    if when:
        print(" " * indent + message, file=sys.stderr, flush=True)


def _elapsed(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.1f}m"


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _ints(value: str) -> list[int]:
    return [int(part) for part in _csv(value)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.perf",
        description="Benchmark LM Studio models on speed alone: TTFT and tokens/second.",
    )
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="model id; repeatable or comma-separated",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="every text model LM Studio has (embeddings excluded)",
    )
    p.add_argument(
        "--exclude", action="append", default=[], help="model id to leave out of --all"
    )

    p.add_argument("--workload", choices=("real", "synthetic", "all"), default="real")
    p.add_argument(
        "--cases",
        type=_csv,
        default=list(corpus.DEFAULT_CASE_IDS),
        help="real: corpus case ids",
    )
    p.add_argument("--prompt-sizes", type=_ints, default=list(DEFAULT_PROMPT_SIZES))
    p.add_argument("--output-sizes", type=_ints, default=list(DEFAULT_OUTPUT_SIZES))

    p.add_argument("--reps", type=int, default=3, help="measured repetitions per cell")
    p.add_argument(
        "--warmup", type=int, default=1, help="discarded requests after each load"
    )
    p.add_argument(
        "--no-nonce",
        dest="nonce",
        action="store_false",
        help="reuse the identical prompt every rep, letting LM Studio's KV cache serve it "
        "(TTFT then measures the cache, not prefill)",
    )

    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="completion cap; 0 uses production's own cap for the real workload",
    )
    p.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default="low"
    )

    p.add_argument(
        "--no-load",
        dest="load",
        action="store_false",
        help="measure whatever is already loaded",
    )
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--load-timeout", type=float, default=900)

    p.add_argument(
        "--timeout", type=float, default=300, help="per-request read timeout"
    )
    p.add_argument(
        "--model-budget", type=float, default=900, help="wall-clock seconds per model"
    )
    p.add_argument(
        "--retries", type=int, default=2, help="retries for the gpt-oss parser flake"
    )

    p.add_argument("--out", type=Path, help="results JSON destination")
    p.add_argument(
        "--dry-run", action="store_true", help="print the plan and call nothing"
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="progress log on stderr: load, warm-up, tokenizer and every rep",
    )

    p.add_argument(
        "--refresh-corpus",
        action="store_true",
        help="re-extract the frozen page text and exit",
    )
    p.add_argument(
        "--node",
        type=Path,
        default=Path("../veganconfirmed-extension/tools/extract_content.js"),
    )
    p.add_argument(
        "--node-pdf",
        dest="node_pdf",
        type=Path,
        default=Path("../veganconfirmed-extension/tools/extract_pdf.js"),
    )
    return p


def _resolve_models(args: argparse.Namespace) -> list[str]:
    named = [m for entry in args.model for m in _csv(entry)]
    if named and args.all:
        raise SystemExit("--model and --all are mutually exclusive")
    if not named and not args.all:
        raise SystemExit("name at least one --model, or pass --all")
    if named:
        return named
    excluded = {m for entry in args.exclude for m in _csv(entry)}
    return [m.id for m in lmstudio.llm_models() if m.id not in excluded]


def _prompts(
    args: argparse.Namespace, tokenizer: workloads.Tokenizer
) -> list[workloads.Prompt]:
    prompts: list[workloads.Prompt] = []
    if args.workload in ("synthetic", "all"):
        prompts += workloads.synthetic_prompts(
            args.prompt_sizes,
            args.output_sizes,
            tokenizer=tokenizer,
            context_length=args.context_length,
        )
    if args.workload in ("real", "all"):
        wanted = set(args.cases)
        cases = [c for c in corpus.load()["cases"] if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            raise SystemExit(
                f"corpus.json has no case(s): {', '.join(sorted(missing))}"
            )
        prompts += workloads.real_prompts(
            cases,
            context_length=args.context_length,
            page_max_tokens=PAGE_MAX_TOKENS,
            output_cap=args.max_tokens or None,
            reasoning_effort=args.reasoning_effort,
        )
    return prompts


def _measure_cell(
    prompt: workloads.Prompt,
    model: str,
    args: argparse.Namespace,
    session: requests.Session,
) -> CellResult:
    cell = CellResult(
        id=prompt.id,
        family=prompt.family,
        nominal_prompt_tokens=prompt.nominal_prompt_tokens,
        meta=dict(prompt.meta, max_tokens=prompt.max_tokens),
    )
    timeouts = 0
    for rep in range(args.reps):
        for attempt in range(args.retries + 1):
            nonce = uuid.uuid4().hex[:8] if args.nonce else None
            sample = lmstudio.chat(
                prompt.body(model, nonce), timeout=args.timeout, session=session
            )
            # A flake is not a measurement: retry it rather than record it.
            if sample.ok or sample.error_class not in ("harmony", "http", "malformed"):
                break
            cell.retries += 1
        detail = (
            f"ttft {sample.ttft_s:.3f}s  decode {sample.decode_tok_s:.1f} t/s  "
            f"ptok {sample.prompt_tokens}  ctok {sample.completion_tokens}"
            if sample.ok
            else f"{sample.error_class}: {sample.error}"
        )
        _log(
            f"rep {rep + 1}/{args.reps} {prompt.id:<34} {detail}",
            indent=10,
            when=args.verbose,
        )
        cell.samples.append(sample)
        if sample.ok:
            timeouts = 0
            continue
        timeouts = timeouts + 1 if sample.error_class == "timeout" else 0
        if timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
            cell.status, cell.error = "abandoned", "timed out repeatedly"
            return cell
    if not cell.ok_samples:
        last = cell.samples[-1] if cell.samples else None
        cell.status = "failed"
        cell.error = last.error if last else "no samples"
    return cell


def _cell_line(cell: CellResult, position: str, elapsed: float) -> str:
    """One finished cell, summarised the way the report will summarise it."""
    if cell.status != "ok":
        detail = f"{cell.status.upper()}" + (f": {cell.error}" if cell.error else "")
        return f"{position} {cell.id:<34} {detail}"
    med = cell.median()
    retries = f"  {cell.retries} retried" if cell.retries else ""
    return (
        f"{position} {cell.id:<34} "
        f"ttft {med['ttft_s']:.3f}s  prefill {med['prefill_tok_s']:>6.0f} t/s  "
        f"decode {med['decode_tok_s']:>5.1f} t/s  "
        f"ptok {int(med['prompt_tokens']):>5}  ctok {int(med['completion_tokens']):>4}"
        f"  [{_elapsed(elapsed)}]{retries}"
    )


def _run_model(model: str, args: argparse.Namespace, position: str = "") -> ModelResult:
    result = ModelResult(model=model)
    session = requests.Session()
    model_started = time.perf_counter()
    _log(f"{position} {model}".strip())

    if args.load:
        _log(
            f"loading (ctx={args.context_length}, parallel={args.parallel})...",
            indent=6,
            when=args.verbose,
        )
        lmscli.unload_all()
        try:
            result.load_s = lmscli.load(
                model,
                context_length=args.context_length,
                parallel=args.parallel,
                timeout=args.load_timeout,
            )
            instance = lmscli.verify_instance(
                model, context_length=args.context_length, parallel=args.parallel
            )
        except lmscli.LoadFailed as e:
            result.status, result.error = "load_failed", str(e)
            _log(f"LOAD FAILED: {e}", indent=6)
            return result
        result.info = {
            "arch": instance.get("architecture"),
            "quant": (instance.get("quantization") or {}).get("name"),
            "context_length": instance.get("contextLength"),
            "parallel": instance.get("parallel"),
            "size_bytes": instance.get("sizeBytes"),
        }
        _log(
            f"loaded in {result.load_s:.1f}s  "
            f"{result.info['arch']} {result.info['quant']}",
            indent=6,
            when=args.verbose,
        )

    # The first request after a load pays one-off costs (kernel setup, weights
    # paged in) that no later one repeats, so it is timed and thrown away.
    for _ in range(args.warmup):
        started = time.perf_counter()
        warm = lmstudio.chat(
            {
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=args.timeout,
            session=session,
        )
        result.warmup_s = time.perf_counter() - started
        if not warm.ok:
            result.status, result.error = "unusable", warm.error
            _log(f"UNUSABLE: {warm.error}", indent=6)
            return result
        result.info.setdefault("arch", (warm.model_info or {}).get("arch"))
        result.info.setdefault("quant", (warm.model_info or {}).get("quant"))
        result.info["runtime"] = warm.runtime
        _log(
            f"warm-up {result.warmup_s:.1f}s (discarded)",
            indent=6,
            when=args.verbose,
        )

    # Prompt sizes are measured against this model's own tokenizer rather than
    # estimated, so the synthetic grid lands on the sizes it claims.
    tokenizer = workloads.Tokenizer(overhead_tokens=0, chars_per_token=CHARS_PER_TOKEN)
    if args.workload != "real":
        tokenizer = workloads.calibrate(model, timeout=args.timeout, session=session)
    result.tokenizer = {
        "overhead_tokens": tokenizer.overhead_tokens,
        "chars_per_token": tokenizer.chars_per_token,
    }
    if args.workload != "real":
        _log(
            f"tokenizer {tokenizer.chars_per_token:.2f} chars/token, "
            f"{tokenizer.overhead_tokens} tokens of prompt overhead",
            indent=6,
            when=args.verbose,
        )
    prompts = _prompts(args, tokenizer)

    started = time.perf_counter()
    for index, prompt in enumerate(prompts):
        if time.perf_counter() - started > args.model_budget:
            _log(
                f"budget of {args.model_budget:.0f}s spent; "
                f"skipping {len(prompts) - index} remaining cell(s)",
                indent=6,
            )
            for remaining in prompts[index:]:
                result.cells.append(
                    CellResult(
                        id=remaining.id,
                        family=remaining.family,
                        status="budget_exceeded",
                        error=f"model budget of {args.model_budget:.0f}s spent",
                    )
                )
            break
        cell_started = time.perf_counter()
        cell = _measure_cell(prompt, model, args, session)
        result.cells.append(cell)
        _log(
            _cell_line(
                cell,
                f"[{index + 1:>2}/{len(prompts)}]",
                time.perf_counter() - cell_started,
            ),
            indent=6,
            when=args.verbose,
        )

    if not any(c.status == "ok" for c in result.cells):
        result.status = "failed"
        result.error = result.error or "every cell failed"
    if args.load:
        lmscli.unload_all()
    _log(
        f"{model} done in {_elapsed(time.perf_counter() - model_started)}",
        indent=6,
        when=args.verbose,
    )
    return result


def run(args: argparse.Namespace) -> int:
    if args.refresh_corpus:
        built = corpus.build(
            tuple(args.cases), corpus.DEFAULT_CASES_DIR, args.node, args.node_pdf
        )
        path = corpus.write(built)
        sizes = ", ".join(f"{c['id']}={c['chars']}c" for c in built["cases"])
        print(f"wrote {path} (sha {built['corpus_sha']}): {sizes}")
        return 0

    if not lmstudio.reachable():
        raise SystemExit(
            f"LM Studio is not answering at {lmstudio.api_base()}; "
            "start it with `lms server start`"
        )

    models = _resolve_models(args)

    if args.dry_run:
        # Only the shape of the grid is wanted here, so the estimator's ratio
        # stands in; a real run remeasures it per model, once loaded.
        planned = _prompts(
            args,
            workloads.Tokenizer(overhead_tokens=0, chars_per_token=CHARS_PER_TOKEN),
        )
        print(f"{len(models)} model(s): {', '.join(models)}")
        for prompt in planned:
            print(f"  {prompt.id:<28} max_tokens={prompt.max_tokens}")
        print(
            f"{len(models)} x {len(planned)} cells x {args.reps} reps = "
            f"{len(models) * len(planned) * args.reps} requests"
            f" (+{len(models) * args.warmup} warm-up)"
        )
        return 0

    config = {
        "workload": args.workload,
        "reps": args.reps,
        "warmup": args.warmup,
        "nonce": args.nonce,
        "context_length": args.context_length,
        "parallel": args.parallel,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "cases": args.cases,
        "prompt_sizes": args.prompt_sizes,
        "output_sizes": args.output_sizes,
        "load": args.load,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_sha": corpus.load()["corpus_sha"]
        if args.workload != "synthetic"
        else None,
    }
    environment: dict[str, Any] = {
        "api_base": lmstudio.api_base(),
        "host": platform.node(),
        "python": platform.python_version(),
    }
    out = args.out or RESULTS_DIR / (
        f"perf-{args.workload}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )

    results: list[ModelResult] = []
    run_started = time.perf_counter()
    try:
        for index, model in enumerate(models):
            results.append(
                _run_model(model, args, position=f"[{index + 1}/{len(models)}]")
            )
            environment.setdefault(
                "runtime", (results[-1].info or {}).get("runtime") or {}
            )
    except KeyboardInterrupt:
        _log("\ninterrupted; writing partial results")
    finally:
        report.write_json(out, results, config, environment)

    print()
    print(report.render(results, config, environment))
    print()
    print(f"results: {out}  ({_elapsed(time.perf_counter() - run_started)} total)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))
