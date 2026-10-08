"""Eval runner: extract → analyze → score → report.

For each case in the corpus:
  1. Run the extension's real extractor over the saved fixture via a Node
     harness, yielding exactly what the extension would send to the backend:
     `extractPageContent` (content.js) over an <id>.html snapshot, or
     `extractPdfText` (pdf_extract.mjs) over an <id>.pdf menu.
  2. Feed it through `AnalysisCore.analyze_page` (enable_database=False) using
     the configured provider (desktop LM Studio or real Gemini) and model.
  3. Compare the predicted analysis against the case's declared expectations.

Then print a metrics report and write a machine-readable results JSON.

Provider/model are selected by setting env vars BEFORE importing AnalysisCore
(it reads them at construction): identical selection to production.

Combos to benchmark are given as `--run provider:model[:effort]` (repeated or
comma-separated); there are no default providers or models.

Usage:
  python -m tests.eval --run gemini:gemini-2.5-flash-lite
  python -m tests.eval --run desktop:openai/gpt-oss-120b
    (desktop needs the Firestore emulator + worker — use tests/eval/run.sh)

  # Compare several models + write an HTML report:
  python -m tests.eval --html \
      --run gemini:gemini-2.5-flash-lite --run gemini:gemini-2.5-flash

  # Mix providers in one run (Gemini vs. a local LM Studio model):
  python -m tests.eval --html \
      --run gemini:gemini-2.5-flash-lite --run desktop:openai/gpt-oss-120b

  # Compare reasoning-effort levels of one local model (desktop only):
  python -m tests.eval --html --run desktop:openai/gpt-oss-20b:low,\
desktop:openai/gpt-oss-20b:medium,desktop:openai/gpt-oss-20b:high

  # Analyze several cases at once (desktop: needs a worker and a model loaded
  # with at least that many prediction slots; run.sh arranges both):
  tests/eval/run.sh --run desktop:openai/gpt-oss-20b --max-batch-size 4

Cases are extracted once and replayed against every combo; the run prints a text
report per combo and (with --html) a combined comparison report. A combo's
provider, model and effort switch the env AnalysisCore reads, identical to
production.

Both providers' responses are cached on disk (keyed on the exact model +
request, so a change of model, reasoning effort or prompt is a different key) —
re-runs don't pay for the same Gemini call, or wait through the same local job,
twice. The call's duration is cached with it so replayed cases still report a
real latency; see tests/eval/gemini_cache.py, tests/eval/desktop_cache.py and
the --cache-dir / --refresh-cache / --no-cache flags.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Optional

from dotenv import load_dotenv

from services.desktop_service import REASONING_EFFORTS
from tests.eval import desktop_cache, gemini_cache, metrics
from tests.eval.response_cache import CacheStats

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES = _REPO_ROOT / "tests" / "eval" / "cases"
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "tests" / "eval" / "results"
_DEFAULT_CACHE_DIR = _REPO_ROOT / "tests" / "eval" / ".cache"
_DEFAULT_NODE_HARNESS = (
    _REPO_ROOT.parent / "veganconfirmed-extension" / "tools" / "extract_content.js"
)
_DEFAULT_PDF_HARNESS = (
    _REPO_ROOT.parent / "veganconfirmed-extension" / "tools" / "extract_pdf.js"
)

# Exit code tools/extract_pdf.js uses for a PDF with no usable text layer, as
# opposed to a harness that genuinely broke.
PDF_NO_TEXT_EXIT = 3

PROVIDERS = ("desktop", "gemini")


class Combo(NamedTuple):
    """One benchmarked configuration, as written on the command line.

    ``effort`` is the reasoning effort to request of a local model (desktop
    only); None leaves the model at its own default, which is what a run
    without an effort measures.
    """

    provider: str
    model: str
    effort: Optional[str] = None

    def label(self) -> str:
        return f"{self.provider}:{self.model}" + (
            f":{self.effort}" if self.effort else ""
        )


def _validate_provider_ready(provider: str) -> None:
    """Fail fast if a provider's prerequisites are missing (env/emulator)."""
    if provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY is required for the gemini provider")
    elif provider == "desktop":
        if not os.getenv("FIRESTORE_EMULATOR_HOST"):
            sys.exit(
                "FIRESTORE_EMULATOR_HOST not set; run desktop evals via "
                "tests/eval/run.sh (brings up the emulator + worker)"
            )
    else:  # pragma: no cover - argparse/_resolve_combos restrict choices
        sys.exit(f"unknown provider: {provider}")


def _configure_provider(combo: Combo) -> None:
    """Set the env vars AnalysisCore reads, before it is imported/constructed.

    Called per combo, so a single run can switch providers: each combo
    reconstructs AnalysisCore against freshly-set env.
    """
    if combo.provider == "gemini":
        os.environ["USE_DESKTOP_SERVER"] = "false"
        os.environ["DISABLE_GEMINI_FALLBACK"] = "false"
        os.environ["GEMINI_MODEL"] = combo.model
    elif combo.provider == "desktop":
        os.environ["USE_DESKTOP_SERVER"] = "true"
        os.environ["DISABLE_GEMINI_FALLBACK"] = "true"
        os.environ["LMS_MODEL"] = combo.model
        # Set for every desktop combo, cleared when the combo names no effort:
        # what a combo is labelled with is then exactly what it ran at, whatever
        # the ambient environment says and whichever combo ran before it.
        if combo.effort:
            os.environ["LMS_REASONING_EFFORT"] = combo.effort
        else:
            os.environ.pop("LMS_REASONING_EFFORT", None)
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "desktop-server-test")
    else:  # pragma: no cover - argparse/_resolve_combos restrict choices
        sys.exit(f"unknown provider: {combo.provider}")
    _validate_provider_ready(combo.provider)


def _load_cases(
    cases_dir: Path, only: Optional[str], types: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """Load case manifests (<id>.json) and locate each sibling fixture.

    A fixture is either an <id>.html page snapshot or an <id>.pdf menu; which
    one it is decides the extractor the case runs through. ``types`` keeps only
    the cases whose expected page_kind is one of them.

    A manifest may set ``skip`` to a reason string, which parks the case: it is
    left out of the corpus entirely rather than scored as a failure, since a
    case the stack cannot yet serve would otherwise drag every aggregate down.
    ``--case <id>`` still runs it, so a parked case stays easy to work on.
    """
    cases = []
    for manifest_path in sorted(cases_dir.glob("*.json")):
        case_id = manifest_path.stem
        if only and case_id != only:
            continue
        manifest = json.loads(manifest_path.read_text())
        if types and metrics.case_type(manifest.get("expect", {})) not in types:
            continue

        skip = manifest.get("skip")
        if skip and not only:
            print(f"[{case_id}] skipped: {skip}", file=sys.stderr)
            continue

        html_path = manifest_path.with_suffix(".html")
        pdf_path = manifest_path.with_suffix(".pdf")
        if html_path.exists():
            fixture, kind = html_path, "html"
        elif pdf_path.exists():
            fixture, kind = pdf_path, "pdf"
        else:
            raise FileNotFoundError(
                f"case '{case_id}': expected a fixture at {html_path} or {pdf_path}"
            )

        cases.append(
            {
                "id": case_id,
                "fixture": fixture,
                "kind": kind,
                "manifest": manifest,
            }
        )
    if not cases:
        raise SystemExit(
            f"no cases found in {cases_dir}"
            + (f" matching '{only}'" if only else "")
            + (f" of type {', '.join(types)}" if types else "")
        )
    return cases


def _extract(
    node_harness: Path, html_path: Path, url: str, title: Optional[str]
) -> dict[str, Any]:
    """Run the Node/jsdom harness and return its extracted payload."""
    cmd = ["node", str(node_harness), "--html", str(html_path), "--url", url]
    if title:
        cmd += ["--title", title]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"extraction failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def _extract_pdf(
    pdf_harness: Path, pdf_path: Path, url: str, title: Optional[str]
) -> dict[str, Any]:
    """Run the extension's PDF harness and return its extracted payload.

    A PDF whose text layer is missing or decorative exits ``PDF_NO_TEXT_EXIT``
    and is reported as a case error — the same thing the extension tells the
    user, rather than a menu analyzed from the prices alone.
    """
    cmd = ["node", str(pdf_harness), "--pdf", str(pdf_path), "--url", url]
    if title:
        cmd += ["--title", title]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == PDF_NO_TEXT_EXIT:
        detail = (json.loads(proc.stdout or "{}")).get("detail", "no text layer")
        raise RuntimeError(f"PDF has no readable text layer: {detail}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"PDF extraction failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return json.loads(proc.stdout)


def _extract_cases(
    node_harness: Path, pdf_harness: Path, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run the (model-independent) Node extraction once per case.

    Returns one record per case carrying either the assembled ``request_data``
    or the extraction ``error``, so it can be replayed against every model
    without re-running jsdom.
    """
    extracted: list[dict[str, Any]] = []
    for case in cases:
        manifest = case["manifest"]
        url = manifest["url"]
        title_override = manifest.get("title")
        avoided = manifest.get("user_avoided_ingredients")
        record: dict[str, Any] = {
            "id": case["id"],
            "category": manifest.get("category", "uncategorized"),
            "expect": manifest.get("expect", {}),
            "request_data": None,
            "error": None,
        }
        print(f"[{case['id']}] extracting...", file=sys.stderr)
        try:
            if case["kind"] == "pdf":
                payload = _extract_pdf(
                    pdf_harness, case["fixture"], url, title_override
                )
            else:
                payload = _extract(node_harness, case["fixture"], url, title_override)
            request_data = {
                "url": payload.get("url", url),
                "title": title_override or payload.get("title", ""),
                "content": payload["content"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # What the fixture declares about its own kind. Comes from the
                # real extractor over the real HTML, so a scope rule written
                # against it is exercised here exactly as in production.
                "page_signals": payload.get("page_signals"),
            }
            if avoided:
                request_data["user_avoided_ingredients"] = avoided
            # Optional passthrough of the hints the extension supplies when it
            # has them (`source`, `place_id`, `restaurant_name`, `language`).
            # The Google Maps extractor sends a restaurant_name that reaches the
            # model as the title, so a case can only exercise that path by
            # setting it here.
            request_data.update(manifest.get("request") or {})
            record["request_data"] = request_data
        except Exception as e:  # noqa: BLE001 - report, never abort the run
            record["error"] = str(e)
        extracted.append(record)
    return extracted


def _scope_of(core: Any, request_data: dict[str, Any]) -> dict[str, Any]:
    """The page-kind scope ``core`` will analyze this request under.

    Asked of the core rather than of ``page_scope`` directly, so a run reports
    the scope the configured deployment really uses — ``PAGE_SIGNAL_SCOPE``
    included. A request the core would reject has no scope to report, which is
    an error the analysis call is about to raise properly.
    """
    from models.content_model import PageAnalysisRequest

    try:
        decision = core.scope_for(PageAnalysisRequest(**request_data))
    except Exception:  # noqa: BLE001 - a scope is never worth failing a case for
        return {}
    return {
        "page_scope_kinds": list(decision.page_kinds),
        "page_scope_rule": decision.rule,
    }


def _run_case(
    core: Any,
    combo: Combo,
    record: dict[str, Any],
    cache_stats: Optional[CacheStats] = None,
) -> metrics.CaseResult:
    """Analyze one (pre-extracted) case and score it. Never raises.

    When a case is served from the provider's cache the call time is skipped, so
    the cached call's original duration is added back into the reported latency
    — otherwise every cached case would report ~0s. The replayed time is read
    per-thread, so a batched run still attributes it to the case that replayed
    it.
    """
    case_id, category, expect = record["id"], record["category"], record["expect"]
    if record["error"] is not None:
        return metrics.score_case(
            case_id, category, expect, None, error=record["error"]
        )

    # The scope the core will run this request under. Asked separately because
    # the response carries no trace of it, and reported per case because it is
    # what decides whether the model was even offered the right branch.
    scope = _scope_of(core, record["request_data"])

    print(f"[{combo.label()}] [{case_id}] analyzing...", file=sys.stderr)
    try:
        replayed_before = cache_stats.thread_replayed_s if cache_stats else 0.0
        t0 = time.time()
        response, status = core.analyze_page(record["request_data"])
        latency = time.time() - t0
        if cache_stats:
            latency += cache_stats.thread_replayed_s - replayed_before

        if status != 200 or "analysis" not in response:
            return metrics.score_case(
                case_id,
                category,
                expect,
                None,
                error=f"analyze_page status {status}: {response}",
                latency_s=latency,
                **scope,
            )

        return metrics.score_case(
            case_id,
            category,
            expect,
            response["analysis"],
            latency_s=latency,
            **scope,
        )
    except Exception as e:  # noqa: BLE001 - report, never abort the run
        return metrics.score_case(
            case_id, category, expect, None, error=str(e), **scope
        )


def _batch_order(extracted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order the cases so a batch draws from one category where it can.

    Cases run concurrently in submission order, so grouping by category keeps
    the cases in flight together similar — a batch of menus, a batch of product
    pages — and only mixes at a category boundary. Categories keep their
    first-seen order, and cases their order within a category.
    """
    by_category: dict[str, list[dict[str, Any]]] = {}
    for record in extracted:
        by_category.setdefault(record["category"], []).append(record)
    return [record for group in by_category.values() for record in group]


def _run_model(
    combo: Combo,
    extracted: list[dict[str, Any]],
    cache_stats: Optional[CacheStats] = None,
    max_batch_size: int = 1,
) -> list[metrics.CaseResult]:
    """Analyze every (pre-extracted) case against one combo.

    With ``max_batch_size`` above 1, that many cases are analyzed at a time,
    fed in the order ``_batch_order`` puts them in. A slot that frees takes the
    next case immediately rather than waiting for the rest of its batch: cases
    differ in cost by an order of magnitude (a product page in 13s against a
    menu in 114s), and on this corpus waiting cost 111s of a 277s run in idle
    slots — the whole benefit of running them concurrently.

    Results are returned in corpus order whichever way they were run, so the
    report does not depend on which case happened to finish first.
    """
    _configure_provider(combo)

    # Import + construct AFTER provider/model env is set, so the services read
    # the selected model identically to production.
    from analysis_core import AnalysisCore

    core = AnalysisCore(enable_database=False)

    if max_batch_size <= 1:
        return [_run_case(core, combo, record, cache_stats) for record in extracted]

    ordered = _batch_order(extracted)
    with ThreadPoolExecutor(max_workers=max_batch_size) as pool:
        results = pool.map(
            lambda record: _run_case(core, combo, record, cache_stats), ordered
        )
        scored = {record["id"]: result for record, result in zip(ordered, results)}
    return [scored[record["id"]] for record in extracted]


def run(args: argparse.Namespace) -> int:
    node_harness = Path(args.node).resolve()
    if not node_harness.exists():
        sys.exit(
            f"Node extraction harness not found at {node_harness}.\n"
            "Set --node, or `npm install` in the extension repo."
        )

    cases = _load_cases(Path(args.cases).resolve(), args.case, args.types)
    cases_dir = str(Path(args.cases).resolve())

    pdf_harness = Path(args.node_pdf).resolve()
    if any(case["kind"] == "pdf" for case in cases) and not pdf_harness.exists():
        sys.exit(
            f"PDF extraction harness not found at {pdf_harness}.\n"
            "Set --node-pdf, or `npm install` in the extension repo."
        )

    # Validate every provider's prerequisites up front so a mixed run fails fast
    # (before the slow extraction) rather than mid-way through.
    for provider in sorted({c.provider for c in args.combos}):
        _validate_provider_ready(provider)

    # Serve repeat calls from disk so re-runs don't pay for — or sit through —
    # the same (model, prompt, page) twice. Installed before AnalysisCore is
    # imported, and only for the providers this run actually uses.
    caches: dict[str, CacheStats] = {}
    if not args.no_cache:
        cache_dir = Path(args.cache_dir).resolve()
        installers = {"gemini": gemini_cache.install, "desktop": desktop_cache.install}
        for provider in sorted({c.provider for c in args.combos}):
            install = installers.get(provider)
            if install is None:  # pragma: no cover - providers are validated above
                continue
            caches[provider] = install(cache_dir / provider, refresh=args.refresh_cache)
            print(f"{provider} cache: {cache_dir / provider}", file=sys.stderr)

    # Extract once (provider-independent); replay against every combo.
    extracted = _extract_cases(node_harness, pdf_harness, cases)

    runs: list[dict[str, Any]] = []
    for combo in args.combos:
        results = _run_model(
            combo, extracted, caches.get(combo.provider), args.max_batch_size
        )
        agg = metrics.aggregate(results)
        config = {
            "provider": combo.provider,
            "model": combo.model,
            "reasoning_effort": combo.effort,
            "max_batch_size": args.max_batch_size,
            "case_types": args.types,
            "cases_dir": cases_dir,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        print()
        print(metrics.render_report(results, agg, config))
        runs.append({"config": config, "results": results, "agg": agg})

    out_paths = _write_results(args, runs)
    for p in out_paths:
        print(f"Results written to {p}", file=sys.stderr)
    for provider in sorted(caches):
        print(caches[provider].summary(), file=sys.stderr)
    return 0


def _write_results(args, runs: list[dict[str, Any]]) -> list[Path]:
    """Write per-model JSON, plus a combined HTML report when requested."""
    written: list[Path] = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON: honour --out only for a single model; otherwise auto-name per model.
    for run in runs:
        config, results, agg = run["config"], run["results"], run["agg"]
        if args.out and len(runs) == 1:
            out_path = Path(args.out)
        else:
            safe_model = config["model"].replace("/", "_")
            effort = config.get("reasoning_effort")
            out_path = _DEFAULT_RESULTS_DIR / (
                f"results-{config['provider']}-{safe_model}"
                f"{('-' + effort) if effort else ''}-{ts}.json"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "config": config,
                    "aggregate": agg,
                    "cases": [r.to_dict() for r in results],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        written.append(out_path)

    if args.html:
        html_path = Path(args.html) if isinstance(args.html, str) else None
        if html_path is None:
            tag = runs[0]["config"]["provider"] if len(runs) == 1 else "compare"
            html_path = _DEFAULT_RESULTS_DIR / f"report-{tag}-{ts}.html"
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(
            metrics.render_html(runs, generated_at=ts), encoding="utf-8"
        )
        written.append(html_path)

    return written


# What --type accepts besides the page_kind itself: the report's short names.
_TYPE_ALIASES = {"item": "shopping_item", "menu": "restaurant_menu"}


def _parse_type(value: str) -> str:
    kind = _TYPE_ALIASES.get(value.strip().lower(), value.strip().lower())
    if kind not in metrics.CASE_TYPES:
        raise argparse.ArgumentTypeError(
            f"unknown type '{value}' (choose from item, menu, other)"
        )
    return kind


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval",
        description="Analysis classification eval (benchmark, report-only).",
    )
    p.add_argument(
        "--run",
        dest="run",
        default=None,
        action="append",
        metavar="PROVIDER:MODEL[:EFFORT]",
        required=True,
        help=(
            "Provider:model combo to benchmark (required). Repeat or comma-separate "
            "to benchmark several combos, mixing providers, in one run, e.g. "
            "--run gemini:gemini-2.5-flash-lite --run desktop:openai/gpt-oss-120b. "
            "A desktop combo may append a reasoning effort (%s) to compare levels of "
            "the same model, e.g. --run desktop:openai/gpt-oss-20b:low --run "
            "desktop:openai/gpt-oss-20b:high; without one the model runs at its own "
            "default, ignoring any LMS_REASONING_EFFORT in the environment. "
            "Providers: %s." % (", ".join(REASONING_EFFORTS), ", ".join(PROVIDERS))
        ),
    )
    p.add_argument(
        "--cases",
        default=str(_DEFAULT_CASES),
        help="Directory of <id>.json + <id>.html|<id>.pdf cases.",
    )
    p.add_argument(
        "--case",
        default=None,
        help="Run only this case id (runs it even when the manifest skips it).",
    )
    p.add_argument(
        "--type",
        dest="types",
        default=None,
        action="append",
        type=_parse_type,
        metavar="TYPE",
        help=(
            "Run only cases of this type — the page_kind they expect: item "
            "(shopping_item), menu (restaurant_menu) or other. Repeat for several."
        ),
    )
    p.add_argument(
        "--node",
        default=str(_DEFAULT_NODE_HARNESS),
        help="Path to the extension's extract_content.js harness.",
    )
    p.add_argument(
        "--node-pdf",
        dest="node_pdf",
        default=str(_DEFAULT_PDF_HARNESS),
        help="Path to the extension's extract_pdf.js harness (PDF cases only).",
    )
    p.add_argument(
        "--max-batch-size",
        dest="max_batch_size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Analyze up to N cases concurrently, fed so that the cases in "
            "flight come from the same category where the corpus allows it. "
            "Default 1 (one at a time, as before). For desktop combos the "
            "worker must allow at "
            "least N jobs at once and the model must be loaded with at least N "
            "prediction slots — tests/eval/run.sh sets both from this flag. "
            "Per-case latencies then include batch contention, so compare "
            "totals rather than per-case times across batch sizes."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help="Results JSON path, single-combo only (default: tests/eval/results/...).",
    )
    p.add_argument(
        "--cache-dir",
        default=str(_DEFAULT_CACHE_DIR),
        help=(
            "Directory holding cached responses, one subdirectory per provider "
            "(default: tests/eval/.cache)."
        ),
    )
    # The --*-gemini-cache spellings predate the desktop cache and still work.
    p.add_argument(
        "--no-cache",
        "--no-gemini-cache",
        dest="no_cache",
        action="store_true",
        help="Call the provider for every case, ignoring (and not writing) the cache.",
    )
    p.add_argument(
        "--refresh-cache",
        "--refresh-gemini-cache",
        dest="refresh_cache",
        action="store_true",
        help="Re-run every case against the provider and overwrite the cached responses.",
    )
    p.add_argument(
        "--html",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Also write an HTML report. Optionally pass a path; with no value "
            "it is auto-named under tests/eval/results/."
        ),
    )
    return p


def _parse_combo(entry: str) -> Combo:
    """Parse one ``PROVIDER:MODEL[:EFFORT]`` entry.

    The optional third field is the reasoning effort, and must be one of
    REASONING_EFFORTS — so a model id carrying its own colon is rejected with a
    message about the effort rather than silently benchmarked under the wrong
    name.
    """
    provider, sep, rest = entry.partition(":")
    provider, rest = provider.strip(), rest.strip()
    if not sep or not rest:
        sys.exit(f"--run expects PROVIDER:MODEL[:EFFORT], got '{entry}'")
    if provider not in PROVIDERS:
        sys.exit(
            f"--run '{entry}': unknown provider '{provider}' (choose from {list(PROVIDERS)})"
        )

    model, sep, effort = rest.rpartition(":")
    if sep:
        model, effort = model.strip(), effort.strip().lower()
        if effort not in REASONING_EFFORTS:
            sys.exit(
                f"--run '{entry}': '{effort}' is not a reasoning effort "
                f"(choose from {list(REASONING_EFFORTS)}); expected PROVIDER:MODEL[:EFFORT]"
            )
    else:
        model, effort = effort, None
    if not model:
        sys.exit(f"--run expects PROVIDER:MODEL[:EFFORT], got '{entry}'")
    if effort and provider != "desktop":
        sys.exit(
            f"--run '{entry}': reasoning effort is only supported by the desktop provider"
        )
    return Combo(provider, model, effort)


def _resolve_combos(args: argparse.Namespace) -> list[Combo]:
    """Build the ordered, de-duplicated list of combos to run.

    Combos come solely from ``--run provider:model[:effort]`` entries (repeated
    or comma-separated). There are no default providers, models or efforts.
    """
    combos = [
        _parse_combo(entry.strip())
        for chunk in args.run or []
        for entry in chunk.split(",")
        if entry.strip()
    ]

    if not combos:
        sys.exit(
            "no combos to benchmark; pass --run PROVIDER:MODEL[:EFFORT] (repeatable)"
        )

    # De-duplicate, preserving first-seen order.
    seen: set[Combo] = set()
    unique: list[Combo] = []
    for combo in combos:
        if combo not in seen:
            seen.add(combo)
            unique.append(combo)
    return unique


def main(argv: Optional[list[str]] = None) -> int:
    # Load the repo-root .env (mainly GEMINI_API_KEY) the same way app.py does,
    # without clobbering vars already set in the environment.
    load_dotenv(_REPO_ROOT / ".env")
    args = build_parser().parse_args(argv)
    args.combos = _resolve_combos(args)
    if args.types:
        args.types = list(dict.fromkeys(args.types))
    if args.max_batch_size < 1:
        sys.exit(f"--max-batch-size must be at least 1, got {args.max_batch_size}")
    if args.out and len(args.combos) > 1:
        sys.exit("--out is single-combo only; drop it to auto-name per-combo results.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
