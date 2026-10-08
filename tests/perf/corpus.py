"""The page text the ``real`` workload replays, frozen into ``corpus.json``.

Extraction runs the extension's own Node harnesses, which need the sibling
``veganconfirmed-extension`` checkout and its ``npm install``. Freezing the
result and committing it buys two things: the benchmark runs with no setup at
all, and the prompt stays byte-identical over time, which is the only way a
TTFT measured today means anything next to one measured after a runtime
upgrade. ``--refresh-corpus`` re-extracts and shows the change as a diff.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent
DEFAULT_CORPUS = _HERE / "corpus.json"
DEFAULT_CASES_DIR = _HERE.parent / "eval" / "cases"

# Four sizes spanning what the extension actually sends, chosen so a run says
# something about prefill scaling rather than about one page.
DEFAULT_CASE_IDS = (
    "non_shopping_article",  # ~0.3k chars: fixed per-request overhead
    "nuts-for-cheese-black-garlic",  # ~1.9k chars: a typical shopping item
    "eureka-restaurant",  # ~5.1k chars: a menu, the long-answer case
    "merit-vegan-menu",  # ~15k chars: fills the page-content budget
)


def corpus_sha(cases: list[dict[str, Any]]) -> str:
    """Short digest of the extracted text, so a results file names its workload."""
    h = hashlib.sha256()
    for case in cases:
        for key in ("id", "url", "title", "content"):
            h.update(str(case.get(key, "")).encode())
    return h.hexdigest()[:12]


def build(
    case_ids: tuple[str, ...], cases_dir: Path, node: Path, node_pdf: Path
) -> dict:
    """Extract ``case_ids`` from the eval corpus using the extension's harnesses."""
    from tests.eval.runner import _extract_cases, _load_cases

    wanted = set(case_ids)
    cases = [c for c in _load_cases(cases_dir, None) if c["id"] in wanted]
    missing = wanted - {c["id"] for c in cases}
    if missing:
        raise SystemExit(
            f"no such case(s) in {cases_dir}: {', '.join(sorted(missing))}"
        )

    extracted = []
    for record in _extract_cases(node, node_pdf, cases):
        if record["error"]:
            raise SystemExit(f"[{record['id']}] extraction failed: {record['error']}")
        data = record["request_data"]
        extracted.append(
            {
                "id": record["id"],
                "url": data["url"],
                "title": data["title"],
                "content": data["content"],
                "chars": len(data["content"]),
                "user_avoided_ingredients": data.get("user_avoided_ingredients"),
                "source": data.get("source"),
                "trigger_type": data.get("trigger_type"),
                "page_signals": data.get("page_signals"),
            }
        )
    # Keep the declared order, so the report reads smallest page first.
    extracted.sort(key=lambda c: case_ids.index(c["id"]))
    return {
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sha": corpus_sha(extracted),
        "cases": extracted,
    }


def write(corpus: dict[str, Any], path: Path = DEFAULT_CORPUS) -> Path:
    path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n")
    return path


def load(path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"{path} is missing; regenerate it with `python -m tests.perf --refresh-corpus`"
        )
    corpus: dict[str, Any] = json.loads(path.read_text())
    return corpus
