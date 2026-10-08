"""Scoring and reporting for the analysis classification eval.

Pure functions over case results: given the per-case expectations and the
predicted analysis dicts, compute per-field accuracy and a few breakdowns, and
render a human-readable report. This harness *reports* — it never raises on a
wrong prediction.
"""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from services.page_scope import describe_scope
from tests.eval import menu_scoring
from tests.eval.menu_scoring import MenuResult

# Fields we score when a case declares an expectation for them. `page_kind` is
# the classification the unified prompt makes first, and every other field is
# downstream of getting it right — a product read as a menu scores zero on the
# rest no matter how well it was analyzed. The others are enums or
# boolean/tri-state; scoring is equality either way (see `_matches`).
SCORED_FIELDS = [
    "page_kind",
    "is_shopping_item",
    "is_vegan",
    "is_cruelty_free",
    "vegan_friendliness",
]

# Each scored field's column header in the (text, HTML) report. Both tables
# take their field columns from here, so a field added to SCORED_FIELDS cannot
# fail a case without showing up in the row.
_FIELD_COLUMNS = {
    "page_kind": ("kind", "page_kind"),
    "is_shopping_item": ("shop", "shopping"),
    "is_vegan": ("vegan", "vegan"),
    "is_cruelty_free": ("cf", "cruelty_free"),
    "vegan_friendliness": ("friend", "friendliness"),
}

# Where each scored field lives in the nested analysis. Case files keep naming
# these fields flat — that is the eval's own vocabulary, and it predates the
# nesting — so the mapping lives here rather than in 25 case manifests. Anything
# absent from this table is read off the top level.
_FIELD_PATHS = {
    "is_vegan": "shopping_item.is_vegan",
    "is_cruelty_free": "shopping_item.is_cruelty_free",
    "cruelty_free_explanation": "shopping_item.cruelty_free_explanation",
    "confidence_level": "shopping_item.confidence_level",
    "vegan_friendliness": "menu.vegan_friendliness",
}


def _analysis_field(analysis: dict[str, Any], name: str) -> Any:
    """Read a field out of a nested analysis by the name a case file uses."""
    if name == "is_shopping_item":
        # Derived, not read: page_kind is what the model actually answered, and
        # the response's flat `is_shopping_item` is only a legacy alias of it.
        return analysis.get("page_kind") == "shopping_item"

    value: Any = analysis
    for part in _FIELD_PATHS.get(name, name).split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


# Key in a case's `expect` block listing avoided ingredients the model should
# surface in analysis["user_avoided_ingredients"].
AVOIDED_INCLUDE_KEY = "user_avoided_ingredients_include"

# Key in a case's `expect` block holding the shopping-item ingredient
# expectations: {"explicit": [...], "inferred": [...], "excluded": [...]}.
# A term in `explicit` / `inferred` must turn up in that list of the analysis;
# a term in `excluded` must turn up in neither — it is how a case catches the
# model inferring an animal ingredient into a product that has none. A term may
# be a list of accepted spellings ("merino" / "wool"). Matching is a
# case-insensitive substring, as for avoided ingredients.
INGREDIENTS_KEY = "ingredients"

# Which list of ``shopping_item`` each expected-term group is checked against.
_INGREDIENT_LISTS = {
    "explicit": "explicit_ingredients",
    "inferred": "inferred_ingredients",
}

# Key in a case's `expect` block holding the restaurant-menu expectations
# (dishes, their verdicts, the restaurant name) — scored by `menu_scoring`.
MENU_KEY = "menu"

# How page_kind renders in the report tables. The full values are too wide for a
# column that appears on every row.
_KIND_ABBR = {
    "shopping_item": "item",
    "restaurant_menu": "menu",
    "other": "other",
}

# A case's type is the page_kind it expects. The reports break their numbers
# down by type, and `--type` runs one alone; this is the order both use.
CASE_TYPES = ("shopping_item", "restaurant_menu", "other")


def case_type(expect: dict[str, Any]) -> str:
    """The case's type, or "untyped" when it expects no single page_kind."""
    kind = expect.get("page_kind")
    return kind if isinstance(kind, str) else "untyped"


def _type_order(case_type: str) -> tuple[int, str]:
    rank = CASE_TYPES.index(case_type) if case_type in CASE_TYPES else len(CASE_TYPES)
    return rank, case_type


# How a page-kind scope renders in the report tables — the scope is a tuple of
# kinds, and only the branch besides "other" distinguishes one from another.
_SCOPE_ABBR = {
    "item_or_other": "item",
    "menu_or_other": "menu",
    "all": "all",
}


def _matches(expected: Any, actual: Any) -> bool:
    """Whether a predicted value satisfies a case's expectation.

    A list expectation means "any of these is acceptable" — needed for the
    genuinely subjective fields (`vegan_friendliness` on a menu with a couple of
    vegan mains is defensibly "medium" or "high"), where insisting on one value
    would score a reasonable answer as wrong.
    """
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _expected_label(expected: Any) -> str:
    """Render an expectation as a hashable, JSON-friendly confusion-matrix key."""
    if isinstance(expected, list):
        return "|".join(str(v) for v in expected)
    return str(expected)


def _mean(values: list[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


@dataclass
class FieldResult:
    expected: Any
    actual: Any
    correct: bool


@dataclass
class IngredientResult:
    """How a shopping item's ingredient lists compare with a case's terms."""

    # (group, accepted spellings) for every expected term the model missed.
    missing: list[tuple[str, list[str]]]
    expected_count: int
    # Excluded terms the model listed anyway.
    excluded_found: list[str]
    explicit: list[str]
    inferred: list[str]

    @property
    def recall(self) -> Optional[float]:
        if not self.expected_count:
            return None
        return (self.expected_count - len(self.missing)) / self.expected_count

    @property
    def all_correct(self) -> bool:
        return not self.missing and not self.excluded_found

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall": self.recall,
            "missing": [{"list": g, "term": t} for g, t in self.missing],
            "excluded_found": self.excluded_found,
            "explicit": self.explicit,
            "inferred": self.inferred,
        }


def _spellings(term: Any) -> list[str]:
    terms = term if isinstance(term, list) else [term]
    return [str(t).lower() for t in terms]


def _contains(entries: list[str], spellings: list[str]) -> bool:
    return any(s in entry for entry in entries for s in spellings)


def score_ingredients(
    expected: dict[str, Any], analysis: dict[str, Any]
) -> IngredientResult:
    """Check a shopping item's ingredient lists against a case's terms."""
    item = analysis.get("shopping_item") or {}
    lists = {
        group: [str(v) for v in item.get(key) or []]
        for group, key in _INGREDIENT_LISTS.items()
    }
    lowered = {group: [v.lower() for v in values] for group, values in lists.items()}

    missing = []
    expected_count = 0
    for group in _INGREDIENT_LISTS:
        for term in expected.get(group, []):
            expected_count += 1
            spellings = _spellings(term)
            if not _contains(lowered[group], spellings):
                missing.append((group, spellings))

    every = lowered["explicit"] + lowered["inferred"]
    excluded_found = [
        "/".join(spellings)
        for spellings in map(_spellings, expected.get("excluded", []))
        if _contains(every, spellings)
    ]
    return IngredientResult(
        missing=missing,
        expected_count=expected_count,
        excluded_found=excluded_found,
        explicit=lists["explicit"],
        inferred=lists["inferred"],
    )


@dataclass
class CaseResult:
    case_id: str
    category: str
    expect: dict[str, Any]
    analysis: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    latency_s: Optional[float] = None
    fields: dict[str, FieldResult] = field(default_factory=dict)
    avoided_expected: list[str] = field(default_factory=list)
    avoided_found: list[str] = field(default_factory=list)
    avoided_recall: Optional[float] = None
    menu: Optional[MenuResult] = None
    ingredients: Optional[IngredientResult] = None
    # The page-kind scope the case actually ran under, and the page_scope rule
    # that chose it. Not scored — a scope is a property of the request, not of
    # the model — but reported, because a wrong page_kind reads very differently
    # depending on whether the right branch was even on offer.
    page_scope_kinds: Optional[list[str]] = None
    page_scope_rule: Optional[str] = None

    @property
    def scope_label(self) -> str:
        """The scope's short name, or empty when the run did not record one."""
        if not self.page_scope_kinds:
            return ""
        return describe_scope(tuple(self.page_scope_kinds))

    @property
    def case_type(self) -> str:
        return case_type(self.expect)

    @property
    def errored(self) -> bool:
        return self.error is not None

    @property
    def all_correct(self) -> bool:
        if self.errored:
            return False
        if (
            not self.fields
            and self.avoided_recall is None
            and self.menu is None
            and self.ingredients is None
        ):
            return False  # nothing was actually checked
        fields_ok = all(fr.correct for fr in self.fields.values())
        avoided_ok = self.avoided_recall is None or self.avoided_recall >= 1.0
        menu_ok = self.menu is None or self.menu.all_correct
        ingredients_ok = self.ingredients is None or self.ingredients.all_correct
        return fields_ok and avoided_ok and menu_ok and ingredients_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "type": self.case_type,
            "error": self.error,
            "latency_s": self.latency_s,
            "all_correct": self.all_correct if not self.errored else None,
            "fields": {
                name: {
                    "expected": fr.expected,
                    "actual": fr.actual,
                    "correct": fr.correct,
                }
                for name, fr in self.fields.items()
            },
            "avoided": {
                "expected": self.avoided_expected,
                "found": self.avoided_found,
                "recall": self.avoided_recall,
            }
            if self.avoided_expected
            else None,
            "menu": self.menu.to_dict() if self.menu is not None else None,
            "ingredients": self.ingredients.to_dict()
            if self.ingredients is not None
            else None,
            "page_scope": {
                "kinds": self.page_scope_kinds,
                "rule": self.page_scope_rule,
            }
            if self.page_scope_kinds
            else None,
            "analysis": self.analysis,
        }


def score_case(
    case_id: str,
    category: str,
    expect: dict[str, Any],
    analysis: Optional[dict[str, Any]],
    error: Optional[str] = None,
    latency_s: Optional[float] = None,
    page_scope_kinds: Optional[list[str]] = None,
    page_scope_rule: Optional[str] = None,
) -> CaseResult:
    """Compare a predicted analysis against a case's expectations."""
    result = CaseResult(
        case_id=case_id,
        category=category,
        expect=expect,
        analysis=analysis,
        error=error,
        latency_s=latency_s,
        page_scope_kinds=page_scope_kinds,
        page_scope_rule=page_scope_rule,
    )
    if error is not None or analysis is None:
        return result

    for name in SCORED_FIELDS:
        if name in expect:
            expected = expect[name]
            actual = _analysis_field(analysis, name)
            result.fields[name] = FieldResult(
                expected=expected, actual=actual, correct=_matches(expected, actual)
            )

    if MENU_KEY in expect:
        result.menu = menu_scoring.score_menu(expect[MENU_KEY], analysis)

    if INGREDIENTS_KEY in expect:
        result.ingredients = score_ingredients(expect[INGREDIENTS_KEY], analysis)

    if AVOIDED_INCLUDE_KEY in expect:
        expected_list = [s.lower() for s in expect[AVOIDED_INCLUDE_KEY]]
        found_raw = analysis.get("user_avoided_ingredients") or []
        found = [str(s).lower() for s in found_raw]
        matched = [e for e in expected_list if any(e in f for f in found)]
        result.avoided_expected = expected_list
        result.avoided_found = found
        result.avoided_recall = (
            len(matched) / len(expected_list) if expected_list else None
        )

    return result


def aggregate(results: list[CaseResult], by_type: bool = True) -> dict[str, Any]:
    """Compute per-field accuracy and breakdowns across all case results.

    With ``by_type``, ``by_type`` holds the same aggregate for each case type's
    results alone.
    """
    scored = [r for r in results if not r.errored]
    errored = [r for r in results if r.errored]

    per_field: dict[str, dict[str, Any]] = {}
    for name in SCORED_FIELDS:
        evaluated = [r for r in scored if name in r.fields]
        if not evaluated:
            continue
        correct = sum(1 for r in evaluated if r.fields[name].correct)
        confusion = Counter(
            (_expected_label(r.fields[name].expected), r.fields[name].actual)
            for r in evaluated
        )
        per_field[name] = {
            "evaluated": len(evaluated),
            "correct": correct,
            "accuracy": correct / len(evaluated),
            # JSON-friendly confusion: "expected->actual": count
            "confusion": {
                f"{exp}->{act}": n
                for (exp, act), n in sorted(
                    confusion.items(), key=lambda kv: str(kv[0])
                )
            },
        }

    avoided_cases = [r for r in scored if r.avoided_recall is not None]
    avoided = None
    if avoided_cases:
        avoided = {
            "cases": len(avoided_cases),
            "mean_recall": sum(r.avoided_recall for r in avoided_cases)
            / len(avoided_cases),
        }

    ingredient_cases = [r for r in scored if r.ingredients is not None]
    ingredients = None
    if ingredient_cases:
        ingredients = {
            "cases": len(ingredient_cases),
            "mean_recall": _mean(
                [
                    r.ingredients.recall
                    for r in ingredient_cases
                    if r.ingredients.recall is not None
                ]
            ),
            "excluded_hits": sum(
                len(r.ingredients.excluded_found) for r in ingredient_cases
            ),
        }

    menu_cases = [r for r in scored if r.menu is not None]
    menu = None
    if menu_cases:
        # The verdict confusion is the point of this block: a model that plays it
        # safe by calling every dish "unclear" scores respectably on recall and
        # only shows up here.
        verdict_confusion = Counter(
            (tuple(d.expected_verdicts), d.actual_verdict)
            for r in menu_cases
            for d in r.menu.found_dishes
            if d.verdict_correct is not None
        )
        menu = {
            "cases": len(menu_cases),
            "expected_dishes": sum(len(r.menu.dishes) for r in menu_cases),
            "found_dishes": sum(len(r.menu.found_dishes) for r in menu_cases),
            "extra_dishes": sum(len(r.menu.extra_dishes) for r in menu_cases),
            "mean_dish_recall": _mean(
                [
                    r.menu.dish_recall
                    for r in menu_cases
                    if r.menu.dish_recall is not None
                ]
            ),
            "mean_verdict_accuracy": _mean(
                [
                    r.menu.verdict_accuracy
                    for r in menu_cases
                    if r.menu.verdict_accuracy is not None
                ]
            ),
            "verdict_confusion": {
                f"{'|'.join(exp)}->{act}": n
                for (exp, act), n in sorted(
                    verdict_confusion.items(), key=lambda kv: str(kv[0])
                )
            },
        }

    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "all_correct": 0, "errored": 0}
    )
    for r in results:
        bucket = by_category[r.category]
        bucket["total"] += 1
        if r.errored:
            bucket["errored"] += 1
        elif r.all_correct:
            bucket["all_correct"] += 1

    latencies = [r.latency_s for r in scored if r.latency_s is not None]

    # How the corpus was scoped, counted by rule under each scope. Errored
    # cases are included: the scope is decided before the model runs, so a case
    # that fell over was still asked (or not asked) the narrowed question.
    scope_counts = Counter(
        (r.scope_label, r.page_scope_rule or "none") for r in results if r.scope_label
    )
    page_scope: dict[str, dict[str, int]] = defaultdict(dict)
    for (label, rule), n in sorted(scope_counts.items()):
        page_scope[label][rule] = n

    agg = {
        "total_cases": len(results),
        "errored": len(errored),
        "fully_correct": sum(1 for r in scored if r.all_correct),
        "per_field": per_field,
        "avoided_ingredients": avoided,
        "ingredients": ingredients,
        "menu": menu,
        "by_category": {k: dict(v) for k, v in sorted(by_category.items())},
        "page_scope": {k: dict(v) for k, v in sorted(page_scope.items())},
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
    }
    if by_type:
        groups: dict[str, list[CaseResult]] = defaultdict(list)
        for r in results:
            groups[r.case_type].append(r)
        agg["by_type"] = {
            t: aggregate(groups[t], by_type=False)
            for t in sorted(groups, key=_type_order)
        }
    return agg


def _fmt(value: Any) -> str:
    return "—" if value is None else str(value)


def _fmt_expected(expected: Any) -> str:
    """An expectation as the report shows it: alternatives joined by " / "."""
    if isinstance(expected, list):
        return " / ".join(_fmt(v) for v in expected)
    return _fmt(expected)


def _mismatch_notes(r: "CaseResult") -> list[str]:
    """One "field=actual (expected …)" note per wrong scored field."""
    return [
        f"{name}={_fmt(fr.actual)} (expected {_fmt_expected(fr.expected)})"
        for name, fr in r.fields.items()
        if not fr.correct
    ]


def _menu_cell(r: "CaseResult") -> tuple[Optional[bool], str]:
    """Compact menu summary for a per-case row: (ok?, "found/expected verdict%")."""
    if r.menu is None:
        return None, "·"
    m = r.menu
    parts = [f"{len(m.found_dishes)}/{len(m.dishes)}"]
    if m.verdict_accuracy is not None:
        parts.append(f"{m.verdict_accuracy:.0%}")
    if m.extra_dishes:
        parts.append(f"+{len(m.extra_dishes)}")
    # Same reasoning: the restaurant name is scored, so a case can be ✗ on it
    # alone while every dish number in this cell reads perfect.
    if m.restaurant_name_correct is False:
        parts.append("name≠")
    return m.all_correct, " ".join(parts)


def _ingredients_summary(r: "CaseResult") -> tuple[Optional[bool], str]:
    """Compact ingredient summary for a per-case row: (ok?, "recall !excluded")."""
    if r.ingredients is None:
        return None, "·"
    ing = r.ingredients
    parts = [_pct(ing.recall)] if ing.recall is not None else []
    if ing.excluded_found:
        parts.append(f"!{len(ing.excluded_found)}")
    return ing.all_correct, " ".join(parts) or "ok"


def _types_label(config: dict[str, Any]) -> str:
    """The case types a run was limited to, as the report abbreviates them."""
    return ",".join(_KIND_ABBR.get(t, t) for t in config.get("case_types") or [])


def render_report(
    results: list[CaseResult], agg: dict[str, Any], config: dict[str, Any]
) -> str:
    """Render the full text report (config header, per-case table, aggregates)."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("Analysis classification eval")
    effort = config.get("reasoning_effort")
    # Batch size is part of how the latencies below were measured, so it is
    # reported whenever it was not the sequential default.
    batch = config.get("max_batch_size") or 1
    lines.append(
        f"  provider={config.get('provider')}  model={config.get('model')}  "
        + (f"effort={effort}  " if effort else "")
        + (f"batch={batch}  " if batch > 1 else "")
        + (f"types={_types_label(config)}  " if config.get("case_types") else "")
        + f"cases={agg['total_cases']}"
    )
    lines.append("=" * 78)

    # Per-case table. `kind` leads the scored columns because everything else is
    # downstream of it.
    field_header = " ".join(f"{text:>7}" for text, _ in _FIELD_COLUMNS.values())
    header = (
        f"{'case':<28} {'scope':>5} {field_header} "
        f"{'avoid':>6} {'ingr':>8} {'menu':>16} {'ok':>4}  notes"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        if r.errored:
            lines.append(f"{r.case_id:<28} {'ERROR':>45}  {r.error}")
            continue

        def cell(name: str) -> str:
            if name not in r.fields:
                return "·"
            fr = r.fields[name]
            mark = "✓" if fr.correct else "✗"
            value = _fmt(fr.actual)
            if name == "page_kind":
                value = _KIND_ABBR.get(fr.actual, value)
            return f"{value}{mark}"

        avoid_cell = "·"
        if r.avoided_recall is not None:
            mark = "✓" if r.avoided_recall >= 1.0 else "✗"
            avoid_cell = f"{r.avoided_recall:.0%}{mark}"

        menu_ok, menu_text = _menu_cell(r)
        if menu_ok is not None:
            menu_text = f"{menu_text}{'✓' if menu_ok else '✗'}"

        ingr_ok, ingr_text = _ingredients_summary(r)
        if ingr_ok is not None:
            ingr_text = f"{ingr_text}{'✓' if ingr_ok else '✗'}"

        ok = "✓" if r.all_correct else "✗"
        # The scope names what the model was allowed to answer; the rule that
        # narrowed it rides along in the notes, where there is room for it.
        scope_cell = _SCOPE_ABBR.get(r.scope_label, r.scope_label or "·")
        notes = " ".join(
            part
            for part in (
                f"{r.latency_s:.1f}s" if r.latency_s is not None else "",
                r.page_scope_rule if r.page_scope_rule not in (None, "none") else "",
                *_mismatch_notes(r),
            )
            if part
        )
        field_cells = " ".join(f"{cell(name):>7}" for name in _FIELD_COLUMNS)
        lines.append(
            f"{r.case_id:<28} {scope_cell:>5} {field_cells} "
            f"{avoid_cell:>6} {ingr_text:>8} {menu_text:>16} {ok:>4}  {notes}"
        )

    # Aggregates.
    lines.append("")
    lines.append("Per-field accuracy:")
    if agg["per_field"]:
        for name, stats in agg["per_field"].items():
            lines.append(
                f"  {name:<18} {stats['accuracy']:.0%} "
                f"({stats['correct']}/{stats['evaluated']})  "
                f"confusion={stats['confusion']}"
            )
    else:
        lines.append("  (no scored fields)")

    if agg["avoided_ingredients"]:
        a = agg["avoided_ingredients"]
        lines.append(
            f"  avoided-ingredient recall {a['mean_recall']:.0%} "
            f"(over {a['cases']} case(s))"
        )

    if agg["ingredients"]:
        i = agg["ingredients"]
        lines.append("")
        lines.append(f"Item ingredients (over {i['cases']} case(s)):")
        lines.append(f"  expected-term recall  {_pct(i['mean_recall'])}")
        lines.append(f"  excluded terms listed {i['excluded_hits']}")

    if agg["menu"]:
        m = agg["menu"]
        lines.append("")
        lines.append(f"Menu analysis (over {m['cases']} case(s)):")
        lines.append(
            f"  dish recall        {_pct(m['mean_dish_recall'])} "
            f"({m['found_dishes']}/{m['expected_dishes']} dishes found)"
        )
        lines.append(f"  verdict accuracy   {_pct(m['mean_verdict_accuracy'])}")
        lines.append(f"  extra dishes       {m['extra_dishes']} (not scored)")
        lines.append(f"  verdict confusion  {m['verdict_confusion']}")

    if agg["page_scope"]:
        lines.append("")
        lines.append("Page-kind scope (cases asked each question, by rule):")
        for label, rules in agg["page_scope"].items():
            detail = ", ".join(f"{rule} {n}" for rule, n in sorted(rules.items()))
            lines.append(f"  {label:<18} {sum(rules.values()):>3}  ({detail})")

    # Only worth a section when there is more than one type to tell apart.
    if len(agg.get("by_type") or {}) > 1:
        lines.append("")
        lines.append("By type (fully-correct / total, errored, per-field accuracy):")
        for t, sub in agg["by_type"].items():
            lines.append(
                f"  {_KIND_ABBR.get(t, t):<6} {sub['fully_correct']}/{sub['total_cases']}"
                + (f"  ({sub['errored']} errored)" if sub["errored"] else "")
                + (
                    f"  mean latency {sub['mean_latency_s']:.1f}s"
                    if sub["mean_latency_s"] is not None
                    else ""
                )
            )
            fields = "  ".join(
                f"{name} {stats['accuracy']:.0%} ({stats['correct']}/{stats['evaluated']})"
                for name, stats in sub["per_field"].items()
            )
            if fields:
                lines.append(f"         {fields}")

    lines.append("")
    lines.append("By category (fully-correct / total, errored):")
    for cat, b in agg["by_category"].items():
        lines.append(
            f"  {cat:<18} {b['all_correct']}/{b['total']}"
            + (f"  ({b['errored']} errored)" if b["errored"] else "")
        )

    lines.append("")
    lines.append(
        f"Fully correct: {agg['fully_correct']}/{agg['total_cases']}"
        f"   errored: {agg['errored']}"
        + (
            f"   mean latency: {agg['mean_latency_s']:.1f}s"
            if agg["mean_latency_s"] is not None
            else ""
        )
    )
    lines.append("=" * 78)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #

# A "run" is one model's pass over the corpus: {"config", "results", "agg"}.
# render_html accepts a list of runs so a single benchmark invocation can
# compare several models side by side. A one-model run renders the same view
# without the cross-model comparison section.

_HTML_STYLE = """
/* Colors are tokens so the dark theme overrides them in one place, regardless
   of where individual rules sit in the sheet. */
:root {
  color-scheme: light dark;
  --bg: #f6f7f9; --fg: #1a1a1a; --surface: #fff; --surface-alt: #fff;
  --th-bg: #f0f2f5; --subhead-bg: #f6f8fa; --border: #e1e4e8;
  --muted: #888; --dim: #57606a; --strong: #424a53; --dot: #aaa;
  --ok: #1a7f37; --bad: #cf222e; --err: #9a6700; --avoid: #9a6700;
  --bar: #2da44e; --barbg: #d0d7de; --legend-border: #d0d7de;
  --conf-bg: #ddf4ff; --conf-fg: #0550ae;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --fg: #e6e6e6; --surface: #1c1f24; --surface-alt: #20242b;
    --th-bg: #262a31; --subhead-bg: #20242b; --border: #2a2f37;
    --muted: #8b949e; --dim: #9aa4af; --strong: #c9d1d9; --dot: #6e7681;
    --ok: #3fb950; --bad: #f85149; --err: #d29922; --avoid: #d4a72c;
    --bar: #2ea043; --barbg: #3a414b; --legend-border: #3a414b;
    --conf-bg: #0b2942; --conf-fg: #6cb6ff;
  }
}
* { box-sizing: border-box; }
body {
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 2rem; background: var(--bg); color: var(--fg);
}
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 .75rem; }
.muted { color: var(--muted); font-size: .85rem; }
.cards { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }
.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: .75rem 1rem; min-width: 9rem;
}
.card .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
.card .value { font-size: 1.6rem; font-weight: 600; }
table { border-collapse: collapse; width: 100%; background: var(--surface); border-radius: 8px; overflow: hidden; }
th, td { padding: .5rem .65rem; text-align: left; border-bottom: 1px solid var(--border); }
th { background: var(--th-bg); font-weight: 600; font-size: .85rem; }
tr:nth-child(even) td { background: var(--surface-alt); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: var(--ok); font-weight: 600; }
.bad { color: var(--bad); font-weight: 600; }
.err { color: var(--err); font-weight: 600; }
.dot { color: var(--dot); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em; }
details { border: 1px solid var(--border); border-radius: 8px; background: var(--surface); margin: .5rem 0; padding: 0 1rem; }
summary { cursor: pointer; padding: .75rem 0; font-weight: 600; }
.bar { display: inline-block; height: .5rem; border-radius: 3px; background: var(--bar); vertical-align: middle; }
.barbg { display: inline-block; width: 60px; height: .5rem; border-radius: 3px; background: var(--barbg); vertical-align: middle; margin-right: .4rem; }
.legend { font-size: .85rem; color: var(--dim); border-left: 3px solid var(--legend-border); padding: .5rem .9rem; margin: 1rem 0; }
.exprow td { border-bottom: 1px solid var(--border); padding-top: 0; }
.exp { display: block; }
.exp b { color: var(--strong); }
.conf { display: inline-block; font-size: .72rem; text-transform: uppercase; letter-spacing: .03em;
  background: var(--conf-bg); color: var(--conf-fg); border-radius: 10px; padding: .05rem .5rem; margin-bottom: .25rem; }
.avoid { color: var(--avoid); }
tr.subhead td {
  background: var(--subhead-bg); color: var(--dim); font-size: .78rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .03em;
}
"""


def _h(value: Any) -> str:
    return html.escape("—" if value is None else str(value), quote=True)


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.0%}"


def _mark_cell(correct: Optional[bool], text: str) -> str:
    if correct is None:
        return f'<span class="dot">{_h(text)}</span>'
    cls = "ok" if correct else "bad"
    glyph = "✓" if correct else "✗"
    return f'<span class="{cls}">{_h(text)} {glyph}</span>'


def _scope_html(r: "CaseResult") -> str:
    """The page-kind scope cell: the scope's short name, the rule as its title."""
    if not r.scope_label:
        return '<span class="dot">·</span>'
    label = _SCOPE_ABBR.get(r.scope_label, r.scope_label)
    rule = r.page_scope_rule or "none"
    return f'<span class="muted" title="{_h(rule)}">{_h(label)}</span>'


def _field_html(r: "CaseResult", name: str) -> str:
    if name not in r.fields:
        return '<span class="dot">·</span>'
    fr = r.fields[name]
    cell = _mark_cell(fr.correct, _fmt(fr.actual))
    if not fr.correct:
        cell += f'<br><span class="muted">expected {_h(_fmt_expected(fr.expected))}</span>'
    return cell


def _run_label(config: dict[str, Any]) -> str:
    """How one run is named in the report — including its reasoning effort, the
    only thing telling two runs of the same model apart."""
    effort = config.get("reasoning_effort")
    return f"{config.get('provider')} / {config.get('model')}" + (
        f" / {effort} effort" if effort else ""
    )


# Free-text rationale the model returns alongside its classification. Surfaced
# under each case row so a wrong (or right) verdict can be understood, not just
# scored. (key in analysis, human label)
EXPLANATION_FIELDS = [
    ("summary", "Summary"),
    ("cruelty_free_explanation", "Cruelty-free rationale"),
]


def _explanation_cell(r: "CaseResult") -> str:
    """Render the model's own confidence + explanatory text for one case."""
    analysis = r.analysis or {}
    bits: list[str] = []
    confidence = _analysis_field(analysis, "confidence_level")
    if confidence:
        bits.append(f'<span class="conf">confidence: {_h(confidence)}</span>')
    for key, label in EXPLANATION_FIELDS:
        value = _analysis_field(analysis, key)
        if value:
            bits.append(f'<span class="exp"><b>{_h(label)}:</b> {_h(value)}</span>')
    item = analysis.get("shopping_item") or {}
    for key, label in (
        ("explicit_ingredients", "Listed ingredients"),
        ("inferred_ingredients", "Inferred ingredients"),
    ):
        if item.get(key):
            bits.append(
                f'<span class="exp"><b>{label}:</b> {_h(", ".join(map(str, item[key])))}</span>'
            )
    if r.ingredients is not None:
        for group, spellings in r.ingredients.missing:
            bits.append(
                f'<span class="exp bad">Missing from {group}: {_h(" / ".join(spellings))}</span>'
            )
        for term in r.ingredients.excluded_found:
            bits.append(f'<span class="exp bad">Listed but excluded: {_h(term)}</span>')
    found = analysis.get("user_avoided_ingredients")
    if found:
        bits.append(
            f'<span class="exp"><b>Avoided found:</b> {_h(", ".join(map(str, found)))}</span>'
        )
    return "".join(bits)


def _ingredients_cell(
    d: "menu_scoring.DishResult | menu_scoring.ExtraDish",
) -> str:
    """The ingredients the model named for one dish, listed then inferred.

    Unscored, so an empty list renders as the answer it is ("none") rather than
    as a missing expectation. Avoid-list hits are shown alongside, since a dish
    can be vegan and still contain something the user avoids.
    """
    bits = []
    if d.explicit_ingredients:
        bits.append(f'listed: {_h(", ".join(d.explicit_ingredients))}')
    if d.inferred_ingredients:
        bits.append(f'inferred: {_h(", ".join(d.inferred_ingredients))}')
    if not bits:
        bits.append('<span class="dot">none</span>')
    if d.avoided_ingredients:
        bits.append(
            f'<span class="avoid">avoid: {_h(", ".join(d.avoided_ingredients))}</span>'
        )
    return "<br>".join(bits)


def _dish_table(menu: MenuResult) -> str:
    """The per-dish breakdown for one menu case, expected against predicted."""
    head = [
        "Expected dish",
        "Model listed it as",
        "Expected verdict",
        "Verdict",
        "Ingredients found",
        "Model's reason",
    ]
    # Colspans of the rows that don't fill the table, derived so the table stays
    # consistent when a column is added.
    rest = len(head) - 1
    rows = ["<tr>" + "".join(f"<th>{_h(h)}</th>" for h in head) + "</tr>"]
    for d in menu.dishes:
        if not d.found:
            rows.append(
                f"<tr><td><code>{_h(d.expected_name)}</code></td>"
                f'<td class="bad" colspan="{rest}">not listed by the model</td></tr>'
            )
            continue
        rows.append(
            "<tr>"
            f"<td><code>{_h(d.expected_name)}</code></td>"
            f"<td>{_h(d.matched_name)}</td>"
            f'<td><span class="muted">{_h(" / ".join(d.expected_verdicts))}</span></td>'
            f"<td>{_mark_cell(d.verdict_correct, _fmt(d.actual_verdict))}</td>"
            f"<td>{_ingredients_cell(d)}</td>"
            f'<td class="muted">{_h(d.reason)}</td>'
            "</tr>"
        )
    # Everything else the model listed. Shown in the same columns as the scored
    # dishes — a case declares the dishes it wants checked, not the whole menu,
    # so these are the model's reading of the rest of it, not mistakes.
    if menu.extra_dishes:
        rows.append(
            f'<tr class="subhead"><td colspan="{len(head)}">'
            f"Also listed by the model — {len(menu.extra_dishes)} dish(es) not in "
            "the case's list, shown for reference and not scored</td></tr>"
        )
    for extra in menu.extra_dishes:
        rows.append(
            "<tr>"
            f'<td class="dot">—</td>'
            f"<td>{_h(extra.name)}</td>"
            f'<td><span class="dot">not expected</span></td>'
            f"<td>{_h(_fmt(extra.verdict))}</td>"
            f"<td>{_ingredients_cell(extra)}</td>"
            f'<td class="muted">{_h(extra.reason)}</td>'
            "</tr>"
        )
    if not rows[1:]:
        rows.append(
            f'<tr><td class="dot" colspan="{len(head)}">'
            "no dishes expected, none returned</td></tr>"
        )

    summary = (
        f"dishes {len(menu.found_dishes)}/{len(menu.dishes)} found"
        + (
            f" · verdicts {_pct(menu.verdict_accuracy)}"
            if menu.verdict_accuracy is not None
            else ""
        )
        + (
            f" · {len(menu.extra_dishes)} extra (not scored)"
            if menu.extra_dishes
            else ""
        )
        # Spelled out rather than flagged: when the name is what failed the
        # case, the two strings are the whole story and usually differ by a
        # word or an apostrophe.
        + (
            f" · name “{_fmt(menu.restaurant_name_actual)}”"
            f" ≠ expected “{_fmt(menu.restaurant_name_expected)}”"
            if menu.restaurant_name_correct is False
            else ""
        )
    )
    return (
        f"<details><summary>Menu breakdown — {_h(summary)}</summary>"
        + "<table>"
        + "".join(rows)
        + "</table></details>"
    )


def _fully_correct_html(agg: dict[str, Any]) -> str:
    total = agg["total_cases"]
    return (
        f'{agg["fully_correct"]}/{total}'
        f' <span class="muted">({_pct(agg["fully_correct"]/total if total else None)})</span>'
    )


def _types_of(aggs: list[dict[str, Any]]) -> list[str]:
    """Every case type across ``aggs``, in report order."""
    types = {t for agg in aggs for t in agg.get("by_type") or {}}
    return sorted(types, key=_type_order)


def _by_type_table(agg: dict[str, Any]) -> str:
    """One row per case type: the run's headline metrics over that type alone."""
    by_type = agg.get("by_type") or {}
    field_names = [
        name
        for name in SCORED_FIELDS
        if any(name in sub["per_field"] for sub in by_type.values())
    ]
    head = ["Type", "Fully correct", "Errored", *(f"{n} acc" for n in field_names)]
    head.append("Mean latency")
    rows = [
        "<tr>"
        + "".join(
            f'<th class="num">{_h(h)}</th>' if i else f"<th>{_h(h)}</th>"
            for i, h in enumerate(head)
        )
        + "</tr>"
    ]
    for t, sub in by_type.items():
        cells = [
            f"<td><code>{_h(t)}</code></td>",
            f'<td class="num">{_fully_correct_html(sub)}</td>',
            f'<td class="num">{sub["errored"]}</td>',
        ]
        for name in field_names:
            stats = sub["per_field"].get(name)
            cells.append(
                f'<td class="num">{_pct(stats["accuracy"])} '
                f'<span class="muted">({stats["correct"]}/{stats["evaluated"]})</span></td>'
                if stats
                else '<td class="num dot">·</td>'
            )
        lat = sub["mean_latency_s"]
        cells.append(
            f'<td class="num">{lat:.1f}s</td>'
            if lat is not None
            else '<td class="num dot">·</td>'
        )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _comparison_table(runs: list[dict[str, Any]]) -> str:
    """One row per model: headline aggregate metrics, side by side."""
    field_names: list[str] = []
    for run in runs:
        for name in SCORED_FIELDS:
            if name in run["agg"]["per_field"] and name not in field_names:
                field_names.append(name)

    any_menu = any(run["agg"]["menu"] for run in runs)
    any_ingredients = any(run["agg"]["ingredients"] for run in runs)
    # Fully correct per type, when there is more than one type to tell apart.
    types = _types_of([run["agg"] for run in runs])
    if len(types) < 2:
        types = []

    head = ["Model", "Fully correct"]
    head += [f"{_KIND_ABBR.get(t, t)} fully correct" for t in types]
    head += ["Errored"]
    head += [f"{n} acc" for n in field_names]
    if any_ingredients:
        head += ["Ingredient recall", "Excluded listed"]
    if any_menu:
        head += ["Dish recall", "Verdict acc"]
    head += ["Avoided recall", "Mean latency"]
    rows = [
        "<tr>"
        + "".join(
            f'<th class="num">{_h(h)}</th>' if i else f"<th>{_h(h)}</th>"
            for i, h in enumerate(head)
        )
        + "</tr>"
    ]
    for run in runs:
        agg = run["agg"]
        cells = [f"<td><code>{_h(_run_label(run['config']))}</code></td>"]
        cells.append(f'<td class="num">{_fully_correct_html(agg)}</td>')
        for t in types:
            sub = (agg.get("by_type") or {}).get(t)
            cells.append(
                f'<td class="num">{_fully_correct_html(sub)}</td>'
                if sub
                else '<td class="num dot">·</td>'
            )
        cells.append(f'<td class="num">{agg["errored"]}</td>')
        for name in field_names:
            stats = agg["per_field"].get(name)
            cells.append(
                f'<td class="num">{_pct(stats["accuracy"])} '
                f'<span class="muted">({stats["correct"]}/{stats["evaluated"]})</span></td>'
                if stats
                else '<td class="num dot">·</td>'
            )
        if any_ingredients:
            ing = agg["ingredients"]
            cells.append(
                f'<td class="num">{_pct(ing["mean_recall"])}</td>'
                f'<td class="num">{ing["excluded_hits"]}</td>'
                if ing
                else '<td class="num dot">·</td><td class="num dot">·</td>'
            )
        if any_menu:
            menu = agg["menu"]
            for key in ("mean_dish_recall", "mean_verdict_accuracy"):
                cells.append(
                    f'<td class="num">{_pct(menu[key])}</td>'
                    if menu
                    else '<td class="num dot">·</td>'
                )
        avoided = agg["avoided_ingredients"]
        cells.append(
            f'<td class="num">{_pct(avoided["mean_recall"])}</td>'
            if avoided
            else '<td class="num dot">·</td>'
        )
        lat = agg["mean_latency_s"]
        cells.append(
            f'<td class="num">{lat:.1f}s</td>'
            if lat is not None
            else '<td class="num dot">·</td>'
        )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _per_case_matrix(runs: list[dict[str, Any]]) -> str:
    """Cases as rows, models as columns; cell = overall pass/fail/error."""
    # Union of case ids, preserving first-seen order.
    case_ids: list[str] = []
    by_run: list[dict[str, "CaseResult"]] = []
    for run in runs:
        idx = {r.case_id: r for r in run["results"]}
        by_run.append(idx)
        for cid in idx:
            if cid not in case_ids:
                case_ids.append(cid)

    head = ["Case"] + [f"<code>{_h(_run_label(r['config']))}</code>" for r in runs]
    rows = ["<tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for cid in case_ids:
        cells = [f"<td><code>{_h(cid)}</code></td>"]
        for idx in by_run:
            r = idx.get(cid)
            if r is None:
                cells.append('<td class="dot">·</td>')
            elif r.errored:
                cells.append('<td class="err" title="error">ERR</td>')
            else:
                cells.append(
                    f'<td>{_mark_cell(r.all_correct, "pass" if r.all_correct else "fail")}</td>'
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _run_detail(run: dict[str, Any]) -> str:
    config, results, agg = run["config"], run["results"], run["agg"]
    total = agg["total_cases"]

    cards = [
        ("Fully correct", f'{agg["fully_correct"]}/{total}'),
        ("Errored", str(agg["errored"])),
    ]
    if agg["mean_latency_s"] is not None:
        cards.append(("Mean latency", f'{agg["mean_latency_s"]:.1f}s'))
    for name, stats in agg["per_field"].items():
        cards.append((f"{name} acc", _pct(stats["accuracy"])))
    if agg["ingredients"]:
        cards.append(("Ingredient recall", _pct(agg["ingredients"]["mean_recall"])))
    if agg["menu"]:
        cards.append(("Dish recall", _pct(agg["menu"]["mean_dish_recall"])))
        cards.append(("Verdict acc", _pct(agg["menu"]["mean_verdict_accuracy"])))
    if agg["page_scope"]:
        narrowed = sum(
            n
            for label, rules in agg["page_scope"].items()
            if label != "all"
            for n in rules.values()
        )
        cards.append(("Scope narrowed", f"{narrowed}/{total}"))
    cards_html = "".join(
        f'<div class="card"><div class="label">{_h(label)}</div>'
        f'<div class="value">{_h(value)}</div></div>'
        for label, value in cards
    )

    head = [
        "Case",
        "Category",
        "scope",
        *(label for _, label in _FIELD_COLUMNS.values()),
        "avoided",
        "ingredients",
        "menu",
        "ok",
        "latency",
    ]
    span = len(head) - 2  # columns after Case + Category
    rows = ["<tr>" + "".join(f"<th>{_h(h)}</th>" for h in head) + "</tr>"]
    for r in results:
        if r.errored:
            rows.append(
                f"<tr><td><code>{_h(r.case_id)}</code></td>"
                f"<td>{_h(r.category)}</td>"
                f'<td class="err" colspan="{span}">ERROR: {_h(r.error)}</td></tr>'
            )
            continue
        if r.avoided_recall is None:
            avoid = '<span class="dot">·</span>'
        else:
            avoid = _mark_cell(r.avoided_recall >= 1.0, _pct(r.avoided_recall))
        ingr_ok, ingr_text = _ingredients_summary(r)
        ingr_cell = (
            '<span class="dot">·</span>'
            if ingr_ok is None
            else _mark_cell(ingr_ok, ingr_text)
        )
        menu_ok, menu_text = _menu_cell(r)
        menu_cell = (
            '<span class="dot">·</span>'
            if menu_ok is None
            else _mark_cell(menu_ok, menu_text)
        )
        lat = f"{r.latency_s:.1f}s" if r.latency_s is not None else ""
        field_cells = "".join(
            f"<td>{_field_html(r, name)}</td>" for name in _FIELD_COLUMNS
        )
        rows.append(
            "<tr>"
            f"<td><code>{_h(r.case_id)}</code></td>"
            f"<td>{_h(r.category)}</td>"
            f"<td>{_scope_html(r)}</td>"
            f"{field_cells}"
            f"<td>{avoid}</td>"
            f"<td>{ingr_cell}</td>"
            f"<td>{menu_cell}</td>"
            f'<td>{_mark_cell(r.all_correct, "yes" if r.all_correct else "no")}</td>'
            f'<td class="num">{_h(lat)}</td>'
            "</tr>"
        )
        detail = _explanation_cell(r)
        if r.menu is not None:
            detail += _dish_table(r.menu)
        if detail:
            rows.append(
                f'<tr class="exprow"><td></td>'
                f'<td colspan="{len(head) - 1}">{detail}</td></tr>'
            )
    table = "<table>" + "".join(rows) + "</table>"
    by_type = (
        _by_type_table(agg) + "<br>" if len(agg.get("by_type") or {}) > 1 else ""
    )

    return (
        f"<details open><summary><code>{_h(_run_label(config))}</code> — "
        f'{agg["fully_correct"]}/{total} fully correct</summary>'
        f'<div class="cards">{cards_html}</div>'
        f"{by_type}{table}</details>"
    )


def render_html(runs: list[dict[str, Any]], generated_at: Optional[str] = None) -> str:
    """Render a standalone HTML report for one or more model runs.

    Each run is a dict with keys ``config``, ``results`` (list[CaseResult]) and
    ``agg`` (the output of :func:`aggregate`). With more than one run, a
    cross-model comparison summary and per-case matrix are emitted first.
    """
    if not runs:
        raise ValueError("render_html requires at least one run")

    providers = sorted({r["config"].get("provider") for r in runs})
    total_cases = runs[0]["agg"]["total_cases"]

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Analysis classification eval</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        "<h1>Analysis classification eval</h1>",
        f'<p class="muted">provider(s): {_h(", ".join(providers))} · '
        f"{len(runs)} model(s) · {total_cases} case(s)"
        + (
            f" · types {_h(_types_label(runs[0]['config']))} only"
            if runs[0]["config"].get("case_types")
            else ""
        )
        + (f" · generated {_h(generated_at)}" if generated_at else "")
        + "</p>",
        '<div class="legend">'
        "<b>How to read this:</b> each case is the extension's extracted page run "
        "through the analysis stack. Per-field cells show the model's value with "
        '<span class="ok">✓</span> (matches the case\'s expectation) or '
        '<span class="bad">✗</span> (mismatch); <span class="dot">·</span> means '
        "the case declares no expectation for that field. <b>ok</b> is true only "
        "when every declared field matches. <b>avoided</b> is recall of the "
        "user-avoided ingredients the model should have surfaced. <b>ingredients</b> "
        "is recall of the terms a product case expects in the model's listed / "
        "inferred ingredients, with <i>!n</i> for terms it listed that the case "
        "excludes. <b>menu</b> is "
        "<i>dishes found / dishes expected</i> then per-dish verdict accuracy, "
        "then <i>+n</i> for dishes the model listed that the case doesn't "
        "declare — those are shown, never scored, since a case declares the "
        "dishes it wants checked rather than the whole menu. Expand "
        "<b>Menu breakdown</b> under the row for the dish-by-dish comparison, "
        "including the ingredients the model named for each dish and everything "
        "else it listed. Under each row, "
        "the model's own <b>confidence</b> and explanation text are shown so a "
        "verdict can be understood, not just scored."
        "</div>",
    ]

    if len(runs) > 1:
        parts.append("<h2>Model comparison</h2>")
        parts.append(_comparison_table(runs))
        parts.append("<h2>Per-case results</h2>")
        parts.append(_per_case_matrix(runs))

    parts.append("<h2>Details</h2>")
    parts.extend(_run_detail(run) for run in runs)

    parts.append("</body></html>")
    return "".join(parts)
