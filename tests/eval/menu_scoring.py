"""Scoring for the restaurant-menu branch of a page analysis.

The product branch answers one question per page, so `metrics.score_case` can
score it by field equality. A menu answers one question per *dish*, and the
dishes are whatever the model chose to list — it may miss one, invent one, or
name one differently from the case's expectation ("Falafel Plate (VG)" against
"Falafel Plate"). So the dishes have to be matched up before anything can be
compared, which is what this module does.

The metrics it produces:

- **dish recall** — how many of the dishes a case declares the model actually
  listed. A menu analysis that silently drops half the menu is the failure mode
  the extension can't detect on its own.
- **verdict accuracy** — over the dishes that *were* found, how many got an
  accepted verdict. Scored separately from recall so a model that lists three
  dishes and rates them perfectly doesn't look better than one that lists all
  nine and gets one wrong.
- **extra dishes** — dishes the model listed that the case doesn't declare.
  Never scored: a case declares the dishes it cares about, and a corpus of
  spot-checked menus would otherwise punish a model for reading the rest of the
  menu correctly. They are carried through in full (verdict, ingredients,
  reason) so the report can show what the model saw beyond the expectations.

Expected verdicts may be a single string or a list of accepted strings: several
dishes are legitimately two-valued (fresh linguine is `likely_vegan` if you
don't worry about egg in the pasta and `unclear` if you do), and forcing one
answer there would score correct behaviour as wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# The schema's own verdict vocabulary, imported rather than re-listed so a typo
# in a case manifest is caught instead of silently scoring every dish wrong.
from services.menu_prompt import ITEM_VERDICTS

_PAREN_SUFFIX = re.compile(r"\([^)]*\)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_dish_name(name: Any) -> str:
    """Reduce a dish name to comparable tokens.

    Drops the labels menus append to dish names — "(VG)", "(v)", "(gf)" — and
    the punctuation that varies between how a menu writes a name and how a case
    manifest does ("Hummus & Flatbread" / "Hummus and Flatbread").
    """
    text = str(name or "").lower()
    text = _PAREN_SUFFIX.sub(" ", text)
    text = text.replace("&", " and ")
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(text.split())


def _tokens(name: Any) -> set[str]:
    return set(normalize_dish_name(name).split())


def match_dishes(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], Optional[dict[str, Any]]]], list[dict[str, Any]]]:
    """Pair each expected dish with at most one predicted dish.

    Two passes, both greedy and order-stable so a run is reproducible: exact
    normalized equality first, then token-subset containment for whatever is
    left ("Latte" against "Iced Latte", "Falafel Plate" against "Falafel Plate
    (VG)"). Exact matches are claimed first precisely so that a menu carrying
    both "Latte" and "Iced Latte" pairs them the obvious way rather than by
    whichever the model happened to list first.

    Containment is on whole tokens, not substrings — "tea" is a substring of
    "steak".

    Returns ``(pairs, extras)`` where a pair's second element is None for an
    expected dish the model never listed, and ``extras`` are the predicted
    dishes that matched nothing.
    """
    candidates = [(normalize_dish_name(item.get("name")), item) for item in actual]
    claimed: set[int] = set()
    pairs: list[list[Any]] = [[exp, None] for exp in expected]

    for pair in pairs:
        want = normalize_dish_name(pair[0].get("name"))
        if not want:
            continue
        for i, (got, _) in enumerate(candidates):
            if i not in claimed and got and got == want:
                pair[1] = i
                claimed.add(i)
                break

    for pair in pairs:
        if pair[1] is not None:
            continue
        want = _tokens(pair[0].get("name"))
        if not want:
            continue
        for i, (got, _) in enumerate(candidates):
            if i in claimed or not got:
                continue
            got_tokens = set(got.split())
            if want <= got_tokens or got_tokens <= want:
                pair[1] = i
                claimed.add(i)
                break

    matched = [(exp, candidates[i][1] if i is not None else None) for exp, i in pairs]
    extras = [item for i, (_, item) in enumerate(candidates) if i not in claimed]
    return matched, extras


@dataclass
class DishResult:
    """One expected dish, and what the model said about it (if anything)."""

    expected_name: str
    matched_name: Optional[str]
    expected_verdicts: list[str]
    actual_verdict: Optional[str]
    verdict_correct: Optional[bool]
    reason: Optional[str] = None
    # What the model said the dish contains, as the menu stated it and as the
    # model inferred it. Unscored — a case declares verdicts, not ingredient
    # lists — but it is the evidence the verdict was supposed to follow from,
    # so the report shows it next to the reason.
    explicit_ingredients: list[str] = field(default_factory=list)
    inferred_ingredients: list[str] = field(default_factory=list)
    avoided_ingredients: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.matched_name is not None

    @property
    def all_correct(self) -> bool:
        if not self.found:
            return False
        if self.verdict_correct is False:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_name": self.expected_name,
            "matched_name": self.matched_name,
            "expected_verdicts": self.expected_verdicts,
            "actual_verdict": self.actual_verdict,
            "verdict_correct": self.verdict_correct,
            "reason": self.reason,
            "explicit_ingredients": self.explicit_ingredients,
            "inferred_ingredients": self.inferred_ingredients,
            "avoided_ingredients": self.avoided_ingredients,
        }


@dataclass
class ExtraDish:
    """A dish the model listed that the case doesn't declare.

    Unscored — it exists so the report can show it. Carries the same fields as
    a scored dish, since "what else did the model find, and what did it make of
    it" is the question these rows answer.
    """

    name: str
    verdict: Optional[str] = None
    reason: Optional[str] = None
    explicit_ingredients: list[str] = field(default_factory=list)
    inferred_ingredients: list[str] = field(default_factory=list)
    avoided_ingredients: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "reason": self.reason,
            "explicit_ingredients": self.explicit_ingredients,
            "inferred_ingredients": self.inferred_ingredients,
            "avoided_ingredients": self.avoided_ingredients,
        }


@dataclass
class MenuResult:
    """Everything scored about one case's menu branch."""

    dishes: list[DishResult] = field(default_factory=list)
    extra_dishes: list[ExtraDish] = field(default_factory=list)
    restaurant_name_expected: Optional[str] = None
    restaurant_name_actual: Optional[str] = None
    restaurant_name_correct: Optional[bool] = None
    dish_recall: Optional[float] = None
    verdict_accuracy: Optional[float] = None

    @property
    def found_dishes(self) -> list[DishResult]:
        return [d for d in self.dishes if d.found]

    @property
    def all_correct(self) -> bool:
        if self.restaurant_name_correct is False:
            return False
        if self.dish_recall is not None and self.dish_recall < 1.0:
            return False
        if self.verdict_accuracy is not None and self.verdict_accuracy < 1.0:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dish_recall": self.dish_recall,
            "verdict_accuracy": self.verdict_accuracy,
            "restaurant_name": {
                "expected": self.restaurant_name_expected,
                "actual": self.restaurant_name_actual,
                "correct": self.restaurant_name_correct,
            }
            if self.restaurant_name_expected is not None
            else None,
            "extra_dishes": [e.to_dict() for e in self.extra_dishes],
            "all_correct": self.all_correct,
            "dishes": [d.to_dict() for d in self.dishes],
        }


def _ingredients(item: Optional[dict[str, Any]], key: str) -> list[str]:
    """Read one of an item's ingredient lists, tolerating it being absent.

    ``user_avoided_ingredients`` is only in the schema when the case supplies an
    avoid-list, and a model can still return null for any of them.
    """
    if not item:
        return []
    return [str(v) for v in (item.get(key) or [])]


def _accepted_verdicts(raw: Any) -> list[str]:
    """Read a dish's expected verdict(s), tolerating a bare string."""
    if raw is None:
        return []
    verdicts = list(raw) if isinstance(raw, list) else [raw]
    unknown = [v for v in verdicts if v not in ITEM_VERDICTS]
    if unknown:
        raise ValueError(
            f"unknown expected verdict(s) {unknown}; "
            f"the schema allows {ITEM_VERDICTS}"
        )
    return verdicts


def score_menu(expect_menu: dict[str, Any], analysis: dict[str, Any]) -> MenuResult:
    """Score a case's ``expect.menu`` block against a predicted analysis.

    ``analysis`` is the page analysis as the API returns it, so the dishes and
    the restaurant name are read off its ``menu`` branch. That branch is absent
    entirely for any page_kind other than ``restaurant_menu``, so a page the model
    didn't call a menu scores as a menu with none of its dishes found.
    """
    expected_dishes = expect_menu.get("dishes") or []
    menu = analysis.get("menu") or {}
    actual_items = [i for i in (menu.get("items") or []) if isinstance(i, dict)]

    result = MenuResult()

    if "restaurant_name" in expect_menu:
        expected_name = expect_menu["restaurant_name"]
        actual_name = menu.get("restaurant_name")
        result.restaurant_name_expected = expected_name
        result.restaurant_name_actual = actual_name
        result.restaurant_name_correct = normalize_dish_name(
            actual_name
        ) == normalize_dish_name(expected_name)

    pairs, extras = match_dishes(expected_dishes, actual_items)
    result.extra_dishes = [
        ExtraDish(
            name=str(item.get("name")),
            verdict=item.get("verdict"),
            reason=item.get("reason"),
            explicit_ingredients=_ingredients(item, "explicit_ingredients"),
            inferred_ingredients=_ingredients(item, "inferred_ingredients"),
            avoided_ingredients=_ingredients(item, "user_avoided_ingredients"),
        )
        for item in extras
    ]

    for expected, actual in pairs:
        accepted = _accepted_verdicts(expected.get("verdict"))
        actual_verdict = actual.get("verdict") if actual else None
        dish = DishResult(
            expected_name=str(expected.get("name")),
            matched_name=str(actual.get("name")) if actual else None,
            expected_verdicts=accepted,
            actual_verdict=actual_verdict,
            verdict_correct=(actual_verdict in accepted)
            if (actual and accepted)
            else None,
            reason=actual.get("reason") if actual else None,
            explicit_ingredients=_ingredients(actual, "explicit_ingredients"),
            inferred_ingredients=_ingredients(actual, "inferred_ingredients"),
            avoided_ingredients=_ingredients(actual, "user_avoided_ingredients"),
        )
        result.dishes.append(dish)

    if expected_dishes:
        result.dish_recall = len(result.found_dishes) / len(expected_dishes)

    verdict_scored = [d for d in result.found_dishes if d.verdict_correct is not None]
    if verdict_scored:
        result.verdict_accuracy = sum(
            1 for d in verdict_scored if d.verdict_correct
        ) / len(verdict_scored)

    return result
