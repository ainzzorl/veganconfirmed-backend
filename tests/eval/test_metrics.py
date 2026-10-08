"""Offline tests for the eval harness's own scoring.

The rest of the harness can only be exercised by running a real model over the
corpus, which costs money (Gemini) or a loaded LM Studio (desktop). These are
pure functions over dicts, so they run anywhere:

    uv run pytest tests/eval/test_metrics.py

The dish matcher is what they mostly cover. It is the one piece of the harness
that can be quietly, plausibly wrong — a matcher that fails to pair "Falafel
Plate" with "Falafel Plate (VG)" reports a missed dish and blames the model.
"""

from __future__ import annotations

from tests.eval import menu_scoring
from tests.eval.metrics import aggregate, score_case


def _dish(name, verdict="vegan", **kwargs):
    return {"name": name, "verdict": verdict, **kwargs}


def _menu_analysis(items, **kwargs):
    """An analysis in the shape the API returns: the menu branch, nested."""
    return {
        "page_kind": "restaurant_menu",
        "shopping_item": None,
        "menu": {
            "restaurant_name": kwargs.pop("restaurant_name", "The Olive Branch"),
            "vegan_friendliness": kwargs.pop("vegan_friendliness", "medium"),
            "items": items,
        },
        **kwargs,
    }


def _item_analysis(**shopping_item):
    """The same, for the other branch."""
    return {
        "page_kind": "shopping_item",
        "shopping_item": shopping_item,
        "menu": None,
    }


# --------------------------------------------------------------------------- #
# Name normalization + matching
# --------------------------------------------------------------------------- #


def test_normalize_strips_menu_labels_and_punctuation():
    assert menu_scoring.normalize_dish_name("Falafel Plate (VG)") == "falafel plate"
    assert (
        menu_scoring.normalize_dish_name("Hummus & Flatbread") == "hummus and flatbread"
    )
    assert (
        menu_scoring.normalize_dish_name("  Soup   of the Day  ") == "soup of the day"
    )
    assert menu_scoring.normalize_dish_name(None) == ""


def test_matches_across_a_menu_label():
    pairs, extras = menu_scoring.match_dishes(
        [_dish("Falafel Plate")], [_dish("Falafel Plate (VG)")]
    )
    assert pairs[0][1]["name"] == "Falafel Plate (VG)"
    assert extras == []


def test_token_containment_does_not_match_on_substrings():
    """The word 'tea' is a substring of 'steak' — matching is on whole tokens."""
    pairs, extras = menu_scoring.match_dishes([_dish("Tea")], [_dish("Steak")])
    assert pairs[0][1] is None
    assert [e["name"] for e in extras] == ["Steak"]


def test_exact_match_wins_over_containment():
    """A menu with both "Latte" and "Iced Latte" must pair them the obvious way."""
    pairs, _ = menu_scoring.match_dishes(
        [_dish("Latte"), _dish("Iced Latte")],
        [_dish("Iced Latte"), _dish("Latte")],
    )
    assert pairs[0][1]["name"] == "Latte"
    assert pairs[1][1]["name"] == "Iced Latte"


def test_a_predicted_dish_is_claimed_only_once():
    pairs, extras = menu_scoring.match_dishes(
        [_dish("Falafel Plate"), _dish("Falafel Wrap")], [_dish("Falafel Plate")]
    )
    assert pairs[0][1]["name"] == "Falafel Plate"
    assert pairs[1][1] is None
    assert extras == []


# --------------------------------------------------------------------------- #
# score_menu
# --------------------------------------------------------------------------- #


def test_verdict_list_accepts_any_listed_value():
    expect = {"dishes": [{"name": "Linguine", "verdict": ["likely_vegan", "unclear"]}]}
    for verdict in ("likely_vegan", "unclear"):
        menu = menu_scoring.score_menu(
            expect, _menu_analysis([_dish("Linguine", verdict)])
        )
        assert menu.verdict_accuracy == 1.0, verdict
    menu = menu_scoring.score_menu(
        expect, _menu_analysis([_dish("Linguine", "not_vegan")])
    )
    assert menu.verdict_accuracy == 0.0
    assert not menu.all_correct


def test_recall_and_verdict_accuracy_are_scored_separately():
    """Listing three dishes perfectly must not beat listing all four with one miss."""
    expect = {
        "dishes": [
            _dish("Hummus"),
            _dish("Kofta", "not_vegan"),
            _dish("Baklava", "not_vegan"),
            _dish("Sorbet"),
        ]
    }
    partial = menu_scoring.score_menu(
        expect,
        _menu_analysis([_dish("Hummus"), _dish("Kofta", "not_vegan"), _dish("Sorbet")]),
    )
    assert partial.dish_recall == 0.75
    assert partial.verdict_accuracy == 1.0
    assert not partial.all_correct  # a dropped dish is a failure, not a rounding


def test_the_ingredients_the_model_named_ride_along_unscored():
    """Both lists reach the report; neither decides whether the dish is correct."""
    expect = {"dishes": [_dish("Cheeseburger", "not_vegan")]}
    menu = menu_scoring.score_menu(
        expect,
        _menu_analysis(
            [
                _dish(
                    "Cheeseburger",
                    "not_vegan",
                    explicit_ingredients=["cheese", "bun"],
                    inferred_ingredients=["beef patty"],
                )
            ]
        ),
    )
    dish = menu.dishes[0]
    assert dish.explicit_ingredients == ["cheese", "bun"]
    assert dish.inferred_ingredients == ["beef patty"]
    assert menu.all_correct


def test_extras_are_reported_but_never_scored():
    """A case declares the dishes it checks, not the whole menu."""
    expect = {"dishes": [_dish("Hummus")]}
    analysis = _menu_analysis(
        [
            _dish("Hummus"),
            _dish(
                "Lamb Kofta",
                "not_vegan",
                explicit_ingredients=["lamb"],
                reason="lamb mince",
            ),
        ]
    )

    menu = menu_scoring.score_menu(expect, analysis)
    assert [e.name for e in menu.extra_dishes] == ["Lamb Kofta"]
    assert menu.all_correct

    # Reported in full, so the HTML breakdown can show what the model made of
    # a dish the case never declared.
    extra = menu.extra_dishes[0]
    assert extra.verdict == "not_vegan"
    assert extra.reason == "lamb mince"
    assert extra.explicit_ingredients == ["lamb"]
    assert menu.to_dict()["extra_dishes"] == [extra.to_dict()]


def test_a_non_menu_page_scores_as_a_menu_with_nothing_found():
    """An "other" page carries no menu branch at all."""
    menu = menu_scoring.score_menu(
        {"dishes": [_dish("Hummus")]},
        {"page_kind": "other", "shopping_item": None, "menu": None},
    )
    assert menu.dish_recall == 0.0
    assert not menu.all_correct


def test_restaurant_name_comparison_ignores_punctuation_and_case():
    analysis = _menu_analysis([], restaurant_name="the olive branch")
    menu = menu_scoring.score_menu(
        {"restaurant_name": "The Olive Branch", "dishes": []}, analysis
    )
    assert menu.restaurant_name_correct


# --------------------------------------------------------------------------- #
# Integration with score_case / aggregate
# --------------------------------------------------------------------------- #


def test_case_is_not_fully_correct_when_only_the_menu_is_wrong():
    expect = {
        "page_kind": "restaurant_menu",
        "menu": {"dishes": [_dish("Kofta", "not_vegan")]},
    }
    result = score_case(
        "menu_case", "restaurant", expect, _menu_analysis([_dish("Kofta", "vegan")])
    )
    assert result.fields["page_kind"].correct  # classification was right
    assert not result.all_correct  # ...but the lamb kofta was called vegan


def test_field_expectation_accepts_a_list_of_values():
    expect = {"vegan_friendliness": ["medium", "high"]}
    assert score_case("c", "restaurant", expect, _menu_analysis([])).all_correct
    assert not score_case(
        "c", "restaurant", expect, _menu_analysis([], vegan_friendliness="none")
    ).all_correct


def test_aggregate_reports_menu_metrics_and_verdict_confusion():
    expect = {"menu": {"dishes": [_dish("Hummus"), _dish("Kofta", "not_vegan")]}}
    results = [
        score_case(
            "menu_case",
            "restaurant",
            expect,
            _menu_analysis([_dish("Hummus"), _dish("Kofta", "unclear")]),
        )
    ]
    agg = aggregate(results)
    assert agg["menu"]["cases"] == 1
    assert agg["menu"]["mean_dish_recall"] == 1.0
    assert agg["menu"]["mean_verdict_accuracy"] == 0.5
    assert agg["menu"]["verdict_confusion"]["not_vegan->unclear"] == 1


def test_aggregate_breaks_down_by_expected_page_kind():
    item = {"page_kind": "shopping_item", "is_vegan": True}
    results = [
        score_case("a", "food", item, _item_analysis(is_vegan=True)),
        score_case("b", "food", item, _item_analysis(is_vegan=False)),
        score_case("c", "x", {"page_kind": "other"}, None, error="boom"),
    ]
    by_type = aggregate(results)["by_type"]
    assert list(by_type) == ["shopping_item", "other"]
    assert by_type["shopping_item"]["fully_correct"] == 1
    assert by_type["shopping_item"]["per_field"]["is_vegan"]["evaluated"] == 2
    assert by_type["other"]["errored"] == 1


def test_aggregate_omits_the_menu_block_when_no_case_declares_one():
    results = [
        score_case("item", "food", {"is_vegan": True}, _item_analysis(is_vegan=True))
    ]
    assert aggregate(results)["menu"] is None


# --------------------------------------------------------------------------- #
# Shopping-item ingredients
# --------------------------------------------------------------------------- #


def test_ingredients_are_checked_per_list_and_excluded_terms_anywhere():
    expect = {
        "ingredients": {
            "explicit": ["wheat gluten", ["merino", "wool"]],
            "inferred": ["casing"],
            "excluded": ["pork"],
        }
    }
    analysis = _item_analysis(
        explicit_ingredients=["Wheat Gluten*", "Casing"],
        inferred_ingredients=["pork fat"],
    )

    ing = score_case("c", "food", expect, analysis).ingredients

    # "casing" was listed, but as explicit: it does not count for inferred.
    assert ing.missing == [("explicit", ["merino", "wool"]), ("inferred", ["casing"])]
    assert ing.recall == 1 / 3
    assert ing.excluded_found == ["pork"]
    assert not ing.all_correct


def test_a_case_with_only_excluded_terms_passes_on_clean_lists():
    expect = {"is_vegan": True, "ingredients": {"excluded": ["milk"]}}
    analysis = _item_analysis(
        is_vegan=True, explicit_ingredients=["cashews"], inferred_ingredients=[]
    )

    result = score_case("c", "food", expect, analysis)

    assert result.ingredients.recall is None
    assert result.all_correct
    assert aggregate([result])["ingredients"]["excluded_hits"] == 0
