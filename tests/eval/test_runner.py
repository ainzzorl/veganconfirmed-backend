"""Offline tests for how `--run` entries are parsed into combos, and how cases
are grouped into batches.

    uv run pytest tests/eval/test_runner.py

What they protect: a combo's label is what every report attributes its numbers
to, so an entry that parses into the wrong model — or an effort that silently
goes nowhere — would misreport which configuration was measured. Batching is
what makes a run concurrent, and cases in flight that mix categories are less
alike — and so waste more of a slot — than ones that don't.
"""

from __future__ import annotations

import argparse

import pytest

from tests.eval.runner import Combo, _batch_order, _resolve_combos


def _combos(*entries: str) -> list[Combo]:
    return _resolve_combos(argparse.Namespace(run=list(entries)))


def test_effort_levels_of_one_model_are_separate_combos():
    assert _combos(
        "desktop:openai/gpt-oss-20b:low,desktop:openai/gpt-oss-20b:high",
        "desktop:openai/gpt-oss-20b",
    ) == [
        Combo("desktop", "openai/gpt-oss-20b", "low"),
        Combo("desktop", "openai/gpt-oss-20b", "high"),
        Combo("desktop", "openai/gpt-oss-20b", None),
    ]


def test_a_third_field_that_is_not_an_effort_is_rejected():
    # Rather than benchmarking "qwen/qwen3.5" and reporting it under that name.
    with pytest.raises(SystemExit, match="not a reasoning effort"):
        _combos("desktop:qwen/qwen3.5:9b")


def test_only_the_desktop_provider_takes_an_effort():
    with pytest.raises(SystemExit, match="only supported by the desktop provider"):
        _combos("gemini:gemini-2.5-flash-lite:high")


def _records(*categories: str) -> list[dict]:
    return [
        {"id": f"case{i}", "category": category}
        for i, category in enumerate(categories)
    ]


def test_batch_order_groups_a_category_together():
    # menu/menu/menu then food/food, so the cases in flight are alike, rather
    # than the corpus order that interleaves them.
    ordered = _batch_order(_records("menu", "food", "menu", "food", "menu"))
    assert [r["category"] for r in ordered] == [
        "menu",
        "menu",
        "menu",
        "food",
        "food",
    ]


def test_batch_order_keeps_every_case_exactly_once():
    records = _records("menu", "food", "food", "clothing")
    assert sorted(r["id"] for r in _batch_order(records)) == [r["id"] for r in records]
