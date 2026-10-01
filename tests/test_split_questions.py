"""Tests for assigning the stratified held-out split."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from rag.eval_schema import EvalItem

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "split_questions", ROOT / "scripts" / "split_questions.py"
)
split_questions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_questions)

QUOTE = "Your landlord must protect your deposit within 30 days."
MIX = {"factual": 60, "multi_passage": 20, "informal": 10, "unanswerable": 10}


def make_items(mix=MIX, examples=3) -> list[EvalItem]:
    items, n = [], 0
    for question_type, count in mix.items():
        for _ in range(count):
            n += 1
            evidence = [] if question_type == "unanswerable" else [{"doc_id": "d", "quote": QUOTE}]
            if question_type == "multi_passage":
                evidence.append({"doc_id": "d", "quote": QUOTE + " Also more."})
            items.append(
                EvalItem.model_validate(
                    {
                        "id": f"q-{n:03d}",
                        "question": f"Question number {n}?",
                        "reference_answer": "A.",
                        "question_type": question_type,
                        "evidence": evidence,
                        "author": "max",
                    }
                )
            )
    for k in range(1, examples + 1):
        items.append(
            EvalItem.model_validate(
                {
                    "id": f"ex-{k:03d}",
                    "question": f"Example question {k}?",
                    "reference_answer": "A.",
                    "question_type": "factual",
                    "evidence": [{"doc_id": "d", "quote": QUOTE}],
                    "author": "example",
                }
            )
        )
    return items


def heldout_types(items):
    return Counter(i.question_type for i in items if i.split == "heldout")


def test_heldout_is_20_items_stratified_by_type():
    updated = split_questions.assign_heldout(make_items())
    assert heldout_types(updated) == {
        "factual": 12,
        "multi_passage": 4,
        "informal": 2,
        "unanswerable": 2,
    }


def test_examples_are_never_held_out():
    updated = split_questions.assign_heldout(make_items())
    assert all(i.split == "dev" for i in updated if i.is_example)


def test_split_is_reproducible_with_the_same_seed():
    first = split_questions.assign_heldout(make_items(), seed=42)
    second = split_questions.assign_heldout(make_items(), seed=42)
    other = split_questions.assign_heldout(make_items(), seed=7)
    ids = lambda items: {i.id for i in items if i.split == "heldout"}  # noqa: E731
    assert ids(first) == ids(second) != ids(other)


def test_refuses_to_change_an_existing_heldout_set():
    once = split_questions.assign_heldout(make_items())
    with pytest.raises(split_questions.SplitError, match="already held out"):
        split_questions.assign_heldout(once)
    assert heldout_types(split_questions.assign_heldout(once, force=True))


def test_refuses_to_split_an_incomplete_set():
    small = make_items({"factual": 30, "unanswerable": 5})
    with pytest.raises(split_questions.SplitError, match="Only 35 of 100"):
        split_questions.assign_heldout(small)
    assert (
        len(
            [
                i
                for i in split_questions.assign_heldout(small, n=7, allow_incomplete=True)
                if i.split == "heldout"
            ]
        )
        == 7
    )
