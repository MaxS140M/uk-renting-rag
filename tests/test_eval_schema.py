"""Tests for the evaluation item schema and JSONL helpers."""

import pytest
from pydantic import ValidationError

from rag.eval_schema import EvalItem, ItemFileError, append_item, load_items, next_id

QUOTE_A = "Your landlord must protect your deposit within 30 days."
QUOTE_B = "Letting agents cannot charge fees for viewings or references."


def make(**overrides) -> dict:
    data = {
        "id": "q-001",
        "question": "How long does my landlord have to protect my deposit?",
        "reference_answer": "30 days.",
        "question_type": "factual",
        "evidence": [{"doc_id": "deposit", "quote": QUOTE_A}],
        "author": "max",
    }
    data.update(overrides)
    return data


def test_valid_item_with_defaults():
    item = EvalItem.model_validate(make())
    assert item.split == "dev" and item.notes == "" and item.answerable


def test_unanswerable_must_have_no_evidence():
    EvalItem.model_validate(make(question_type="unanswerable", evidence=[]))
    with pytest.raises(ValidationError, match="no evidence"):
        EvalItem.model_validate(make(question_type="unanswerable"))


@pytest.mark.parametrize("question_type", ["factual", "multi_passage", "informal"])
def test_answerable_items_need_evidence(question_type):
    with pytest.raises(ValidationError, match="at least one evidence"):
        EvalItem.model_validate(make(question_type=question_type, evidence=[]))


def test_multi_passage_needs_two_different_quotes():
    one = [{"doc_id": "deposit", "quote": QUOTE_A}]
    with pytest.raises(ValidationError, match="two different"):
        EvalItem.model_validate(make(question_type="multi_passage", evidence=one * 2))
    two = one + [{"doc_id": "fees", "quote": QUOTE_B}]
    EvalItem.model_validate(make(question_type="multi_passage", evidence=two))


@pytest.mark.parametrize(
    "overrides",
    [
        {"question_type": "trivia"},
        {"split": "test"},
        {"author": "chatgpt"},
        {"id": "Q1"},
        {"question": "Deposit?"},
        {"reference_answer": ""},
        {"evidence": [{"doc_id": "deposit", "quote": "30 days"}]},  # quote too short
        {"evidence": [{"doc_id": "deposit", "quote": QUOTE_A, "page": 3}]},  # unknown field
        {"quesion": "typo in a field name"},
    ],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValidationError):
        EvalItem.model_validate(make(**overrides))


@pytest.mark.parametrize(("item_id", "author"), [("q-001", "example"), ("ex-001", "max")])
def test_id_prefix_must_match_author(item_id, author):
    with pytest.raises(ValidationError, match="should start with"):
        EvalItem.model_validate(make(id=item_id, author=author))


def test_load_reports_every_bad_line_with_its_number(tmp_path):
    path = tmp_path / "questions.jsonl"
    good = EvalItem.model_validate(make())
    append_item(path, good)
    with path.open("a", encoding="utf-8") as f:
        f.write("{not json\n")
        f.write('{"id": "q-002"}\n')
    with pytest.raises(ItemFileError) as err:
        load_items(path)
    assert [p.split(":")[0] for p in err.value.problems] == ["line 2", "line 3"]


def test_round_trip_and_next_id(tmp_path):
    path = tmp_path / "questions.jsonl"
    items = [EvalItem.model_validate(make(id=f"q-{n:03d}")) for n in (1, 2, 7)]
    items.append(EvalItem.model_validate(make(id="ex-001", author="example")))
    for item in items:
        append_item(path, item)
    loaded = load_items(path)
    assert loaded == items
    assert next_id(loaded, "max") == "q-008"
    assert next_id(loaded, "llm_draft_reviewed") == "q-008"  # same numbering as "max"
    assert next_id(loaded, "example") == "ex-002"
    assert next_id([], "llm_draft") == "draft-001"
