"""Tests for the interactive authoring CLI: helper functions plus scripted end-to-end runs."""

import subprocess
import sys
from pathlib import Path

import pytest

from rag.evaluation.authoring import parse_selection, sentence_units
from rag.evaluation.schema import load_items

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "testset" / "add_question.py"
SAMPLE = ROOT / "data" / "sample"


@pytest.mark.parametrize(
    ("selection", "expected"),
    [("2", [1]), ("2-4", [1, 2, 3]), ("a", [0, 1, 2, 3, 4]), ("A", [0, 1, 2, 3, 4])],
)
def test_parse_selection(selection, expected):
    assert parse_selection(selection, 5) == expected


@pytest.mark.parametrize("selection", ["0", "6", "4-2", "1-9", "two", "Your landlord must"])
def test_parse_selection_rejects_non_selections(selection):
    assert parse_selection(selection, 5) is None


def test_sentence_units_split_sentences_and_keep_list_items_whole():
    text = "First sentence here. Second one.\n- a list item, with a comma\n- another item"
    assert sentence_units(text) == [
        "First sentence here.",
        "Second one.",
        "- a list item, with a comma",
        "- another item",
    ]


def run(tmp_path: Path, answers: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--questions",
            str(tmp_path / "q.jsonl"),
            "--corpus",
            str(SAMPLE),
        ],
        input="\n".join(answers) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_scripted_factual_question_is_saved(tmp_path):
    result = run(
        tmp_path,
        [
            "How long does my landlord have to protect my deposit?",
            "1",  # factual
            "scheme within 30 days",  # search
            "1",  # first passage
            "1",  # first sentence
            "",  # finish searching
            "Within 30 days of getting it.",
            "",  # end of reference answer
            "",  # no notes
            "y",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (item,) = load_items(tmp_path / "q.jsonl")
    assert item.id == "q-001" and item.author == "max" and item.split == "dev"
    assert item.evidence[0].doc_id == "tenancy-deposit-protection"
    assert "within 30 days of getting it" in item.evidence[0].quote


def test_scripted_unanswerable_question_uses_default_answer(tmp_path):
    result = run(tmp_path, ["What is the average rent in Manchester?", "4", "", "", "y"])
    assert result.returncode == 0, result.stdout + result.stderr
    (item,) = load_items(tmp_path / "q.jsonl")
    assert item.question_type == "unanswerable" and item.evidence == []
    assert "does not cover" in item.reference_answer
    assert "confirm none of them answers" in result.stdout


def test_pasted_text_not_in_the_document_is_rejected(tmp_path):
    result = run(
        tmp_path,
        [
            "How long does my landlord have to protect my deposit?",
            "1",
            "scheme within 30 days",
            "1",
            "Your landlord must protect your deposit within 45 days.",  # not in the document
            "1",
            "",
            "Within 30 days.",
            "",
            "",
            "y",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "does not appear exactly in the document" in result.stdout
    assert len(load_items(tmp_path / "q.jsonl")) == 1


def test_input_ending_early_saves_nothing(tmp_path):
    result = run(tmp_path, ["How long does my landlord have to protect my deposit?"])
    assert result.returncode == 1
    assert not (tmp_path / "q.jsonl").exists()
