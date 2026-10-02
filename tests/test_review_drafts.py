"""End-to-end tests for the draft review tool, driven by scripted keyboard input."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rag.evaluation.schema import EvalItem, append_item, load_items

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "testset" / "review_drafts.py"
SAMPLE = ROOT / "data" / "sample"
QUOTE = (
    "Your landlord or letting agent must put your deposit in the scheme within 30 days of "
    "getting it."
)


def draft(n, question, question_type="factual", quote=QUOTE):
    evidence = (
        []
        if question_type == "unanswerable"
        else [{"doc_id": "tenancy-deposit-protection", "quote": quote}]
    )
    return EvalItem(
        id=f"draft-{n:03d}",
        question=question,
        reference_answer="Within 30 days of getting it."
        if evidence
        else "The guidance does not cover this.",
        question_type=question_type,
        evidence=evidence,
        author="llm_draft",
        notes="Persona-generated.",
    )


@pytest.fixture
def files(tmp_path):
    drafts = tmp_path / "drafts.jsonl"
    for d in (
        draft(1, "How long does my landlord have to protect my deposit?"),
        draft(
            2,
            "When must a deposit go into a protection scheme?",
            quote="Landlords must protect deposits within thirty days of receiving them.",
        ),
        draft(3, "What is the average rent in Leeds?", "unanswerable"),
    ):
        append_item(drafts, d)
    return {
        "drafts": drafts,
        "questions": tmp_path / "questions.jsonl",
        "log": tmp_path / "review_log.jsonl",
        "summary": tmp_path / "summary.md",
    }


def run(files, answers, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--drafts",
            str(files["drafts"]),
            "--questions",
            str(files["questions"]),
            "--log",
            str(files["log"]),
            "--summary-file",
            str(files["summary"]),
            "--corpus",
            str(SAMPLE),
            *extra,
        ],
        input="\n".join(answers) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_accept_edit_reject_then_resume(files):
    files["questions"].touch()
    answers = [
        "a",
        "",  # draft 1: accept unchanged, no note
        "a",  # draft 2: refused, its quote is not in the document
        "e",
        "v",  # edit the evidence
        "scheme within 30 days",
        "1",
        "1",
        "",  # search, pick passage 1, sentence 1, finish
        "d",
        "fixed paraphrased quote",  # done editing, note
        "r",
        "duplicate of draft 1",  # draft 3: reject with a reason
    ]
    result = run(files, answers)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NOT FOUND VERBATIM" in result.stdout
    assert "not in the document word for word" in result.stdout

    items = load_items(files["questions"])
    assert [i.id for i in items] == ["q-001", "q-002"]
    assert all(i.author == "llm_draft_reviewed" for i in items)
    assert items[0].notes == "Reviewed from draft-001."
    assert "within 30 days of getting it" in items[1].evidence[0].quote
    assert items[1].notes == "Reviewed from draft-002. fixed paraphrased quote"

    log = read_log(files["log"])
    assert [e["decision"] for e in log] == ["accepted", "edited", "rejected"]
    assert log[1]["changed"] == ["evidence"]
    assert log[2]["note"] == "duplicate of draft 1"

    summary = files["summary"].read_text(encoding="utf-8")
    assert "Accepted unchanged: 1" in summary and "Accepted after editing: 1" in summary
    assert "Rejected: 1" in summary and "evidence 1" in summary

    again = run(files, [])
    assert "Nothing left to review." in again.stdout
    assert len(load_items(files["questions"])) == 2


def test_quitting_saves_nothing_more_and_resumes_at_the_same_draft(files):
    files["questions"].touch()
    first = run(files, ["a", "", "q"])  # accept draft 1, then quit at draft 2
    assert "Progress is saved" in first.stdout
    assert len(read_log(files["log"])) == 1

    second = run(files, ["s", "s"])  # skip both remaining drafts
    assert "[2/3]  draft-002" in second.stdout
    assert len(read_log(files["log"])) == 1  # skipping is not a decision


def test_editing_the_type_to_unanswerable_clears_evidence(files):
    files["questions"].touch()
    result = run(files, ["e", "t", "4", "a", "The guidance does not cover this.", "", "d", "", "q"])
    assert result.returncode == 0, result.stdout + result.stderr
    (item,) = load_items(files["questions"])
    assert item.question_type == "unanswerable" and item.evidence == []
    assert read_log(files["log"])[0]["changed"] == ["question_type", "reference_answer", "evidence"]


def test_summary_only(files):
    files["questions"].touch()
    result = run(files, [], "--summary")
    assert "Drafts: 3; reviewed: 0; still to review: 3" in result.stdout
