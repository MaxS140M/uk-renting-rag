"""Tests for test-set validation: the checks themselves and the script's exit codes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rag.chunking import chunk_document
from rag.evaluation.schema import EvalItem, append_item
from rag.evaluation.utils import load_corpus, validate_items

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"
QUOTE = (
    "Your landlord or letting agent must put your deposit in the scheme within 30 days of "
    "getting it."
)


@pytest.fixture(scope="module")
def docs():
    return load_corpus(SAMPLE)


def item(
    item_id="q-001",
    question="How long does my landlord have to protect my deposit?",
    quote=QUOTE,
    doc_id="tenancy-deposit-protection",
    **overrides,
):
    data = {
        "id": item_id,
        "question": question,
        "reference_answer": "Within 30 days.",
        "question_type": "factual",
        "evidence": [{"doc_id": doc_id, "quote": quote}],
        "author": "max",
    }
    data.update(overrides)
    return EvalItem.model_validate(data)


def test_valid_item_passes(docs):
    report = validate_items([item()], docs)
    assert report.ok and report.warnings == []


def test_quote_not_in_document_is_an_error_showing_the_closest_text(docs):
    report = validate_items([item(quote=QUOTE.replace("30 days", "45 days"))], docs)
    (error,) = report.errors
    assert "quote not found" in error and "closest" in error and "30 days" in error


def test_unknown_document_is_an_error(docs):
    assert "unknown doc_id" in validate_items([item(doc_id="no-such-doc")], docs).errors[0]


def test_duplicate_ids_and_duplicate_questions_are_errors(docs):
    report = validate_items([item(), item()], docs)
    assert any("id used 2 times" in e for e in report.errors)
    assert any("duplicate question" in e for e in report.errors)


def test_near_duplicates_are_warnings_not_errors(docs):
    second = item("q-002", question="How long has my landlord got to protect my deposit?")
    report = validate_items([item(), second], docs)
    assert report.ok
    assert any("similar" in w for w in report.warnings)


def test_unreviewed_drafts_are_not_allowed_in_the_question_file(docs):
    report = validate_items([item("draft-001", author="llm_draft")], docs)
    assert any("unreviewed LLM draft" in e for e in report.errors)


def test_unanswerable_answer_should_say_the_guidance_does_not_cover_it(docs):
    vague = item(question_type="unanswerable", evidence=[], reference_answer="About £900.")
    clear = item(
        "q-002",
        question="What is the average rent in Manchester?",
        question_type="unanswerable",
        evidence=[],
        reference_answer="The guidance doesn't cover average rents.",
    )
    report = validate_items([vague, clear], docs)
    assert [w.split(":")[0] for w in report.warnings] == ["q-001"]


def test_examples_are_flagged(docs):
    report = validate_items([item("ex-001", author="example")], docs)
    assert report.ok and "example item" in report.warnings[0]


def test_quote_must_map_to_a_chunk_when_chunks_are_given(docs):
    other_doc = docs["renters-rights-act-overview-for-tenants"]
    unrelated_chunks = [c.to_dict() for c in chunk_document(other_doc)]
    report = validate_items([item()], docs, chunks=unrelated_chunks)
    assert any("does not map to any chunk" in e for e in report.errors)


# --- Script exit codes ----------------------------------------------------------------------


def run_script(path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_questions.py"),
            "--questions",
            str(path),
            "--corpus",
            str(SAMPLE),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )


def test_script_exit_codes(tmp_path):
    path = tmp_path / "questions.jsonl"
    append_item(path, item())
    ok = run_script(path)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "OK: 0 error(s)" in ok.stdout

    append_item(path, item("ex-001", question="An example question here?", author="example"))
    assert run_script(path, "--skip-chunk-check").returncode == 0  # warning only
    assert run_script(path, "--skip-chunk-check", "--strict").returncode == 1

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "q-003", "question": "Broken?"}) + "\n")
    broken = run_script(path, "--skip-chunk-check")
    assert broken.returncode == 1 and "line 3" in broken.stdout


@pytest.mark.parametrize(
    "answer",
    [
        "The guidance does not explain which scheme covers Northern Ireland.",
        "The guidance does not give current market rents for Leeds.",
        "I can't find that in the guidance.",
    ],
)
def test_other_ways_of_saying_not_covered_are_accepted(docs, answer):
    unanswerable = item(question_type="unanswerable", evidence=[], reference_answer=answer)
    assert validate_items([unanswerable], docs).warnings == []
