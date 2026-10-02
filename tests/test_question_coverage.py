"""Tests for the coverage report."""

import importlib.util
from pathlib import Path

from rag.evaluation.schema import EvalItem
from rag.evaluation.utils import load_corpus

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "question_coverage", ROOT / "scripts" / "testset" / "question_coverage.py"
)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

QUOTE = "Your landlord or letting agent must put your deposit in the scheme"


def item(item_id, question_type="factual", doc_id="tenancy-deposit-protection", **extra):
    evidence = [] if question_type == "unanswerable" else [{"doc_id": doc_id, "quote": QUOTE}]
    if question_type == "multi_passage":
        evidence.append({"doc_id": "renters-rights-act-overview-for-tenants", "quote": QUOTE + "!"})
    return EvalItem.model_validate(
        {
            "id": item_id,
            "question": f"Question {item_id}?",
            "reference_answer": "Answer.",
            "question_type": question_type,
            "evidence": evidence,
            "author": "example" if item_id.startswith("ex") else "max",
            **extra,
        }
    )


def test_coverage_counts_types_splits_and_documents():
    docs = load_corpus(ROOT / "data" / "sample")
    items = [
        item("q-001"),
        item("q-002", split="heldout"),
        item("q-003", "multi_passage"),
        item("q-004", "unanswerable"),
        item("ex-001"),  # excluded from counts
    ]
    report = coverage.build_coverage(items, docs)

    assert "**4 of 100 questions written.**" in report
    assert "| factual | 2 | 60 |" in report
    assert "| multi_passage | 1 | 20 |" in report
    assert "| unanswerable | 1 | 10 |" in report
    assert "dev 3, held-out 1" in report
    assert "| 3 | Tenancy deposit protection |" in report  # q-001, q-002, q-003
    assert "Counts exclude 1 example item(s)" in report


def test_documents_without_questions_are_listed():
    docs = load_corpus(ROOT / "data" / "sample")
    report = coverage.build_coverage([item("q-001")], docs)
    gaps = report.split("## Documents with no questions")[1]
    assert "(2 of 3)" in gaps
    assert "damp-and-mould-in-the-private-rented-sector" in gaps
    assert "tenancy-deposit-protection" not in gaps
