"""Tests for the judge agreement tool, with a synthetic results file and scripted labels."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "judge_agreement.py"
spec = importlib.util.spec_from_file_location("judge_agreement", SCRIPT)
judge_agreement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge_agreement)


def record(n, faithful=True, verdict="correct", refused=False, qtype="factual"):
    return {
        "id": f"q-{n:03d}",
        "question": f"Question {n}?",
        "question_type": qtype,
        "retrieved": [{"chunk_id": "c1"}],
        "generation": {
            "answer": f"Answer {n} [1].",
            "refused": refused,
            "faithfulness": None
            if refused
            else {
                "score": 1.0 if faithful else 0.5,
                "faithful": faithful,
                "reasoning": "judge says",
            },
            "correctness": {"verdict": verdict, "score": 1.0, "reasoning": "judge says"},
        },
    }


def test_sample_uses_only_answered_answerable_questions_and_is_reproducible():
    records = [record(n) for n in range(1, 41)]
    records += [record(41, refused=True), record(42, qtype="unanswerable")]
    first, second = judge_agreement.sample(records), judge_agreement.sample(records)
    assert len(first) == 30 and [r["id"] for r in first] == [r["id"] for r in second]
    assert not {"q-041", "q-042"} & {r["id"] for r in first}


@pytest.fixture
def files(tmp_path):
    # 4 answers. The judge says: faithful T, T, F, F and correct, correct, partial, incorrect.
    records = [
        record(1, True, "correct"),
        record(2, True, "correct"),
        record(3, False, "partially_correct"),
        record(4, False, "incorrect"),
    ]
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    chunk = {
        "chunk_id": "c1",
        "doc_id": "d",
        "title": "Deposits",
        "section": "Overview",
        "text": "Your landlord must protect your deposit within 30 days.",
    }
    (index / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "".join(
            json.dumps({"id": r["id"], "reference_answer": "Within 30 days."}) + "\n"
            for r in records
        ),
        encoding="utf-8",
    )
    return {
        "results": results,
        "chunks": index / "chunks.jsonl",
        "questions": questions,
        "labels": tmp_path / "labels.jsonl",
        "out": tmp_path / "agreement.md",
    }


def run(files, answers, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            "hybrid_rerank",
            "--results",
            str(files["results"]),
            "--chunks",
            str(files["chunks"]),
            "--questions",
            str(files["questions"]),
            "--labels",
            str(files["labels"]),
            "--out",
            str(files["out"]),
            *extra,
        ],
        input="\n".join(answers) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def test_labelling_is_blind_and_agreement_is_computed(files):
    # Human: faithful y, y, n, y (disagrees on q-004) and c, p, p, i (disagrees on q-002).
    # Items are shown in the sampled order, so answer by id.
    order = [
        r["id"]
        for r in judge_agreement.sample(
            [json.loads(line) for line in files["results"].read_text().splitlines()]
        )
    ]
    human = {"q-001": ("y", "c"), "q-002": ("y", "p"), "q-003": ("n", "p"), "q-004": ("y", "i")}
    answers = [x for item_id in order for x in (*human[item_id], "")]
    result = run(files, answers)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "judge says" not in result.stdout.split("# LLM judge agreement")[0]  # never shown

    stats = json.loads(files["out"].with_suffix(".json").read_text())
    # Faithfulness: 3 of 4 agree. Human T,T,F,T (3 T); judge T,T,F,F (2 T).
    # p_e = 0.75 * 0.5 + 0.25 * 0.5 = 0.5, kappa = (0.75 - 0.5) / 0.5 = 0.5
    assert stats["faithfulness_agreement"] == 0.75
    assert stats["faithfulness_kappa"] == pytest.approx(0.5)
    assert stats["correctness_agreement"] == 0.75
    assert stats["trusted"]["faithfulness"] is False
    report = files["out"].read_text(encoding="utf-8")
    assert "q-004" in report and "Faithful: human True, judge False" in report


def test_labels_resume_and_report_only_mode(files):
    first = run(files, ["y", "c", "", "q"])  # label one item, then quit
    assert "Labels so far are saved" in first.stdout
    assert len(files["labels"].read_text().splitlines()) == 1
    report = run(files, [], "--report")
    assert report.returncode == 0
    assert json.loads(files["out"].with_suffix(".json").read_text())["n"] == 1
