"""Tests for the results table and error analysis reports."""

import importlib.util
from pathlib import Path

from rag.eval_schema import EvalItem
from rag.experiment import load_experiments

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


results_table = load_script("make_results_table")
error_analysis = load_script("error_analysis")
EXPERIMENTS = load_experiments(ROOT / "eval" / "configs.yaml")


def row(config, split="dev", **values):
    base = {
        "config": config,
        "split": split,
        "n_questions": "110",
        "n_answerable": "97",
        "n_unanswerable": "13",
        "corpus": "abc",
        "prompt_version": "v1",
    }
    return {**base, **{k: str(v) for k, v in values.items()}}


ROWS = [
    row(
        "dense_baseline",
        recall_at_5=0.60,
        mrr=0.45,
        faithfulness=0.90,
        correctness=0.70,
        refusal_accuracy=0.85,
        total_ms_p50=2500,
    ),
    row("bm25", recall_at_5=0.55, mrr=0.40, retrieval_ms_p50=1),
    row("hybrid", recall_at_5=0.70, mrr=0.52, retrieval_ms_p50=9),
    row(
        "hybrid_rerank",
        recall_at_5=0.78,
        mrr=0.61,
        faithfulness=0.93,
        correctness=0.79,
        refusal_accuracy=0.92,
        total_ms_p50=3200,
    ),
]


def test_headline_reports_absolute_point_differences():
    md = results_table.build_markdown(ROWS, EXPERIMENTS, None, [])
    headline = md.split("## Headline")[1].split("##")[0]
    assert "Best configuration: `hybrid_rerank`" in headline
    # 78% - 60% = +18.0 points (not "+30%"); correctness 79% - 70% = +9.0 points
    assert "Recall@5 +18.0 pts (60.0% → 78.0%)" in headline
    assert "correctness +9.0 pts (70.0% → 79.0%)" in headline
    assert "MRR +0.160" in headline


def test_main_table_marks_configs_without_generation():
    md = results_table.build_markdown(ROWS, EXPERIMENTS, None, [])
    bm25_line = next(line for line in md.splitlines() if line.startswith("| `bm25`"))
    assert "| — | — | — |" in bm25_line and "(retrieval only)" in bm25_line


def test_ablation_table_compares_each_experiment_with_its_reference():
    md = results_table.build_markdown(ROWS, EXPERIMENTS, None, [])
    rerank = next(line for line in md.splitlines() if line.startswith("| `hybrid_rerank` |"))
    assert "`hybrid`" in rerank and "use_reranker: False → True" in rerank
    assert "+8.0 pts" in rerank  # 78% - 70%


def test_judge_status_and_heldout_section():
    md = results_table.build_markdown(ROWS, EXPERIMENTS, None, [])
    assert "not yet been checked against human labels" in md
    assert "Not run yet" in md
    stats = {
        "n": 30,
        "faithfulness_agreement": 0.9,
        "faithfulness_kappa": 0.7,
        "correctness_agreement": 0.8,
        "correctness_kappa": 0.5,
        "trusted": {"faithfulness": True, "correctness": False},
    }
    rows = ROWS + [
        row(
            "hybrid_rerank",
            "heldout",
            recall_at_5=0.75,
            mrr=0.6,
            correctness=0.8,
            faithfulness=0.9,
            refusal_accuracy=1.0,
            total_ms_p50=3000,
        )
    ]
    md = results_table.build_markdown(
        rows, EXPERIMENTS, stats, [{"run_at": "2026-10-02", "forced": False}]
    )
    assert "below the trust threshold (kappa 0.6) for correctness" in md
    heldout = md.split("## Held-out result (run once)")[1]
    assert "2026-10-02" in heldout and "| `hybrid_rerank`" in heldout


# --- Error analysis -------------------------------------------------------------------------

QUOTE = "Your landlord must protect your deposit in a scheme within 30 days."


def item(item_id, question_type="factual"):
    return EvalItem(
        id=item_id,
        question="A test question?",
        reference_answer="30 days.",
        question_type=question_type,
        evidence=[] if question_type == "unanswerable" else [{"doc_id": "d", "quote": QUOTE}],
        author="llm_generated",
    )


def record(item_id, rank=None, verdict="correct", faith=1.0, qtype="factual", refused=False):
    retrieved = [{"chunk_id": f"c{n}", "title": "T", "section": "S"} for n in range(1, 11)]
    return {
        "id": item_id,
        "question": "A test question?",
        "question_type": qtype,
        "retrieved": retrieved,
        "gold": [[f"c{rank}"]] if rank else [["zz"]],
        "first_gold_rank": rank,
        "generation": {
            "answer": "An answer.",
            "refused": refused,
            "refusal_correct": refused if qtype == "unanswerable" else None,
            "correctness": None
            if qtype == "unanswerable"
            else {"verdict": verdict, "reasoning": "r"},
            "faithfulness": {"score": faith, "claims": [{"claim": "x", "supported": faith == 1.0}]},
        },
    }


def test_failures_are_classified_by_cause():
    assert (
        error_analysis.classify(record("q-1", rank=None, verdict="incorrect"), 5) == "not_retrieved"
    )
    assert (
        error_analysis.classify(record("q-1", rank=8, verdict="incorrect"), 5) == "ranked_too_low"
    )
    assert error_analysis.classify(record("q-1", rank=2, verdict="incorrect"), 5) == "misread"
    unanswered = record("q-1", qtype="unanswerable", refused=False)
    assert error_analysis.classify(unanswered, 5) == "answered_unanswerable"
    assert not error_analysis.failed(record("q-1", rank=1))
    assert error_analysis.failed(record("q-1", rank=1, faith=0.5))  # an unsupported claim


def test_report_spreads_examples_across_causes_and_leaves_placeholders():
    records = [record(f"q-{n:03d}", rank=None, verdict="incorrect") for n in range(1, 21)]
    records += [
        record("q-101", rank=7, verdict="partially_correct"),
        record("q-102", qtype="unanswerable", refused=False),
    ]
    items = {r["id"]: item(r["id"], r["question_type"]) for r in records}
    report = error_analysis.build_report("hybrid_rerank", records, items, 5)
    assert "22 of 22 dev questions failed" in report
    assert "## Examples (15 of 22)" in report
    assert "#### q-101" in report and "#### q-102" in report  # rarer causes still shown
    assert report.count("**My diagnosis:** _TODO") == 15
    assert "## Conclusions" in report and "Reference answer wrong" in report
