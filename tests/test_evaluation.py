"""Tests for the evaluation runner's building blocks, with every LLM call mocked."""

import csv
import json
from types import SimpleNamespace

import pytest

from rag.config import RAGConfig
from rag.evaluation.judges import Judge
from rag.evaluation.runner import (
    HeldoutRefused,
    check_heldout_allowed,
    estimate_generation_cost,
    evaluate_generation,
    evaluate_retrieval,
    log_heldout_run,
    summarise,
    update_summary_csv,
)
from rag.evaluation.schema import EvalItem
from rag.evaluation.utils import EvidenceMapper
from rag.generate import Generator
from rag.llm_cache import CachingClient, LLMCache, LLMReply
from rag.retrieval import RetrievalResult, Retriever

QUOTE = "Your landlord must protect your deposit in a scheme within 30 days."


def chunk(cid, text, doc="deposit"):
    return {
        "chunk_id": cid,
        "doc_id": doc,
        "title": "Deposits",
        "url": f"https://www.gov.uk/{doc}",
        "section": "Overview",
        "date_retrieved": "2026-10-01",
        "text": text,
    }


CHUNKS = [
    chunk("c1", "Unrelated text about repairs and boilers."),
    chunk("c2", "More unrelated text about fees."),
    chunk("c3", f"Intro sentence. {QUOTE} More detail."),
    chunk("c4", "Text about notice periods for evictions."),
]


class FixedRetriever(Retriever):
    name = "fixed"

    def __init__(self, order):
        self.order = order

    def retrieve(self, query, k, timings=None):
        if timings is not None:
            timings["dense"] = 5.0
        by_id = {c["chunk_id"]: c for c in CHUNKS}
        return [
            RetrievalResult.from_chunk(by_id[c], 1 / r, r, "dense")
            for r, c in enumerate(self.order[:k], start=1)
        ]


def item(item_id="q-001", question_type="factual", evidence=((("deposit", QUOTE)),)):
    return EvalItem(
        id=item_id,
        question="How long to protect my deposit?",
        reference_answer="30 days.",
        question_type=question_type,
        evidence=[]
        if question_type == "unanswerable"
        else [{"doc_id": d, "quote": q} for d, q in evidence],
        author="llm_generated",
    )


# --- Retrieval ------------------------------------------------------------------------------


def test_retrieval_metrics_for_gold_at_rank_3():
    record = evaluate_retrieval(
        item(), FixedRetriever(["c1", "c2", "c3", "c4"]), EvidenceMapper(CHUNKS)
    )
    assert record["gold"] == [["c3"]]
    assert record["first_gold_rank"] == 3
    assert (record["recall_at_1"], record["recall_at_5"]) == (0.0, 1.0)
    assert record["reciprocal_rank"] == pytest.approx(1 / 3)
    assert record["retrieval_ms"] == {"dense": 5.0}


def test_unanswerable_items_get_no_retrieval_metrics():
    record = evaluate_retrieval(
        item(question_type="unanswerable"), FixedRetriever(["c1"]), EvidenceMapper(CHUNKS)
    )
    assert "recall_at_5" not in record and record["retrieved"]


def test_unmappable_evidence_is_excluded_not_scored_as_a_miss():
    missing = item(evidence=(("deposit", "A quote that is in none of the chunks at all."),))
    record = evaluate_retrieval(missing, FixedRetriever(["c1"]), EvidenceMapper(CHUNKS))
    assert record["unmapped_evidence"] == 1 and "recall_at_5" not in record


def test_summary_averages_hand_worked():
    # Two answerable questions with gold at rank 1 and rank 4, plus one unanswerable:
    # R@1 = (1 + 0) / 2 = 0.5, R@5 = 1.0, MRR = (1 + 1/4) / 2 = 0.625
    mapper = EvidenceMapper(CHUNKS)
    records = [
        evaluate_retrieval(item("q-001"), FixedRetriever(["c3", "c1"]), mapper),
        evaluate_retrieval(item("q-002"), FixedRetriever(["c1", "c2", "c4", "c3"]), mapper),
        evaluate_retrieval(item("q-003", "unanswerable"), FixedRetriever(["c1"]), mapper),
    ]
    s = summarise(records)
    assert (s["n_answerable"], s["n_unanswerable"]) == (2, 1)
    assert (s["recall_at_1"], s["recall_at_5"], s["mrr"]) == (0.5, 1.0, 0.625)
    assert s["generated"] == 0 and s["correctness"] is None


# --- Generation with mocked LLMs ------------------------------------------------------------


class FakeMessages:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            model=kwargs["model"],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def fake_judge(cache):
    calls = []

    def send(request):
        calls.append(request)
        if "claims" in json.dumps(request["output_config"]):
            payload = {
                "claims": [
                    {"claim": "30 days", "supported": True, "explanation": "ok"},
                    {"claim": "fine", "supported": False, "explanation": "no"},
                ],
                "reasoning": "One claim unsupported.",
            }
        else:
            payload = {"verdict": "partially_correct", "reasoning": "Missing detail."}
        return LLMReply(json.dumps(payload), "judge", "end_turn", 50, 20, latency_ms=900.0)

    return Judge(send, cache, "judge", "medium"), calls


def run_generation(tmp_path, answer, test_item, retrieval_order=("c3", "c1")):
    cache = LLMCache(tmp_path)
    caching = CachingClient(SimpleNamespace(messages=FakeMessages(answer)), cache)
    generator = Generator(RAGConfig(), client=caching)
    judge, judge_calls = fake_judge(cache)
    record = evaluate_retrieval(
        test_item, FixedRetriever(list(retrieval_order)), EvidenceMapper(CHUNKS)
    )
    passages = FixedRetriever(list(retrieval_order)).retrieve("q", 5)
    evaluate_generation(test_item, record, passages, generator, caching, judge)
    return record, judge_calls


def test_answered_question_is_judged(tmp_path):
    record, judge_calls = run_generation(tmp_path, "Within 30 days [1].", item())
    gen = record["generation"]
    assert gen["refused"] is False and gen["cited_chunk_ids"] == ["c3"]
    assert gen["faithfulness"]["score"] == 0.5 and gen["faithfulness"]["faithful"] is False
    assert (
        gen["correctness"]["verdict"] == "partially_correct" and gen["correctness"]["score"] == 0.5
    )
    assert len(judge_calls) == 2
    assert (
        record["latency_ms"]["total"]
        == record["latency_ms"]["retrieval"] + record["latency_ms"]["generation"]
    )


def test_false_refusal_is_incorrect_without_calling_the_judge(tmp_path):
    record, judge_calls = run_generation(tmp_path, "I can't find that in the guidance.", item())
    assert record["generation"]["refused"] is True
    assert record["generation"]["correctness"]["verdict"] == "incorrect"
    assert judge_calls == []


def test_unanswerable_refusal_is_correct(tmp_path):
    record, judge_calls = run_generation(
        tmp_path, "I can't find that in the guidance.", item(question_type="unanswerable")
    )
    assert record["generation"]["refusal_correct"] is True and judge_calls == []
    s = summarise([record])
    assert s["refusal_accuracy"] == 1.0 and s["false_refusal_rate"] is None


def test_answering_an_unanswerable_question_is_checked_for_faithfulness(tmp_path):
    record, judge_calls = run_generation(
        tmp_path, "About £900 a month.", item(question_type="unanswerable")
    )
    assert record["generation"]["refusal_correct"] is False
    assert record["generation"]["faithfulness"] is not None and len(judge_calls) == 1


def test_rerun_uses_cached_answers_and_judgements(tmp_path):
    run_generation(tmp_path, "Within 30 days [1].", item())
    record, judge_calls = run_generation(tmp_path, "A DIFFERENT ANSWER", item())
    assert record["generation"]["answer"] == "Within 30 days [1]."  # from the cache
    assert record["generation"]["llm_cached"] is True and judge_calls == []


# --- Summary file, held-out guard, estimate -------------------------------------------------


def test_summary_csv_replaces_the_row_for_the_same_config_and_split(tmp_path):
    path = tmp_path / "summary.csv"
    update_summary_csv(path, {"config": "a", "split": "dev", "mrr": 0.5})
    update_summary_csv(path, {"config": "b", "split": "dev", "mrr": 0.6})
    update_summary_csv(path, {"config": "a", "split": "dev", "mrr": 0.7})
    update_summary_csv(path, {"config": "a", "split": "heldout", "mrr": 0.4})
    with path.open(encoding="utf-8") as f:
        rows = {(r["config"], r["split"]): r["mrr"] for r in csv.DictReader(f)}
    assert rows == {("b", "dev"): "0.6", ("a", "dev"): "0.7", ("a", "heldout"): "0.4"}


def test_heldout_runs_once_unless_forced_with_a_reason(tmp_path):
    log = tmp_path / "heldout_runs.jsonl"
    assert check_heldout_allowed(log, force=False, reason=None) == []
    log_heldout_run(log, ["hybrid_rerank"], forced=False, reason=None)
    with pytest.raises(HeldoutRefused, match="already run"):
        check_heldout_allowed(log, force=False, reason=None)
    with pytest.raises(HeldoutRefused, match="needs a --reason"):
        check_heldout_allowed(log, force=True, reason="  ")
    assert len(check_heldout_allowed(log, force=True, reason="judge prompt bug fixed")) == 1


def test_estimate_excludes_cached_answers():
    requests = [{"model": "m", "messages": [{"content": f"{n} " + "x" * 8000}]} for n in range(100)]
    cached = {id(r) for r in requests[:50]}
    models = ("claude-haiku-4-5-20251001", "claude-opus-5-5")
    all_new = estimate_generation_cost(requests, lambda r: False, *models)
    half_cached = estimate_generation_cost(requests, lambda r: id(r) in cached, *models)
    assert (all_new.answers_uncached, half_cached.answers_uncached) == (100, 50)
    assert half_cached.cost_low < all_new.cost_low < all_new.cost_high
    assert all_new.judge_calls == 200
