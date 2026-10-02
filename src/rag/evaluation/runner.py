"""Run evaluations: retrieval metrics for every config, generation + judging for a few.

Retrieval is evaluated without the LLM (cheap, run for every config). Generation and the
LLM judges cost money, so they run only for selected configs, with every call cached on
disk so reruns are free and reproducible.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag.evaluation.judges import Judge, verdict_dict
from rag.evaluation.metrics import (
    evidence_recall_at_k,
    first_relevant_rank,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from rag.evaluation.schema import EvalItem
from rag.evaluation.utils import EvidenceMapper
from rag.pipeline import cited_sources, is_refusal
from rag.retrieval import RetrievalResult, Retriever

RETRIEVE_K = 10  # depth retrieved for metrics; the first final_k results go to the LLM
RETRIEVAL_STAGES = ("dense", "bm25", "fusion", "rerank")


# --- Retrieval --------------------------------------------------------------------------------


def evaluate_retrieval(
    item: EvalItem, retriever: Retriever, mapper: EvidenceMapper, k: int = RETRIEVE_K
) -> dict:
    """Retrieve for one item and compute its retrieval metrics (answerable items only)."""
    timings: dict[str, float] = {}
    results = retriever.retrieve(item.question, k, timings)
    record = {
        "id": item.id,
        "question": item.question,
        "question_type": item.question_type,
        "split": item.split,
        "retrieved": [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "section": r.section,
                "score": round(r.score, 4),
                "ranks": r.ranks,
            }
            for r in results
        ],
        "retrieval_ms": timings,
    }
    if not item.answerable:
        return record

    gold_per_evidence = [set(ids) for ids in mapper.map_each(item)]
    mapped = [g for g in gold_per_evidence if g]
    record["gold"] = [sorted(g) for g in gold_per_evidence]
    record["unmapped_evidence"] = len(gold_per_evidence) - len(mapped)
    if not mapped:  # no quote could be located in this chunking; excluded from metrics
        return record
    gold = set().union(*mapped)
    ids = [r.chunk_id for r in results]
    record.update(
        first_gold_rank=first_relevant_rank(ids, gold),
        recall_at_1=recall_at_k(ids, gold, 1),
        recall_at_5=recall_at_k(ids, gold, 5),
        reciprocal_rank=reciprocal_rank(ids, gold),
        evidence_recall_at_5=evidence_recall_at_k(ids, mapped, 5),
    )
    return record


# --- Generation and judging -------------------------------------------------------------------


def evaluate_generation(
    item: EvalItem,
    record: dict,
    results: Sequence[RetrievalResult],
    generator,
    caching_client,
    judge: Judge,
) -> dict:
    """Answer one item from its retrieved passages, then judge the answer."""
    generation = generator.generate(item.question, results)
    reply = caching_client.last_reply
    refused = is_refusal(generation.text)
    out = {
        "answer": generation.text,
        "refused": refused,
        "cited_chunk_ids": [s.chunk_id for s in cited_sources(generation.text, results)]
        if not refused
        else [],
        "generation_ms": reply.latency_ms,
        "llm_cached": reply.cached,
        "llm_tokens": {"input": reply.input_tokens, "output": reply.output_tokens},
        "faithfulness": None,
        "correctness": None,
    }
    if item.answerable:
        if refused:  # a false refusal: the guidance answers this question
            out["correctness"] = {
                "verdict": "incorrect",
                "score": 0.0,
                "reasoning": "Refused an answerable question (false refusal).",
            }
        else:
            out["faithfulness"] = verdict_dict(
                judge.faithfulness(item.question, results, generation.text)
            )
            out["correctness"] = verdict_dict(
                judge.correctness(item.question, item.reference_answer, generation.text)
            )
    else:
        out["refusal_correct"] = refused
        if not refused:  # answered a question the guidance does not cover: check the claims
            out["faithfulness"] = verdict_dict(
                judge.faithfulness(item.question, results, generation.text)
            )
    record["generation"] = out
    record["latency_ms"] = {
        **record["retrieval_ms"],
        "retrieval": round(sum(record["retrieval_ms"].values()), 1),
        "generation": reply.latency_ms,
    }
    record["latency_ms"]["total"] = round(
        record["latency_ms"]["retrieval"] + record["latency_ms"]["generation"], 1
    )
    return record


# --- Summaries --------------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "config",
    "split",
    "run_at",
    "corpus",
    "prompt_version",
    "n_questions",
    "n_answerable",
    "n_unanswerable",
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "evidence_recall_at_5_multi",
    "unmapped_evidence",
    "generated",
    "faithfulness",
    "faithful_rate",
    "correctness",
    "correct_rate",
    "refusal_accuracy",
    "false_refusal_rate",
    "retrieval_ms_p50",
    "retrieval_ms_p95",
    "generation_ms_p50",
    "generation_ms_p95",
    "total_ms_p50",
    "total_ms_p95",
] + [f"{stage}_ms_p50" for stage in RETRIEVAL_STAGES]


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def summarise(records: Sequence[dict]) -> dict:
    """Aggregate per-question records into the metrics reported in summary.csv."""
    answerable = [r for r in records if r["question_type"] != "unanswerable"]
    scored = [r for r in answerable if "recall_at_5" in r]
    multi = [r for r in scored if r["question_type"] == "multi_passage"]
    generated = [r for r in records if "generation" in r]
    gen_answerable = [r for r in generated if r["question_type"] != "unanswerable"]
    gen_unanswerable = [r for r in generated if r["question_type"] == "unanswerable"]
    faith = [
        r["generation"]["faithfulness"]
        for r in generated
        if r["generation"]["faithfulness"] and r["generation"]["faithfulness"]["score"] is not None
    ]
    correctness = [r["generation"]["correctness"] for r in gen_answerable]

    retrieval_ms = [sum(r["retrieval_ms"].values()) for r in records]
    summary = {
        "n_questions": len(records),
        "n_answerable": len(answerable),
        "n_unanswerable": len(records) - len(answerable),
        "recall_at_1": _round(mean([r["recall_at_1"] for r in scored])),
        "recall_at_5": _round(mean([r["recall_at_5"] for r in scored])),
        "mrr": _round(mean([r["reciprocal_rank"] for r in scored])),
        "evidence_recall_at_5_multi": _round(mean([r["evidence_recall_at_5"] for r in multi])),
        "unmapped_evidence": sum(r.get("unmapped_evidence", 0) for r in answerable),
        "generated": len(generated),
        "faithfulness": _round(mean([f["score"] for f in faith])),
        "faithful_rate": _round(mean([float(f["faithful"]) for f in faith])),
        "correctness": _round(mean([c["score"] for c in correctness])),
        "correct_rate": _round(mean([float(c["verdict"] == "correct") for c in correctness])),
        "refusal_accuracy": _round(
            mean([float(r["generation"]["refusal_correct"]) for r in gen_unanswerable])
        ),
        "false_refusal_rate": _round(
            mean([float(r["generation"]["refused"]) for r in gen_answerable])
        ),
        "retrieval_ms_p50": _round(percentile(retrieval_ms, 50), 1),
        "retrieval_ms_p95": _round(percentile(retrieval_ms, 95), 1),
    }
    for name in ("generation", "total"):
        values = [r["latency_ms"][name] for r in generated]
        summary[f"{name}_ms_p50"] = _round(percentile(values, 50), 1)
        summary[f"{name}_ms_p95"] = _round(percentile(values, 95), 1)
    for stage in RETRIEVAL_STAGES:
        values = [r["retrieval_ms"][stage] for r in records if stage in r["retrieval_ms"]]
        summary[f"{stage}_ms_p50"] = _round(percentile(values, 50), 1)
    return summary


def write_records(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )


def update_summary_csv(path: Path, row: dict) -> None:
    """Insert or replace the row for (config, split), keeping other rows."""
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            rows = [
                r
                for r in csv.DictReader(f)
                if (r["config"], r["split"]) != (row["config"], row["split"])
            ]
    rows.append({k: ("" if row.get(k) is None else row[k]) for k in SUMMARY_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# --- Held-out guard ---------------------------------------------------------------------------


class HeldoutRefused(RuntimeError):
    pass


def check_heldout_allowed(log_path: Path, force: bool, reason: str | None) -> list[dict]:
    """Refuse a second held-out run unless forced with a reason; returns earlier runs."""
    previous = []
    if log_path.exists():
        previous = [json.loads(line) for line in log_path.read_text("utf-8").splitlines() if line]
    if previous and not force:
        raise HeldoutRefused(
            f"The held-out split was already run on {previous[-1]['run_at']} "
            f"({', '.join(previous[-1]['configs'])}). Running it again means it is no longer "
            "unseen. If you must, pass --force with --reason explaining why."
        )
    if previous and not (reason and reason.strip()):
        raise HeldoutRefused("--force needs a --reason, which is logged.")
    return previous


def log_heldout_run(
    log_path: Path, configs: Sequence[str], forced: bool, reason: str | None
) -> None:
    entry = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "configs": list(configs),
        "forced": forced,
        "reason": reason or "",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# --- Cost estimate ----------------------------------------------------------------------------

PRICES = {  # US$ per million tokens (input, output)
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
}


@dataclass
class GenerationEstimate:
    answers_uncached: int
    judge_calls: int
    cost_low: float
    cost_high: float


def estimate_generation_cost(
    answer_requests: Sequence[dict],
    is_cached: Callable[[dict], bool],
    answer_model: str,
    judge_model: str,
    judge_calls_per_answer: float = 2.0,
) -> GenerationEstimate:
    """Rough cost range for answering and judging, at about 4 characters per token.

    Cached answers are free. Judge calls are all counted as uncached, because they depend on
    answers that may not exist yet. Assumptions: an answer is about 300 output tokens; the
    faithfulness judge reads about as much as the answering prompt (it sees the same
    passages), the correctness judge about 400 tokens; judge output (with thinking) is 500
    to 2,000 tokens; the high end allows a tokenizer counting 35% more tokens.
    """
    a_in, a_out = PRICES[answer_model]
    j_in, j_out = PRICES[judge_model]
    sizes = {id(r): len(json.dumps(r, ensure_ascii=False)) / 4 for r in answer_requests}
    uncached = [r for r in answer_requests if not is_cached(r)]
    answer_cost = sum(sizes[id(r)] * a_in + 300 * a_out for r in uncached) / 1e6

    avg_prompt = sum(sizes.values()) / len(sizes) if sizes else 0.0
    judge_input = (avg_prompt + 400) / 2  # average of the faithfulness and correctness calls
    n_judge = round(len(answer_requests) * judge_calls_per_answer)
    low = answer_cost + n_judge * (judge_input * j_in + 500 * j_out) / 1e6
    high = answer_cost + n_judge * (judge_input * 1.35 * j_in + 2000 * j_out) / 1e6
    return GenerationEstimate(len(uncached), n_judge, round(low, 2), round(high, 2))
