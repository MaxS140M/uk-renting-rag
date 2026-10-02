"""Evaluate retrieval and generation for the experiments in eval/configs.yaml.

Retrieval metrics (Recall@1, Recall@5, MRR, evidence recall) run for every config: no LLM,
no cost. Generation (answers judged for faithfulness and correctness, refusal accuracy,
latency) runs only for the configs named with --generate, after showing a cost estimate.
Every LLM call is cached on disk, so reruns are free and give identical results.

Runs use the DEV split. The held-out split is run once, at the very end, for the final
config only (--heldout); a second held-out run needs --force and a logged --reason.

Usage:
    python scripts/evaluation/run_eval.py                    # retrieval, all configs
    python scripts/evaluation/run_eval.py --generate auto    # + answers: baseline + top 2
    python scripts/evaluation/run_eval.py --generate dense_baseline,hybrid_rerank
    python scripts/evaluation/run_eval.py --heldout --configs hybrid_rerank --generate hybrid_rerank
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag.config import (
    JUDGE_EFFORT,
    JUDGE_MODEL,
    LLM_CACHE_DIR,
    PROJECT_ROOT,
    RAW_DIR,
)
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
    write_records,
)
from rag.evaluation.schema import load_items
from rag.evaluation.utils import EvidenceMapper, corpus_fingerprint, load_corpus
from rag.experiment import ensure_index, load_experiments, load_index_chunks
from rag.factory import build_retriever
from rag.generate import Generator, create_client
from rag.llm_cache import CachingClient, LLMCache, streaming_sender
from rag.prompts import PROMPT_VERSION
from rag.retrieval import RetrievalResult

EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
SEED = 0


def set_seeds(seed: int = SEED) -> None:
    """Fix every random number generator involved, for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def results_from_record(
    record: dict, chunks_by_id: dict[str, dict], k: int
) -> list[RetrievalResult]:
    """Rebuild the top-k RetrievalResults stored in a record, to pass to the generator."""
    out = []
    for rank, r in enumerate(record["retrieved"][:k], start=1):
        chunk = chunks_by_id[r["chunk_id"]]
        out.append(RetrievalResult.from_chunk(chunk, r["score"], rank, "eval"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", help="comma-separated experiment names (default: all)")
    parser.add_argument(
        "--generate",
        default="",
        help="experiments to run generation for: comma-separated names, or 'auto' for the "
        "baseline plus the top 2 by MRR",
    )
    parser.add_argument("--heldout", action="store_true", help="run on the held-out split")
    parser.add_argument("--force", action="store_true", help="allow a repeat held-out run")
    parser.add_argument("--reason", help="why a repeat held-out run is needed (logged)")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-effort", default=JUDGE_EFFORT)
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    parser.add_argument("--estimate-only", action="store_true", help="stop after the estimate")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--experiments", type=Path, default=EVAL_DIR / "configs.yaml")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    set_seeds()

    split = "heldout" if args.heldout else "dev"
    experiments = load_experiments(args.experiments)
    if args.configs:
        wanted = args.configs.split(",")
        unknown = set(wanted) - {e.name for e in experiments}
        if unknown:
            parser.error(f"unknown configs: {sorted(unknown)}")
        experiments = [e for e in experiments if e.name in wanted]

    heldout_log = RESULTS_DIR / "heldout_runs.jsonl"
    if args.heldout:
        if len(experiments) != 1:
            parser.error("--heldout runs exactly one final config: use --configs NAME")
        try:
            check_heldout_allowed(heldout_log, args.force, args.reason)
        except HeldoutRefused as err:
            print(f"Refused: {err}", file=sys.stderr)
            return 1

    items = [i for i in load_items(args.questions) if not i.is_example and i.split == split]
    docs = load_corpus(RAW_DIR)
    fingerprint = corpus_fingerprint(docs)
    print(f"{len(items)} {split} questions, corpus {fingerprint}, prompt {PROMPT_VERSION}")

    # --- Retrieval for every config ----------------------------------------------------------
    records: dict[str, list[dict]] = {}
    chunk_maps: dict[str, dict[str, dict]] = {}
    for experiment in experiments:
        ensure_index(experiment.config, list(docs.values()))
        chunks = load_index_chunks(experiment.config.index_dir)
        chunk_maps[experiment.name] = {c["chunk_id"]: c for c in chunks}
        mapper = EvidenceMapper(chunks)
        retriever = build_retriever(experiment.config)
        retriever.retrieve("warm-up question about deposits", 5)  # one-off loading costs
        records[experiment.name] = [evaluate_retrieval(i, retriever, mapper) for i in items]
        s = summarise(records[experiment.name])
        print(
            f"  {experiment.name:24} R@1 {s['recall_at_1']:.3f}  R@5 {s['recall_at_5']:.3f}  "
            f"MRR {s['mrr']:.3f}  multi-evidence R@5 {s['evidence_recall_at_5_multi'] or 0:.3f}  "
            f"retrieval p50 {s['retrieval_ms_p50']:.0f} ms"
        )

    # --- Choose configs for generation -------------------------------------------------------
    by_name = {e.name: e for e in experiments}
    if args.generate == "auto":
        ranked = sorted(experiments, key=lambda e: -summarise(records[e.name])["mrr"])
        chosen = ["dense_baseline"] if "dense_baseline" in by_name else []
        chosen += [e.name for e in ranked if e.name not in chosen][:2]
    else:
        chosen = [n for n in args.generate.split(",") if n]
        unknown = set(chosen) - set(by_name)
        if unknown:
            parser.error(f"--generate names not in the selected configs: {sorted(unknown)}")

    cache = LLMCache(LLM_CACHE_DIR)
    if chosen:
        generators = {n: Generator(by_name[n].config) for n in chosen}
        requests = [
            generators[n].build_request(
                i.question,
                results_from_record(r, chunk_maps[n], by_name[n].config.final_k),
            )
            for n in chosen
            for i, r in zip(items, records[n], strict=True)
        ]
        estimate = estimate_generation_cost(
            requests,
            lambda r: cache.get(r) is not None,
            by_name[chosen[0]].config.llm_model,
            args.judge_model,
        )
        print(
            f"\nGeneration for {', '.join(chosen)}: {len(requests)} answers "
            f"({estimate.answers_uncached} not cached) and up to {estimate.judge_calls} judge "
            f"calls with {args.judge_model}.\nEstimated cost: ${estimate.cost_low:.2f} to "
            f"${estimate.cost_high:.2f} (cached calls are free)."
        )
        if args.estimate_only:
            return 0
        if not args.yes and input("Proceed? (y/n): ").strip().lower() != "y":
            print("Generation skipped.")
            chosen = []

    if chosen:
        client = create_client()
        judge = Judge(streaming_sender(client), cache, args.judge_model, args.judge_effort)
        for name in chosen:
            caching = CachingClient(client, cache)
            generator = Generator(by_name[name].config, client=caching)
            k = by_name[name].config.final_k
            print(f"\nGenerating and judging: {name}")
            for n, (item, record) in enumerate(zip(items, records[name], strict=True), start=1):
                passages = results_from_record(record, chunk_maps[name], k)
                evaluate_generation(item, record, passages, generator, caching, judge)
                if n % 10 == 0 or n == len(items):
                    print(f"  {n}/{len(items)}")

    # --- Save --------------------------------------------------------------------------------
    run_at = datetime.now(UTC).isoformat(timespec="seconds")
    suffix = "_heldout" if args.heldout else ""
    for experiment in experiments:
        name = experiment.name
        write_records(RESULTS_DIR / f"{name}{suffix}.jsonl", records[name])
        row = {
            "config": name,
            "split": split,
            "run_at": run_at,
            "corpus": fingerprint,
            "prompt_version": PROMPT_VERSION,
            **summarise(records[name]),
        }
        update_summary_csv(RESULTS_DIR / "summary.csv", row)
        if name in chosen:
            s = row
            print(
                f"{name}: faithfulness {s['faithfulness']}, correctness {s['correctness']}, "
                f"refusal accuracy {s['refusal_accuracy']}, false refusals "
                f"{s['false_refusal_rate']}, total p50 {s['total_ms_p50']} ms"
            )
    if args.heldout:
        log_heldout_run(heldout_log, [e.name for e in experiments], args.force, args.reason)
    print(f"\nSaved per-question results to {RESULTS_DIR} and the summary to summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
