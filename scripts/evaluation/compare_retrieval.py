"""Compare retrieval setups side by side for one or more questions. No LLM calls.

Shows the top-k chunks from dense, BM25, hybrid and hybrid + reranker, with scores and
timings. A qualitative check, not an evaluation: formal metrics come from the eval script.

Usage:
    python scripts/evaluation/compare_retrieval.py "How long to protect my deposit?"
    python scripts/evaluation/compare_retrieval.py --smoke \
        --out eval/early_checks/phase3_comparison.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag.config import PROJECT_ROOT, RAGConfig
from rag.factory import build_retriever
from rag.retrieval import RetrievalResult, Retriever

SMOKE_QUESTIONS = PROJECT_ROOT / "eval" / "early_checks" / "smoke_questions.json"
SETUPS = [("dense", False), ("bm25", False), ("hybrid", False), ("hybrid", True)]


def cell(result: RetrievalResult, baseline_ids: set[str]) -> str:
    title = result.title if len(result.title) <= 45 else result.title[:44] + "…"
    section = result.section.split(" > ")[-1]
    section = section if len(section) <= 35 else section[:34] + "…"
    new = " **(new)**" if result.chunk_id not in baseline_ids else ""
    text = f"{title} / {section} ({result.score:.2f}){new}" if section else f"{title}{new}"
    return text.replace("|", "/")


def run(retriever: Retriever, question: str, k: int, repeats: int):
    """Return the results and the median per-stage timings over several runs."""
    runs = []
    for _ in range(repeats):
        timings: dict[str, float] = {}
        results = retriever.retrieve(question, k, timings)
        runs.append(timings)
    median = {stage: statistics.median(t[stage] for t in runs) for stage in runs[0]}
    return results, median


def compare(question: str, retrievers: dict[str, Retriever], k: int, repeats: int) -> str:
    outputs = {name: run(r, question, k, repeats) for name, r in retrievers.items()}
    baseline_ids = {r.chunk_id for r in outputs["dense"][0]}
    names = list(outputs)
    lines = [
        f"### {question}",
        "",
        "| Rank | " + " | ".join(names) + " |",
        "|---:|" + "---|" * len(names),
    ]
    for i in range(k):
        cells = []
        for name in names:
            results = outputs[name][0]
            cells.append(cell(results[i], baseline_ids) if i < len(results) else "")
        lines.append(f"| {i + 1} | " + " | ".join(cells) + " |")

    def timing(t: dict[str, float]) -> str:
        total = sum(t.values())
        parts = " + ".join(f"{stage} {ms:.0f}" for stage, ms in t.items())
        return f"**{total:.0f} ms**" + (f" ({parts})" if len(t) > 1 else "")

    lines.append("| ms | " + " | ".join(timing(outputs[n][1]) for n in names) + " |")
    lines.append("")
    lines.append(
        "Chunk IDs: "
        + "; ".join(f"{n}: " + ", ".join(r.chunk_id for r in outputs[n][0]) for n in names)
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="*", help="questions to compare")
    parser.add_argument(
        "--smoke", action="store_true", help="use eval/early_checks/smoke_questions.json"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3, help="runs per setup for timing")
    parser.add_argument("--out", type=Path, help="also save the output as Markdown")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    questions = list(args.questions)
    if args.smoke:
        questions += [q["question"] for q in json.loads(SMOKE_QUESTIONS.read_text("utf-8"))]
    if not questions:
        parser.error("give at least one question, or --smoke")

    base = RAGConfig(final_k=args.k)
    configs = [dataclasses.replace(base, retrieval_mode=m, use_reranker=r) for m, r in SETUPS]
    retrievers = {c.label: build_retriever(c) for c in configs}
    for retriever in retrievers.values():  # warm-up: first calls include one-off setup costs
        retriever.retrieve("warm-up query about deposits", args.k)

    reranker = base.reranker_model
    header = [
        "# Retrieval comparison",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by "
        "`scripts/evaluation/compare_retrieval.py` "
        f"(no LLM calls). Top {args.k} chunks per setup; candidate pool {base.candidate_k}, "
        f"RRF k = {base.rrf_k}, reranker `{reranker}` on {base.reranker_device}. "
        f"**(new)** marks a chunk that is not in the dense top {args.k}. Scores are on "
        "different scales per setup (cosine, BM25, RRF, cross-encoder logit) and are only "
        f"comparable within a column. Timings are the median of {args.repeats} warm runs on a "
        "laptop CPU, in milliseconds.",
        "",
    ]
    sections = [compare(q, retrievers, args.k, args.repeats) for q in questions]
    output = "\n".join(header + sections)
    print(output)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
