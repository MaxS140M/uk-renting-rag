"""Ask the RAG assistant a question from the command line.

Usage:
    python scripts/ask.py "How long does my landlord have to protect my deposit?"
    python scripts/ask.py "Can I be evicted without a reason?" --retrieval-only
    python scripts/ask.py "..." --mode hybrid --rerank
    python scripts/ask.py "..." --mode bm25 --k 8 --json
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

import anthropic

from rag.cli import add_retrieval_args, config_from_args
from rag.generate import MissingAPIKeyError
from rag.pipeline import RAGPipeline


def format_timings(timings: dict[str, float]) -> str:
    return ", ".join(f"{stage} {ms:.0f} ms" for stage, ms in timings.items())


def print_retrieval(pipeline: RAGPipeline, question: str) -> None:
    timings: dict[str, float] = {}
    for r in pipeline.retrieve(question, timings):
        stages = ", ".join(f"{stage} #{rank}" for stage, rank in r.ranks.items())
        print(f"\n#{r.rank}  {r.scored_by} score {r.score:.3f}  ({stages})")
        print(f"    {r.title}")
        if r.section:
            print(f"    Section: {r.section}")
        print(f"    {r.url}")
        print(f"    chunk_id: {r.chunk_id}")
        snippet = " ".join(r.text.split())[:300]
        print(textwrap.indent(textwrap.fill(snippet + " ...", width=96), "    "))
    print(f"\nTimings: {format_timings(timings)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="show retrieved chunks; no LLM call"
    )
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    add_retrieval_args(parser)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    config = config_from_args(args)
    pipeline = RAGPipeline(config)
    if not args.json:
        print(f"Retrieval: {config.label}, top {config.final_k} of {config.candidate_k} candidates")

    if args.retrieval_only:
        print_retrieval(pipeline, args.question)
        return 0

    try:
        result = pipeline.answer(args.question)
    except MissingAPIKeyError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except anthropic.APIError as err:
        print(f"Error from the Anthropic API: {err.__class__.__name__}: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(result.answer)
    print("\n" + "-" * 60)
    print("Retrieved passages (rank, score, cited?):")
    cited = {s.chunk_id for s in result.sources}
    for r in result.retrieved:
        mark = "cited" if r.chunk_id in cited else ""
        print(f"  [{r.rank}] {r.score:.3f}  {r.title} | {r.section[:50]}  {mark}")
    print(f"Timings: {format_timings(result.latency_ms)}")
    print(
        f"Model: {result.model} | prompt {result.prompt_version} | refused: {result.refused} | "
        f"tokens in/out: {result.usage.get('input_tokens')}/{result.usage.get('output_tokens')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
