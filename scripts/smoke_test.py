"""Run a fixed set of questions through the pipeline and save the answers as Markdown.

A quick manual check, not an evaluation: rerun it after each phase and compare the outputs.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --out eval/hybrid_smoke_test.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from rag.config import PROJECT_ROOT, RAGConfig
from rag.generate import MissingAPIKeyError
from rag.pipeline import AnswerResult, RAGPipeline
from rag.prompts import PROMPT_VERSION

# (question, what it tests). The last two are deliberately not answerable from the corpus.
QUESTIONS = [
    ("How long does my landlord have to protect my deposit?", "single fact: deposits"),
    ("Can my landlord evict me without giving a reason?", "Renters' Rights Act change"),
    (
        "How much notice does my landlord have to give before increasing my rent, "
        "and can I challenge it?",
        "two-part question: rent increases",
    ),
    ("Who is responsible for fixing a broken boiler in my rented flat?", "repairs"),
    (
        "Can a landlord refuse to rent to me because I get Universal Credit?",
        "discrimination (new rules)",
    ),
    ("What fees is a letting agent allowed to charge me?", "Tenant Fees Act"),
    (
        "Does a landlord need a licence to rent a house to five people who aren't related?",
        "HMO licensing",
    ),
    ("What can I do about damp and mould in my council flat?", "social housing"),
    ("What is the average rent for a one-bedroom flat in Manchester?", "unanswerable"),
    ("How do I apply for a mortgage to buy my first home?", "unanswerable (off-topic)"),
]


def format_result(n: int, purpose: str, result: AnswerResult) -> str:
    cited = {s.chunk_id for s in result.sources}
    lines = [
        f"## {n}. {result.question}",
        "",
        f"*Tests: {purpose}*",
        "",
        "**Answer:**",
        "",
        *[f"> {line}" if line else ">" for line in result.answer.splitlines()],
        "",
        "**Retrieved passages:**",
        "",
        "| Rank | Score | Document | Section | Cited |",
        "|---:|---:|---|---|:---:|",
    ]
    for r in result.retrieved:
        mark = "yes" if r.chunk_id in cited else ""
        section = r.section.replace("|", "/")[:60]
        lines.append(f"| {r.rank} | {r.score:.3f} | {r.title} | {section} | {mark} |")
    t = result.latency_ms
    lines += [
        "",
        f"Refused: **{result.refused}** · retrieval {t['retrieval']:.0f} ms · "
        f"generation {t['generation']:.0f} ms · total {t['total']:.0f} ms · "
        f"tokens in/out {result.usage.get('input_tokens')}/{result.usage.get('output_tokens')}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "eval" / "baseline_smoke_test.md"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    config = RAGConfig()
    pipeline = RAGPipeline(config)
    sections = []
    for n, (question, purpose) in enumerate(QUESTIONS, start=1):
        print(f"[{n}/{len(QUESTIONS)}] {question}")
        try:
            result = pipeline.answer(question)
        except (MissingAPIKeyError, anthropic.APIError) as err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        sections.append(format_result(n, purpose, result))

    header = [
        "# Baseline smoke test",
        "",
        f"Run {datetime.now(UTC):%Y-%m-%d %H:%M} UTC with `scripts/smoke_test.py`.",
        "",
        f"Configuration: retriever `{config.retriever}`, embedding model "
        f"`{config.embedding_model}`, top_k {config.top_k}, chunks up to "
        f"{config.chunk_max_tokens} tokens, LLM `{config.llm_model}` "
        f"(temperature {config.llm_temperature}), prompt `{PROMPT_VERSION}`.",
        "",
        "Timings are from one laptop run and include no warm-up, so treat them as rough.",
        "",
    ]
    args.out.write_text("\n".join(header + sections), encoding="utf-8")
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
