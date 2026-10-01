"""Run a fixed set of questions through the pipeline and save the answers as Markdown.

A quick manual check, not an evaluation: rerun it after each phase and compare the outputs.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --mode hybrid --rerank   # -> eval/smoke_test_hybrid+rerank.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from rag.cli import add_retrieval_args, config_from_args
from rag.config import PROJECT_ROOT, RAGConfig
from rag.generate import MissingAPIKeyError
from rag.pipeline import AnswerResult, RAGPipeline
from rag.prompts import PROMPT_VERSION

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "smoke_questions.json"


def load_questions() -> list[tuple[str, str]]:
    """The fixed smoke-test questions: (question, what it tests). The last two are
    deliberately not answerable from the corpus."""
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    return [(q["question"], q["tests"]) for q in data]


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
    parser.add_argument("--out", type=Path, help="output file (default depends on the setup)")
    add_retrieval_args(parser)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    config = config_from_args(args)
    if args.out is None:
        name = "baseline_smoke_test" if config == RAGConfig() else f"smoke_test_{config.label}"
        args.out = PROJECT_ROOT / "eval" / f"{name}.md"
    pipeline = RAGPipeline(config)
    questions = load_questions()
    sections = []
    for n, (question, purpose) in enumerate(questions, start=1):
        print(f"[{n}/{len(questions)}] {question}")
        try:
            result = pipeline.answer(question)
        except (MissingAPIKeyError, anthropic.APIError) as err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        sections.append(format_result(n, purpose, result))

    header = [
        f"# Smoke test: {config.label}",
        "",
        f"Run {datetime.now(UTC):%Y-%m-%d %H:%M} UTC with `scripts/smoke_test.py`.",
        "",
        f"Configuration: retrieval `{config.label}`, embedding model "
        f"`{config.embedding_model}`, final_k {config.final_k}, chunks up to "
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
