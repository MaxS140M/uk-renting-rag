"""Validate eval/questions.jsonl against the schema and the frozen corpus.

Checks every item's format, that every evidence quote appears exactly in its document and
maps to a chunk, and that no two questions are duplicates. Exits with code 1 on errors, so
it can run in CI.

Usage:
    python scripts/validate_questions.py
    python scripts/validate_questions.py --show-unanswerable       # check unanswerable items
    python scripts/validate_questions.py --strict                  # warnings also fail
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import textwrap
from pathlib import Path

from rag.chunking import chunk_documents
from rag.cli import add_retrieval_args, config_from_args
from rag.config import PROJECT_ROOT, RAW_DIR, RAGConfig
from rag.evaluation.schema import parse_items
from rag.evaluation.utils import corpus_fingerprint, load_corpus, type_counts, validate_items

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"


def show_unanswerable(items, args) -> None:
    """Show the top passages for each unanswerable item, to confirm the corpus lacks the answer."""
    from rag.factory import build_retriever  # loads models, so only imported when needed

    base = dataclasses.replace(RAGConfig(), retrieval_mode="hybrid", use_reranker=True)
    config = config_from_args(args, base)
    retriever = build_retriever(config)
    print(f"\nUnanswerable items: top passages from {config.label} (check none answers it)")
    for item in (i for i in items if not i.answerable):
        print(f"\n{item.id}: {item.question}")
        for r in retriever.retrieve(item.question, 3):
            snippet = " ".join(r.text.split())[:220]
            print(f"  #{r.rank} ({r.score:.2f}) {r.title} > {r.section}")
            print(textwrap.indent(textwrap.fill(snippet + " ...", 90), "      "))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--corpus", type=Path, default=RAW_DIR)
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    parser.add_argument(
        "--drafts", action="store_true", help="the file holds unreviewed drafts (eval/drafts.jsonl)"
    )
    parser.add_argument(
        "--skip-chunk-check", action="store_true", help="skip mapping quotes to chunks (faster)"
    )
    parser.add_argument(
        "--show-unanswerable",
        action="store_true",
        help="retrieve passages for unanswerable items (needs the index; default hybrid+rerank)",
    )
    add_retrieval_args(parser)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    items, problems = parse_items(args.questions)
    docs = load_corpus(args.corpus)
    if not docs:
        print(f"No documents in {args.corpus}.", file=sys.stderr)
        return 1

    chunks = None
    if not args.skip_chunk_check:
        cited = {e.doc_id for i in items for e in i.evidence if e.doc_id in docs}
        config = RAGConfig()
        chunks = [
            c.to_dict()
            for c in chunk_documents(
                [docs[d] for d in sorted(cited)],
                max_tokens=config.chunk_max_tokens,
                overlap_tokens=config.chunk_overlap_tokens,
            )
        ]

    report = validate_items(items, docs, chunks, allow_drafts=args.drafts)
    errors = problems + report.errors

    counts = type_counts(items)
    print(
        f"{args.questions}: {len(items) + len(problems)} items, corpus {corpus_fingerprint(docs)}"
    )
    print("  " + ", ".join(f"{t} {n}" for t, n in sorted(counts.items())) if counts else "  empty")
    for warning in report.warnings:
        print(f"WARNING {warning}")
    for error in errors:
        print(f"ERROR   {error}")

    if args.show_unanswerable:
        show_unanswerable(items, args)

    failed = bool(errors) or (args.strict and bool(report.warnings))
    print(
        f"\n{'FAILED' if failed else 'OK'}: {len(errors)} error(s), "
        f"{len(report.warnings)} warning(s)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
