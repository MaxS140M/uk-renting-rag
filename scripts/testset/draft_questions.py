"""Ask the LLM to draft candidate questions for you to review.

Two modes:
  --test-set   Draft a whole test set (~130 questions) in two steps: questions written from
               a tenant persona that sees only the topic list, then evidence quotes and
               reference answers found in the full corpus by a separate call.
  --doc ID     Draft a few questions for one document (shows the model the document).

Drafts go to eval/testset/drafts.jsonl (never to eval/testset/questions.jsonl) with author
"llm_draft".
Review them with:  python scripts/testset/review_drafts.py

Usage:
    python scripts/testset/draft_questions.py --test-set --estimate-only
    python scripts/testset/draft_questions.py --test-set
    python scripts/testset/draft_questions.py --doc tenancy-deposit-protection --n 5
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import anthropic

from rag.config import (
    DRAFT_EFFORT,
    DRAFT_MODEL,
    LLM_CACHE_DIR,
    LLM_MODEL,
    PROJECT_ROOT,
    RAW_DIR,
)
from rag.evaluation.drafting import draft_questions
from rag.evaluation.schema import append_item, parse_items
from rag.evaluation.testset_generation import (
    DEFAULT_MIX,
    estimate_cost,
    generate_test_set,
    topic_list,
)
from rag.evaluation.utils import load_corpus
from rag.generate import MissingAPIKeyError, create_client
from rag.llm_cache import LLMCache

DRAFTS_FILE = PROJECT_ROOT / "eval" / "testset" / "drafts.jsonl"
OVERVIEW_FILE = PROJECT_ROOT / "eval" / "testset" / "corpus_overview.md"


def run_test_set(args, docs, existing) -> int:
    topics = topic_list(OVERVIEW_FILE.read_text(encoding="utf-8"))
    mix = dict(DEFAULT_MIX)
    estimate = estimate_cost(args.model, topics, docs, mix)
    print(
        f"Drafting {sum(mix.values())} questions ({', '.join(f'{t} {n}' for t, n in mix.items())})"
        f" with {args.model}, effort {args.effort}, in {estimate.calls} calls.\n"
        f"  Input ~{estimate.input_tokens:,} tokens; corpus ~{estimate.cache_write_tokens:,} "
        f"tokens written to the cache once, then read ~{estimate.cache_read_tokens:,} tokens "
        f"from cache; output (including thinking) up to ~{estimate.output_tokens:,} tokens.\n"
        f"  Estimated cost: ${estimate.low:.2f} to ${estimate.high:.2f}. Cached calls are free."
    )
    if args.estimate_only:
        return 0
    if not args.yes and input("Proceed? (y/n): ").strip().lower() != "y":
        print("Cancelled.")
        return 0

    result = generate_test_set(
        create_client(),
        LLMCache(LLM_CACHE_DIR),
        topics,
        docs,
        args.model,
        args.effort,
        mix,
        existing,
    )
    for draft in result.drafts:
        append_item(args.drafts, draft)

    fresh = [r for r in result.replies if not r.cached]
    print(f"\nSaved {len(result.drafts)} drafts to {args.drafts}. These are NOT test items.")
    print(
        "By type: "
        + ", ".join(f"{t} {n}" for t, n in Counter(d.question_type for d in result.drafts).items())
    )
    print(
        f"API calls: {len(fresh)} new, {len(result.replies) - len(fresh)} from cache; tokens: "
        f"input {sum(r.input_tokens for r in fresh):,}, cache write "
        f"{sum(r.cache_write_tokens for r in fresh):,}, cache read "
        f"{sum(r.cache_read_tokens for r in fresh):,}, "
        f"output {sum(r.output_tokens for r in fresh):,}"
    )
    for line in result.dropped_duplicates:
        print(f"Dropped duplicate: {line}")
    for line in result.rejected:
        print(f"Rejected: {line}")
    print(
        "Next: python scripts/testset/validate_questions.py "
        "--questions eval/testset/drafts.jsonl --drafts"
    )
    return 0


def run_single_doc(args, docs, existing) -> int:
    if args.doc not in docs:
        print(f"Unknown doc_id '{args.doc}'. See eval/testset/corpus_overview.md.", file=sys.stderr)
        return 1
    result = draft_questions(docs[args.doc], create_client(), args.model, args.n, existing)
    for draft in result.drafts:
        append_item(args.drafts, draft)
        flag = "  (quote needs fixing)" if "not found verbatim" in draft.notes else ""
        print(f"{draft.id} [{draft.question_type}] {draft.question}{flag}")
    for reason in result.rejected:
        print(f"Rejected {reason}")
    print(f"\nSaved {len(result.drafts)} draft(s) to {args.drafts}. These are NOT test items.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-set", action="store_true", help="draft a whole test set")
    mode.add_argument("--doc", help="doc_id of one document to draft questions for")
    parser.add_argument("--n", type=int, default=5, help="questions for --doc mode")
    parser.add_argument("--model", help=f"default: {DRAFT_MODEL} (--test-set), {LLM_MODEL} (--doc)")
    parser.add_argument("--effort", default=DRAFT_EFFORT, help="thinking effort for --test-set")
    parser.add_argument("--estimate-only", action="store_true", help="show the cost estimate only")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--corpus", type=Path, default=RAW_DIR)
    parser.add_argument("--drafts", type=Path, default=DRAFTS_FILE)
    args = parser.parse_args()
    args.model = args.model or (DRAFT_MODEL if args.test_set else LLM_MODEL)
    sys.stdout.reconfigure(encoding="utf-8")

    docs = load_corpus(args.corpus)
    existing, problems = parse_items(args.drafts)
    if problems:
        print(f"{args.drafts} has invalid lines; fix them first:\n" + "\n".join(problems))
        return 1
    try:
        return (
            run_test_set(args, docs, existing)
            if args.test_set
            else run_single_doc(args, docs, existing)
        )
    except (MissingAPIKeyError, anthropic.APIError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
