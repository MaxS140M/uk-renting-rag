"""Ask the LLM to draft candidate questions for one document, for you to review.

Drafts go to eval/drafts.jsonl (never to eval/questions.jsonl) with author "llm_draft".
Review each one with:  python scripts/add_question.py --from-draft draft-001

Usage:
    python scripts/draft_questions.py --doc tenancy-deposit-protection
    python scripts/draft_questions.py --doc private-renting --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic

from rag.config import LLM_MODEL, PROJECT_ROOT, RAW_DIR
from rag.drafting import draft_questions
from rag.eval_schema import append_item, parse_items
from rag.eval_utils import load_corpus
from rag.generate import MissingAPIKeyError, create_client

DRAFTS_FILE = PROJECT_ROOT / "eval" / "drafts.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--doc", required=True, help="doc_id of the document to draft for")
    parser.add_argument("--n", type=int, default=5, help="number of questions to propose")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--corpus", type=Path, default=RAW_DIR)
    parser.add_argument("--drafts", type=Path, default=DRAFTS_FILE)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    docs = load_corpus(args.corpus)
    if args.doc not in docs:
        print(f"Unknown doc_id '{args.doc}'. See eval/corpus_overview.md.", file=sys.stderr)
        return 1
    existing, problems = parse_items(args.drafts)
    if problems:
        print(f"{args.drafts} has invalid lines; fix them first:\n" + "\n".join(problems))
        return 1

    try:
        result = draft_questions(docs[args.doc], create_client(), args.model, args.n, existing)
    except (MissingAPIKeyError, anthropic.APIError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    for draft in result.drafts:
        append_item(args.drafts, draft)
        flag = "  (quote needs fixing)" if "not found verbatim" in draft.notes else ""
        print(f"{draft.id} [{draft.question_type}] {draft.question}{flag}")
    for reason in result.rejected:
        print(f"Rejected {reason}")
    print(f"\nSaved {len(result.drafts)} draft(s) to {args.drafts}. These are NOT test items.")
    if result.drafts:
        print(
            f"Review each with: python scripts/add_question.py --from-draft {result.drafts[0].id}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
