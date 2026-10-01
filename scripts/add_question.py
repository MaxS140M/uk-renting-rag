"""Interactively write one evaluation question and append it to eval/questions.jsonl.

You type the question, pick its type, search the corpus by keyword to find supporting
passages, select the exact quote, and write the reference answer. The item is validated
before it is saved, and gets the next free id automatically.

Usage:
    python scripts/add_question.py
    python scripts/add_question.py --from-draft draft-003   # review an LLM draft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from rag.authoring import (
    Aborted,
    ask,
    ask_multiline,
    build_passage_searcher,
    choose_type,
    collect_evidence,
    show_passages,
)
from rag.config import PROJECT_ROOT, RAW_DIR
from rag.eval_schema import (
    EvalItem,
    Evidence,
    append_item,
    item_to_json,
    load_items,
    next_id,
    parse_items,
)
from rag.eval_utils import load_corpus, question_similarity

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"
DRAFTS_FILE = PROJECT_ROOT / "eval" / "drafts.jsonl"
SIMILARITY_WARNING = 0.85  # same threshold the validator uses for near-duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--corpus", type=Path, default=RAW_DIR)
    parser.add_argument("--from-draft", metavar="DRAFT_ID", help="review a draft from drafts.jsonl")
    parser.add_argument("--drafts", type=Path, default=DRAFTS_FILE)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    items = load_items(args.questions)
    docs = load_corpus(args.corpus)
    searcher = build_passage_searcher(docs)

    draft = None
    if args.from_draft:
        drafts, problems = parse_items(args.drafts)
        draft = next((d for d in drafts if d.id == args.from_draft), None)
        if draft is None:
            print(f"Draft {args.from_draft} not found in {args.drafts}.", file=sys.stderr)
            return 1
        print(f"Reviewing {draft.id}. Press Enter to keep a value, or type a new one.")

    try:
        question = ask("\nQuestion", draft.question if draft else "")
        for existing in items:
            similarity = question_similarity(question, existing.question)
            if similarity >= SIMILARITY_WARNING:
                print(f"  Warning: {similarity:.0%} similar to {existing.id}: {existing.question}")

        question_type = choose_type(draft.question_type if draft else None)
        if question_type == "unanswerable":
            print("\nTop matching passages, to confirm none of them answers the question:")
            show_passages(searcher.retrieve(question, 3))
            evidence: list[Evidence] = []
        else:
            evidence = collect_evidence(question_type, searcher, docs, draft)

        default_answer = (
            draft.reference_answer
            if draft
            else ("The guidance does not cover this." if question_type == "unanswerable" else "")
        )
        reference_answer = ask_multiline("\nReference answer", default_answer)
        notes = ask("Notes (optional)", f"Reviewed from {draft.id}" if draft else "")

        author = "llm_draft_reviewed" if draft else "max"
        item = EvalItem(
            id=next_id(items, author),
            question=question,
            reference_answer=reference_answer,
            question_type=question_type,
            evidence=evidence,
            author=author,
            notes=notes,
        )
        print("\n" + item_to_json(item))
        if not ask("Save this item? (y/n)", "y").lower().startswith("y"):
            print("Not saved.")
            return 0
    except Aborted:
        print("\nInput ended; nothing saved.")
        return 1
    except ValidationError as err:
        print(f"\nThe item is not valid, so it was not saved:\n{err}", file=sys.stderr)
        return 1

    append_item(args.questions, item)
    print(f"Saved {item.id} to {args.questions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
