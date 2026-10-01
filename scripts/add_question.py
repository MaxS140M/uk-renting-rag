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
import textwrap
from pathlib import Path

from pydantic import ValidationError

from rag.chunking import split_sentences
from rag.config import PROJECT_ROOT, RAW_DIR
from rag.eval_schema import (
    MIN_QUOTE_CHARS,
    QUESTION_TYPES,
    EvalItem,
    Evidence,
    append_item,
    item_to_json,
    load_items,
    next_id,
    parse_items,
)
from rag.eval_utils import document_passages, load_corpus, question_similarity, quote_count
from rag.retrieval import BM25Retriever

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"
DRAFTS_FILE = PROJECT_ROOT / "eval" / "drafts.jsonl"
SIMILARITY_WARNING = 0.85  # same threshold the validator uses for near-duplicates
TYPE_HELP = {
    "factual": "one fact, answered by one passage",
    "multi_passage": "needs two or more passages (two or more quotes)",
    "informal": "casual or messy wording, as a real tenant might ask",
    "unanswerable": "the guidance does not answer it (no evidence)",
}


class Aborted(Exception):
    """The user ended input early; nothing is saved."""


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError as err:
        raise Aborted from err
    return answer or default


def ask_multiline(prompt: str, default: str = "") -> str:
    print(
        f"{prompt} (finish with an empty line"
        + (", or just Enter to keep it" if default else "")
        + "):"
    )
    if default:
        print(textwrap.indent(default, "  | "))
    lines = []
    while True:
        try:
            line = input("  > ")
        except EOFError as err:
            raise Aborted from err
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip() or default


def sentence_units(passage_text: str) -> list[str]:
    """Split a passage into numbered units: sentences, with each list item kept as one."""
    return [s for line in passage_text.splitlines() if line.strip() for s in split_sentences(line)]


def parse_selection(selection: str, n_units: int) -> list[int] | None:
    """Turn '3', '2-4' or 'a' into 0-based unit indices; None if it is not a selection."""
    selection = selection.strip().lower()
    if selection == "a":
        return list(range(n_units))
    parts = selection.split("-")
    if not all(p.strip().isdigit() for p in parts) or len(parts) > 2:
        return None
    start, end = int(parts[0]), int(parts[-1])
    if not 1 <= start <= end <= n_units:
        return None
    return list(range(start - 1, end))


def choose_type(default: str | None) -> str:
    print("\nQuestion type:")
    for n, name in enumerate(QUESTION_TYPES, start=1):
        print(f"  {n}. {name:14} {TYPE_HELP[name]}")
    default_no = str(QUESTION_TYPES.index(default) + 1) if default else ""
    while True:
        choice = ask("Type number", default_no)
        if choice.isdigit() and 1 <= int(choice) <= len(QUESTION_TYPES):
            return QUESTION_TYPES[int(choice) - 1]
        print("  Please enter a number from the list.")


def show_passages(results) -> None:
    for n, r in enumerate(results, start=1):
        where = f"{r.title} > {r.section}" if r.section else r.title
        print(f"\n  [{n}] {where}")
        snippet = " ".join(r.text.split())
        print(
            textwrap.indent(
                textwrap.fill(snippet[:280] + ("…" if len(snippet) > 280 else ""), 92), "      "
            )
        )


def pick_quote(passage: dict, docs: dict) -> Evidence | None:
    units = sentence_units(passage["text"])
    print(f"\n  {passage['title']} > {passage['section']}")
    for n, unit in enumerate(units, start=1):
        print(textwrap.indent(textwrap.fill(f"{n:>2}. {unit}", 92, subsequent_indent="    "), "  "))
    while True:
        choice = ask(
            "Quote: sentence number(s) e.g. 2 or 2-4, 'a' for all, paste exact text, "
            "or Enter to go back"
        )
        if not choice:
            return None
        indices = parse_selection(choice, len(units))
        quote = " ".join(units[i] for i in indices) if indices is not None else choice
        if len(quote) < MIN_QUOTE_CHARS:
            print(f"  Quote must be at least {MIN_QUOTE_CHARS} characters.")
            continue
        if quote_count(quote, docs[passage["doc_id"]]["text"]) == 0:
            print("  That text does not appear exactly in the document. Try again.")
            continue
        print(f'  Selected: "{quote}"')
        return Evidence(doc_id=passage["doc_id"], quote=quote)


def collect_evidence(
    question_type: str, searcher: BM25Retriever, docs: dict, draft: EvalItem | None
) -> list[Evidence]:
    evidence: list[Evidence] = []
    if draft:
        for e in draft.evidence:
            valid = e.doc_id in docs and quote_count(e.quote, docs[e.doc_id]["text"]) > 0
            print(f'\nDraft evidence from {e.doc_id}:\n  "{e.quote}"')
            if not valid:
                print("  This quote does not appear exactly in the document; find it again below.")
            elif ask("Keep this quote? (y/n)", "y").lower().startswith("y"):
                evidence.append(e)

    minimum = 2 if question_type == "multi_passage" else 1
    while True:
        done = len({e.quote for e in evidence}) >= minimum
        prompt = "\nSearch keywords" + (" (Enter to finish)" if done else "")
        query = ask(prompt)
        if not query:
            if done:
                return evidence
            print(f"  This question type needs at least {minimum} different quote(s).")
            continue
        results = searcher.retrieve(query, 8)
        if not results:
            print("  No passages contain those words. Try other keywords.")
            continue
        show_passages(results)
        choice = ask("\nPassage number (Enter to search again)")
        if choice.isdigit() and 1 <= int(choice) <= len(results):
            picked = searcher_passage(searcher, results[int(choice) - 1].chunk_id)
            quote = pick_quote(picked, docs)
            if quote:
                evidence.append(quote)
                print(f"  Evidence so far: {len(evidence)} quote(s).")


def searcher_passage(searcher: BM25Retriever, passage_id: str) -> dict:
    return next(p for p in searcher.chunks if p["chunk_id"] == passage_id)


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
    searcher = BM25Retriever([p for doc in docs.values() for p in document_passages(doc)])

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
