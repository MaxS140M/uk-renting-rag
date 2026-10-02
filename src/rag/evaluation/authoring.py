"""Interactive helpers for writing and reviewing evaluation items at the terminal.

Shared by scripts/testset/add_question.py and scripts/testset/review_drafts.py: prompts that can be
aborted cleanly, the question-type menu, keyword search over the documents' own
paragraphs, and selecting an exact quote from a passage.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping

from rag.chunking import split_sentences
from rag.evaluation.schema import MIN_QUOTE_CHARS, QUESTION_TYPES, EvalItem, Evidence
from rag.evaluation.utils import document_passages, normalise, quote_count
from rag.retrieval import BM25Retriever

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


def build_passage_searcher(docs: Mapping[str, Mapping]) -> BM25Retriever:
    """Keyword search over every paragraph of every document (independent of chunking)."""
    return BM25Retriever([p for doc in docs.values() for p in document_passages(doc)])


def quote_in_context(doc: Mapping, quote: str, context: int = 250) -> str:
    """The quote with some surrounding document text, the quote marked with >>> <<<."""
    text, target = normalise(doc["text"]), normalise(quote)
    start = text.find(target)
    if start == -1:
        return f"(quote not found in {doc['doc_id']})"
    before = text[max(0, start - context) : start]
    after = text[start + len(target) : start + len(target) + context]
    return f"...{before}>>> {target} <<<{after}..."
