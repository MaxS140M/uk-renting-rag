"""Review LLM-drafted questions one by one: accept, edit or reject each.

Accepted items are added to eval/testset/questions.jsonl with author "llm_draft_reviewed". Every
decision is logged to eval/testset/review_log.jsonl, so you can stop at any time and resume later,
and the log provides the accepted / edited / rejected counts reported in the README.

Usage:
    python scripts/testset/review_drafts.py              # review (or resume reviewing)
    python scripts/testset/review_drafts.py --summary    # just print the counts so far
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from rag.config import PROJECT_ROOT, RAW_DIR
from rag.evaluation.authoring import (
    Aborted,
    ask,
    ask_multiline,
    build_passage_searcher,
    choose_type,
    collect_evidence,
    quote_in_context,
)
from rag.evaluation.schema import EvalItem, append_item, load_items, next_id, parse_items
from rag.evaluation.utils import load_corpus, question_similarity, quote_count

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "testset" / "questions.jsonl"
DRAFTS_FILE = PROJECT_ROOT / "eval" / "testset" / "drafts.jsonl"
LOG_FILE = PROJECT_ROOT / "eval" / "testset" / "review_log.jsonl"
SUMMARY_FILE = PROJECT_ROOT / "eval" / "testset" / "review_summary.md"
EDITABLE = ("question", "question_type", "reference_answer", "evidence")


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def log_decision(path: Path, entry: dict) -> None:
    entry = {**entry, "timestamp": datetime.now(UTC).isoformat(timespec="seconds")}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def broken_quotes(item: EvalItem, docs: dict) -> list[int]:
    return [
        n
        for n, e in enumerate(item.evidence, start=1)
        if e.doc_id not in docs or quote_count(e.quote, docs[e.doc_id]["text"]) == 0
    ]


def show_draft(draft: EvalItem, docs: dict, position: str, accepted: list[EvalItem]) -> None:
    print("\n" + "=" * 96)
    print(f"{position}  {draft.id}  [{draft.question_type}]")
    print(f"\nQUESTION:  {draft.question}")
    for existing in accepted:
        similarity = question_similarity(draft.question, existing.question)
        if similarity >= 0.85:
            print(f"  ! {similarity:.0%} similar to {existing.id}: {existing.question}")
    if draft.notes:
        print(textwrap.indent(textwrap.fill(f"Notes: {draft.notes}", 92), "  "))
    if not draft.evidence:
        print("\nEVIDENCE:  none (unanswerable)")
    for n, e in enumerate(draft.evidence, start=1):
        doc = docs.get(e.doc_id)
        title = doc["title"] if doc else "UNKNOWN DOCUMENT"
        status = "" if n not in broken_quotes(draft, docs) else "  ** NOT FOUND VERBATIM **"
        print(f"\nEVIDENCE {n}: {title} ({e.doc_id}){status}")
        context = quote_in_context(doc, e.quote) if doc else e.quote
        print(textwrap.indent(textwrap.fill(context, 92), "    "))
    print("\nREFERENCE ANSWER:")
    print(textwrap.indent(textwrap.fill(draft.reference_answer, 92), "    "))


def edit_draft(draft: EvalItem, docs: dict, searcher) -> EvalItem | None:
    """Edit fields until the reviewer is done; returns the edited copy or None if cancelled."""
    current = draft.model_copy()
    while True:
        choice = ask(
            "\nEdit: [q]uestion, [t]ype, [a]nswer, [v]evidence, [d]one, [c]ancel", "d"
        ).lower()[:1]
        if choice == "q":
            current = current.model_copy(update={"question": ask("Question", current.question)})
        elif choice == "t":
            new_type = choose_type(current.question_type)
            evidence = [] if new_type == "unanswerable" else current.evidence
            current = current.model_copy(update={"question_type": new_type, "evidence": evidence})
            if new_type != "unanswerable" and not evidence:
                print("  This type needs evidence: choose [v] to add it.")
        elif choice == "a":
            answer = ask_multiline("Reference answer", current.reference_answer)
            current = current.model_copy(update={"reference_answer": answer})
        elif choice == "v":
            if current.question_type == "unanswerable":
                print("  Unanswerable items have no evidence; change the type first.")
                continue
            evidence = collect_evidence(current.question_type, searcher, docs, current)
            current = current.model_copy(update={"evidence": evidence})
        elif choice == "c":
            return None
        elif choice == "d":
            return current


def make_item(base: EvalItem, items: list[EvalItem], draft_id: str, note: str) -> EvalItem:
    notes = f"Reviewed from {draft_id}." + (f" {note}" if note else "")
    return EvalItem(
        id=next_id(items, "llm_draft_reviewed"),
        question=base.question,
        reference_answer=base.reference_answer,
        question_type=base.question_type,
        evidence=base.evidence,
        author="llm_draft_reviewed",
        notes=notes,
    )


def summarise(log: list[dict], n_drafts: int) -> str:
    accepted = [e for e in log if e["decision"] in ("accepted", "edited")]
    edited = [e for e in log if e["decision"] == "edited"]
    rejected = [e for e in log if e["decision"] == "rejected"]
    by_type = Counter(e["question_type"] for e in accepted)
    fields = Counter(f for e in edited for f in e["changed"])
    lines = [
        "# Draft review summary",
        "",
        "Generated by `scripts/testset/review_drafts.py` from `eval/testset/review_log.jsonl`.",
        "",
        f"- Drafts: {n_drafts}; reviewed: {len(log)}; still to review: {n_drafts - len(log)}",
        f"- Accepted unchanged: {len(accepted) - len(edited)}",
        f"- Accepted after editing: {len(edited)}"
        + (
            f" (fields changed: {', '.join(f'{k} {v}' for k, v in fields.most_common())})"
            if fields
            else ""
        ),
        f"- Rejected: {len(rejected)}",
        "- Accepted by type: "
        + (", ".join(f"{t} {n}" for t, n in sorted(by_type.items())) or "none"),
    ]
    reasons = [e["note"] for e in rejected if e.get("note")]
    if reasons:
        lines += ["", "Rejection reasons:", ""] + [f"- {r}" for r in reasons]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drafts", type=Path, default=DRAFTS_FILE)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--log", type=Path, default=LOG_FILE)
    parser.add_argument("--summary-file", type=Path, default=SUMMARY_FILE)
    parser.add_argument("--corpus", type=Path, default=RAW_DIR)
    parser.add_argument("--summary", action="store_true", help="print the counts and exit")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    drafts, problems = parse_items(args.drafts)
    if problems:
        print(f"{args.drafts} has invalid lines:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    log = read_log(args.log)

    if not args.summary:
        docs = load_corpus(args.corpus)
        searcher = build_passage_searcher(docs)
        items = load_items(args.questions)
        done = {e["draft_id"] for e in log}
        todo = [d for d in drafts if d.id not in done]
        if not todo:
            print("Nothing left to review.")
        print("Keys: [a]ccept  [e]dit  [r]eject  [s]kip for now  [q]uit (progress is saved)")
        try:
            for k, draft in enumerate(todo, start=1):
                reviewed = [i for i in items if not i.is_example]
                show_draft(draft, docs, f"[{len(done) + k}/{len(drafts)}]", reviewed)
                while True:
                    action = ask("\nAction (a/e/r/s/q)").lower()[:1]
                    if action == "q":
                        raise Aborted
                    if action == "s":
                        break
                    if action == "r":
                        note = ask("Why reject? (optional)")
                        entry = {
                            "draft_id": draft.id,
                            "decision": "rejected",
                            "item_id": None,
                            "question_type": draft.question_type,
                            "changed": [],
                            "note": note,
                        }
                        log_decision(args.log, entry)
                        log.append(entry)
                        break
                    if action not in ("a", "e"):
                        continue
                    final = draft if action == "a" else edit_draft(draft, docs, searcher)
                    if final is None:
                        continue
                    broken = broken_quotes(final, docs)
                    if broken:
                        print(
                            f"  Quote(s) {broken} are not in the document word for word: "
                            "edit the evidence [e] > [v] first."
                        )
                        continue
                    note = ask("Note (optional)")
                    try:
                        item = make_item(final, items, draft.id, note)
                    except ValidationError as err:
                        print(f"  Not valid yet: {err.errors()[0]['msg']}")
                        continue
                    changed = [f for f in EDITABLE if getattr(final, f) != getattr(draft, f)]
                    append_item(args.questions, item)
                    items.append(item)
                    entry = {
                        "draft_id": draft.id,
                        "decision": "edited" if changed else "accepted",
                        "item_id": item.id,
                        "question_type": item.question_type,
                        "changed": changed,
                        "note": note,
                    }
                    log_decision(args.log, entry)
                    log.append(entry)
                    print(
                        f"  Saved as {item.id}"
                        + (f" (edited: {', '.join(changed)})" if changed else "")
                    )
                    break
        except Aborted:
            print("\nStopped. Progress is saved; run again to continue.")

    summary = summarise(log, len(drafts))
    args.summary_file.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
