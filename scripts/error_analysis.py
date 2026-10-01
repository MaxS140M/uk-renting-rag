"""Draft eval/ERROR_ANALYSIS.md: failures from one config, grouped by likely cause.

Causes are assigned automatically from the per-question results and are only a starting
point: each example has placeholders for your own diagnosis and proposed fix, and two
causes (judge error, reference answer wrong) can only be found by reading the examples.

Usage:
    python scripts/error_analysis.py --config hybrid_rerank
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

from rag.config import PROJECT_ROOT
from rag.eval_schema import load_items

EVAL_DIR = PROJECT_ROOT / "eval"
OUT = EVAL_DIR / "ERROR_ANALYSIS.md"
MAX_EXAMPLES = 15

CAUSES = {
    "not_retrieved": "Correct passage not retrieved (not in the top 10)",
    "ranked_too_low": "Correct passage retrieved but ranked too low (top 10, not in the "
    "passages given to the LLM)",
    "answered_unanswerable": "Answer not in the corpus, but the model answered anyway",
    "misread": "Correct passage given to the model, but it ignored or misread it",
}
MANUAL_CAUSES = {
    "judge_error": "Judge error: the answer was actually fine",
    "reference_wrong": "Reference answer wrong (the test set was not human-reviewed)",
}


def failed(record: dict) -> bool:
    gen = record.get("generation")
    if gen is None:
        return False
    if record["question_type"] == "unanswerable":
        return not gen["refusal_correct"]
    faith = gen["faithfulness"]
    return gen["correctness"]["verdict"] != "correct" or (
        faith is not None and faith["score"] is not None and faith["score"] < 1.0
    )


def classify(record: dict, final_k: int) -> str:
    if record["question_type"] == "unanswerable":
        return "answered_unanswerable"
    rank = record.get("first_gold_rank")
    if rank is None:
        return "not_retrieved"
    if rank > final_k:
        return "ranked_too_low"
    return "misread"


def pick_examples(by_cause: dict[str, list[dict]], limit: int) -> list[tuple[str, dict]]:
    """Take examples round-robin across causes, so every cause is represented."""
    queues = {cause: list(records) for cause, records in by_cause.items() if records}
    picked: list[tuple[str, dict]] = []
    while queues and len(picked) < limit:
        for cause in list(queues):
            if len(picked) >= limit:
                break
            picked.append((cause, queues[cause].pop(0)))
            if not queues[cause]:
                del queues[cause]
    return picked


def describe(record: dict, item, final_k: int) -> list[str]:
    gen = record["generation"]
    rank = record.get("first_gold_rank")
    lines = [f"**Question** ({record['question_type']}): {record['question']}", ""]
    if item.evidence:
        where = f"first correct passage at rank {rank}" if rank else "not in the top 10"
        lines.append(f"**Gold evidence** ({where}):")
        lines += [f"> {e.quote} (`{e.doc_id}`)" for e in item.evidence]
        lines.append("")
    lines.append(f"**Passages given to the model** (top {final_k}):")
    for n, r in enumerate(record["retrieved"][:final_k], start=1):
        gold = " ✓" if r["chunk_id"] in {c for g in record.get("gold", []) for c in g} else ""
        lines.append(f"{n}. {r['title']} > {r['section']}{gold}")
    lines += ["", "**Answer:**", ""]
    lines += [f"> {line}" if line else ">" for line in gen["answer"].splitlines()]
    lines += ["", f"**Reference answer:** {item.reference_answer}", ""]
    if gen.get("correctness"):
        lines.append(
            f"**Correctness judge:** {gen['correctness']['verdict']}: "
            f"{gen['correctness']['reasoning']}"
        )
    if gen.get("faithfulness") and gen["faithfulness"]["score"] is not None:
        unsupported = [c["claim"] for c in gen["faithfulness"]["claims"] if not c["supported"]]
        lines.append(
            f"**Faithfulness judge:** {gen['faithfulness']['score']:.0%} of claims "
            "supported" + (f"; unsupported: {'; '.join(unsupported)}" if unsupported else "")
        )
    lines += [
        "",
        "**My diagnosis:** _TODO: confirm the cause, or move this example to judge error / "
        "reference answer wrong._",
        "",
        "**Proposed fix:** _TODO_",
        "",
    ]
    return lines


def build_report(config: str, records: list[dict], items: dict, final_k: int) -> str:
    generated = [r for r in records if r.get("generation")]
    failures = [r for r in generated if failed(r)]
    by_cause: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(failures, key=lambda r: r["id"]):
        by_cause[classify(r, final_k)].append(r)
    examples = pick_examples(by_cause, MAX_EXAMPLES)

    lines = [
        f"# Error analysis: `{config}`",
        "",
        "Drafted by `scripts/error_analysis.py`. Causes were assigned automatically and must be "
        "checked by reading each example: **the diagnoses, fixes and conclusions below are mine "
        "to write** (marked TODO).",
        "",
        f"{len(failures)} of {len(generated)} dev questions failed: a correctness verdict other "
        "than correct, an unsupported claim, or a missed refusal.",
        "",
        "## Failures by likely cause",
        "",
        "| Cause | Count |",
        "|---|---:|",
        *[f"| {label} | {len(by_cause.get(cause, []))} |" for cause, label in CAUSES.items()],
        *[f"| {label} | _TODO after review_ |" for label in MANUAL_CAUSES.values()],
        "",
        f"## Examples ({len(examples)} of {len(failures)})",
        "",
    ]
    current = None
    for cause, record in sorted(examples, key=lambda x: list(CAUSES).index(x[0])):
        if cause != current:
            lines += [f"### {CAUSES[cause]}", ""]
            current = cause
        lines += [f"#### {record['id']}", ""]
        lines += describe(record, items[record["id"]], final_k)
    for label in MANUAL_CAUSES.values():
        lines += [f"### {label}", "", "_TODO: move examples here after checking them._", ""]
    lines += [
        "## Conclusions",
        "",
        "_TODO: the main weaknesses, in order of impact, and which fixes to try first._",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", type=Path, help="default: eval/results/<config>.jsonl")
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    path = args.results or EVAL_DIR / "results" / f"{args.config}.jsonl"
    if not path.exists():
        print(f"{path} not found: run eval/run_eval.py --generate {args.config} first.")
        return 1
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    items = {i.id: i for i in load_items(EVAL_DIR / "questions.jsonl")}
    if not any(r.get("generation") for r in records):
        print(f"No generated answers in {path}.")
        return 1
    report = build_report(args.config, records, items, args.final_k)
    args.out.write_text(report, encoding="utf-8")
    print(textwrap.shorten(report, 1500))
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
