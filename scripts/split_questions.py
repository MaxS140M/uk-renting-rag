"""Assign the held-out split: 20 items, stratified by question type, with a fixed seed.

Run this ONCE, when the test set is complete. Held-out items are evaluated only once, at
the very end, and never used for tuning. Re-splitting later could move an item that was
already used for tuning into the held-out set, so the script refuses to change an existing
held-out set unless --force is given.

Usage:
    python scripts/split_questions.py --dry-run     # show what would be held out
    python scripts/split_questions.py               # write the split
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from rag.config import PROJECT_ROOT
from rag.evaluation.schema import EvalItem, load_items, write_items
from rag.evaluation.utils import stratified_sample

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"
HELDOUT_SIZE = 20
EXPECTED_TOTAL = 100
SEED = 42


class SplitError(RuntimeError):
    pass


def assign_heldout(
    items: list[EvalItem],
    n: int = HELDOUT_SIZE,
    seed: int = SEED,
    force: bool = False,
    allow_incomplete: bool = False,
) -> list[EvalItem]:
    """Return the items with ``n`` real (non-example) items marked held-out, the rest dev."""
    real = [i for i in items if not i.is_example]
    already = [i.id for i in real if i.split == "heldout"]
    if already and not force:
        raise SplitError(
            f"{len(already)} items are already held out ({', '.join(already[:5])}...). The "
            "held-out set is fixed once assigned; use --force only if no tuning has used the "
            "test set yet."
        )
    if len(real) < EXPECTED_TOTAL and not allow_incomplete:
        raise SplitError(
            f"Only {len(real)} of {EXPECTED_TOTAL} questions are written. Split once the set "
            "is complete, or pass --allow-incomplete."
        )
    heldout = stratified_sample(real, n, seed)
    return [
        item.model_copy(update={"split": "heldout" if item.id in heldout else "dev"})
        for item in items
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--n", type=int, default=HELDOUT_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="show the split; don't save it")
    parser.add_argument("--force", action="store_true", help="replace an existing held-out set")
    parser.add_argument("--allow-incomplete", action="store_true", help="split before 100 items")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    items = load_items(args.questions)
    try:
        updated = assign_heldout(items, args.n, args.seed, args.force, args.allow_incomplete)
    except (SplitError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    real = [i for i in updated if not i.is_example]
    for split in ("dev", "heldout"):
        counts = Counter(i.question_type for i in real if i.split == split)
        total = sum(counts.values())
        print(
            f"{split:8} {total:3} items: "
            + ", ".join(f"{t} {n}" for t, n in sorted(counts.items()))
        )
    print("Held out: " + ", ".join(sorted(i.id for i in real if i.split == "heldout")))

    if args.dry_run:
        print("Dry run: nothing saved.")
        return 0
    write_items(args.questions, updated)
    print(f"Saved split to {args.questions} (seed {args.seed}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
