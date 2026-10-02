"""Check the LLM judge against your own labels on a random sample of 30 answers.

You label each answer yourself, without seeing the judge's verdict: first faithfulness
(is every claim supported by the passages?), then correctness against the reference
answer. Labels are saved after each item, so you can stop and resume. At the end, it
reports agreement and Cohen's kappa between you and the judge.

Usage:
    python scripts/judge_agreement.py --config hybrid_rerank      # label, then report
    python scripts/judge_agreement.py --config hybrid_rerank --report
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path

from rag.authoring import Aborted, ask
from rag.config import PROJECT_ROOT
from rag.experiment import load_experiments, load_index_chunks
from rag.metrics import cohens_kappa, percent_agreement

EVAL_DIR = PROJECT_ROOT / "eval"
LABELS_FILE = EVAL_DIR / "judge_labels.jsonl"
REPORT_FILE = EVAL_DIR / "judge_agreement.md"
SAMPLE_SIZE = 30
SEED = 42
KAPPA_TRUST_THRESHOLD = 0.6  # "substantial" agreement or better (Landis and Koch, 1977)
CORRECTNESS_KEYS = {"c": "correct", "p": "partially_correct", "i": "incorrect"}


def eligible(records: list[dict]) -> list[dict]:
    """Answered (not refused) answerable questions: the ones with both judge verdicts."""
    return sorted(
        (
            r
            for r in records
            if r.get("generation")
            and r["question_type"] != "unanswerable"
            and not r["generation"]["refused"]
            and r["generation"]["faithfulness"]
            and r["generation"]["faithfulness"]["score"] is not None
        ),
        key=lambda r: r["id"],
    )


def sample(records: list[dict], n: int = SAMPLE_SIZE, seed: int = SEED) -> list[dict]:
    pool = eligible(records)
    return random.Random(seed).sample(pool, min(n, len(pool)))


def read_labels(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    labels = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {(label["config"], label["id"]): label for label in labels}


def show_item(record: dict, chunks: dict[str, dict], k: int, position: str) -> None:
    print("\n" + "=" * 96)
    print(f"{position}  {record['id']}  [{record['question_type']}]")
    print(f"\nQUESTION: {record['question']}")
    for n, retrieved in enumerate(record["retrieved"][:k], start=1):
        chunk = chunks[retrieved["chunk_id"]]
        print(f"\n[{n}] {chunk['title']} > {chunk['section']}")
        print(textwrap.indent(textwrap.fill(" ".join(chunk["text"].split()), 92), "    "))
    print("\nANSWER:")
    print(textwrap.indent(record["generation"]["answer"], "    "))


def collect_labels(config: str, records, chunks, k, labels_path, reference_by_id) -> None:
    labels = read_labels(labels_path)
    todo = [r for r in records if (config, r["id"]) not in labels]
    done = len(records) - len(todo)
    print(
        f"{done} of {len(records)} already labelled. Keys are shown at each prompt; "
        "q quits (progress is saved)."
    )
    for k_item, record in enumerate(todo, start=1):
        show_item(record, chunks, k, f"[{done + k_item}/{len(records)}]")
        while True:
            faithful = ask(
                "\nIs EVERY factual claim in the answer supported by the passages "
                "above? (y/n, q to quit)"
            ).lower()[:1]
            if faithful == "q":
                raise Aborted
            if faithful in ("y", "n"):
                break
        print("\nREFERENCE ANSWER:")
        print(textwrap.indent(textwrap.fill(reference_by_id[record["id"]], 92), "    "))
        while True:
            verdict = ask(
                "Compared with the reference: c = correct, p = partially correct, "
                "i = incorrect (q to quit)"
            ).lower()[:1]
            if verdict == "q":
                raise Aborted
            if verdict in CORRECTNESS_KEYS:
                break
        note = ask("Note (optional)")
        entry = {
            "config": config,
            "id": record["id"],
            "faithful": faithful == "y",
            "correctness": CORRECTNESS_KEYS[verdict],
            "note": note,
        }
        with labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


BLIND_METHOD = "labelled by hand without seeing the judge's verdict"
ADJUDICATED_METHOD = (
    "blind labels, then every disagreement reviewed with the judge's reasoning visible "
    "and the label corrected where the judge was right (not blind, so agreement is "
    "optimistic)"
)


def agreement_report(
    config: str,
    records: list[dict],
    labels: dict,
    method: str = BLIND_METHOD,
    title: str = "# LLM judge agreement with human labels",
) -> tuple[str, dict]:
    pairs = [(r, labels[(config, r["id"])]) for r in records if (config, r["id"]) in labels]
    if not pairs:
        return "No labels yet.\n", {}
    human_f = [label["faithful"] for _, label in pairs]
    judge_f = [r["generation"]["faithfulness"]["faithful"] for r, _ in pairs]
    human_c = [label["correctness"] for _, label in pairs]
    judge_c = [r["generation"]["correctness"]["verdict"] for r, _ in pairs]
    human_cb = [c == "correct" for c in human_c]
    judge_cb = [c == "correct" for c in judge_c]

    stats = {
        "n": len(pairs),
        "faithfulness_agreement": percent_agreement(human_f, judge_f),
        "faithfulness_kappa": cohens_kappa(human_f, judge_f),
        "correctness_agreement": percent_agreement(human_c, judge_c),
        "correctness_kappa": cohens_kappa(human_c, judge_c),
        "correct_binary_agreement": percent_agreement(human_cb, judge_cb),
        "correct_binary_kappa": cohens_kappa(human_cb, judge_cb),
    }
    trusted = {
        "faithfulness": stats["faithfulness_kappa"] >= KAPPA_TRUST_THRESHOLD,
        "correctness": stats["correctness_kappa"] >= KAPPA_TRUST_THRESHOLD,
    }
    stats["trusted"] = trusted

    def matrix(human, judge, labels_order):
        counts = Counter(zip(human, judge, strict=True))
        rows = [
            "| human \\ judge | " + " | ".join(labels_order) + " |",
            "|---|" + "---:|" * len(labels_order),
        ]
        for h in labels_order:
            rows.append(f"| {h} | " + " | ".join(str(counts[(h, j)]) for j in labels_order) + " |")
        return rows

    lines = [
        title,
        "",
        f"Generated by `scripts/judge_agreement.py`. {stats['n']} answers from `{config}`, "
        f"sampled at random (seed {SEED}) from answered questions; {method}.",
        "",
        "| Judgement | Agreement | Cohen's kappa | Trust the judge? |",
        "|---|---:|---:|---|",
        f"| Faithfulness (all claims supported: yes/no) | {stats['faithfulness_agreement']:.0%} | "
        f"{stats['faithfulness_kappa']:.2f} | {'yes' if trusted['faithfulness'] else '**no**'} |",
        f"| Correctness (correct / partial / incorrect) | {stats['correctness_agreement']:.0%} | "
        f"{stats['correctness_kappa']:.2f} | {'yes' if trusted['correctness'] else '**no**'} |",
        f"| Correctness (correct vs not) | {stats['correct_binary_agreement']:.0%} | "
        f"{stats['correct_binary_kappa']:.2f} | |",
        "",
        f"Kappa corrects agreement for chance. The judge is trusted at kappa >= "
        f'{KAPPA_TRUST_THRESHOLD} ("substantial" agreement). With {stats["n"]} items, these '
        "estimates are themselves uncertain.",
        "",
        "## Faithfulness",
        "",
        *matrix([str(x) for x in human_f], [str(x) for x in judge_f], ["True", "False"]),
        "",
        "## Correctness",
        "",
        *matrix(human_c, judge_c, ["correct", "partially_correct", "incorrect"]),
        "",
        "## Disagreements",
        "",
    ]
    for (record, label), hf, jf, hc, jc in zip(
        pairs, human_f, judge_f, human_c, judge_c, strict=True
    ):
        if hf != jf or hc != jc:
            gen = record["generation"]
            lines.append(f"- **{record['id']}** {record['question']}")
            if hf != jf:
                lines.append(
                    f"  - Faithful: human {hf}, judge {jf}. Judge: "
                    f"{gen['faithfulness']['reasoning']}"
                )
            if hc != jc:
                lines.append(
                    f"  - Correctness: human {hc}, judge {jc}. Judge: "
                    f"{gen['correctness']['reasoning']}"
                )
            if label.get("note"):
                lines.append(f"  - Human note: {label['note']}")
    return "\n".join(lines) + "\n", stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment whose answers to check")
    parser.add_argument("--results", type=Path, help="default: eval/results/<config>.jsonl")
    parser.add_argument("--chunks", type=Path, help="default: the config's index chunks")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--labels", type=Path, default=LABELS_FILE)
    parser.add_argument("--out", type=Path, default=REPORT_FILE)
    parser.add_argument("--report", action="store_true", help="report only; no labelling")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    experiments = {e.name: e for e in load_experiments(EVAL_DIR / "configs.yaml")}
    results_path = args.results or EVAL_DIR / "results" / f"{args.config}.jsonl"
    if not results_path.exists():
        print(f"{results_path} not found: run eval/run_eval.py --generate {args.config} first.")
        return 1
    records = [json.loads(line) for line in results_path.read_text("utf-8").splitlines() if line]
    chosen = sample(records)
    if not chosen:
        print("No judged answers in the results file.")
        return 1

    if not args.report:
        chunks_path = args.chunks or experiments[args.config].config.index_dir / "chunks.jsonl"
        chunks = {c["chunk_id"]: c for c in load_index_chunks(chunks_path.parent)}
        k = experiments[args.config].config.final_k if args.config in experiments else 5
        references = {}
        for line in args.questions.read_text("utf-8").splitlines():
            if line:
                item = json.loads(line)
                references[item["id"]] = item["reference_answer"]
        try:
            collect_labels(args.config, chosen, chunks, k, args.labels, references)
        except Aborted:
            print("\nStopped. Labels so far are saved; run again to continue.")

    report, stats = agreement_report(args.config, chosen, read_labels(args.labels))
    stats["method"] = BLIND_METHOD
    adjudicated = args.labels.with_name("judge_adjudicated_labels.jsonl")
    if adjudicated.exists() and stats:
        adj_report, adj_stats = agreement_report(
            args.config,
            chosen,
            read_labels(adjudicated),
            ADJUDICATED_METHOD,
            "## After reviewing the disagreements",
        )
        adj_stats["method"] = ADJUDICATED_METHOD
        adj_stats["blind"] = stats
        report = (
            report.replace(
                "# LLM judge agreement with human labels",
                "# LLM judge agreement with human labels\n\n## Blind labels",
                1,
            )
            + "\n"
            + adj_report.replace("\n## ", "\n### ")
        )
        stats = adj_stats
    args.out.write_text(report, encoding="utf-8")
    (args.out.with_suffix(".json")).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n" + report.split("## Faithfulness")[0])
    if stats and not all(stats["trusted"].values()):
        print(
            "WARNING: agreement with the judge is low for at least one metric. Do not trust "
            "those judge scores until the judge prompt is improved and re-checked."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
