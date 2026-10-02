"""Write eval/RESULTS.md from eval/results/summary.csv.

Includes the main results table, a one-variable-at-a-time comparison table, a short
auto-generated headline (best config vs the baseline, in absolute points), the judge's
agreement with human labels, and, clearly separated, the single held-out run.

Usage:
    python scripts/make_results_table.py
"""

from __future__ import annotations

import csv
import json
import math
import sys

from rag.config import PROJECT_ROOT
from rag.experiment import Experiment, changed_settings, load_experiments
from rag.metrics import paired_bootstrap_ci

EVAL_DIR = PROJECT_ROOT / "eval"
SUMMARY = EVAL_DIR / "results" / "summary.csv"
OUT = EVAL_DIR / "RESULTS.md"
BASELINE = "dense_baseline"


def num(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def points(new: float | None, old: float | None) -> str:
    if new is None or old is None:
        return "—"
    diff = (new - old) * 100
    return f"{diff:+.1f} pts"


def latency(row: dict) -> str:
    total = num(row, "total_ms_p50")
    if total is not None:
        return f"{total / 1000:.2f} s"
    retrieval = num(row, "retrieval_ms_p50")
    return "—" if retrieval is None else f"{retrieval:.0f} ms (retrieval only)"


def margin(p: float | None, n: int) -> float | None:
    """Half-width of an approximate 95% confidence interval for a proportion."""
    if p is None or n == 0:
        return None
    return 1.96 * math.sqrt(p * (1 - p) / n)


def main_table(rows: dict[str, dict], experiments: list[Experiment]) -> list[str]:
    lines = [
        "| Configuration | Recall@5 | MRR | Faithfulness | Correctness | Refusal accuracy "
        "| Median latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for e in experiments:
        row = rows.get(e.name)
        if row is None:
            continue
        lines.append(
            f"| `{e.name}`: {e.description} | {pct(num(row, 'recall_at_5'))} | "
            f"{num(row, 'mrr'):.3f} | {pct(num(row, 'faithfulness'))} | "
            f"{pct(num(row, 'correctness'))} | {pct(num(row, 'refusal_accuracy'))} | "
            f"{latency(row)} |"
        )
    return lines


def paired_diff(per_question: dict, new: str, old: str, metric: str, scale: float) -> str:
    """95% paired bootstrap interval for the difference, as text; '*' if it excludes zero."""
    a_records, b_records = per_question.get(old), per_question.get(new)
    if not a_records or not b_records:
        return ""
    ids = sorted(i for i in a_records if i in b_records and metric in a_records[i])
    if not ids:
        return ""
    _, low, high = paired_bootstrap_ci(
        [a_records[i][metric] for i in ids], [b_records[i][metric] for i in ids]
    )
    star = " *" if low > 0 or high < 0 else ""
    fmt = "{:+.1f}" if scale == 100 else "{:+.3f}"
    return f" [{fmt.format(low * scale)}, {fmt.format(high * scale)}]{star}"


def ablation_table(
    rows: dict[str, dict], experiments: list[Experiment], per_question: dict | None = None
) -> list[str]:
    per_question = per_question or {}
    by_name = {e.name: e for e in experiments}
    lines = [
        "| Experiment | Compared with | Change | Δ Recall@5 (95% CI) | Δ MRR (95% CI) "
        "| Δ Correctness |",
        "|---|---|---|---:|---:|---:|",
    ]
    for e in experiments:
        if not e.compare_to or e.name not in rows or e.compare_to not in rows:
            continue
        new, old = rows[e.name], rows[e.compare_to]
        change = ", ".join(
            f"{k}: {a} → {b}"
            for k, (a, b) in changed_settings(e, by_name[e.compare_to]).items()
            if k != "chunk_overlap_tokens"
        )
        d_mrr = (num(new, "mrr") or 0) - (num(old, "mrr") or 0)
        ci_r5 = paired_diff(per_question, e.name, e.compare_to, "recall_at_5", 100)
        ci_mrr = paired_diff(per_question, e.name, e.compare_to, "reciprocal_rank", 1)
        lines.append(
            f"| `{e.name}` | `{e.compare_to}` | {change} | "
            f"{points(num(new, 'recall_at_5'), num(old, 'recall_at_5'))}{ci_r5} | "
            f"{d_mrr:+.3f}{ci_mrr} | "
            f"{points(num(new, 'correctness'), num(old, 'correctness'))} |"
        )
    return lines


def correctness_scores(records: dict) -> dict[str, float]:
    return {
        i: r["generation"]["correctness"]["score"]
        for i, r in records.items()
        if r.get("generation") and r["generation"].get("correctness")
    }


def headline(rows: dict[str, dict], per_question: dict | None = None) -> list[str]:
    per_question = per_question or {}
    generated = [r for r in rows.values() if num(r, "correctness") is not None]
    pool = generated or list(rows.values())
    key = "correctness" if generated else "mrr"
    best = max(pool, key=lambda r: (num(r, key) or 0, num(r, "mrr") or 0))
    base = rows.get(BASELINE)
    lines = [f"**Highest-scoring configuration: `{best['config']}`** (by {key}, dev split)."]
    if base is None or best["config"] == BASELINE:
        lines.append("It is the baseline itself; no improvement over the baseline was found.")
        return lines

    intervals = {}  # metric -> (low, high) of the paired difference, where computable
    if best["config"] in per_question and BASELINE in per_question:
        new, old = per_question[best["config"]], per_question[BASELINE]
        ids = sorted(i for i in old if i in new and "recall_at_5" in old[i])
        if ids:
            _, low, high = paired_bootstrap_ci(
                [old[i]["recall_at_5"] for i in ids], [new[i]["recall_at_5"] for i in ids]
            )
            intervals["recall_at_5"] = (low, high)
        old_c, new_c = correctness_scores(old), correctness_scores(new)
        ids = sorted(set(old_c) & set(new_c))
        if ids:
            _, low, high = paired_bootstrap_ci([old_c[i] for i in ids], [new_c[i] for i in ids])
            intervals["correctness"] = (low, high)

    def ci(metric: str) -> str:
        if metric not in intervals:
            return ""
        low, high = intervals[metric]
        return f", 95% CI {low * 100:+.1f} to {high * 100:+.1f}"

    parts = [
        f"Recall@5 {points(num(best, 'recall_at_5'), num(base, 'recall_at_5'))} "
        f"({pct(num(base, 'recall_at_5'))} → {pct(num(best, 'recall_at_5'))}{ci('recall_at_5')})",
        f"MRR {(num(best, 'mrr') or 0) - (num(base, 'mrr') or 0):+.3f} "
        f"({num(base, 'mrr'):.3f} → {num(best, 'mrr'):.3f})",
    ]
    if num(best, "correctness") is not None and num(base, "correctness") is not None:
        parts.append(
            f"correctness {points(num(best, 'correctness'), num(base, 'correctness'))} "
            f"({pct(num(base, 'correctness'))} → {pct(num(best, 'correctness'))}"
            f"{ci('correctness')})"
        )
        parts.append(f"faithfulness {points(num(best, 'faithfulness'), num(base, 'faithfulness'))}")
    lines.append("Compared with the dense baseline: " + "; ".join(parts) + ".")
    unclear = [m for m, (low, high) in intervals.items() if low <= 0 <= high]
    if unclear:
        lines.append(
            "**The 95% interval includes zero for "
            + " and ".join(m.replace("_at_", "@").replace("recall", "Recall") for m in unclear)
            + ", so these improvements are not statistically clear on this test set: "
            "they could be chance.**"
        )
    lines.append(
        f"Median latency: {latency(base)} for the baseline, {latency(best)} for `{best['config']}`."
    )
    return lines


def judge_section(stats: dict | None) -> list[str]:
    if not stats:
        return [
            "The LLM judge has **not yet been checked against human labels** "
            "(`scripts/judge_agreement.py`), so faithfulness and correctness are provisional.",
        ]
    trusted = stats["trusted"]
    blind = stats.get("blind")
    caveat = (
        f" My initial blind labels marked every answer faithful and correct, so they could not "
        f"measure agreement (kappa {blind['faithfulness_kappa']:.2f}); these figures are after "
        "reviewing each disagreement with the judge's reasoning visible, which is not blind and "
        "therefore optimistic. In that review the judge was right in most disputes, and its "
        "errors were mostly over-strict faithfulness calls."
        if blind
        else ""
    )
    return [
        f"Judge checked against my own labels on {stats['n']} randomly sampled answers "
        f"([details](judge_agreement.md)): faithfulness agreement "
        f"{stats['faithfulness_agreement']:.0%} (kappa {stats['faithfulness_kappa']:.2f}), "
        f"correctness agreement {stats['correctness_agreement']:.0%} "
        f"(kappa {stats['correctness_kappa']:.2f}).{caveat}",
        ""
        if all(trusted.values())
        else "**Agreement is below the trust threshold (kappa 0.6) for "
        + " and ".join(k for k, v in trusted.items() if not v)
        + "; treat those scores with caution.**",
    ]


def build_markdown(
    rows: list[dict],
    experiments: list[Experiment],
    judge_stats: dict | None,
    heldout_runs: list[dict],
    per_question: dict | None = None,
) -> str:
    dev = {r["config"]: r for r in rows if r["split"] == "dev"}
    heldout = {r["config"]: r for r in rows if r["split"] == "heldout"}
    order = [e for e in experiments if e.name in dev]
    any_row = next(iter(dev.values()), {})
    n_answerable = int(num(any_row, "n_answerable") or 0)
    recall = num(dev.get(BASELINE, any_row), "recall_at_5")
    moe = margin(recall, n_answerable)

    lines = [
        "# Evaluation results",
        "",
        "Generated by `scripts/make_results_table.py` from `eval/results/summary.csv`; do not "
        "edit by hand. Method: [`eval/README.md`](README.md). Failures: "
        "[`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md).",
        "",
        f"Dev split: {int(num(any_row, 'n_questions') or 0)} questions "
        f"({n_answerable} answerable, {int(num(any_row, 'n_unanswerable') or 0)} unanswerable), "
        f"corpus `{any_row.get('corpus', '?')}`, prompt `{any_row.get('prompt_version', '?')}`.",
        "",
        "## Headline",
        "",
        *headline(dev, per_question),
        "",
        "## All configurations (dev split)",
        "",
        *main_table(dev, order),
        "",
        "Recall@5: share of answerable questions with a correct passage in the top 5. MRR: mean "
        "of 1 / rank of the first correct passage. Faithfulness: share of answer claims "
        "supported by the retrieved passages (LLM judge). Correctness: answers graded against "
        "the reference (correct = 1, partially correct = 0.5). Refusal accuracy: share of "
        "unanswerable questions correctly refused. Latency: median end to end (retrieval + "
        "generation) where answers were generated, otherwise retrieval only. — means not run "
        "(generation runs only for the most promising configurations, to save cost).",
        "",
        "## One change at a time",
        "",
        "Each experiment changes one setting from the one it is compared with, so the "
        "difference can be attributed to that setting. Brackets give a 95% paired bootstrap "
        "interval for the difference (the same questions resampled 5,000 times); * marks an "
        "interval that excludes zero, i.e. a difference unlikely to be chance.",
        "",
        *ablation_table(dev, order, per_question),
        "",
        "## How much to trust these numbers",
        "",
        f"- **Sample size.** With {n_answerable} answerable questions, a single Recall@5 figure "
        + (f"is uncertain by roughly ±{moe * 100:.0f} points (95% interval). " if moe else "")
        + "Differences of a few points between configurations may be noise.",
        "- **Reference answers were not reviewed by a person** (see "
        "[`README.md`](README.md#status-of-human-review)), so correctness has unknown noise "
        "from errors in the answer key. Retrieval metrics depend only on the verified quotes.",
        "- " + " ".join(s for s in judge_section(judge_stats) if s),
        "",
    ]

    lines += ["## Held-out result (run once)", ""]
    if not heldout:
        lines.append(
            "Not run yet. The held-out split is run once, for the final configuration "
            "only, after all choices are made on the dev split."
        )
    else:
        run = heldout_runs[-1] if heldout_runs else {}
        lines += [
            f"Run on {run.get('run_at', '?')}"
            + (" (forced repeat: " + run["reason"] + ")" if run.get("forced") else "")
            + ". These questions were never used for any decision. With only "
            + f"{int(num(next(iter(heldout.values())), 'n_answerable') or 0)} answerable "
            "questions, one question moves Recall@5 or correctness by several points, so the "
            "held-out figures confirm the dev results' order of magnitude rather than refine "
            "them.",
            "",
            *main_table(heldout, [e for e in experiments if e.name in heldout]),
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not SUMMARY.exists():
        print(f"{SUMMARY} not found: run eval/run_eval.py first.")
        return 1
    with SUMMARY.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    experiments = load_experiments(EVAL_DIR / "configs.yaml")
    stats_path = EVAL_DIR / "judge_agreement.json"
    judge_stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    log = EVAL_DIR / "results" / "heldout_runs.jsonl"
    heldout_runs = (
        [json.loads(line) for line in log.read_text().splitlines() if line] if log.exists() else []
    )
    per_question = {}
    for e in experiments:
        path = EVAL_DIR / "results" / f"{e.name}.jsonl"
        if path.exists():
            records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
            per_question[e.name] = {r["id"]: r for r in records}
    markdown = build_markdown(rows, experiments, judge_stats, heldout_runs, per_question)
    OUT.write_text(markdown, encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
