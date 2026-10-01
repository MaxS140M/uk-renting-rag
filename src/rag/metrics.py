"""Evaluation metrics: retrieval (Recall@k, MRR, evidence recall), latency and agreement.

Gold labels are sets of chunk_ids per evidence quote (from ``EvidenceMapper``), so the same
metrics work for any chunking.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence


def first_relevant_rank(retrieved: Sequence[str], gold: set[str]) -> int | None:
    """1-based rank of the first retrieved chunk that is gold, or None."""
    return next((rank for rank, c in enumerate(retrieved, start=1) if c in gold), None)


def recall_at_k(retrieved: Sequence[str], gold: set[str], k: int) -> float:
    """1.0 if any gold chunk is in the top k (a "hit"), else 0.0."""
    rank = first_relevant_rank(retrieved, gold)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(retrieved: Sequence[str], gold: set[str]) -> float:
    """1 / rank of the first gold chunk; 0.0 if none was retrieved (averaged into MRR)."""
    rank = first_relevant_rank(retrieved, gold)
    return 1.0 / rank if rank is not None else 0.0


def evidence_recall_at_k(
    retrieved: Sequence[str], gold_per_evidence: Sequence[set[str]], k: int
) -> float:
    """Share of evidence pieces with at least one matching chunk in the top k.

    For a multi-passage question this shows whether *all* the needed passages were found,
    which a single hit does not.
    """
    if not gold_per_evidence:
        raise ValueError("no evidence pieces")
    top = set(retrieved[:k])
    return sum(bool(gold & top) for gold in gold_per_evidence) / len(gold_per_evidence)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Percentile with linear interpolation (q in [0, 100]), e.g. 50 = median, 95 = p95."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def percent_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of the same length")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Agreement between two raters corrected for chance: (p_o - p_e) / (1 - p_e).

    1 = perfect agreement, 0 = no better than chance, negative = worse than chance. If both
    raters always give the same single label, chance agreement is 1 and kappa is undefined;
    1.0 is returned when they also agree on every item.
    """
    observed = percent_agreement(a, b)
    n = len(a)
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[label] * counts_b[label] for label in counts_a) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
