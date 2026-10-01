"""Hand-worked examples for the evaluation metrics."""

import pytest

from rag.metrics import (
    cohens_kappa,
    evidence_recall_at_k,
    first_relevant_rank,
    mean,
    percent_agreement,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

# Retrieved list used throughout: gold chunk "g" appears at rank 3.
RETRIEVED = ["a", "b", "g", "c", "d", "e"]
GOLD = {"g", "z"}  # "z" (another chunk holding the same quote) was not retrieved


def test_first_relevant_rank():
    assert first_relevant_rank(RETRIEVED, GOLD) == 3
    assert first_relevant_rank(RETRIEVED, {"x"}) is None


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.0), (3, 1.0), (5, 1.0)])
def test_recall_at_k_is_a_hit_within_the_top_k(k, expected):
    assert recall_at_k(RETRIEVED, GOLD, k) == expected


def test_reciprocal_rank_and_mrr():
    # Three questions with first gold chunk at ranks 1, 3 and not found:
    # MRR = (1/1 + 1/3 + 0) / 3 = 0.4444
    ranks = [
        reciprocal_rank(["g", "a"], {"g"}),
        reciprocal_rank(RETRIEVED, GOLD),
        reciprocal_rank(RETRIEVED, {"x"}),
    ]
    assert ranks == [1.0, pytest.approx(1 / 3), 0.0]
    assert mean(ranks) == pytest.approx(4 / 9)


def test_evidence_recall_counts_each_evidence_piece():
    # Multi-passage question with two evidence pieces: the first is found at rank 3, the
    # second (chunks "p" or "q") only at rank 6.
    retrieved = ["a", "b", "g", "c", "d", "p"]
    gold = [{"g"}, {"p", "q"}]
    assert evidence_recall_at_k(retrieved, gold, 5) == 0.5
    assert evidence_recall_at_k(retrieved, gold, 6) == 1.0
    assert evidence_recall_at_k(retrieved, gold, 1) == 0.0


def test_evidence_recall_needs_evidence():
    with pytest.raises(ValueError):
        evidence_recall_at_k(RETRIEVED, [], 5)


def test_percentiles_interpolate():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 95) == pytest.approx(48)  # 40 + 0.8 * (50 - 40)
    assert percentile([], 50) is None
    assert mean([]) is None


def test_cohens_kappa_hand_worked():
    # 10 items. They disagree on items 6 and 10, so p_o = 0.8.
    # Rater A: 6 yes, 4 no. Rater B: 6 yes, 4 no.
    # p_e = 0.6 * 0.6 + 0.4 * 0.4 = 0.52, so kappa = (0.8 - 0.52) / (1 - 0.52) = 0.5833.
    a = ["y", "y", "y", "y", "y", "y", "n", "n", "n", "n"]
    b = ["y", "y", "y", "y", "y", "n", "n", "n", "n", "y"]
    assert percent_agreement(a, b) == 0.8
    assert cohens_kappa(a, b) == pytest.approx(0.28 / 0.48)


def test_cohens_kappa_edge_cases():
    assert cohens_kappa(["y", "n"], ["y", "n"]) == 1.0
    assert cohens_kappa(["y", "y"], ["y", "y"]) == 1.0  # one label only, full agreement
    assert cohens_kappa(["y", "n", "y", "n"], ["n", "y", "n", "y"]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohens_kappa(["y"], ["y", "n"])
