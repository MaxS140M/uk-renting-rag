"""Tests for the cross-encoder reranker, using a stub model so CI never downloads one."""

import numpy as np
import pytest

from rag.rerank import CrossEncoderReranker
from rag.retrieval import BM25Retriever, DenseRetriever


class StubCrossEncoder:
    """Scores a (query, passage) pair by how many query words appear in the passage."""

    def __init__(self):
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append(pairs)
        return np.array(
            [
                sum(word in passage.lower() for word in query.lower().split())
                for query, passage in pairs
            ],
            dtype=float,
        )


@pytest.fixture
def base(loaded_index, embedder):
    return DenseRetriever(loaded_index, embedder)


@pytest.mark.parametrize("final_k", [1, 2, 3])
def test_returns_exactly_final_k_in_descending_score_order(base, final_k):
    reranker = CrossEncoderReranker(base, StubCrossEncoder(), candidate_k=6)
    results = reranker.retrieve("landlord must protect deposit scheme", final_k)
    assert len(results) == final_k
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == list(range(1, final_k + 1))


def test_scores_the_whole_candidate_pool_and_records_ranks(base):
    model = StubCrossEncoder()
    reranker = CrossEncoderReranker(base, model, candidate_k=6)
    timings = {}
    results = reranker.retrieve("your landlord must give notice", 2, timings)

    assert len(model.calls) == 1  # one batched call for all candidates
    assert len(model.calls[0]) == len(base.retrieve("your landlord must give notice", 6))
    assert all(query == "your landlord must give notice" for query, _ in model.calls[0])
    for r in results:
        assert r.scored_by == "rerank"
        assert r.ranks["rerank"] == r.rank
        assert "dense" in r.ranks  # rank before reranking is kept
    assert "rerank" in timings and "dense" in timings


def test_reranking_can_change_the_order(base):
    class ReverseStub:
        def predict(self, pairs, **kwargs):
            return np.arange(len(pairs), dtype=float)  # last candidate scores highest

    before = [r.chunk_id for r in base.retrieve("landlord", 3)]
    after = [
        r.chunk_id for r in CrossEncoderReranker(base, ReverseStub(), 3).retrieve("landlord", 3)
    ]
    assert after == list(reversed(before))


def test_no_candidates_means_no_model_call(chunks):
    model = StubCrossEncoder()
    assert CrossEncoderReranker(BM25Retriever(chunks), model).retrieve("zebra", 5) == []
    assert model.calls == []
