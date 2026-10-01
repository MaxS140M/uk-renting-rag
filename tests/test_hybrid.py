"""Tests for Reciprocal Rank Fusion and the hybrid retriever."""

import pytest

from rag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalResult,
    Retriever,
    reciprocal_rank_fusion,
)


def test_rrf_matches_hand_calculated_example():
    # With rrf_k = 60:
    #   a: dense rank 1, bm25 rank 3 -> 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    #   b: dense rank 2, bm25 rank 1 -> 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    #   c: dense rank 3 only         -> 1/63        = 0.015873
    #   d: bm25 rank 2 only          -> 1/62        = 0.016129
    # So b (good in both lists) beats a (top of one list), and the order is b, a, d, c.
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["b", "d", "a"]}, rrf_k=60)
    assert [item for item, _ in fused] == ["b", "a", "d", "c"]
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)


def test_rrf_uses_ranks_not_raw_scores():
    # Only the order of each list matters, so wildly different score scales cannot let one
    # retriever dominate: the inputs here are plain ranked ID lists.
    assert reciprocal_rank_fusion({"x": ["a", "b"], "y": ["b", "a"]}, rrf_k=60)[0][1] == (
        pytest.approx(1 / 61 + 1 / 62)
    )


def test_rrf_ties_are_broken_deterministically():
    fused = reciprocal_rank_fusion({"x": ["a", "b"], "y": ["b", "a"]}, rrf_k=60)
    assert [item for item, _ in fused] == ["a", "b"]  # equal score and best rank -> by ID


class StaticRetriever(Retriever):
    """Returns a fixed ranked list of chunk IDs, for testing fusion in isolation."""

    def __init__(self, name, chunk_ids, chunks):
        self.name = name
        self.ids = chunk_ids
        self.by_id = {c["chunk_id"]: c for c in chunks}

    def retrieve(self, query, k, timings=None):
        return [
            RetrievalResult.from_chunk(self.by_id[cid], 1.0 / rank, rank, self.name)
            for rank, cid in enumerate(self.ids[:k], start=1)
        ]


def test_hybrid_fuses_deduplicates_and_keeps_original_ranks(chunks):
    dense = StaticRetriever("dense", ["deposit-000", "repairs-000", "damp-000"], chunks)
    bm25 = StaticRetriever("bm25", ["repairs-000", "fees-000", "deposit-000"], chunks)
    results = HybridRetriever(dense, bm25, candidate_k=3).retrieve("q", 10)

    ids = [r.chunk_id for r in results]
    assert len(ids) == len(set(ids)) == 4  # 6 candidates, 2 shared -> 4 unique chunks
    assert ids[0] == "repairs-000"  # dense #2 + bm25 #1 beats dense #1 + bm25 #3
    top = results[0]
    assert top.ranks == {"dense": 2, "bm25": 1, "hybrid": 1}
    assert top.scored_by == "hybrid"
    only_bm25 = next(r for r in results if r.chunk_id == "fees-000")
    assert "dense" not in only_bm25.ranks and only_bm25.ranks["bm25"] == 2


def test_hybrid_with_real_retrievers_has_no_duplicates(loaded_index, embedder, chunks):
    hybrid = HybridRetriever(DenseRetriever(loaded_index, embedder), BM25Retriever(chunks))
    timings = {}
    results = hybrid.retrieve("landlord must protect deposit", 5, timings)
    assert len(results) == 5
    assert len({r.chunk_id for r in results}) == 5
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    assert {"dense", "bm25", "fusion"} <= set(timings)


def test_hybrid_requests_the_candidate_pool_from_each_retriever(chunks):
    calls = []

    class Recording(StaticRetriever):
        def retrieve(self, query, k, timings=None):
            calls.append((self.name, k))
            return super().retrieve(query, k, timings)

    ids = [c["chunk_id"] for c in chunks]
    HybridRetriever(Recording("dense", ids, chunks), Recording("bm25", ids, chunks), 4).retrieve(
        "q", 2
    )
    assert calls == [("dense", 4), ("bm25", 4)]
