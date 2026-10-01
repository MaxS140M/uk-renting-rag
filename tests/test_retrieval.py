"""Tests for the FAISS index helpers and the dense retriever (using a fake embedder)."""

import numpy as np
import pytest

from rag.indexing import IndexMismatchError, build_index, load_index, save_index
from rag.retrieval import DenseRetriever, RetrievalResult


@pytest.fixture
def retriever(loaded_index, embedder) -> DenseRetriever:
    return DenseRetriever(loaded_index, embedder)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_returns_exactly_k_results(retriever, k):
    assert len(retriever.retrieve("landlord deposit", k)) == k


def test_k_larger_than_index_returns_every_chunk(retriever, chunks):
    assert len(retriever.retrieve("landlord", 100)) == len(chunks)


def test_results_sorted_by_score_with_sequential_ranks(retriever):
    results = retriever.retrieve("landlord repairs heating", 5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]


def test_metadata_is_carried_through_intact(retriever, chunks):
    by_id = {c["chunk_id"]: c for c in chunks}
    for result in retriever.retrieve("landlord deposit scheme", 5):
        assert isinstance(result, RetrievalResult)
        chunk = by_id[result.chunk_id]
        for field in ("text", "doc_id", "title", "url", "section", "date_retrieved"):
            assert getattr(result, field) == chunk[field]


def test_most_relevant_chunk_ranks_first(retriever):
    assert retriever.retrieve("protect my deposit in a scheme", 3)[0].chunk_id == "deposit-000"


def test_invalid_k_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("deposit", 0)


def test_build_index_rejects_unnormalised_vectors():
    with pytest.raises(ValueError):
        build_index(np.full((2, 4), 3.0, dtype=np.float32))


def test_save_and_load_round_trip(tmp_path, chunks, embedder):
    vectors = embedder.encode([c["text"] for c in chunks])
    save_index(tmp_path, build_index(vectors), chunks, embedding_model=embedder.model_name)
    loaded = load_index(tmp_path, embedder.model_name)
    assert loaded.chunks == chunks
    assert loaded.index.ntotal == len(chunks)


def test_loading_with_a_different_model_fails_clearly(tmp_path, chunks, embedder):
    vectors = embedder.encode([c["text"] for c in chunks])
    save_index(tmp_path, build_index(vectors), chunks, embedding_model=embedder.model_name)
    with pytest.raises(IndexMismatchError, match="fake-embedder"):
        load_index(tmp_path, "sentence-transformers/some-other-model")


def test_missing_index_explains_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        load_index(tmp_path / "nowhere", "any-model")
