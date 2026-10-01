"""Tests for config validation and the retriever factory (with model loading stubbed out)."""

import dataclasses
import itertools

import pytest

import rag.factory as factory
from rag.config import RETRIEVAL_MODES, RAGConfig
from rag.rerank import CrossEncoderReranker
from rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


@pytest.fixture
def stub_loaders(monkeypatch, loaded_index, embedder, chunks):
    """Replace the slow loaders, so building a retriever loads no real models."""
    loads = []

    def load_dense(*args):
        loads.append("dense")
        return DenseRetriever(loaded_index, embedder)

    def load_bm25(*args):
        loads.append("bm25")
        return BM25Retriever(chunks)

    def load_cross_encoder(*args):
        loads.append("cross-encoder")
        return object()

    monkeypatch.setattr(factory, "load_dense", load_dense)
    monkeypatch.setattr(factory, "load_bm25", load_bm25)
    monkeypatch.setattr(factory, "load_cross_encoder", load_cross_encoder)
    return loads


EXPECTED_TYPE = {"dense": DenseRetriever, "bm25": BM25Retriever, "hybrid": HybridRetriever}


@pytest.mark.parametrize(
    ("mode", "use_reranker"), list(itertools.product(RETRIEVAL_MODES, [False, True]))
)
def test_build_retriever_returns_the_right_type(stub_loaders, mode, use_reranker):
    config = RAGConfig(retrieval_mode=mode, use_reranker=use_reranker, candidate_k=12)
    retriever = factory.build_retriever(config)

    if use_reranker:
        assert isinstance(retriever, CrossEncoderReranker)
        assert retriever.candidate_k == 12
        retriever = retriever.base
    assert type(retriever) is EXPECTED_TYPE[mode]
    if mode == "hybrid":
        assert retriever.candidate_k == 12
        assert isinstance(retriever.retrievers["dense"], DenseRetriever)
        assert isinstance(retriever.retrievers["bm25"], BM25Retriever)


def test_only_the_needed_components_are_loaded(stub_loaders):
    factory.build_retriever(RAGConfig(retrieval_mode="bm25"))
    assert stub_loaders == ["bm25"]  # no embedding model or cross-encoder for BM25 alone


@pytest.mark.parametrize(
    "overrides",
    [
        {"retrieval_mode": "keyword"},
        {"final_k": 0},
        {"final_k": 10, "candidate_k": 5},
        {"rrf_k": -1},
    ],
)
def test_invalid_configs_are_rejected(overrides):
    with pytest.raises(ValueError):
        RAGConfig(**overrides)


def test_config_label_and_dict():
    config = dataclasses.replace(RAGConfig(), retrieval_mode="hybrid", use_reranker=True)
    assert config.label == "hybrid+rerank"
    assert isinstance(config.to_dict()["index_dir"], str)
