"""Build the retriever a config describes, so nothing else needs to know which mode is active.

Loaded components (embedding model + FAISS index, BM25 statistics, cross-encoder) are cached
per process, so building many configurations, as the evaluation does, loads each model once.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rag.config import RAGConfig
from rag.indexing import SentenceTransformerEmbedder, load_index
from rag.rerank import CrossEncoderReranker, load_cross_encoder
from rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever


@lru_cache(maxsize=4)
def load_dense(index_dir: Path, embedding_model: str, max_seq_length: int) -> DenseRetriever:
    loaded = load_index(index_dir, embedding_model)
    return DenseRetriever(loaded, SentenceTransformerEmbedder(embedding_model, max_seq_length))


@lru_cache(maxsize=4)
def load_bm25(index_dir: Path) -> BM25Retriever:
    return BM25Retriever.from_index_dir(index_dir)


def build_retriever(config: RAGConfig) -> Retriever:
    """Return the retriever for ``config``: dense, BM25 or hybrid, optionally reranked."""
    if config.retrieval_mode == "dense":
        retriever: Retriever = load_dense(
            config.index_dir, config.embedding_model, config.embedding_max_seq_length
        )
    elif config.retrieval_mode == "bm25":
        retriever = load_bm25(config.index_dir)
    elif config.retrieval_mode == "hybrid":
        retriever = HybridRetriever(
            dense=load_dense(
                config.index_dir, config.embedding_model, config.embedding_max_seq_length
            ),
            bm25=load_bm25(config.index_dir),
            candidate_k=config.candidate_k,
            rrf_k=config.rrf_k,
        )
    else:  # RAGConfig validates the mode, so this only guards against future edits
        raise ValueError(f"Unknown retrieval_mode '{config.retrieval_mode}'")

    if config.use_reranker:
        model = load_cross_encoder(config.reranker_model, config.reranker_device)
        retriever = CrossEncoderReranker(retriever, model, candidate_k=config.candidate_k)
    return retriever
