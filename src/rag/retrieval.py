"""Retrieval behind a common interface: given a query, return the top-k chunks with scores.

Every retriever (dense now; BM25, hybrid and reranked later) implements ``Retriever``, so
the pipeline and evaluation code never need to know which one they are using.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np

from rag.config import RAGConfig
from rag.indexing import Embedder, LoadedIndex, SentenceTransformerEmbedder, load_index


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk: its ID, score and rank, plus everything needed to cite it."""

    chunk_id: str
    score: float
    rank: int  # 1 = best
    text: str
    doc_id: str
    title: str
    url: str
    section: str
    date_retrieved: str

    @classmethod
    def from_chunk(cls, chunk: dict, score: float, rank: int) -> RetrievalResult:
        return cls(
            chunk_id=chunk["chunk_id"],
            score=float(score),
            rank=rank,
            text=chunk["text"],
            doc_id=chunk["doc_id"],
            title=chunk["title"],
            url=chunk["url"],
            section=chunk.get("section", ""),
            date_retrieved=chunk["date_retrieved"],
        )

    def to_dict(self) -> dict:
        return asdict(self)


class Retriever(ABC):
    """Common interface for all retrievers."""

    name: str

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """Return up to ``k`` results, best first (highest score, rank 1)."""


class DenseRetriever(Retriever):
    """Semantic search: embed the query and find the nearest chunk vectors in FAISS."""

    name = "dense"

    def __init__(self, loaded: LoadedIndex, embedder: Embedder) -> None:
        if loaded.meta["embedding_model"] != embedder.model_name:
            raise ValueError("index and query embedder use different models")
        self.index = loaded.index
        self.chunks = loaded.chunks
        self.embedder = embedder

    @classmethod
    def from_config(cls, config: RAGConfig) -> DenseRetriever:
        loaded = load_index(config.index_dir, config.embedding_model)
        embedder = SentenceTransformerEmbedder(
            config.embedding_model, config.embedding_max_seq_length
        )
        return cls(loaded, embedder)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        k = min(k, self.index.ntotal)
        query_vector = np.ascontiguousarray(self.embedder.encode([query]), dtype=np.float32)
        scores, rows = self.index.search(query_vector, k)
        return [
            RetrievalResult.from_chunk(self.chunks[row], score, rank)
            for rank, (score, row) in enumerate(zip(scores[0], rows[0], strict=True), start=1)
            if row != -1
        ]


def get_retriever(config: RAGConfig) -> Retriever:
    """Build the retriever named in the config."""
    if config.retriever == "dense":
        return DenseRetriever.from_config(config)
    raise ValueError(f"Unknown retriever '{config.retriever}'. Available: dense")
