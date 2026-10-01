"""Rerank retrieved candidates with a cross-encoder for higher precision.

A bi-encoder (the dense retriever) embeds the query and each chunk *separately*, so chunk
vectors can be computed once in advance and searched in milliseconds, but the model never
sees the query and chunk together. A cross-encoder reads the (query, chunk) pair *together*
through every transformer layer, so it can judge whether the chunk actually answers the
question, not just whether it is about the same topic. That makes it more accurate, but it
must run once per pair at query time, so it is only affordable on a small candidate pool.
"""

from __future__ import annotations

import time
from dataclasses import replace
from functools import lru_cache
from typing import Protocol

import numpy as np

from rag.retrieval import RetrievalResult, Retriever, record_time


class PairScorer(Protocol):
    """Anything that scores (query, passage) pairs, like sentence-transformers' CrossEncoder."""

    def predict(self, pairs: list[tuple[str, str]], **kwargs) -> np.ndarray: ...


@lru_cache(maxsize=2)
def load_cross_encoder(model_name: str, device: str = "cpu") -> PairScorer:
    """Load a cross-encoder once per process and reuse it (loading takes seconds)."""
    from sentence_transformers import CrossEncoder  # slow import, so done lazily

    return CrossEncoder(model_name, device=device, max_length=512)


class CrossEncoderReranker(Retriever):
    """Wrap any retriever: take its top candidates, rescore them with a cross-encoder."""

    name = "rerank"

    def __init__(self, base: Retriever, model: PairScorer, candidate_k: int = 20) -> None:
        self.base = base
        self.model = model
        self.candidate_k = candidate_k

    def retrieve(
        self, query: str, k: int, timings: dict[str, float] | None = None
    ) -> list[RetrievalResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        candidates = self.base.retrieve(query, max(k, self.candidate_k), timings)
        if not candidates:
            return []

        start = time.perf_counter()
        pairs = [(query, c.text) for c in candidates]
        scores = np.asarray(self.model.predict(pairs, show_progress_bar=False), dtype=float)
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].rank))
        results = [
            replace(
                candidates[i],
                score=float(scores[i]),
                rank=rank,
                scored_by=self.name,
                ranks={**candidates[i].ranks, self.name: rank},
            )
            for rank, i in enumerate(order[:k], start=1)
        ]
        record_time(timings, self.name, start)
        return results
