"""Shared test fixtures: a fake embedder, a fake LLM client and a guard against real API calls."""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

from rag.indexing import LoadedIndex, build_index


@pytest.fixture(autouse=True)
def block_real_anthropic_client(monkeypatch):
    """Fail loudly if any test tries to create a real Anthropic client (and spend money)."""

    def _refuse(*args, **kwargs):
        raise RuntimeError("Tests must not create a real Anthropic client; inject a fake one.")

    monkeypatch.setattr("rag.generate.anthropic.Anthropic", _refuse)


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors.

    Uses hashlib rather than hash(), because Python's hash() of a string changes between runs.
    """

    model_name = "fake-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                vectors[i, bucket] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


CHUNKS = [
    (
        "deposit-000",
        "Tenancy deposit protection",
        "Overview",
        "Your landlord must protect your deposit in a scheme within 30 days.",
    ),
    (
        "repairs-000",
        "Private renting",
        "Repairs",
        "Your landlord is responsible for repairs to the heating and hot water.",
    ),
    (
        "evictions-000",
        "Private renting evictions",
        "Notices",
        "Your landlord must give you notice before they can evict you.",
    ),
    (
        "hmo-000",
        "House in multiple occupation licence",
        "Overview",
        "A large house in multiple occupation must be licensed by the council.",
    ),
    (
        "fees-000",
        "Tenant Fees Act",
        "Banned fees",
        "Letting agents cannot charge fees for viewings or references.",
    ),
    (
        "damp-000",
        "Damp and mould",
        "Health risks",
        "Damp and mould in the home can harm your health.",
    ),
]


@pytest.fixture
def chunks() -> list[dict]:
    return [
        {
            "chunk_id": cid,
            "doc_id": cid.rsplit("-", 1)[0],
            "title": title,
            "url": f"https://www.gov.uk/{cid.rsplit('-', 1)[0]}",
            "section": section,
            "date_retrieved": "2026-10-01",
            "text": text,
            "n_tokens": len(text.split()),
        }
        for cid, title, section, text in CHUNKS
    ]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def loaded_index(chunks, embedder) -> LoadedIndex:
    vectors = embedder.encode([c["text"] for c in chunks])
    return LoadedIndex(build_index(vectors), chunks, {"embedding_model": embedder.model_name})


class FakeMessages:
    def __init__(self, reply: str, stop_reason: str = "end_turn") -> None:
        self.reply = reply
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            model=kwargs["model"],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


class FakeAnthropicClient:
    """Stands in for anthropic.Anthropic: records each request and returns a canned reply."""

    def __init__(self, reply: str = "Answer [1].", stop_reason: str = "end_turn") -> None:
        self.messages = FakeMessages(reply, stop_reason)


@pytest.fixture
def fake_client_factory():
    return FakeAnthropicClient
