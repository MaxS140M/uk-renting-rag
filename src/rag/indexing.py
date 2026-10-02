"""Embed chunks and build, save and load the FAISS vector index used for dense retrieval.

Vectors are L2-normalised, so the inner product of two vectors equals their cosine
similarity. That lets an exact inner-product index (IndexFlatIP) do cosine search directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


class IndexMismatchError(RuntimeError):
    """Raised when an index was built with different settings from those requested."""


class Embedder(Protocol):
    """Anything that turns texts into a (len(texts), dim) array of unit-length float32 rows."""

    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model."""

    def __init__(self, model_name: str, max_seq_length: int, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # slow import, so done lazily

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class LoadedIndex:
    """A FAISS index plus the chunk records it was built from, in the same order."""

    index: faiss.Index
    chunks: list[dict]
    meta: dict


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build an exact inner-product index from unit-length vectors."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("vectors must be L2-normalised so inner product equals cosine similarity")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(
    out_dir: Path, index: faiss.Index, chunks: list[dict], embedding_model: str, **extra_meta
) -> dict:
    """Save the index, its chunk records (the row -> chunk lookup) and build metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))

    chunk_lines = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
    (out_dir / CHUNKS_FILE).write_text(chunk_lines, encoding="utf-8")

    meta = {
        "embedding_model": embedding_model,
        "dimension": index.d,
        "n_vectors": index.ntotal,
        "metric": "inner_product_on_normalised_vectors (cosine)",
        "chunks_sha256": hashlib.sha256(chunk_lines.encode("utf-8")).hexdigest(),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra_meta,
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_index(index_dir: Path, embedding_model: str) -> LoadedIndex:
    """Load a saved index, checking it was built with the embedding model the query will use.

    Query and document vectors must come from the same model: vectors from different models
    live in unrelated spaces, so a mismatch would return confident but meaningless results.
    """
    meta_path = index_dir / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No index found in {index_dir}. Build it with: python scripts/data/build_index.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["embedding_model"] != embedding_model:
        raise IndexMismatchError(
            f"Index in {index_dir} was built with '{meta['embedding_model']}', but queries "
            f"would be embedded with '{embedding_model}'. Rebuild it with: "
            f"python scripts/data/build_index.py --model {embedding_model}"
        )

    index = faiss.read_index(str(index_dir / INDEX_FILE))
    lines = (index_dir / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in lines if line]
    if len(chunks) != index.ntotal:
        raise IndexMismatchError(
            f"Index has {index.ntotal} vectors but {len(chunks)} chunk records; rebuild it."
        )
    return LoadedIndex(index=index, chunks=chunks, meta=meta)
