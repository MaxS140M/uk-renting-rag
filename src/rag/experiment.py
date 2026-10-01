"""Experiment configurations for the evaluation: loading them, and building their indexes.

Each experiment is a named set of overrides to ``RAGConfig`` (eval/configs.yaml). Chunk
size and embedding model change the index, so each distinct (chunking, embedding model)
pair gets its own index directory, built automatically the first time it is needed.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag.chunking import chunk_documents
from rag.config import DATA_DIR, RAGConfig
from rag.indexing import SentenceTransformerEmbedder, build_index, save_index
from rag.retrieval import TOKENIZER_VERSION, save_bm25

# Settings an experiment may change. index_dir is derived, never set by hand.
ALLOWED_SETTINGS = {f.name for f in dataclasses.fields(RAGConfig) if f.name not in {"index_dir"}}


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    config: RAGConfig
    compare_to: str | None = None  # the experiment this one differs from by one variable


def index_dir_for(config: RAGConfig, root: Path = DATA_DIR) -> Path:
    """Index directory for a chunking + embedding model, e.g. data/index_c400_o50_all-minilm."""
    model = re.sub(r"[^a-z0-9.-]+", "-", config.embedding_model.split("/")[-1].lower())
    return root / f"index_c{config.chunk_max_tokens}_o{config.chunk_overlap_tokens}_{model}"


def load_experiments(path: Path, root: Path = DATA_DIR) -> list[Experiment]:
    """Read and validate eval/configs.yaml."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = data.get("experiments") if isinstance(data, dict) else None
    if not entries:
        raise ValueError(f"{path}: expected a non-empty 'experiments' list")
    experiments: list[Experiment] = []
    names: set[str] = set()
    for n, entry in enumerate(entries, start=1):
        name = entry.get("name")
        if not name or not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"experiment {n}: name must be lower_snake_case")
        if name in names:
            raise ValueError(f"experiment '{name}' is defined twice")
        settings = entry.get("settings") or {}
        unknown = set(settings) - ALLOWED_SETTINGS
        if unknown:
            raise ValueError(f"experiment '{name}': unknown settings {sorted(unknown)}")
        config = dataclasses.replace(RAGConfig(), **settings)  # RAGConfig validates values
        config = dataclasses.replace(config, index_dir=index_dir_for(config, root))
        compare_to = entry.get("compare_to")
        if compare_to is not None and compare_to not in names:
            raise ValueError(f"experiment '{name}': compare_to '{compare_to}' must come earlier")
        names.add(name)
        experiments.append(Experiment(name, entry.get("description", ""), config, compare_to))
    return experiments


def changed_settings(experiment: Experiment, reference: Experiment) -> dict[str, tuple]:
    """Settings that differ between two experiments (index_dir excluded)."""
    a, b = experiment.config.to_dict(), reference.config.to_dict()
    return {k: (b[k], a[k]) for k in a if k != "index_dir" and a[k] != b[k]}


def index_is_current(config: RAGConfig) -> bool:
    meta_path = config.index_dir / "meta.json"
    if not meta_path.exists() or not (config.index_dir / "bm25.json").exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bm25 = json.loads((config.index_dir / "bm25.json").read_text(encoding="utf-8"))
    return (
        meta.get("embedding_model") == config.embedding_model
        and meta.get("chunk_max_tokens") == config.chunk_max_tokens
        and meta.get("chunk_overlap_tokens") == config.chunk_overlap_tokens
        and bm25.get("tokenizer_version") == TOKENIZER_VERSION
    )


def ensure_index(config: RAGConfig, docs: list[dict], log=print) -> Path:
    """Chunk, embed and index the corpus for this config, unless a matching index exists."""
    if index_is_current(config):
        return config.index_dir
    log(
        f"Building index {config.index_dir.name}: {config.chunk_max_tokens}-token chunks, "
        f"{config.embedding_model}"
    )
    chunks = [
        c.to_dict()
        for c in chunk_documents(
            docs, max_tokens=config.chunk_max_tokens, overlap_tokens=config.chunk_overlap_tokens
        )
    ]
    embedder = SentenceTransformerEmbedder(config.embedding_model, config.embedding_max_seq_length)
    vectors = embedder.encode([c["text"] for c in chunks], show_progress_bar=True)
    save_index(
        config.index_dir,
        build_index(vectors),
        chunks,
        embedding_model=config.embedding_model,
        embedding_max_seq_length=config.embedding_max_seq_length,
        chunk_max_tokens=config.chunk_max_tokens,
        chunk_overlap_tokens=config.chunk_overlap_tokens,
    )
    save_bm25(config.index_dir, chunks)
    log(f"  {len(chunks)} chunks indexed")
    return config.index_dir


def load_index_chunks(index_dir: Path) -> list[dict]:
    lines = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]
