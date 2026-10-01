"""Central settings (models, chunk sizes, retrieval and generation parameters, paths).

Module-level constants are the defaults. ``RAGConfig`` bundles them into one immutable
object that is passed through the pipeline, so an experiment can change a setting with
``dataclasses.replace(config, top_k=10)`` without editing code.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_DIR = DATA_DIR / "index"

# Chunks are measured in this model's tokens, so chunk limits match what the embedder sees.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# sentence-transformers truncates all-MiniLM-L6-v2 at 256 tokens by default, which would
# silently drop the end of chunks up to CHUNK_MAX_TOKENS long. Its architecture supports 512
# positions, so the limit is raised to embed every chunk in full. (The model was trained on
# shorter texts; comparing this against 256-token chunks is a planned ablation.)
EMBEDDING_MAX_SEQ_LENGTH = 512

CHUNK_MAX_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

TOP_K = 5

LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.0  # deterministic as possible: we want faithful answers, not creative ones


@dataclass(frozen=True)
class RAGConfig:
    """Every setting that affects a pipeline run, with defaults from the constants above."""

    embedding_model: str = EMBEDDING_MODEL
    embedding_max_seq_length: int = EMBEDDING_MAX_SEQ_LENGTH
    chunk_max_tokens: int = CHUNK_MAX_TOKENS
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS
    index_dir: Path = INDEX_DIR
    retriever: str = "dense"
    top_k: int = TOP_K
    llm_model: str = LLM_MODEL
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_temperature: float = LLM_TEMPERATURE
