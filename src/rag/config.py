"""Central settings (model names, chunk sizes, paths) shared by the library and scripts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

# Chunks are measured in this model's tokens, so chunk limits match what the embedder sees.
# Note: sentence-transformers truncates this model's input at 256 tokens by default
# (max_seq_length), although its architecture supports 512. Chunks larger than 256 tokens
# need model.max_seq_length raised at embedding time, or they will be silently cut short.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_MAX_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
