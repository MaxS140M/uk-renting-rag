"""Central settings (models, chunk sizes, retrieval and generation parameters, paths).

Module-level constants are the defaults. ``RAGConfig`` bundles them into one immutable
object that fully describes a run, so an experiment can change a setting with
``dataclasses.replace(config, retrieval_mode="hybrid")`` without editing code.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GITHUB_URL = "https://github.com/MaxS140M/uk-renting-rag"
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

# Retrieval. These library defaults are the Phase 2 baseline (dense only, no reranker),
# kept so earlier results stay reproducible. The deployed configuration, chosen by the
# Phase 5 evaluation, is the named experiment `hybrid_rerank_bge` in eval/configs.yaml:
# the demo uses it, and scripts accept --config hybrid_rerank_bge.
RETRIEVAL_MODES = ("dense", "bm25", "hybrid")
RETRIEVAL_MODE = "dense"
USE_RERANKER = False
CANDIDATE_K = 20  # candidates fetched from each retriever before fusion / reranking
FINAL_K = 5  # passages passed to the LLM
RRF_K = 60  # Reciprocal Rank Fusion constant, from Cormack et al. (2009)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE = "cpu"

LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 1024
# Low temperature: we want faithful, repeatable answers, not creative ones. Set to None for
# models that reject the parameter (Claude Opus 4.7 and later).
LLM_TEMPERATURE: float | None = 0.0

# Test-set drafting uses a stronger model than the answering model, so the questions and
# reference answers are not limited by the system being evaluated.
DRAFT_MODEL = "claude-opus-5-5"
DRAFT_EFFORT = "high"  # careful evidence-finding matters more than speed here
LLM_CACHE_DIR = DATA_DIR / "llm_cache"

# LLM-as-judge for faithfulness and correctness: a stronger model than the one answering.
# Its verdicts are checked against human labels (scripts/judge_agreement.py) before use.
JUDGE_MODEL = "claude-opus-5-5"
JUDGE_EFFORT = "medium"


@dataclass(frozen=True)
class RAGConfig:
    """Every setting that affects a pipeline run, with defaults from the constants above."""

    embedding_model: str = EMBEDDING_MODEL
    embedding_max_seq_length: int = EMBEDDING_MAX_SEQ_LENGTH
    chunk_max_tokens: int = CHUNK_MAX_TOKENS
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS
    index_dir: Path = INDEX_DIR
    retrieval_mode: str = RETRIEVAL_MODE
    use_reranker: bool = USE_RERANKER
    candidate_k: int = CANDIDATE_K
    final_k: int = FINAL_K
    rrf_k: int = RRF_K
    reranker_model: str = RERANKER_MODEL
    reranker_device: str = RERANKER_DEVICE
    llm_model: str = LLM_MODEL
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_temperature: float | None = LLM_TEMPERATURE

    def __post_init__(self) -> None:
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"retrieval_mode must be one of {RETRIEVAL_MODES}, not '{self.retrieval_mode}'"
            )
        if self.final_k < 1:
            raise ValueError("final_k must be at least 1")
        if self.candidate_k < self.final_k:
            raise ValueError("candidate_k must be at least final_k")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must not be negative")

    @property
    def label(self) -> str:
        """Short human-readable name for this retrieval setup, e.g. 'hybrid+rerank'."""
        return self.retrieval_mode + ("+rerank" if self.use_reranker else "")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["index_dir"] = str(self.index_dir)
        return data
