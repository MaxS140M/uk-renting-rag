"""Retrieval behind a common interface: given a query, return the top-k chunks with scores.

Every retriever (dense, BM25, hybrid, and the reranker in rerank.py) implements
``Retriever``, so the pipeline and evaluation code never need to know which one is active.
Retrievers can be stacked: a hybrid retriever wraps a dense and a BM25 retriever, and a
reranker can wrap any retriever.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag.indexing import Embedder, LoadedIndex

BM25_FILE = "bm25.json"


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk: its ID, score and rank, plus everything needed to cite it.

    ``ranks`` records the chunk's rank at every stage it passed through, e.g.
    {"dense": 3, "bm25": 1, "hybrid": 1, "rerank": 2}, so results can be analysed later.
    ``score`` is from the final stage named in ``scored_by``; scores from different stages
    are on different scales and must not be compared with each other.
    """

    chunk_id: str
    score: float
    rank: int  # 1 = best
    text: str
    doc_id: str
    title: str
    url: str
    section: str
    date_retrieved: str
    scored_by: str = ""
    ranks: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: dict, score: float, rank: int, scored_by: str) -> RetrievalResult:
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
            scored_by=scored_by,
            ranks={scored_by: rank},
        )

    def to_dict(self) -> dict:
        return asdict(self)


def record_time(timings: dict[str, float] | None, stage: str, start: float) -> None:
    """Add the milliseconds since ``start`` to ``timings[stage]``, if timings are wanted."""
    if timings is not None:
        elapsed = (time.perf_counter() - start) * 1000
        timings[stage] = round(timings.get(stage, 0.0) + elapsed, 1)


class Retriever(ABC):
    """Common interface for all retrievers."""

    name: str

    @abstractmethod
    def retrieve(
        self, query: str, k: int, timings: dict[str, float] | None = None
    ) -> list[RetrievalResult]:
        """Return up to ``k`` results, best first (highest score, rank 1).

        If ``timings`` is given, the retriever adds its stage latency in milliseconds to it
        (e.g. timings["bm25"]). Passing the dict down the chain, rather than storing timings
        on the retriever, keeps retrievers safe to share between concurrent requests.
        """


# --- Dense ------------------------------------------------------------------------------------


class DenseRetriever(Retriever):
    """Semantic search: embed the query and find the nearest chunk vectors in FAISS."""

    name = "dense"

    def __init__(self, loaded: LoadedIndex, embedder: Embedder) -> None:
        if loaded.meta["embedding_model"] != embedder.model_name:
            raise ValueError("index and query embedder use different models")
        self.index = loaded.index
        self.chunks = loaded.chunks
        self.embedder = embedder

    def retrieve(
        self, query: str, k: int, timings: dict[str, float] | None = None
    ) -> list[RetrievalResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        start = time.perf_counter()
        k = min(k, self.index.ntotal)
        query_vector = np.ascontiguousarray(self.embedder.encode([query]), dtype=np.float32)
        scores, rows = self.index.search(query_vector, k)
        results = [
            RetrievalResult.from_chunk(self.chunks[row], score, rank, self.name)
            for rank, (score, row) in enumerate(zip(scores[0], rows[0], strict=True), start=1)
            if row != -1
        ]
        record_time(timings, self.name, start)
        return results


# --- BM25 -------------------------------------------------------------------------------------

TOKENIZER_VERSION = "v1"  # bump when tokenize() changes, so stale saved indexes are rejected

# Very common words that carry no topic information. Negations ("no", "not", "cannot") and
# modal verbs ("must", "can", "should") are deliberately kept: they matter in legal text.
_STOPWORDS = frozenset(
    "a an and are as at be been but by did do does doing for from had has have how i if in "
    "into is it its me my "
    "of on or our so that the their them then there these they this to was we were what "
    "when where which who why will with you your".split()
)
# Legal references whose word and number belong together, e.g. "section 21", "form 4a".
_LEGAL_REF_RE = re.compile(
    r"\b(section|ground|form|schedule|part|regulation|article|clause)\s+(\d+[a-z]?)\b"
)
_TOKEN_RE = re.compile(r"\d+\.\d+|[a-z0-9]+(?:'[a-z]+)?")


def _normalise(token: str) -> str:
    """Strip possessives and fold simple plurals, so 'HMOs' matches 'HMO'."""
    if token.endswith("'s"):
        token = token[:-2]
    if not token.isalpha():  # leave numbers and codes like "4a" untouched
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"  # tenancies -> tenancy
    if token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
        return token[:-1]  # deposits -> deposit, hmos -> hmo (but not "process", "status")
    return token


def tokenize(text: str) -> list[str]:
    """Split text into BM25 terms: lowercased, punctuation removed, numbers kept.

    Legal references also produce a combined token ("section 21" -> "section", "21",
    "section_21"), because BM25 treats text as a bag of words: without it, a query about
    section 21 would match any passage containing "section" and "21 days".
    """
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # "£7,000" -> "7000"
    legal_refs = [f"{word}_{number}" for word, number in _LEGAL_REF_RE.findall(text)]
    words = [_normalise(t) for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS]
    return [w for w in words if w and w not in _STOPWORDS] + legal_refs


def bm25_document(chunk: Mapping) -> str:
    """The text BM25 indexes for a chunk: its title and section heading plus its body.

    Titles are short, precise topic labels ("Tenancy deposit protection"), which is exactly
    the kind of exact-term evidence BM25 is good at using.
    """
    return " ".join(filter(None, [chunk["title"], chunk.get("section", ""), chunk["text"]]))


def save_bm25(index_dir: Path, chunks: Sequence[Mapping]) -> None:
    """Tokenise every chunk once at build time and save the tokens next to the FAISS index.

    The tokenised corpus is saved rather than a pickled BM25 object: it is plain JSON (safe to
    load, unlike pickle), independent of the rank-bm25 version, and computing BM25's term
    statistics from it at startup takes well under a second.
    """
    data = {
        "tokenizer_version": TOKENIZER_VERSION,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "tokens": [tokenize(bm25_document(c)) for c in chunks],
    }
    (index_dir / BM25_FILE).write_text(json.dumps(data), encoding="utf-8")


class BM25Retriever(Retriever):
    """Keyword search: rank chunks by BM25 term-overlap score with the query."""

    name = "bm25"

    def __init__(
        self,
        chunks: Sequence[Mapping],
        tokens: Sequence[list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = list(chunks)
        tokens = tokens if tokens is not None else [tokenize(bm25_document(c)) for c in chunks]
        if len(tokens) != len(self.chunks):
            raise ValueError("need exactly one token list per chunk")
        # k1 controls how quickly repeated terms stop adding score; b how much long chunks
        # are penalised. These are rank-bm25's standard defaults.
        self.bm25 = BM25Okapi(tokens, k1=k1, b=b)

    @classmethod
    def from_index_dir(cls, index_dir: Path) -> BM25Retriever:
        path = index_dir / BM25_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"No BM25 index in {index_dir}. Build it with: python scripts/build_index.py"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["tokenizer_version"] != TOKENIZER_VERSION:
            raise ValueError(
                f"BM25 index uses tokenizer {data['tokenizer_version']}, but the code uses "
                f"{TOKENIZER_VERSION}. Rebuild it with: python scripts/build_index.py"
            )
        lines = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [json.loads(line) for line in lines if line]
        if [c["chunk_id"] for c in chunks] != data["chunk_ids"]:
            raise ValueError("BM25 index and chunk file are out of sync; rebuild the index")
        return cls(chunks, data["tokens"])

    def retrieve(
        self, query: str, k: int, timings: dict[str, float] | None = None
    ) -> list[RetrievalResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        start = time.perf_counter()
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:k]
        # A chunk with score <= 0 shares no informative terms with the query, so it is not
        # returned: it is not a match, just the least-bad non-match.
        results = [
            RetrievalResult.from_chunk(self.chunks[i], scores[i], rank, self.name)
            for rank, i in enumerate((i for i in order if scores[i] > 0), start=1)
        ]
        record_time(timings, self.name, start)
        return results


# --- Hybrid (Reciprocal Rank Fusion) ----------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked lists of IDs: score(id) = sum over lists of 1 / (rrf_k + rank in list).

    Only ranks are used, never raw scores, because scores from different retrievers are on
    incompatible scales (cosine similarity in [-1, 1], BM25 unbounded). An ID missing from a
    list simply gets nothing from it. Ties are broken by best single rank, then by ID, so the
    output order is deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked_ids in rankings.values():
        for rank, item in enumerate(ranked_ids, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


class HybridRetriever(Retriever):
    """Run dense and BM25 retrieval, then merge their candidates with Reciprocal Rank Fusion."""

    name = "hybrid"

    def __init__(
        self,
        dense: Retriever,
        bm25: Retriever,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.retrievers = {"dense": dense, "bm25": bm25}
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, k: int, timings: dict[str, float] | None = None
    ) -> list[RetrievalResult]:
        if k < 1:
            raise ValueError("k must be at least 1")
        pool = max(k, self.candidate_k)
        candidates = {
            name: retriever.retrieve(query, pool, timings)
            for name, retriever in self.retrievers.items()
        }

        start = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {name: [r.chunk_id for r in results] for name, results in candidates.items()},
            self.rrf_k,
        )
        # Deduplicate by chunk_id: keep one result object per chunk, and collect the rank it
        # had in every list it appeared in.
        by_id: dict[str, RetrievalResult] = {}
        ranks: dict[str, dict[str, int]] = {}
        for name, results in candidates.items():
            for r in results:
                by_id.setdefault(r.chunk_id, r)
                ranks.setdefault(r.chunk_id, {})[name] = r.rank
        results = [
            replace(
                by_id[chunk_id],
                score=score,
                rank=rank,
                scored_by=self.name,
                ranks={**ranks[chunk_id], self.name: rank},
            )
            for rank, (chunk_id, score) in enumerate(fused[:k], start=1)
        ]
        record_time(timings, "fusion", start)
        return results
