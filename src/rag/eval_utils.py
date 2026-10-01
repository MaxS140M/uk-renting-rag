"""Helpers for the evaluation set: quote checking, mapping evidence to chunks, passages for
authoring, near-duplicate detection and the stratified held-out split.

Gold evidence is (doc_id, exact quote). ``EvidenceMapper`` turns it into chunk_ids for
whatever chunking a run uses, so one set of labels scores every chunk-size ablation.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from rag.eval_schema import EvalItem, Evidence
from rag.retrieval import tokenize

# --- Text normalisation and quote matching ----------------------------------------------------

_TYPOGRAPHIC = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalise(text: str) -> str:
    """Normalise text for quote matching without changing its words.

    Collapses all whitespace (line breaks in a document vs spaces in a pasted quote), removes
    Markdown heading markers (documents keep '## ', chunks do not) and straightens curly
    quotes. Case and punctuation are kept, so a match is still an exact quote.
    """
    text = text.translate(_TYPOGRAPHIC)
    text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_count(quote: str, text: str) -> int:
    """How many times a quote appears in a text, after normalisation."""
    return normalise(text).count(normalise(quote))


def load_corpus(raw_dir: Path) -> dict[str, dict]:
    """Load every document in a directory of JSON files, keyed by doc_id."""
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.json"))]
    return {d["doc_id"]: d for d in docs}


def corpus_fingerprint(docs: Mapping[str, Mapping]) -> str:
    """Short hash of the corpus text, to record which snapshot an evaluation ran against."""
    digest = hashlib.sha256()
    for doc_id in sorted(docs):
        digest.update(doc_id.encode() + b"\0" + docs[doc_id]["text"].encode() + b"\0")
    return digest.hexdigest()[:12]


# --- Evidence to chunks -----------------------------------------------------------------------


def _field(chunk, name: str):
    return chunk[name] if isinstance(chunk, Mapping) else getattr(chunk, name)


class EvidenceMapper:
    """Map evidence quotes to the chunk_ids that contain them, for one chunking of the corpus.

    A chunk matches if it contains the whole quote. If no chunk does (the quote spans a chunk
    boundary), every chunk containing at least ``min_fraction`` of the quote as one
    contiguous piece matches instead.
    """

    def __init__(self, chunks: Iterable, min_fraction: float = 0.5) -> None:
        self.min_fraction = min_fraction
        self.by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for chunk in chunks:
            self.by_doc[_field(chunk, "doc_id")].append(
                (_field(chunk, "chunk_id"), normalise(_field(chunk, "text")))
            )

    def map_evidence(self, evidence: Evidence) -> list[str]:
        quote = normalise(evidence.quote)
        candidates = self.by_doc.get(evidence.doc_id, [])
        exact = [chunk_id for chunk_id, text in candidates if quote in text]
        if exact:
            return exact
        partial = []
        for chunk_id, text in candidates:
            match = SequenceMatcher(None, quote, text, autojunk=False).find_longest_match(
                0, len(quote), 0, len(text)
            )
            if match.size / len(quote) >= self.min_fraction:
                partial.append(chunk_id)
        return partial

    def map_each(self, item: EvalItem) -> list[list[str]]:
        """Matching chunk_ids for each evidence quote separately (for per-quote recall)."""
        return [self.map_evidence(e) for e in item.evidence]

    def map_item(self, item: EvalItem) -> list[str]:
        """All chunk_ids matching any of the item's evidence, without duplicates."""
        seen: dict[str, None] = {}
        for chunk_ids in self.map_each(item):
            seen.update(dict.fromkeys(chunk_ids))
        return list(seen)


def map_evidence_to_chunks(
    item: EvalItem, chunks: Iterable, min_fraction: float = 0.5
) -> list[str]:
    """Return the chunk_ids whose text contains the item's evidence (see EvidenceMapper)."""
    return EvidenceMapper(chunks, min_fraction).map_item(item)


# --- Passages for authoring -------------------------------------------------------------------


def document_passages(doc: Mapping) -> list[dict]:
    """Split a document into paragraph-level passages, each tagged with its heading path.

    Used to search for evidence while writing questions. Passages follow the document's own
    paragraphs, not any chunking, so quotes chosen from them are chunking-independent.
    The dicts have the same fields as chunks, so the BM25 retriever can search them.
    """
    passages, headings = [], []
    for block in re.split(r"\n\s*\n", doc["text"]):
        block = block.strip()
        heading = re.match(r"^(#{1,6})\s+(.+)$", block)
        if heading and "\n" not in block:
            level = len(heading.group(1))
            headings = [h for h in headings if h[0] < level] + [(level, heading.group(2))]
            continue
        if block:
            passages.append(
                {
                    "chunk_id": f"{doc['doc_id']}#p{len(passages):03d}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "section": " > ".join(h for _, h in headings),
                    "date_retrieved": doc["date_retrieved"],
                    "text": block,
                }
            )
    return passages


# --- Near-duplicate questions -----------------------------------------------------------------


@dataclass(frozen=True)
class SimilarPair:
    first: str
    second: str
    similarity: float
    exact: bool


def question_similarity(a: str, b: str) -> float:
    """Similarity of two questions in [0, 1]: the higher of character-level similarity
    (catches rewording of the same sentence) and content-word overlap (catches reordering)."""
    a_norm, b_norm = normalise(a).lower(), normalise(b).lower()
    char_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    a_terms, b_terms = set(tokenize(a)), set(tokenize(b))
    jaccard = len(a_terms & b_terms) / len(a_terms | b_terms) if a_terms | b_terms else 0.0
    return max(char_ratio, jaccard)


def _comparable(question: str) -> str:
    """A question with case and punctuation removed, for exact-duplicate checks."""
    return re.sub(r"[^\w\s]", "", normalise(question).lower()).strip()


def find_similar_questions(items: Sequence[EvalItem], threshold: float = 0.85) -> list[SimilarPair]:
    """Pairs of questions that are identical (ignoring case and punctuation) or nearly so."""
    pairs = []
    for i, first in enumerate(items):
        for second in items[i + 1 :]:
            exact = _comparable(first.question) == _comparable(second.question)
            score = 1.0 if exact else question_similarity(first.question, second.question)
            if exact or score >= threshold:
                pairs.append(SimilarPair(first.id, second.id, round(score, 3), exact))
    return pairs


# --- Stratified held-out split ----------------------------------------------------------------


def stratified_sample(items: Sequence[EvalItem], n: int, seed: int) -> set[str]:
    """Pick ``n`` item ids, keeping each question type's share the same as in ``items``.

    Allocation uses the largest-remainder method (e.g. 60/20/10/10 items and n=20 gives
    12/4/2/2), then picks randomly within each type with a fixed seed, so the same input
    always gives the same split.
    """
    if not 0 < n <= len(items):
        raise ValueError(f"n must be between 1 and {len(items)}")
    by_type: dict[str, list[str]] = defaultdict(list)
    for item in sorted(items, key=lambda i: i.id):
        by_type[item.question_type].append(item.id)

    exact = {t: n * len(ids) / len(items) for t, ids in by_type.items()}
    counts = {t: int(share) for t, share in exact.items()}
    by_remainder = sorted(exact, key=lambda t: (-(exact[t] - counts[t]), t))
    for t in by_remainder[: n - sum(counts.values())]:
        counts[t] += 1

    rng = random.Random(seed)
    chosen: set[str] = set()
    for question_type in sorted(by_type):
        chosen.update(rng.sample(by_type[question_type], counts[question_type]))
    return chosen


def type_counts(items: Iterable[EvalItem]) -> Counter:
    return Counter(i.question_type for i in items)
