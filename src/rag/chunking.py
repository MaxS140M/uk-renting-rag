"""Split cleaned GOV.UK guidance pages into overlapping, citation-ready text chunks.

Chunk sizes are measured in tokens of the embedding model's own tokenizer, not words, so a
chunk's size is exactly what the embedder sees. Documents are split at the most natural
boundary available:

1. headings (a new section starts a new chunk once the current one is at least half full),
2. paragraphs (a paragraph that does not fit starts a new chunk),
3. sentences and list items (only when a single paragraph is too large).

Text is never cut mid-sentence, unless one sentence alone exceeds the maximum chunk size.
Consecutive chunks within a section share up to ``overlap_tokens`` of whole sentences, so a
fact that straddles a boundary is fully contained in at least one chunk.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache

from rag.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, EMBEDDING_MODEL

TokenCounter = Callable[[str], int]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
# A sentence ends at . ! or ? followed by whitespace and a capital, digit or opening quote.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'‘“(]?[A-Z0-9])")


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document, carrying everything needed to cite its source."""

    chunk_id: str
    doc_id: str
    title: str
    url: str
    section: str
    date_retrieved: str
    text: str
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Piece:
    """The smallest unit the packer moves around: a sentence, list item or heading line."""

    text: str
    n_tokens: int
    sep: str  # inserted before this piece when joined: "\n\n" new block, "\n" new line, " "
    section: str
    is_heading: bool = False


@lru_cache(maxsize=4)
def get_token_counter(model_name: str = EMBEDDING_MODEL) -> TokenCounter:
    """Return a function counting tokens with the given Hugging Face model's tokenizer."""
    from transformers import AutoTokenizer  # imported lazily because it is slow to load

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count_tokens


def split_sentences(text: str) -> list[str]:
    """Split a line of prose into sentences."""
    return [s for s in _SENTENCE_BOUNDARY_RE.split(text.strip()) if s]


def _split_oversized(text: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    """Split a single sentence longer than ``max_tokens`` at word boundaries (last resort)."""
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in text.split():
        n = count_tokens(word)
        if current and current_tokens + n > max_tokens:
            parts.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += n
    if current:
        parts.append(" ".join(current))
    return parts


def _parse_blocks(text: str, max_tokens: int, count_tokens: TokenCounter) -> list[list[_Piece]]:
    """Turn document text into blocks (heading lines or paragraphs) made of pieces."""
    blocks: list[list[_Piece]] = []
    headings: list[tuple[int, str]] = []  # stack of (level, heading) currently in force

    for raw_block in re.split(r"\n\s*\n", text):
        block_text = raw_block.strip()
        if not block_text:
            continue

        match = _HEADING_RE.match(block_text)
        if match and "\n" not in block_text:
            level, heading = len(match.group(1)), match.group(2).strip()
            headings = [h for h in headings if h[0] < level] + [(level, heading)]
            section = " > ".join(h for _, h in headings)
            blocks.append([_Piece(heading, count_tokens(heading), "\n\n", section, True)])
            continue

        section = " > ".join(h for _, h in headings)
        pieces: list[_Piece] = []
        for line_no, line in enumerate(block_text.split("\n")):
            for sent_no, sentence in enumerate(split_sentences(line)):
                n = count_tokens(sentence)
                parts = (
                    _split_oversized(sentence, max_tokens, count_tokens)
                    if n > max_tokens
                    else [sentence]
                )
                for part_no, part in enumerate(parts):
                    if not pieces:
                        sep = "\n\n"
                    elif line_no > 0 and sent_no == 0 and part_no == 0:
                        sep = "\n"
                    else:
                        sep = " "
                    pieces.append(
                        _Piece(part, n if len(parts) == 1 else count_tokens(part), sep, section)
                    )
        if pieces:
            blocks.append(pieces)
    return blocks


def _pack(blocks: list[list[_Piece]], max_tokens: int, overlap_tokens: int) -> list[list[_Piece]]:
    """Greedily pack pieces into chunks, preferring heading then paragraph boundaries."""
    chunks: list[list[_Piece]] = []
    current: list[_Piece] = []
    n_overlap = 0  # number of leading pieces in `current` copied from the previous chunk

    def tokens(pieces: list[_Piece]) -> int:
        return sum(p.n_tokens for p in pieces)

    def flush(overlap: bool) -> None:
        nonlocal current, n_overlap
        # Never end a chunk on a heading: carry trailing headings into the next chunk.
        carry: list[_Piece] = []
        while current[n_overlap:] and current[-1].is_heading:
            carry.insert(0, current.pop())
        body = current
        if len(body) > n_overlap:  # only emit if there is new (non-overlap) content
            chunks.append(body)
        tail: list[_Piece] = []
        if overlap and not carry and len(body) > n_overlap:
            for piece in reversed(body):
                if piece.is_heading or tokens(tail) + piece.n_tokens > overlap_tokens:
                    break
                tail.insert(0, piece)
        current = tail + carry
        n_overlap = len(tail)

    def add(piece: _Piece) -> None:
        nonlocal current, n_overlap
        if tokens(current) + piece.n_tokens > max_tokens:
            flush(overlap=True)
            if tokens(current) + piece.n_tokens > max_tokens:
                current, n_overlap = [], 0  # overlap does not fit alongside this piece
        current.append(piece)

    half_full = max_tokens // 2
    for block in blocks:
        if block[0].is_heading and tokens(current[n_overlap:]) >= half_full:
            flush(overlap=False)  # start each sufficiently large section on a new chunk
        elif (
            tokens(current) + tokens(block) > max_tokens
            and tokens(current[n_overlap:]) >= half_full
        ):
            flush(overlap=True)  # end the chunk at a paragraph boundary
        for piece in block:
            add(piece)
    flush(overlap=False)
    return chunks


def _render(pieces: list[_Piece]) -> str:
    text = pieces[0].text
    for piece in pieces[1:]:
        text += piece.sep + piece.text
    return text.strip()


def chunk_document(
    doc: dict,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Split one document (as saved by scripts/download_docs.py) into chunks.

    Args:
        doc: dict with at least doc_id, title, url, text and date_retrieved.
        max_tokens: hard upper limit on tokens per chunk.
        overlap_tokens: maximum tokens of whole sentences repeated between consecutive chunks.
        count_tokens: token counting function; defaults to the embedding model's tokenizer.
    """
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be at least 0 and less than max_tokens")
    count_tokens = count_tokens or get_token_counter()

    blocks = _parse_blocks(doc["text"], max_tokens, count_tokens)
    chunks = []
    for index, pieces in enumerate(_pack(blocks, max_tokens, overlap_tokens)):
        text = _render(pieces)
        # Content hash in the ID: stable across runs, but changes if GOV.UK edits the text,
        # so stale references (e.g. in evaluation labels) are easy to detect.
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        first_new = next((p for p in pieces if not p.is_heading), pieces[0])
        chunks.append(
            Chunk(
                chunk_id=f"{doc['doc_id']}-{index:03d}-{digest}",
                doc_id=doc["doc_id"],
                title=doc["title"],
                url=doc["url"],
                section=first_new.section,
                date_retrieved=doc["date_retrieved"],
                text=text,
                n_tokens=count_tokens(text),
            )
        )
    return chunks


def chunk_documents(docs: Iterable[dict], **kwargs) -> list[Chunk]:
    """Chunk many documents with the same settings (see ``chunk_document``)."""
    return [chunk for doc in docs for chunk in chunk_document(doc, **kwargs)]
