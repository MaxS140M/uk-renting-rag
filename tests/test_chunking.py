"""Tests for token-based document chunking, run against the committed sample documents."""

import json
from pathlib import Path

import pytest

from rag.chunking import chunk_document, chunk_documents, get_token_counter

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture(scope="module")
def sample_docs() -> list[dict]:
    paths = sorted(SAMPLE_DIR.glob("*.json"))
    assert paths, f"no sample documents found in {SAMPLE_DIR}"
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


@pytest.fixture(scope="module")
def count_tokens():
    return get_token_counter()


def make_doc(text: str) -> dict:
    return {
        "doc_id": "test-doc",
        "title": "Test document",
        "url": "https://www.gov.uk/test-doc",
        "date_retrieved": "2026-10-01",
        "text": text,
    }


def test_every_chunk_has_citation_metadata(sample_docs):
    chunks = chunk_documents(sample_docs)
    assert chunks
    for chunk in chunks:
        assert chunk.title
        assert chunk.url.startswith("https://www.gov.uk/")
        assert chunk.date_retrieved
        assert chunk.text.strip()


@pytest.mark.parametrize(("max_tokens", "overlap"), [(128, 16), (256, 32), (400, 50)])
def test_no_chunk_exceeds_max_tokens(sample_docs, count_tokens, max_tokens, overlap):
    chunks = chunk_documents(sample_docs, max_tokens=max_tokens, overlap_tokens=overlap)
    for chunk in chunks:
        # Recount from the text itself rather than trusting the stored n_tokens.
        assert count_tokens(chunk.text) <= max_tokens, chunk.chunk_id


def test_oversized_sentence_is_split_within_limit(count_tokens):
    long_sentence = " ".join(["tenancy"] * 300) + "."
    chunks = chunk_document(make_doc(long_sentence), max_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    sentences = [f"Rule number {i} says the landlord must do task {i}." for i in range(40)]
    chunks = chunk_document(make_doc(" ".join(sentences)), max_tokens=80, overlap_tokens=30)
    assert len(chunks) > 2
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_sentence = current.text.split(". ")[0]
        assert first_sentence in previous.text, "next chunk should start with repeated text"


def test_zero_overlap_shares_no_text():
    sentences = [f"Rule number {i} says the landlord must do task {i}." for i in range(40)]
    chunks = chunk_document(make_doc(" ".join(sentences)), max_tokens=80, overlap_tokens=0)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.text.split(". ")[0] not in previous.text


def test_chunks_never_end_mid_sentence():
    sentences = [f"Rule number {i} says the landlord must do task {i}." for i in range(40)]
    chunks = chunk_document(make_doc(" ".join(sentences)), max_tokens=80, overlap_tokens=30)
    assert all(c.text.endswith(".") for c in chunks)


def test_heading_starts_new_chunk_once_current_is_half_full():
    # The Deposits section (~70 tokens) fills over half of a 120-token chunk, so the
    # Repairs heading should start a fresh chunk instead of being packed in after it.
    deposits = " ".join(f"Deposit fact {i} applies to every tenancy." for i in range(8))
    repairs = " ".join(f"Repair fact {i} applies to every tenancy." for i in range(30))
    text = f"# Deposits\n\n{deposits}\n\n# Repairs\n\n{repairs}"
    chunks = chunk_document(make_doc(text), max_tokens=120, overlap_tokens=20)

    assert chunks[0].section == "Deposits"
    assert "Repair" not in chunks[0].text
    repair_chunks = [c for c in chunks if c.section == "Repairs"]
    assert len(repair_chunks) == len(chunks) - 1
    assert repair_chunks[0].text.startswith("Repairs\n\n")
    assert all("Deposit" not in c.text for c in repair_chunks)


def test_chunk_ids_are_stable_and_unique(sample_docs):
    first = [c.chunk_id for c in chunk_documents(sample_docs)]
    second = [c.chunk_id for c in chunk_documents(sample_docs)]
    assert first == second
    assert len(set(first)) == len(first)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_document(make_doc("Some text."), max_tokens=50, overlap_tokens=50)
