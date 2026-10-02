"""Tests for quote matching, evidence-to-chunk mapping, near-duplicates and splitting."""

import json
from pathlib import Path

import pytest

from rag.chunking import chunk_document
from rag.evaluation.schema import EvalItem
from rag.evaluation.utils import (
    EvidenceMapper,
    corpus_fingerprint,
    document_passages,
    find_similar_questions,
    map_evidence_to_chunks,
    normalise,
    quote_count,
    stratified_sample,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample"
DEPOSIT_QUOTE = (
    "Your landlord or letting agent must put your deposit in the scheme within 30 days of "
    "getting it."
)


@pytest.fixture(scope="module")
def deposit_doc() -> dict:
    return json.loads((SAMPLE / "tenancy-deposit-protection.json").read_text(encoding="utf-8"))


def item(evidence, question_type="factual", item_id="q-001", question="A test question?"):
    return EvalItem(
        id=item_id,
        question=question,
        reference_answer="Answer.",
        question_type=question_type,
        evidence=[{"doc_id": d, "quote": q} for d, q in evidence],
        author="max",
    )


# --- Normalisation and quote matching -------------------------------------------------------


def test_normalise_collapses_whitespace_headings_and_curly_quotes():
    text = "## Holding deposits\n\nYour landlord’s   duty\n- item"
    assert normalise(text) == "Holding deposits Your landlord's duty - item"


def test_quote_matches_across_line_breaks_and_apostrophe_styles(deposit_doc):
    assert quote_count(DEPOSIT_QUOTE, deposit_doc["text"]) == 1
    reflowed = DEPOSIT_QUOTE.replace(" in the scheme ", "\n  in the scheme\n")
    assert quote_count(reflowed, deposit_doc["text"]) == 1


def test_quote_matching_is_otherwise_exact(deposit_doc):
    assert quote_count(DEPOSIT_QUOTE.replace("30 days", "28 days"), deposit_doc["text"]) == 0
    assert quote_count(DEPOSIT_QUOTE.lower(), deposit_doc["text"]) == 0


# --- Evidence to chunks ---------------------------------------------------------------------


@pytest.mark.parametrize(("max_tokens", "overlap"), [(64, 16), (128, 16), (256, 32), (400, 50)])
def test_gold_evidence_survives_rechunking(deposit_doc, max_tokens, overlap):
    chunks = [
        c.to_dict()
        for c in chunk_document(deposit_doc, max_tokens=max_tokens, overlap_tokens=overlap)
    ]
    test_item = item([("tenancy-deposit-protection", DEPOSIT_QUOTE)])
    chunk_ids = map_evidence_to_chunks(test_item, chunks)

    assert chunk_ids, f"quote not found with {max_tokens}-token chunks"
    by_id = {c["chunk_id"]: c for c in chunks}
    for chunk_id in chunk_ids:
        assert by_id[chunk_id]["doc_id"] == "tenancy-deposit-protection"
        assert normalise(DEPOSIT_QUOTE) in normalise(by_id[chunk_id]["text"])


def test_chunk_ids_differ_between_chunkings_but_labels_still_map(deposit_doc):
    small = [c.to_dict() for c in chunk_document(deposit_doc, max_tokens=64, overlap_tokens=16)]
    large = [c.to_dict() for c in chunk_document(deposit_doc, max_tokens=400, overlap_tokens=50)]
    test_item = item([("tenancy-deposit-protection", DEPOSIT_QUOTE)])
    small_ids, large_ids = (
        map_evidence_to_chunks(test_item, small),
        map_evidence_to_chunks(test_item, large),
    )
    assert small_ids and large_ids and not set(small_ids) & set(large_ids)


def test_quote_spanning_a_boundary_maps_to_the_chunk_holding_most_of_it():
    sentences = [f"Rule {i} says the landlord must complete task number {i}." for i in range(30)]
    doc = {
        "doc_id": "rules",
        "title": "Rules",
        "url": "https://www.gov.uk/rules",
        "date_retrieved": "2026-10-01",
        "text": " ".join(sentences),
    }
    chunks = [c.to_dict() for c in chunk_document(doc, max_tokens=60, overlap_tokens=0)]
    first_chunk_sentences = chunks[0]["text"].split(". ")
    last_in_first = first_chunk_sentences[-1].rstrip(".") + "."
    index = sentences.index(last_in_first)
    # One sentence from the end of chunk 0 plus two from the start of chunk 1.
    quote = " ".join(sentences[index : index + 3])
    assert not any(quote in c["text"] for c in chunks)  # precondition: really spans a boundary

    mapped = map_evidence_to_chunks(item([("rules", quote)]), chunks)
    assert mapped == [chunks[1]["chunk_id"]]


def test_multi_passage_items_map_each_quote(deposit_doc):
    chunks = [c.to_dict() for c in chunk_document(deposit_doc, max_tokens=64, overlap_tokens=0)]
    other = "Your landlord must return your deposit within 10 days of you both agreeing"
    test_item = item(
        [("tenancy-deposit-protection", DEPOSIT_QUOTE), ("tenancy-deposit-protection", other)],
        question_type="multi_passage",
    )
    per_quote = EvidenceMapper(chunks).map_each(test_item)
    assert len(per_quote) == 2 and all(per_quote)
    assert set(per_quote[0]) != set(per_quote[1])


def test_quote_from_another_document_maps_to_nothing(deposit_doc):
    chunks = [c.to_dict() for c in chunk_document(deposit_doc)]
    assert map_evidence_to_chunks(item([("some-other-doc", DEPOSIT_QUOTE)]), chunks) == []


# --- Passages, fingerprint, duplicates, split -----------------------------------------------


def test_document_passages_carry_heading_paths(deposit_doc):
    passages = document_passages(deposit_doc)
    assert all(not p["text"].startswith("#") for p in passages)
    holding = next(p for p in passages if p["text"].startswith("Your landlord does not have"))
    assert holding["section"] == "Overview > Holding deposits"
    assert any(DEPOSIT_QUOTE in p["text"] for p in passages)


def test_corpus_fingerprint_changes_when_text_changes(deposit_doc):
    docs = {"a": deposit_doc}
    edited = {"a": {**deposit_doc, "text": deposit_doc["text"] + " Edited."}}
    assert corpus_fingerprint(docs) == corpus_fingerprint(dict(docs))
    assert corpus_fingerprint(docs) != corpus_fingerprint(edited)


def test_exact_and_near_duplicates_are_flagged_but_distinct_questions_are_not():
    items = [
        item(
            [("d", DEPOSIT_QUOTE)],
            item_id="q-001",
            question="How long does my landlord have to protect my deposit?",
        ),
        item(
            [("d", DEPOSIT_QUOTE)],
            item_id="q-002",
            question="how long does my landlord have to protect my deposit",
        ),
        item(
            [("d", DEPOSIT_QUOTE)],
            item_id="q-003",
            question="How long has my landlord got to protect my deposit?",
        ),
        item(
            [("d", DEPOSIT_QUOTE)],
            item_id="q-004",
            question="Can a letting agent charge me for a viewing?",
        ),
    ]
    pairs = {(p.first, p.second): p for p in find_similar_questions(items)}
    assert pairs[("q-001", "q-002")].exact
    assert ("q-001", "q-003") in pairs and not pairs[("q-001", "q-003")].exact
    assert not any("q-004" in key for key in pairs)


def make_pool(counts: dict[str, int]) -> list[EvalItem]:
    pool, n = [], 0
    for question_type, count in counts.items():
        for _ in range(count):
            n += 1
            evidence = [] if question_type == "unanswerable" else [("d", DEPOSIT_QUOTE)]
            if question_type == "multi_passage":
                evidence.append(("d", "A second quote that is long enough."))
            pool.append(item(evidence, question_type, f"q-{n:03d}", f"Question number {n}?"))
    return pool


def test_stratified_sample_keeps_type_proportions():
    pool = make_pool({"factual": 60, "multi_passage": 20, "informal": 10, "unanswerable": 10})
    chosen = stratified_sample(pool, 20, seed=42)
    by_id = {i.id: i for i in pool}
    counts = {}
    for item_id in chosen:
        counts[by_id[item_id].question_type] = counts.get(by_id[item_id].question_type, 0) + 1
    assert counts == {"factual": 12, "multi_passage": 4, "informal": 2, "unanswerable": 2}


def test_stratified_sample_is_deterministic_for_a_seed():
    pool = make_pool({"factual": 30, "multi_passage": 10, "unanswerable": 7})
    assert stratified_sample(pool, 10, seed=1) == stratified_sample(
        list(reversed(pool)), 10, seed=1
    )
    assert stratified_sample(pool, 10, seed=1) != stratified_sample(pool, 10, seed=2)
    assert len(stratified_sample(pool, 10, seed=1)) == 10


def test_stratified_sample_rejects_impossible_sizes():
    with pytest.raises(ValueError):
        stratified_sample(make_pool({"factual": 3}), 5, seed=1)
