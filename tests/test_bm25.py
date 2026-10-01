"""Tests for the BM25 tokeniser and retriever."""

import json

import pytest

from rag.retrieval import BM25Retriever, save_bm25, tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Section 21 notice", ["section", "21", "notice", "section_21"]),
        ("You must use Form 4A", ["must", "use", "form", "4a", "form_4a"]),
        ("ground 8 and Ground 1A", ["ground", "8", "ground", "1a", "ground_8", "ground_1a"]),
    ],
)
def test_legal_references_stay_intact(text, expected):
    assert tokenize(text) == expected


def test_numbers_are_kept_and_thousands_separators_removed():
    tokens = tokenize("A fine of up to £7,000 within 30 days, or 2.5% interest")
    assert {"7000", "30", "2.5"} <= set(tokens)


def test_case_and_punctuation_are_normalised():
    assert tokenize("HMO! (Licence), hmo?") == ["hmo", "licence", "hmo"]


def test_acronym_plurals_and_possessives_fold_together():
    assert tokenize("HMOs") == tokenize("HMO") == ["hmo"]
    assert tokenize("tenancies") == ["tenancy"]
    assert tokenize("landlord's deposits") == ["landlord", "deposit"]
    assert tokenize("landlord’s") == ["landlord"]  # curly apostrophe


def test_words_ending_in_s_that_are_not_plurals_are_kept():
    assert tokenize("process status analysis gas") == ["process", "status", "analysis", "gas"]


def test_stopwords_removed_but_negations_and_modals_kept():
    assert tokenize("Does the landlord have to do it?") == ["landlord"]
    assert tokenize("You must not and cannot") == ["must", "not", "cannot"]


def test_multiword_terms_match_term_by_term():
    assert tokenize("Tenancy Deposit Scheme") == ["tenancy", "deposit", "scheme"]


# --- Retriever ------------------------------------------------------------------------------


@pytest.fixture
def bm25(chunks) -> BM25Retriever:
    return BM25Retriever(chunks)


def test_exact_terms_rank_the_matching_chunk_first(bm25):
    assert bm25.retrieve("licensed house in multiple occupation", 3)[0].chunk_id == "hmo-000"


def test_results_sorted_with_ranks_and_metadata(bm25, chunks):
    results = bm25.retrieve("landlord deposit repairs", 3)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    by_id = {c["chunk_id"]: c for c in chunks}
    for r in results:
        assert r.scored_by == "bm25"
        assert r.ranks == {"bm25": r.rank}
        assert r.url == by_id[r.chunk_id]["url"]


def test_chunks_sharing_no_terms_with_the_query_are_not_returned(bm25):
    assert bm25.retrieve("zebra giraffe", 5) == []


def test_timings_are_recorded(bm25):
    timings = {}
    bm25.retrieve("deposit", 2, timings)
    assert "bm25" in timings and timings["bm25"] >= 0


def test_saved_index_round_trip(tmp_path, chunks):
    (tmp_path / "chunks.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8"
    )
    save_bm25(tmp_path, chunks)
    loaded = BM25Retriever.from_index_dir(tmp_path)
    query = "letting agents fees"
    assert [r.chunk_id for r in loaded.retrieve(query, 3)] == [
        r.chunk_id for r in BM25Retriever(chunks).retrieve(query, 3)
    ]


def test_missing_bm25_index_explains_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        BM25Retriever.from_index_dir(tmp_path)
