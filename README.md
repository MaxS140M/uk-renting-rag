# UK Renting Guidance Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about renting and
tenancy in England using official [GOV.UK](https://www.gov.uk) guidance, with citations back
to the source pages.

> **Status:** Phase 3 – dense, BM25 and hybrid retrieval with an optional cross-encoder
> reranker, all switchable in config. Formal evaluation and the web API are next.

## Project description

Tenancy rules are spread across many GOV.UK pages and are hard to search. This project builds
a question-answering assistant that:

- retrieves relevant guidance with **hybrid search** (BM25 keyword + dense embeddings),
- re-orders candidates with a **cross-encoder reranker**,
- generates an answer with an **LLM that cites its sources**,
- is served through a **FastAPI** backend and packaged with **Docker**,
- is measured by a dedicated **evaluation suite**.

## Live demo

Coming soon.

## Results

Coming soon. Retrieval and answer-quality metrics will be reported here.

## Architecture

```
Offline:  GOV.UK pages ─► download ─► chunk ─┬─► embed (MiniLM) ─► FAISS index
                                             └─► tokenise ─────► BM25 index

Query:    question ─┬─► dense search (FAISS) ─┐
                    │                         ├─► RRF fusion ─► cross-encoder rerank ─► top 5
                    └─► BM25 keyword search ──┘   (top 20 each)  (20 candidates)        │
                                                                                         ▼
          AnswerResult ◄── citations parsed ◄── Claude Haiku 4.5 ◄── numbered passages
          (answer, cited sources, retrieved chunk IDs, per-stage latency, config)
```

Each stage is switchable in `RAGConfig`: `retrieval_mode` (`dense`, `bm25` or `hybrid`) and
`use_reranker`, plus `candidate_k`, `final_k` and `rrf_k`. The default is the Phase 2
baseline (dense, no reranker) until the evaluation shows which setup is best.

How it works:

- **Indexing** (`scripts/build_index.py`): each chunk is embedded with
  `all-MiniLM-L6-v2`. Vectors are normalised to unit length, so the inner product equals
  cosine similarity, and stored in an exact FAISS inner-product index. The embedding model
  name is saved with the index, and loading fails clearly if the query model differs. The
  same script tokenises every chunk for BM25 and saves the tokens as JSON (safe to load and
  independent of library versions, unlike a pickled object).
- **Retrieval** (`src/rag/retrieval.py`, `src/rag/rerank.py`): every retriever implements
  one interface, `retrieve(query, k) -> list[RetrievalResult]`, returning chunk IDs, scores,
  text, citation metadata and the chunk's rank at every stage. `build_retriever(config)`
  (`src/rag/factory.py`) assembles the right combination, so nothing else knows which is
  active. Retrieval runs without the LLM, so retrieval metrics are cheap.
  - **Dense** search finds passages with similar *meaning*, even with different wording.
  - **BM25** scores *exact term* overlap, which dense search blurs: numbers ("30 days"),
    form names ("Form 4A") and legal references. Its tokeniser keeps "section 21" together
    as one term as well as two words, keeps numbers, and folds simple plurals ("HMOs").
  - **Hybrid** merges the two candidate lists with Reciprocal Rank Fusion,
    `score = Σ 1 / (60 + rank)`. It uses ranks, not raw scores, because cosine similarity and
    BM25 scores are on incompatible scales.
  - **Reranker**: a cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads each (question, passage)
    pair together and rescores the top 20 candidates. More accurate than embedding
    similarity, but about 0.7 s per question on CPU, so it only sees a short list.
- **Generation** (`src/rag/generate.py`, prompt in `src/rag/prompts.py`): Claude answers only
  from the numbered passages, cites them as `[n]` with URLs, states the retrieval date, says
  "I can't find that in the guidance" instead of guessing, and ends with a not-legal-advice
  note. The prompt is versioned so results can be traced to it.
- **Pipeline** (`src/rag/pipeline.py`): `answer(question, config)` returns a structured
  `AnswerResult`. All settings live in one `RAGConfig` (`src/rag/config.py`).

| Path            | Purpose                                                    |
| --------------- | ---------------------------------------------------------- |
| `src/rag/`      | Core library: chunking, retrieval, reranking, generation   |
| `app/`          | FastAPI application                                        |
| `scripts/`      | One-off jobs: scraping, cleaning, building indexes         |
| `eval/`         | Evaluation datasets and scoring scripts                    |
| `notebooks/`    | Exploratory analysis                                       |
| `tests/`        | Unit tests (pytest)                                        |

## Data

The knowledge base is **47 GOV.UK guidance pages on renting in England** (about 155,000 words),
covering private renting, the Renters' Rights Act 2025, evictions, deposits, rent increases,
repairs and safety, HMOs, social housing and help with housing costs. See
[`data/README.md`](data/README.md) for the full details and
[`eval/corpus_overview.md`](eval/corpus_overview.md) for a per-document topic list.

- **Collection:** `scripts/download_docs.py` fetches each page from the GOV.UK Content API,
  which returns structured content without menus or footers, and handles multi-part guides
  and HTML publications.
- **Chunking:** `src/rag/chunking.py` splits documents into chunks of up to 400 tokens with
  up to 50 tokens of overlap, measured with the embedding model's tokenizer. Chunks break at
  headings and paragraphs where possible and never mid-sentence. Each chunk keeps its title,
  URL, section heading and retrieval date for citations.

Contains public sector information licensed under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Quick start

Requires Python 3.11+ and Git.

```bash
git clone https://github.com/MaxS140M/uk-renting-rag.git
cd uk-renting-rag

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install pinned dependencies, then this project as an editable package
pip install -r requirements.txt
pip install -e . --no-deps

# Configure secrets
cp .env.example .env             # then add your ANTHROPIC_API_KEY to .env

# Check everything works
ruff check .
pytest

# Build the dataset and index (a few minutes; see data/README.md)
python scripts/download_docs.py
python scripts/chunk_corpus.py
python scripts/build_index.py

# Ask a question
python scripts/ask.py "How long does my landlord have to protect my deposit?"

# Choose a retrieval setup
python scripts/ask.py "Can my landlord evict me without a reason?" --mode hybrid --rerank

# Inspect retrieval only (no LLM call, no API key needed)
python scripts/ask.py "Can my landlord evict me without a reason?" --retrieval-only --mode bm25

# Compare dense, BM25, hybrid and hybrid + reranker side by side (no LLM calls)
python scripts/compare_retrieval.py "How long does my landlord have to protect my deposit?"
```

Tests mock the LLM, so `pytest` needs no API key and makes no paid calls.

## Evaluation

A first manual check of the baseline is in
[`eval/baseline_smoke_test.md`](eval/baseline_smoke_test.md) (10 questions, including two the
guidance cannot answer), produced by `scripts/smoke_test.py`.

A qualitative comparison of the four retrieval setups on the same questions, with
latency, is in [`eval/phase3_comparison.md`](eval/phase3_comparison.md). A full evaluation is
coming. The plan is to measure retrieval (recall@k, MRR), compare the effect of the
reranker, and score answer faithfulness and citation accuracy against a labelled question set.

## Limitations

- **This is a demo, not legal advice.** Always check the official GOV.UK guidance or get
  professional advice for your situation.
- **Renting law in England has been changing.** The Renters' Rights Act 2025 replaced assured
  shorthold tenancies and abolished Section 21 "no-fault" evictions from 1 May 2026, and
  further changes are being phased in. Answers show the date each source was retrieved, so
  check GOV.UK for anything that may have changed since.
- Covers only the 47 indexed GOV.UK pages. Some official content (e.g. the Renters' Rights
  Act Information Sheet) is published only as a PDF and is not fully included.
- Covers England only; rules differ in Scotland, Wales and Northern Ireland.
- LLM answers can be wrong even when citations are provided.

## Next steps

- [x] Collect and clean GOV.UK renting guidance
- [x] Implement token-based chunking
- [x] Dense retrieval baseline with FAISS
- [x] Generate cited answers with an LLM
- [x] Add BM25 and hybrid retrieval (Reciprocal Rank Fusion)
- [x] Add cross-encoder reranking
- [ ] Expose a FastAPI endpoint and containerise with Docker
- [ ] Build the evaluation suite and publish results
