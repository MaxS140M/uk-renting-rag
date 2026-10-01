# UK Renting Guidance Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about renting and
tenancy in England using official [GOV.UK](https://www.gov.uk) guidance, with citations back
to the source pages.

> **Status:** Phase 1 – document collection and chunking done. Retrieval not yet implemented.

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
GOV.UK pages ──► chunking ──► BM25 index  ─┐
                         └──► FAISS index ─┴─► hybrid retrieval ──► cross-encoder rerank
                                                                          │
                       FastAPI /ask ◄── answer + citations ◄── LLM ◄──────┘
```

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

# Build the dataset (about a minute; see data/README.md)
python scripts/download_docs.py
python scripts/chunk_corpus.py
```

## Evaluation

Coming soon. The plan is to measure retrieval (recall@k, MRR), compare the effect of the
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
- [ ] Implement hybrid retrieval (BM25 + dense embeddings)
- [ ] Add cross-encoder reranking
- [ ] Generate cited answers with an LLM
- [ ] Expose a FastAPI endpoint and containerise with Docker
- [ ] Build the evaluation suite and publish results
