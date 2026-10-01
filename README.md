# UK Renting Guidance Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about renting and
tenancy in England using official [GOV.UK](https://www.gov.uk) guidance, with citations back
to the source pages.

> **Status:** Phase 0 – project scaffold. RAG pipeline not yet implemented.

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

## Quick start

Requires Python 3.11+ and Git.

```bash
git clone https://github.com/<your-username>/uk-renting-rag.git
cd uk-renting-rag

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt

# Configure secrets
cp .env.example .env             # then add your ANTHROPIC_API_KEY to .env

# Check everything works
ruff check .
pytest
```

## Evaluation

Coming soon. The plan is to measure retrieval (recall@k, MRR), compare the effect of the
reranker, and score answer faithfulness and citation accuracy against a labelled question set.

## Limitations

- **This is a demo, not legal advice.** Always check the official GOV.UK guidance or get
  professional advice for your situation.
- Covers only the GOV.UK pages that have been indexed, which may be out of date.
- Guidance mostly applies to England; rules differ in Scotland, Wales and Northern Ireland.
- LLM answers can be wrong even when citations are provided.

## Next steps

- [ ] Scrape and clean GOV.UK renting guidance
- [ ] Implement chunking and hybrid retrieval
- [ ] Add cross-encoder reranking
- [ ] Generate cited answers with an LLM
- [ ] Expose a FastAPI endpoint and containerise with Docker
- [ ] Build the evaluation suite and publish results
