# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [1.0.1] - 2026-10-02

Repository reorganised; no change to behaviour or results.

### Changed
- Library: evaluation modules moved into the `rag.evaluation` package (`schema`, `utils`,
  `runner`, `metrics`, `judges`, `testset_generation`, `drafting`, `authoring`).
- Scripts grouped by purpose into `scripts/data/`, `scripts/testset/`,
  `scripts/evaluation/` (now including `run_eval.py`) and `scripts/deploy/`.
- `eval/` split into `testset/`, `judge/` (files renamed without the `judge_` prefix),
  `results/` and `early_checks/`.
- `DEPLOY.md` and the demo GIF moved into `docs/`; the README shows the demo GIF.
- The demo page shows its cold-start notice only when hosted on Hugging Face.

## [1.0.0] - 2026-10-02

First release: a deployable, evaluated question-answering assistant for renting in England,
built in seven phases.

### Phase 0: project setup
- Python 3.11 project with pinned dependencies, ruff linting and formatting, pytest, and a
  GitHub Actions workflow running both on every push.

### Phase 1: documents and chunking
- Downloader for 47 GOV.UK pages via the Content API, handling multi-part guides and HTML
  publications, with polite rate limiting, retries and a rerunnable cache.
- Token-based chunking (about 400 tokens, 50-token overlap) that breaks at headings,
  paragraphs and sentences, never mid-sentence, with stable chunk IDs.

### Phase 2: baseline pipeline
- FAISS index over normalised sentence-transformer embeddings and a dense retriever behind a
  common `Retriever` interface.
- Grounded answers from Claude Haiku 4.5 with numbered citations, the retrieval date, a
  refusal when the guidance does not cover the question, and a not-legal-advice note.
- Structured `answer()` result with cited sources, retrieved chunk IDs and per-stage latency;
  command-line tool with a retrieval-only mode; LLM mocked in every test.

### Phase 3: better retrieval
- BM25 keyword retriever with a tokeniser that keeps legal references ("section 21") and
  numbers intact.
- Hybrid retrieval with Reciprocal Rank Fusion, and a cross-encoder reranker that can wrap
  any retriever; every option switchable in configuration through a retriever factory.

### Phase 4: evaluation tooling
- Test-set schema with gold evidence stored as exact quotes (not chunk IDs), mapped to chunks
  at run time so labels work for any chunk size.
- Frozen, committed corpus snapshot; validator run in CI; authoring and review tools;
  coverage report; stratified held-out split with a fixed seed.

### Phase 5: evaluation and ablations
- 130-question test set generated in two steps (questions from topics only; evidence found
  in the full corpus), verified automatically, not reviewed by a person.
- Evaluation runner with Recall@1/5, MRR, evidence recall, LLM-judged faithfulness and
  correctness, refusal accuracy and latency; on-disk cache for every LLM call; cost estimates
  before spending.
- Seven configurations compared one change at a time, with paired bootstrap confidence
  intervals; judge checked against human labels (blind, then by reviewing disagreements).
- Final configuration `hybrid_rerank_bge` chosen on the dev split and run once on the
  held-out split. Results in `eval/RESULTS.md`, failures in `eval/ERROR_ANALYSIS.md`.

### Phase 6: deployment
- FastAPI service: `POST /ask` and `GET /health`, input validation, friendly JSON errors,
  models loaded once at startup.
- Abuse and cost protection: per-IP rate limit, daily cap on LLM calls, LLM timeout and low
  `max_tokens`; prompt v2 separates the user's question from the instructions; question text
  and IP addresses are never logged.
- Gradio demo page mounted on the same server, with example questions, numbered source
  links, the retrieval date, a "how it works" panel and a privacy notice.
- Production Dockerfile (slim base, non-root user, cached dependency layer, models and index
  baked in, offline at run time); CI builds the image and checks it starts with no network.
- Deployment guide for Hugging Face Spaces and other Docker hosts; MIT licence.
