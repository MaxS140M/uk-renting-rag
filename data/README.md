# Dataset

47 GOV.UK guidance pages about renting in England (about 155,000 words): private renting,
the Renters' Rights Act 2025, evictions and possession notices, deposits, rent increases and
disputes, repairs and safety, HMOs, social housing, and help with housing costs.

The full page list is in [`scripts/data/sources.txt`](../scripts/data/sources.txt) and a per-document
topic inventory is in [`eval/testset/corpus_overview.md`](../eval/testset/corpus_overview.md).

## Source and licence

All documents come from [GOV.UK](https://www.gov.uk) via the public
[Content API](https://content-api.publishing.service.gov.uk/). GOV.UK content is Crown
copyright and published under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/),
which allows reuse provided the source is attributed:

> Contains public sector information licensed under the Open Government Licence v3.0.

## Layout

| Path | Committed? | Contents |
| --- | --- | --- |
| `raw/` | **Yes** (frozen snapshot) | One JSON file per page, written by `scripts/data/download_docs.py` |
| `chunks.jsonl` | No | One chunk per line, written by `scripts/data/chunk_corpus.py` |
| `index/` | No | FAISS and BM25 indexes, written by `scripts/data/build_index.py` |
| `sample/` | Yes | Three small documents used by the unit tests |

## Frozen snapshot

`raw/` is committed as a **frozen snapshot** of the corpus, retrieved on 2026-10-01. The
evaluation set (`eval/testset/questions.jsonl`) cites exact quotes from these documents, and GOV.UK
edits its pages over time, so the test set is only valid against a fixed version of the
text. Committing the snapshot means anyone who clones the repo evaluates against the same
documents, and CI can check that every quote still exists.

Updating the corpus is therefore a deliberate change, not a side effect: re-download with
`--refresh`, review the git diff, run `python scripts/testset/validate_questions.py` to find quotes
that no longer match, fix them, and commit everything together.

Each document is a JSON object:

| Field | Description |
| --- | --- |
| `doc_id` | Stable ID: the last segment of the GOV.UK path |
| `title` | Page title |
| `url` | Canonical GOV.UK URL, used for citations |
| `text` | Plain text body. Headings are kept as Markdown `#` lines; list items start with `- ` |
| `date_retrieved` | Date the page was downloaded (UTC) |
| `last_updated` | GOV.UK's `public_updated_at`: the date of the last *major* change |
| `document_type` | GOV.UK format, e.g. `guide`, `detailed_guide`, `guidance` |
| `licence` | Always `Open Government Licence v3.0` |

`last_updated` is the date GOV.UK displays, but editors often make minor changes without
updating it. Several guides last marked a major change before 2020, yet already describe the
Renters' Rights Act changes from 1 May 2026. Treat `date_retrieved` as the reliable "as of" date.

## Rebuilding the dataset

```bash
python scripts/data/download_docs.py        # fetch any pages missing from data/raw/
python scripts/data/download_docs.py --refresh   # re-download everything (changes the snapshot)
python scripts/data/chunk_corpus.py         # write data/chunks.jsonl
python scripts/data/summarise_corpus.py     # regenerate eval/testset/corpus_overview.md
```

The downloader sends a descriptive User-Agent, waits one second between requests, retries
rate-limited and server errors with exponential backoff, and logs every page that fails.

## Known gaps

- *The Renters' Rights Act Information Sheet 2026* is published only as a PDF, so the dataset
  contains its landing page summary rather than the full sheet.
- *Guide to the Renters' Rights Act* was written while the Act was a bill (last updated
  November 2025), so parts describe changes in the future tense.
- Only England is covered. Renting law differs in Scotland, Wales and Northern Ireland.
