# Dataset

47 GOV.UK guidance pages about renting in England (about 155,000 words): private renting,
the Renters' Rights Act 2025, evictions and possession notices, deposits, rent increases and
disputes, repairs and safety, HMOs, social housing, and help with housing costs.

The full page list is in [`scripts/sources.txt`](../scripts/sources.txt) and a per-document
topic inventory is in [`eval/corpus_overview.md`](../eval/corpus_overview.md).

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
| `raw/` | No | One JSON file per page, written by `scripts/download_docs.py` |
| `chunks.jsonl` | No | One chunk per line, written by `scripts/chunk_corpus.py` |
| `sample/` | Yes | Three example documents, showing the format; also used by the tests |

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
python scripts/download_docs.py        # fetch pages into data/raw/ (skips existing files)
python scripts/download_docs.py --refresh   # re-download everything
python scripts/chunk_corpus.py         # write data/chunks.jsonl
python scripts/summarise_corpus.py     # regenerate eval/corpus_overview.md
```

The downloader sends a descriptive User-Agent, waits one second between requests, retries
rate-limited and server errors with exponential backoff, and logs every page that fails.

## Known gaps

- *The Renters' Rights Act Information Sheet 2026* is published only as a PDF, so the dataset
  contains its landing page summary rather than the full sheet.
- *Guide to the Renters' Rights Act* was written while the Act was a bill (last updated
  November 2025), so parts describe changes in the future tense.
- Only England is covered. Renting law differs in Scotland, Wales and Northern Ireland.
