# Evaluation test set

> [!WARNING]
> **Held-out rule.** Items with `"split": "heldout"` are run **once, at the very end**, to
> report final results. They are never used to choose settings, tune prompts, or debug.
> All development and tuning uses the `dev` split only. If held-out items influence any
> decision, they stop measuring how the system performs on unseen questions.

`questions.jsonl` is a hand-written test set of questions about renting in England, used to
measure both retrieval (does the system find the right passages?) and generation (is the
answer correct, faithful to its sources, and does it refuse when it should?).

## Files

| File | Contents |
| --- | --- |
| `questions.jsonl` | The test set: one item per line |
| `coverage.md` | Progress against the target mix, and documents with no questions yet |
| `corpus_overview.md` | Titles and headings of every document, used to plan questions |
| `smoke_questions.json` | 10 informal smoke-test questions (not part of the test set) |
| `drafts.jsonl` | Unreviewed LLM drafts (git-ignored; never used directly) |

## Item format

```json
{"id": "q-001", "question": "...", "reference_answer": "...", "question_type": "factual",
 "evidence": [{"doc_id": "tenancy-deposit-protection", "quote": "exact text from the document"}],
 "split": "dev", "author": "max", "notes": ""}
```

| Field | Meaning |
| --- | --- |
| `id` | `q-NNN` for test items, `ex-NNN` for format examples, `draft-NNN` for drafts |
| `question` | The question, as a user would ask it |
| `reference_answer` | A correct answer, written from the evidence |
| `question_type` | `factual`, `multi_passage`, `informal` or `unanswerable` (below) |
| `evidence` | The passages that support the answer: a document id and an exact quote |
| `split` | `dev` (development and tuning) or `heldout` (final evaluation only) |
| `author` | `max` (written by hand) or `llm_draft_reviewed` (LLM draft, reviewed and edited) |
| `notes` | Free text, e.g. which draft an item came from |

The schema (`src/rag/eval_schema.py`) also enforces the rules between fields: unanswerable
items have no evidence; every other type needs at least one quote; multi-passage items need
at least two different quotes; quotes are at least 20 characters; unknown fields are errors.

## Question types and target mix

| Type | Target | What it tests |
| --- | ---: | --- |
| `factual` | 60 | One fact from one passage: the core retrieval and answering task |
| `multi_passage` | 20 | Needs two or more passages combined, e.g. a rule and its exception |
| `informal` | 10 | Casual or vague wording ("can my landlord just kick me out?"), unlike the document's wording |
| `unanswerable` | 10 | Plausible questions the guidance does not answer; the system should refuse |

Most questions are factual because single-fact lookup is what users mostly need, and it gives
the most stable retrieval metrics. Multi-passage questions test whether retrieval finds all
the needed passages. Informal questions counter a known bias: questions written while reading
a document tend to reuse its wording, which flatters keyword search. Unanswerable questions
measure refusal: an assistant giving legal information must say "I can't find that" rather
than guess.

## How the questions are written

1. **By hand, from the documents.** Questions are written with `scripts/add_question.py`,
   which searches the corpus by keyword, shows matching paragraphs, and lets the author select
   the exact supporting sentences, so every quote is copied from the document, not retyped.
2. **LLM drafts are optional and always reviewed.** `scripts/draft_questions.py` asks the LLM
   to propose questions for one document. Drafts go to `drafts.jsonl` (never to the test set)
   with author `llm_draft`. Each is reviewed with `add_question.py --from-draft draft-NNN`,
   where the question, type, quotes and answer are checked and edited; the saved item has
   author `llm_draft_reviewed`, and its notes record which draft it came from. Draft quotes
   that are not found word for word in the document are flagged, because LLMs often
   paraphrase when asked to quote.
3. **Unanswerable questions are checked against the corpus**, not just one document:
   `validate_questions.py --show-unanswerable` shows the top passages from the strongest
   retriever, to confirm none of them answers the question.

## Why evidence is a quote, not a chunk id

Retrieval metrics (Recall@5, MRR) need to know which chunks are correct for each question.
Chunk ids change whenever the chunking changes (chunk size is one of the settings compared in
the evaluation), so labels stored as chunk ids would be wrong for every chunking but one.

Instead, each item stores the **document id and an exact quote**. At evaluation time,
`EvidenceMapper` (`src/rag/eval_utils.py`) finds the chunks that contain each quote under
whatever chunking that run uses:

- matching ignores differences in whitespace, Markdown heading markers and curly vs straight
  quotes, but is otherwise exact;
- if a quote crosses a chunk boundary, any chunk containing at least half of it, as one
  continuous piece, counts as correct.

One set of labels therefore scores every chunking fairly. The tests chunk the same document
at 64, 128, 256 and 400 tokens and check that the same label maps correctly each time.

## Frozen corpus

Quotes are only valid against the exact text they were copied from, and GOV.UK edits its
pages. The corpus is therefore committed as a frozen snapshot (`data/raw/`, retrieved
2026-10-01), and `validate_questions.py` prints a fingerprint of it (currently
`c1f26012c3e5`) so results can record which version they used. Updating the corpus is a
deliberate change: re-download, validate, fix broken quotes, commit together.

## The held-out split

20 items are held out, **stratified by question type**: each type keeps the same share in
the held-out set as in the whole set (with the target mix: 12 factual, 4 multi-passage,
2 informal, 2 unanswerable). With only 20 items, a purely random draw could easily include
0 or 4 unanswerable questions, which would make held-out refusal scores meaningless.
Stratifying keeps held-out scores comparable with dev scores.

The split is made by `scripts/split_questions.py` with a fixed seed (42), so it is
reproducible. It is assigned **once**, when the set is complete, and the script refuses to
change an existing held-out set: re-splitting later could move an item that was already used
for tuning into the held-out set.

## Validation

`scripts/validate_questions.py` runs in CI on every push and fails if:

- any line does not match the schema, or two items share an id;
- an evidence document does not exist, or a quote is not found in it (the closest text in
  the document is shown, to make fixing easy);
- a quote does not map to any chunk under the current chunking;
- two questions are identical apart from case and punctuation;
- an unreviewed LLM draft is in the test set.

It warns (without failing, unless `--strict`) about near-duplicate questions, which may be
deliberate, such as an informal rewording of a factual question; quotes that appear more than
once in a document; and unanswerable items whose reference answer doesn't say the guidance
doesn't cover the question.

## Workflow

```bash
python scripts/add_question.py                    # write a question
python scripts/draft_questions.py --doc private-renting   # optional: LLM drafts to review
python scripts/add_question.py --from-draft draft-001     # review a draft
python scripts/validate_questions.py              # check everything (also runs in CI)
python scripts/question_coverage.py               # progress and gaps -> coverage.md
python scripts/split_questions.py --dry-run       # once the 100 are written, preview...
python scripts/split_questions.py                 # ...and assign the held-out split
```

## Limitations

- **One author.** All questions and reference answers are written by one person, so they
  reflect one reading of the guidance; there is no inter-annotator agreement measure.
- **Written from the documents.** Questions written while reading the corpus tend to match
  its wording; the informal type exists partly to counter this.
- **Point in time.** Reference answers describe the guidance as retrieved on 2026-10-01.
- **Examples.** The `ex-` items in `questions.jsonl` only show the format. They are excluded
  from every metric and will be replaced by real questions.
