"""Use the LLM to propose draft evaluation questions for one document, for human review.

Drafts are a starting point, never test items: they are written to eval/testset/drafts.jsonl with
author "llm_draft", and only become test items after review and editing in
scripts/testset/add_question.py --from-draft. Every draft quote is checked against the document,
because LLMs often paraphrase when asked to quote.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from rag.evaluation.schema import EvalItem, next_id
from rag.evaluation.utils import quote_count

DRAFT_SYSTEM_PROMPT = """\
You help build an evaluation set for a question-answering system over UK GOV.UK guidance on \
renting in England. Given one document, propose questions that a tenant or landlord might \
realistically ask and that this document answers.

For each question, return:
- "question": the question as a person would ask it.
- "question_type": "factual" (one fact from one passage), "informal" (casual, vague or \
messy wording, as a worried tenant might type it), or "multi_passage" (needs two separate \
passages of the document combined).
- "quotes": the passage(s) that support the answer, copied EXACTLY, word for word, as one \
contiguous piece of the document each. One quote for factual and informal questions, two or \
more for multi_passage. Each quote must be at least one full sentence.
- "reference_answer": a short, correct answer based only on the quotes.

Vary the question types and cover different parts of the document. Return only a JSON \
array of objects, with no other text."""


@dataclass
class DraftResult:
    drafts: list[EvalItem]
    rejected: list[str]  # why each unusable proposal was dropped


def build_draft_prompt(doc: Mapping, n: int) -> str:
    return (
        f"Propose {n} questions for this document.\n\n"
        f'<document title="{doc["title"]}" doc_id="{doc["doc_id"]}">\n{doc["text"]}\n</document>'
    )


def parse_json_array(text: str) -> list:
    """Extract the JSON array from a reply, tolerating any text around it."""
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError("reply contains no JSON array")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("reply is not a JSON array")
    return data


def proposals_to_drafts(
    proposals: list, doc: Mapping, existing_drafts: Sequence[EvalItem] = ()
) -> DraftResult:
    """Validate proposals as EvalItems with author 'llm_draft', flagging unverified quotes."""
    drafts: list[EvalItem] = []
    rejected: list[str] = []
    for n, proposal in enumerate(proposals, start=1):
        if not isinstance(proposal, dict):
            rejected.append(f"proposal {n}: not an object")
            continue
        quotes = proposal.get("quotes") or []
        missing = [q for q in quotes if quote_count(str(q), doc["text"]) == 0]
        notes = "LLM draft: review and edit before use."
        if missing:
            notes += f" {len(missing)} quote(s) not found verbatim in the document; fix them."
        try:
            draft = EvalItem(
                id=next_id([*existing_drafts, *drafts], "llm_draft"),
                question=proposal.get("question", ""),
                reference_answer=proposal.get("reference_answer", ""),
                question_type=proposal.get("question_type"),
                evidence=[{"doc_id": doc["doc_id"], "quote": str(q)} for q in quotes],
                author="llm_draft",
                notes=notes,
            )
        except ValidationError as err:
            reasons = "; ".join(e["msg"] for e in err.errors())
            rejected.append(f"proposal {n} ({proposal.get('question', '?')!r}): {reasons}")
            continue
        if draft.question_type == "unanswerable":
            rejected.append(f"proposal {n}: unanswerable drafts are not accepted")
            continue
        drafts.append(draft)
    return DraftResult(drafts, rejected)


def draft_questions(
    doc: Mapping,
    client,
    model: str,
    n: int = 5,
    existing_drafts: Sequence[EvalItem] = (),
    max_tokens: int = 4096,
) -> DraftResult:
    """Ask the LLM for ``n`` draft questions about ``doc`` and validate them."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=DRAFT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_draft_prompt(doc, n)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    if response.stop_reason == "max_tokens":
        raise ValueError(f"reply was cut off at max_tokens={max_tokens}; ask for fewer drafts")
    return proposals_to_drafts(parse_json_array(text), doc, existing_drafts)
