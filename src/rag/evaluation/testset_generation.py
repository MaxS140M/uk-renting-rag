"""Generate a draft evaluation set in two separate steps, for human review.

Step 1 writes questions from a tenant persona, seeing only the list of topics the guidance
covers (eval/testset/corpus_overview.md) and never the passages. Questions written while reading
a passage tend to reuse its exact wording, which unfairly favours keyword search; real
tenants describe their problem in their own words.

Step 2, in separate calls, gives the model the whole corpus and asks it to find exact
evidence quotes for each question and write a reference answer from those quotes only. The
evidence is found by reading the full corpus, not with this project's retriever: using the
system under test to find the gold evidence would drop every question it fails on, and
inflate its scores.

Every output is a draft (author "llm_draft") for review in scripts/testset/review_drafts.py.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from rag.evaluation.schema import EvalItem, next_id
from rag.evaluation.utils import question_similarity, quote_count
from rag.llm_cache import LLMCache, LLMReply, streaming_sender

DEFAULT_MIX = {"factual": 80, "multi_passage": 25, "informal": 12, "unanswerable": 13}
COMMON_TOPICS = (
    "deposits, repairs, rent increases, evictions and notice periods, tenancy agreements, "
    "letting fees, moving out, HMOs and shared houses, landlord access to the property, "
    "rights and responsibilities of tenants and landlords"
)
QUESTIONS_PER_EVIDENCE_CALL = 10

# --- Step 1: questions from a persona ---------------------------------------------------------

PERSONA_SYSTEM = f"""\
You are helping build a realistic test set for an assistant that answers questions about \
renting a home in England. Imagine the many different people who would use it: private \
tenants (students, families, people on benefits, older renters, flat sharers), some \
social housing tenants, and a few small landlords.

You have NOT read the guidance the assistant uses. You only know the list of topics it \
covers, which is given below. Write questions the way these people would really ask them, \
in their own words, about their own situations. Do not copy wording from the topic list.

Focus on the most COMMON real-life questions, spread across: {COMMON_TOPICS}. Each question \
must be self-contained (no "my previous question") and different from the others."""

TYPE_BRIEFS = {
    "factual": (
        "Each asks about ONE specific fact: a deadline, an amount, a notice period, who is "
        "responsible for something, or whether something is allowed."
    ),
    "multi_passage": (
        "Each needs TWO OR MORE separate pieces of information combined to answer fully, for "
        "example a rule and how to challenge it, a notice period and what happens after it "
        "ends, or the situation for tenants who started before and after a change in the law."
    ),
    "informal": (
        "Each is phrased casually, the way a stressed person types into a phone: lower case, "
        "slang, a typo or two, vague or emotional wording (e.g. 'can my landlord just kick me "
        "out??', 'landlord wont give my deposit back what do i do'). Still a real question "
        "with a real answer about renting in England."
    ),
    "unanswerable": (
        "Each is a plausible question a renter might ask that this guidance does NOT answer, "
        "judging from the topic list: for example tenancy law in Scotland, Wales or Northern "
        "Ireland, commercial or business leases, rent levels in a particular town, buying a "
        "home, or very specific legal disputes. They must sound like the other questions, not "
        "obviously off-topic."
    ),
}

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "topic": {"type": "string"}},
                "required": ["question", "topic"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


def topic_list(corpus_overview_md: str) -> str:
    """Document titles and headings from eval/testset/corpus_overview.md (no URLs or counts)."""
    lines = []
    for line in corpus_overview_md.splitlines():
        if line.startswith("## ") and re.match(r"## \d+\. ", line):
            lines.append("\n" + re.sub(r"^## \d+\. ", "Topic: ", line))
        elif re.match(r"^\s*- ", line) and "more headings" not in line:
            lines.append(line)
    return "\n".join(lines).strip()


def question_request(model: str, effort: str, topics: str, question_type: str, n: int) -> dict:
    return {
        "model": model,
        "max_tokens": 16000,
        "system": f"{PERSONA_SYSTEM}\n\n<topics>\n{topics}\n</topics>",
        "messages": [
            {
                "role": "user",
                "content": f"Write {n} questions. {TYPE_BRIEFS[question_type]} For each, give "
                "the question and a short topic label.",
            }
        ],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": QUESTION_SCHEMA},
        },
    }


# --- Step 2: evidence from the whole corpus ---------------------------------------------------

EVIDENCE_INSTRUCTIONS = """\
You are building the answer key for a test set. The documents below are the complete \
guidance an assistant uses to answer questions about renting in England. For each question \
you are given, decide whether these documents answer it, and if so find the evidence.

Rules:
- Quotes must be copied EXACTLY, character for character, from one document, including \
punctuation and curly apostrophes, as one contiguous piece of text. Each quote should be \
one or more full sentences: as short as possible but enough to support the answer.
- Use the quotes that most directly answer the question. A question marked multi_passage \
needs at least two separate quotes; if one passage answers it fully, give one quote and \
say so in the comment.
- Write the reference answer using ONLY the information in your quotes, in plain English, \
in one to three sentences. Do not add anything from your own knowledge.
- If the documents do not answer the question, set answerable to false, give no quotes, \
and write a reference answer saying the guidance does not cover it. Be strict for \
questions marked unanswerable: only mark them answerable if the documents clearly answer \
them. If the documents answer only part of a question, mark it answerable, quote that \
part, and explain what is missing in the comment.
- Keep the comment to one sentence."""

EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "answerable": {"type": "boolean"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["doc_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                    "reference_answer": {"type": "string"},
                    "comment": {"type": "string"},
                },
                "required": ["id", "answerable", "evidence", "reference_answer", "comment"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def corpus_block(docs: Mapping[str, Mapping]) -> str:
    return "\n\n".join(
        f'<document doc_id="{d["doc_id"]}" title="{d["title"]}">\n{d["text"]}\n</document>'
        for d in (docs[k] for k in sorted(docs))
    )


def evidence_request(
    model: str, effort: str, docs: Mapping[str, Mapping], batch: Sequence[dict]
) -> dict:
    listing = "\n".join(f"{q['key']} [{q['intended_type']}]: {q['question']}" for q in batch)
    return {
        "model": model,
        "max_tokens": 32000,
        # The corpus is identical in every evidence call, so it is cached (1 hour) and only
        # paid for in full once; later calls read it at a tenth of the price or less.
        "system": [
            {"type": "text", "text": EVIDENCE_INSTRUCTIONS},
            {
                "type": "text",
                "text": f"<documents>\n{corpus_block(docs)}\n</documents>",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": "Find the evidence for each of these questions. Use the id before "
                f"each question.\n\n{listing}",
            }
        ],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": EVIDENCE_SCHEMA},
        },
    }


# --- Calling the model ------------------------------------------------------------------------


def parse_reply(reply: LLMReply) -> dict:
    if reply.stop_reason == "max_tokens":
        raise ValueError("reply was cut off at max_tokens")
    if reply.stop_reason == "refusal":
        raise ValueError("the model declined the request")
    return json.loads(reply.text)


# --- Assembling drafts ------------------------------------------------------------------------


@dataclass
class GenerationResult:
    drafts: list[EvalItem] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    dropped_duplicates: list[str] = field(default_factory=list)
    replies: list[LLMReply] = field(default_factory=list)


def dedupe_questions(
    questions: list[dict], threshold: float = 0.85
) -> tuple[list[dict], list[str]]:
    """Drop near-duplicate generated questions, keeping the first of each similar group."""
    kept: list[dict] = []
    dropped: list[str] = []
    for q in questions:
        match = next(
            (k for k in kept if question_similarity(q["question"], k["question"]) >= threshold),
            None,
        )
        if match:
            dropped.append(f"{q['question']!r} (similar to {match['question']!r})")
        else:
            kept.append(q)
    return kept, dropped


def build_draft(
    question: dict,
    result: dict,
    docs: Mapping[str, Mapping],
    existing: Sequence[EvalItem],
    model: str,
) -> EvalItem:
    intended = question["intended_type"]
    evidence = [e for e in result["evidence"] if e["quote"].strip()] if result["answerable"] else []
    notes = [f"Persona-generated as {intended} (topic: {question['topic']}); evidence by {model}."]

    if not evidence:
        question_type = "unanswerable"
        if intended != "unanswerable":
            notes.append("Generated as answerable, but no evidence was found: check the corpus.")
    elif intended == "unanswerable":
        question_type = "factual"
        notes.append("Generated as unanswerable, but the corpus answers it: retyped as factual.")
    elif intended == "multi_passage" and len({e["quote"] for e in evidence}) < 2:
        question_type = "factual"
        notes.append("Generated as multi_passage, but one passage answers it: retyped as factual.")
    else:
        question_type = intended

    unverified = [
        e
        for e in evidence
        if e["doc_id"] not in docs or quote_count(e["quote"], docs[e["doc_id"]]["text"]) == 0
    ]
    if unverified:
        notes.append(f"{len(unverified)} quote(s) not found verbatim in the document: fix them.")
    if result.get("comment"):
        notes.append(f"Model comment: {result['comment']}")

    return EvalItem(
        id=next_id(existing, "llm_draft"),
        question=question["question"],
        reference_answer=result["reference_answer"],
        question_type=question_type,
        evidence=evidence,
        author="llm_draft",
        notes=" ".join(notes),
    )


def generate_test_set(
    client,
    cache: LLMCache,
    topics: str,
    docs: Mapping[str, Mapping],
    model: str,
    effort: str,
    mix: Mapping[str, int] = DEFAULT_MIX,
    existing_drafts: Sequence[EvalItem] = (),
    progress: Callable[[str], None] = print,
) -> GenerationResult:
    """Run both steps and return validated drafts (not saved: the caller decides)."""
    result = GenerationResult()
    send = streaming_sender(client)

    questions: list[dict] = []
    for question_type, n in mix.items():
        if n <= 0:
            continue
        progress(f"Step 1: writing {n} {question_type} questions from the persona prompt")
        reply = cache.call(question_request(model, effort, topics, question_type, n), send)
        result.replies.append(reply)
        for q in parse_reply(reply)["questions"]:
            questions.append({**q, "intended_type": question_type})

    questions, result.dropped_duplicates = dedupe_questions(questions)
    for n, q in enumerate(questions, start=1):
        q["key"] = f"Q{n}"

    by_key: dict[str, dict] = {}
    batches = [
        questions[i : i + QUESTIONS_PER_EVIDENCE_CALL]
        for i in range(0, len(questions), QUESTIONS_PER_EVIDENCE_CALL)
    ]
    for b, batch in enumerate(batches, start=1):
        progress(f"Step 2: finding evidence, batch {b} of {len(batches)}")
        reply = cache.call(evidence_request(model, effort, docs, batch), send)
        result.replies.append(reply)
        for item in parse_reply(reply)["results"]:
            by_key[item["id"]] = item

    drafts: list[EvalItem] = list(existing_drafts)
    for q in questions:
        found = by_key.get(q["key"])
        if found is None:
            result.rejected.append(f"{q['question']!r}: no evidence result returned")
            continue
        try:
            draft = build_draft(q, found, docs, drafts, model)
        except ValidationError as err:
            reasons = "; ".join(e["msg"] for e in err.errors())
            result.rejected.append(f"{q['question']!r}: {reasons}")
            continue
        drafts.append(draft)
        result.drafts.append(draft)
    return result


# --- Cost estimate ----------------------------------------------------------------------------

# Claude Opus 5.5 list prices, US$ per million tokens (checked 2026-09-25).
PRICES = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write_1h": 8.00, "cache_read": 0.20},
}


@dataclass
class CostEstimate:
    calls: int
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    low: float
    high: float


def estimate_cost(
    model: str, topics: str, docs: Mapping[str, Mapping], mix: Mapping[str, int]
) -> CostEstimate:
    """Rough cost range, counting about 4 characters per token.

    The low end assumes 4 characters per token and modest thinking; the high end allows a
    tokenizer that counts up to 35% more tokens and long thinking at high effort.
    """
    price = PRICES[model]
    n_questions = sum(mix.values())
    n_q_calls = sum(1 for n in mix.values() if n > 0)
    n_e_calls = -(-n_questions // QUESTIONS_PER_EVIDENCE_CALL)
    corpus = len(corpus_block(docs)) // 4 + len(EVIDENCE_INSTRUCTIONS) // 4
    step1_in = n_q_calls * (len(topics) + len(PERSONA_SYSTEM) + 600) // 4
    step2_in = n_e_calls * 400  # the question list in each call
    out_low = n_q_calls * 3000 + n_e_calls * 6000
    out_high = n_q_calls * 8000 + n_e_calls * 16000

    def total(scale: float, output: int) -> float:
        return (
            (step1_in + step2_in) * scale * price["input"]
            + corpus * scale * price["cache_write_1h"]
            + corpus * scale * (n_e_calls - 1) * price["cache_read"]
            + output * price["output"]
        ) / 1_000_000

    return CostEstimate(
        calls=n_q_calls + n_e_calls,
        input_tokens=step1_in + step2_in,
        cache_write_tokens=corpus,
        cache_read_tokens=corpus * (n_e_calls - 1),
        output_tokens=out_high,
        low=round(total(1.0, out_low), 2),
        high=round(total(1.35, out_high), 2),
    )
