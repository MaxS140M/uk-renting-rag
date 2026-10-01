"""LLM-as-judge for faithfulness (is every claim supported by the passages?) and
correctness (does the answer match the reference answer?).

The judge is a stronger model than the one answering. Its verdicts are only trusted after
checking them against human labels on a sample (scripts/judge_agreement.py), because an
LLM judge can be confidently wrong or biased, for example towards answers that sound like
its own writing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from rag.llm_cache import LLMCache, LLMReply
from rag.prompts import format_passage
from rag.retrieval import RetrievalResult

FAITHFULNESS_SYSTEM = """\
You check whether an answer is supported by the source passages it was written from.

Split the answer into its individual factual claims. Ignore: the line saying when guidance \
was retrieved, the list of sources, the "not legal advice" sentence, citation numbers, and \
generic suggestions such as "contact Citizens Advice" unless they state a fact. For each \
claim, decide whether the passages support it: it is stated in, or directly follows from, \
the passages. A claim that is true in general but not in the passages is NOT supported. Be \
strict and literal; do not use your own knowledge of the law.

If the answer only says it cannot find the information, return an empty list of claims."""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "explanation": {"type": "string"},
                },
                "required": ["claim", "supported", "explanation"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["claims", "reasoning"],
    "additionalProperties": False,
}

CORRECTNESS_SYSTEM = """\
You grade an answer to a question about renting in England against a reference answer \
written from the official guidance. Treat the reference answer as correct.

- correct: the answer gives the key facts of the reference answer and nothing that \
contradicts it. Extra accurate detail, different wording and a different order are fine.
- partially_correct: the answer gets some key facts right but misses another key fact, or \
has a minor error that does not change the main conclusion.
- incorrect: the answer contradicts the reference, misses its main point, gives the wrong \
conclusion, or says it cannot find information that the reference answer gives.

Judge only the facts, not the style, length, citations or disclaimers."""

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "reasoning"],
    "additionalProperties": False,
}

CORRECTNESS_SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


@dataclass(frozen=True)
class FaithfulnessVerdict:
    score: float | None  # share of claims supported; None if the answer makes no claims
    claims: list[dict]
    reasoning: str

    @property
    def faithful(self) -> bool | None:
        """True if every claim is supported (the binary label compared with human labels)."""
        return None if self.score is None else self.score == 1.0


@dataclass(frozen=True)
class CorrectnessVerdict:
    verdict: str
    reasoning: str

    @property
    def score(self) -> float:
        return CORRECTNESS_SCORES[self.verdict]


def _request(model: str, effort: str, system: str, user: str, schema: dict) -> dict:
    return {
        "model": model,
        "max_tokens": 8000,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    }


def _parse(reply: LLMReply) -> dict:
    if reply.stop_reason in ("max_tokens", "refusal"):
        raise ValueError(f"judge reply unusable (stop_reason={reply.stop_reason})")
    return json.loads(reply.text)


def faithfulness_request(
    model: str, effort: str, question: str, passages: Sequence[RetrievalResult], answer: str
) -> dict:
    sources = "\n\n".join(format_passage(n, p) for n, p in enumerate(passages, start=1))
    user = (
        f"<passages>\n{sources}\n</passages>\n\n<question>\n{question}\n</question>\n\n"
        f"<answer>\n{answer}\n</answer>\n\nCheck every factual claim in the answer."
    )
    return _request(model, effort, FAITHFULNESS_SYSTEM, user, FAITHFULNESS_SCHEMA)


def correctness_request(
    model: str, effort: str, question: str, reference: str, answer: str
) -> dict:
    user = (
        f"<question>\n{question}\n</question>\n\n<reference_answer>\n{reference}\n"
        f"</reference_answer>\n\n<answer>\n{answer}\n</answer>\n\nGrade the answer."
    )
    return _request(model, effort, CORRECTNESS_SYSTEM, user, CORRECTNESS_SCHEMA)


class Judge:
    """Runs both judges through the disk cache, so re-judging the same answer is free."""

    def __init__(
        self, send: Callable[[dict], LLMReply], cache: LLMCache, model: str, effort: str
    ) -> None:
        self.send, self.cache, self.model, self.effort = send, cache, model, effort
        self.replies: list[LLMReply] = []

    def _call(self, request: dict) -> dict:
        reply = self.cache.call(request, self.send)
        self.replies.append(reply)
        return _parse(reply)

    def faithfulness(
        self, question: str, passages: Sequence[RetrievalResult], answer: str
    ) -> FaithfulnessVerdict:
        data = self._call(faithfulness_request(self.model, self.effort, question, passages, answer))
        claims = data["claims"]
        score = sum(c["supported"] for c in claims) / len(claims) if claims else None
        return FaithfulnessVerdict(score, claims, data["reasoning"])

    def correctness(self, question: str, reference: str, answer: str) -> CorrectnessVerdict:
        data = self._call(correctness_request(self.model, self.effort, question, reference, answer))
        return CorrectnessVerdict(data["verdict"], data["reasoning"])


def verdict_dict(verdict: FaithfulnessVerdict | CorrectnessVerdict | None) -> dict | None:
    if verdict is None:
        return None
    data = asdict(verdict)
    data["score"] = verdict.score
    if isinstance(verdict, FaithfulnessVerdict):
        data["faithful"] = verdict.faithful
    return data
