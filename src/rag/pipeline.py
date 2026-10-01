"""End-to-end RAG pipeline: retrieve passages, generate a cited answer, time each stage.

``answer()`` returns a structured ``AnswerResult`` rather than a string, so the CLI, the API
and the evaluation scripts can all use the same output (text, sources, chunk IDs, timings).
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from rag.config import RAGConfig
from rag.generate import Generator
from rag.prompts import PROMPT_VERSION, REFUSAL_TEXT
from rag.retrieval import RetrievalResult, Retriever, get_retriever

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Source:
    """A passage the answer actually cited, with its citation number."""

    number: int
    chunk_id: str
    title: str
    url: str
    section: str
    date_retrieved: str


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    sources: list[Source]  # cited passages, in order of first citation
    retrieved: list[RetrievalResult]  # everything passed to the LLM, in rank order
    refused: bool  # True if the model said the guidance does not contain the answer
    latency_ms: dict[str, float]  # retrieval, generation and total
    model: str | None
    prompt_version: str
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [r.chunk_id for r in self.retrieved]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["retrieved_chunk_ids"] = self.retrieved_chunk_ids
        return data


def is_refusal(text: str) -> bool:
    """True if the answer opens with the refusal sentence (a full refusal, not a partial one)."""
    normalised = text.replace("’", "'").strip().lower()  # model may use a curly apostrophe
    return normalised.startswith(REFUSAL_TEXT.lower())


def cited_sources(answer: str, results: Sequence[RetrievalResult]) -> list[Source]:
    """Map citation markers like [2] or [1, 3] back to the passages they refer to."""
    sources: dict[int, Source] = {}
    for match in _CITATION_RE.finditer(answer):
        for number in (int(n) for n in match.group(1).split(",")):
            if 1 <= number <= len(results) and number not in sources:
                r = results[number - 1]
                sources[number] = Source(
                    number, r.chunk_id, r.title, r.url, r.section, r.date_retrieved
                )
    return list(sources.values())


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class RAGPipeline:
    """Retrieve-then-generate. Components can be injected for tests and experiments."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        retriever: Retriever | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.retriever = retriever or get_retriever(self.config)
        self.generator = generator or Generator(self.config)

    def retrieve(self, question: str) -> list[RetrievalResult]:
        """Retrieval only, with no LLM call: cheap enough to run many times for evaluation."""
        return self.retriever.retrieve(question, self.config.top_k)

    def answer(self, question: str) -> AnswerResult:
        total_start = time.perf_counter()

        start = time.perf_counter()
        results = self.retrieve(question)
        retrieval_ms = _ms(start)

        if not results:  # nothing to ground an answer in, so do not call the LLM at all
            return AnswerResult(
                question=question,
                answer=REFUSAL_TEXT,
                sources=[],
                retrieved=[],
                refused=True,
                latency_ms={
                    "retrieval": retrieval_ms,
                    "generation": 0.0,
                    "total": _ms(total_start),
                },
                model=None,
                prompt_version=PROMPT_VERSION,
            )

        start = time.perf_counter()
        generation = self.generator.generate(question, results)
        generation_ms = _ms(start)

        refused = is_refusal(generation.text)
        return AnswerResult(
            question=question,
            answer=generation.text,
            sources=[] if refused else cited_sources(generation.text, results),
            retrieved=results,
            refused=refused,
            latency_ms={
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": _ms(total_start),
            },
            model=generation.model,
            prompt_version=PROMPT_VERSION,
            stop_reason=generation.stop_reason,
            usage={
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
            },
        )


@lru_cache(maxsize=4)
def _cached_pipeline(config: RAGConfig) -> RAGPipeline:
    return RAGPipeline(config)


def answer(question: str, config: RAGConfig | None = None) -> AnswerResult:
    """Answer a question, reusing a loaded pipeline (model and index) for each config."""
    return _cached_pipeline(config or RAGConfig()).answer(question)
