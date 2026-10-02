"""The demo's core: loads the pipeline once, then answers questions with abuse protection.

Shared by the API (app/api.py) and the Gradio UI (app/ui.py), so both get the same
validation, rate limit, daily cap, error handling and privacy-preserving logging.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import anthropic

from rag.config import GITHUB_URL, PROJECT_ROOT  # noqa: F401 (re-exported for the UI)
from rag.experiment import Experiment, load_experiments, load_index_chunks
from rag.generate import Generator, MissingAPIKeyError, create_client
from rag.pipeline import AnswerResult, RAGPipeline

log = logging.getLogger("uk_renting_rag")
MAX_QUESTION_CHARS = 500


@dataclass(frozen=True)
class Settings:
    """Deployment settings, read from environment variables so they can change per host."""

    config_name: str = "hybrid_rerank_bge"  # the best configuration in eval/RESULTS.md
    rate_limit_per_minute: int = 10
    daily_llm_cap: int = 200
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 1
    llm_max_tokens: int = 700

    @classmethod
    def from_env(cls) -> Settings:
        def get(name: str, cast, default):
            value = os.environ.get(name)
            return default if value in (None, "") else cast(value)

        defaults = cls()
        return cls(
            config_name=get("RAG_CONFIG", str, defaults.config_name),
            rate_limit_per_minute=get("RATE_LIMIT_PER_MINUTE", int, defaults.rate_limit_per_minute),
            daily_llm_cap=get("DAILY_LLM_CAP", int, defaults.daily_llm_cap),
            llm_timeout_seconds=get("LLM_TIMEOUT_SECONDS", float, defaults.llm_timeout_seconds),
            llm_max_retries=get("LLM_MAX_RETRIES", int, defaults.llm_max_retries),
            llm_max_tokens=get("LLM_MAX_TOKENS", int, defaults.llm_max_tokens),
        )


class DemoError(Exception):
    """An expected failure with a message that is safe and friendly to show to a visitor."""

    def __init__(self, status: int, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


def validate_question(question: str | None) -> str:
    """Return the cleaned question, or raise DemoError with a clear message."""
    question = (question or "").strip()
    if not question:
        raise DemoError(422, "Please type a question.")
    if len(question) > MAX_QUESTION_CHARS:
        raise DemoError(
            422,
            f"Please keep questions to {MAX_QUESTION_CHARS} characters or fewer "
            f"(yours is {len(question)}).",
        )
    return question


class RateLimiter:
    """Sliding-window limit per client: at most ``limit`` requests in any ``window`` seconds.

    Client keys (IP addresses) are held in memory only for the length of the window.
    """

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """Record a request; return None if allowed, else seconds until the next is allowed."""
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                if not hits:  # a limit of 0 blocks everything
                    return int(self.window)
                return max(1, int(self.window - (now - hits[0])) + 1)
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle clients so memory stays bounded
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return None


class DailyCap:
    """A global limit on LLM calls per UTC day, the hard ceiling on what the demo can cost."""

    def __init__(self, limit: int, today: Callable[[], date] = lambda: datetime.now(UTC).date()):
        self.limit, self.today = limit, today
        self._day, self._used = today(), 0
        self._lock = threading.Lock()

    def try_use(self) -> bool:
        with self._lock:
            if self.today() != self._day:
                self._day, self._used = self.today(), 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.limit if self.today() != self._day else self.limit - self._used


@dataclass
class DemoAnswer:
    answer: str
    refused: bool
    sources: list[dict]
    passages: list[dict]
    latency_ms: dict[str, float]
    config_name: str
    prompt_version: str
    date_retrieved: str | None = None
    extra: dict = field(default_factory=dict)


class DemoService:
    def __init__(
        self,
        settings: Settings | None = None,
        pipeline: RAGPipeline | None = None,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], date] = lambda: datetime.now(UTC).date(),
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.pipeline = pipeline
        self.llm_configured = pipeline is not None
        self.load_error: str | None = None
        self.guidance_date: str | None = None  # ISO date the corpus was retrieved from GOV.UK
        self.rate_limiter = RateLimiter(self.settings.rate_limit_per_minute, clock=clock)
        self.daily_cap = DailyCap(self.settings.daily_llm_cap, today=today)
        self._retrieval_lock = threading.Lock()

    # --- Startup ---------------------------------------------------------------------------

    def experiment(self) -> Experiment:
        experiments = {e.name: e for e in load_experiments(PROJECT_ROOT / "eval" / "configs.yaml")}
        if self.settings.config_name not in experiments:
            raise ValueError(f"Unknown RAG_CONFIG '{self.settings.config_name}'")
        return experiments[self.settings.config_name]

    def load(self) -> None:
        """Load the index, embedding model, reranker and LLM client, once, at startup."""
        if self.pipeline is not None:
            return
        try:
            config = dataclasses.replace(
                self.experiment().config, llm_max_tokens=self.settings.llm_max_tokens
            )
            try:
                client = create_client(
                    timeout=self.settings.llm_timeout_seconds,
                    max_retries=self.settings.llm_max_retries,
                )
                self.llm_configured = True
            except MissingAPIKeyError:
                client = None  # retrieval still works; /ask explains the demo is not configured
                log.warning("ANTHROPIC_API_KEY is not set: answers are disabled")
            pipeline = RAGPipeline(config, generator=Generator(config, client=client))
            pipeline.retrieve("warm-up question about tenancy deposits")  # first-call costs
            chunks = load_index_chunks(config.index_dir)
            self.guidance_date = max(c["date_retrieved"] for c in chunks)
            self.pipeline = pipeline
            log.info("Loaded config %s", self.settings.config_name)
        except Exception as err:  # keep serving /health, which reports the problem
            self.load_error = f"{type(err).__name__}: {err}"
            log.error("Failed to load the pipeline: %s", self.load_error)

    def health(self) -> dict:
        config = self.pipeline.config if self.pipeline else None
        ready = self.pipeline is not None and self.llm_configured
        return {
            "status": "ok" if ready else "degraded",
            "index_loaded": self.pipeline is not None,
            "models_loaded": self.pipeline is not None,
            "llm_configured": self.llm_configured,
            "config_name": self.settings.config_name,
            "guidance_date": self.guidance_date,
            "embedding_model": config.embedding_model if config else None,
            "daily_llm_calls_remaining": self.daily_cap.remaining,
            "error": "The search index failed to load." if self.load_error else None,
        }

    # --- Answering -------------------------------------------------------------------------

    def ask(self, question: str | None, client_id: str) -> DemoAnswer:
        start = time.perf_counter()
        outcome = "error"
        result: AnswerResult | None = None
        try:
            question = validate_question(question)
            retry_after = self.rate_limiter.check(client_id)
            if retry_after is not None:
                outcome = "rate_limited"
                raise DemoError(
                    429,
                    f"You're asking questions faster than this demo allows "
                    f"({self.settings.rate_limit_per_minute} per minute). Please wait "
                    f"{retry_after} seconds and try again.",
                    retry_after,
                )
            if self.pipeline is None:
                raise DemoError(503, "The demo is still starting up. Please try again in a minute.")
            if not self.llm_configured:
                raise DemoError(503, "The demo's answer service isn't configured right now.")
            if not self.daily_cap.try_use():
                outcome = "daily_cap"
                raise DemoError(
                    429,
                    "This demo has reached its daily limit of questions, which keeps it free "
                    "to run. Please try again tomorrow.",
                )

            total_start = time.perf_counter()
            timings: dict[str, float] = {}
            with self._retrieval_lock:  # retrieval is CPU-bound; the LLM wait is not locked
                passages = self.pipeline.retrieve(question, timings)
            timings["retrieval"] = round(sum(timings.values()), 1)
            result = self._generate(question, passages, timings, total_start)
            outcome = "ok"
            return self._to_answer(result)
        except DemoError as err:
            if outcome == "error":
                outcome = "rejected" if err.status == 422 else "unavailable"
            raise
        finally:
            self._log_metrics(outcome, result, time.perf_counter() - start)

    def _generate(self, question, passages, timings, total_start) -> AnswerResult:
        try:
            return self.pipeline.answer_from(question, passages, timings, total_start)
        except anthropic.APITimeoutError as err:
            raise DemoError(504, "The answer took too long to generate. Please try again.") from err
        except anthropic.RateLimitError as err:
            raise DemoError(
                503, "The answer service is busy right now. Please try again shortly."
            ) from err
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as err:
            raise DemoError(
                502, "The answer service is unavailable right now. Please try again later."
            ) from err

    def _to_answer(self, result: AnswerResult) -> DemoAnswer:
        dates = sorted({s.date_retrieved for s in result.sources})
        return DemoAnswer(
            answer=result.answer,
            refused=result.refused,
            sources=[
                {
                    "number": s.number,
                    "title": s.title,
                    "url": s.url,
                    "section": s.section,
                    "date_retrieved": s.date_retrieved,
                }
                for s in result.sources
            ],
            passages=[
                {
                    "rank": r.rank,
                    "title": r.title,
                    "section": r.section,
                    "url": r.url,
                    "score": round(r.score, 3),
                    "text": r.text,
                }
                for r in result.retrieved
            ],
            latency_ms=result.latency_ms,
            config_name=self.settings.config_name,
            prompt_version=result.prompt_version,
            date_retrieved=dates[-1] if dates else None,
        )

    def _log_metrics(self, outcome: str, result: AnswerResult | None, seconds: float) -> None:
        """Log anonymous metrics only: never the question text or the client's IP address."""
        record = {"event": "ask", "outcome": outcome, "ms": round(seconds * 1000)}
        if result is not None:
            record.update(
                refused=result.refused,
                stages=result.latency_ms,
                output_tokens=result.usage.get("output_tokens"),
            )
        log.info(json.dumps(record))
