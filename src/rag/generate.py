"""Generate a grounded, cited answer from retrieved passages with the Anthropic API.

The API key is read from the environment (loaded from .env by python-dotenv) by the
Anthropic SDK itself; this module never reads, prints or logs it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from rag.config import PROJECT_ROOT, RAGConfig
from rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.retrieval import RetrievalResult

log = logging.getLogger(__name__)


class MissingAPIKeyError(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not available."""


@dataclass(frozen=True)
class Generation:
    """The model's reply plus the details needed to monitor cost and failures."""

    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int


def create_client() -> anthropic.Anthropic:
    """Create an Anthropic client, loading ANTHROPIC_API_KEY from the project's .env file."""
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic()


class Generator:
    """Wraps the LLM call. The client can be injected, which lets tests use a fake one."""

    def __init__(self, config: RAGConfig, client: anthropic.Anthropic | None = None) -> None:
        self.config = config
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:  # created on first use, so retrieval-only runs need no key
            self._client = create_client()
        return self._client

    def generate(self, question: str, results: Sequence[RetrievalResult]) -> Generation:
        # Anthropic SDK 1.x dropped `temperature` from messages.create() because the newest
        # models reject it. Older models such as Haiku 4.5 still accept it, so it is sent as
        # raw JSON via extra_body, and only when configured (None means "do not send").
        extra_body = {}
        if self.config.llm_temperature is not None:
            extra_body["temperature"] = self.config.llm_temperature

        response = self.client.messages.create(
            model=self.config.llm_model,
            max_tokens=self.config.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(question, results)}],
            extra_body=extra_body or None,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "max_tokens":
            log.warning("Answer was cut off at max_tokens=%d", self.config.llm_max_tokens)
        return Generation(
            text=text.strip(),
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
