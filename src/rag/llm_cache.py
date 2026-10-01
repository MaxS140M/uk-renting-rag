"""On-disk cache for LLM calls, so reruns are free and results are reproducible.

Each request is keyed by a hash of everything that affects the reply (model, system
prompt, messages and every other parameter). A cached reply is returned instead of calling
the API again. Cached calls cost nothing, and a crashed run can resume where it stopped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LLMReply:
    """The parts of an API response the project uses, in a form that can be cached."""

    text: str
    model: str
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cached: bool = False  # True if this came from the disk cache (no API call made)


def request_key(request: dict) -> str:
    """Stable hash of a request: same model, prompt and parameters -> same key."""
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def reply_from_response(response) -> LLMReply:
    usage = response.usage
    return LLMReply(
        text="".join(block.text for block in response.content if block.type == "text"),
        model=response.model,
        stop_reason=response.stop_reason,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


class LLMCache:
    """A directory of JSON files, one per cached request."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, request: dict) -> LLMReply | None:
        path = self._path(request_key(request))
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMReply(**{**data["reply"], "cached": True})

    def put(self, request: dict, reply: LLMReply) -> None:
        path = self._path(request_key(request))
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"request_summary": _summary(request), "reply": asdict(reply) | {"cached": False}}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")

    def call(self, request: dict, send) -> LLMReply:
        """Return the cached reply for ``request``, or call ``send(request)`` and cache it.

        Replies cut off by max_tokens are not cached, so a rerun can try again.
        """
        cached = self.get(request)
        if cached is not None:
            return cached
        reply = send(request)
        if reply.stop_reason != "max_tokens":
            self.put(request, reply)
        return reply


def _summary(request: dict) -> dict:
    """A small readable summary stored with each cache entry, for debugging."""
    messages = request.get("messages", [])
    last = messages[-1]["content"] if messages else ""
    text = last if isinstance(last, str) else json.dumps(last)[:500]
    return {"model": request.get("model"), "last_message_start": text[:300]}
