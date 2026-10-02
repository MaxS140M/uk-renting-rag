"""Tests for the LLM cache and two-step test-set generation, with the API mocked."""

import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from anthropic.resources.beta.messages import Messages as BetaMessages

from rag.evaluation.testset_generation import (
    DEFAULT_MIX,
    estimate_cost,
    evidence_request,
    generate_test_set,
    question_request,
    topic_list,
)
from rag.evaluation.utils import load_corpus
from rag.llm_cache import LLMCache, LLMReply, request_key

ROOT = Path(__file__).resolve().parent.parent
QUOTE_30 = (
    "Your landlord or letting agent must put your deposit in the scheme within 30 days of "
    "getting it."
)
QUOTE_10 = "Your landlord must return your deposit within 10 days of you both agreeing"


@pytest.fixture(scope="module")
def docs():
    return load_corpus(ROOT / "data" / "sample")


# --- Cache ----------------------------------------------------------------------------------


def reply(text="{}", stop_reason="end_turn") -> LLMReply:
    return LLMReply(text, "claude-test", stop_reason, 10, 5)


def test_cache_returns_stored_reply_without_calling_again(tmp_path):
    cache, calls = LLMCache(tmp_path), []
    send = lambda request: calls.append(request) or reply("hello")  # noqa: E731
    request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    first, second = cache.call(request, send), cache.call(request, send)
    assert len(calls) == 1
    assert first.text == second.text == "hello"
    assert not first.cached and second.cached


def test_cache_key_depends_on_every_parameter_but_not_key_order():
    base = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
    reordered = {"messages": base["messages"], "max_tokens": 10, "model": "m"}
    assert request_key(base) == request_key(reordered)
    assert request_key(base) != request_key({**base, "model": "other"})
    assert request_key(base) != request_key({**base, "max_tokens": 11})


def test_truncated_replies_are_not_cached(tmp_path):
    cache, calls = LLMCache(tmp_path), []
    send = lambda request: calls.append(1) or reply(stop_reason="max_tokens")  # noqa: E731
    cache.call({"model": "m"}, send)
    cache.call({"model": "m"}, send)
    assert len(calls) == 2


# --- Fake API that answers both steps -------------------------------------------------------


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


DISTINCT_QUESTIONS = [
    "How long does my landlord have to protect my deposit?",
    "Who pays for a broken boiler in a rented flat?",
    "Can my rent go up twice in one year?",
    "What fees can a letting agent legally charge?",
    "Does a shared house with five tenants need a licence?",
    "can my landlord just turn up without telling me",
    "Is the landlord allowed to keep my holding deposit?",
    "What happens if mould is making my kids ill?",
    "Do Scottish tenancy rules apply to me in Glasgow?",
    "How much is rent in central Leeds right now?",
]


class FakeBetaMessages:
    """Returns persona questions for step 1 and evidence for step 2, recording requests."""

    def __init__(self, evidence_for):
        self.requests = []
        self.evidence_for = evidence_for

        self.pool = list(DISTINCT_QUESTIONS)

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        content = kwargs["messages"][0]["content"]
        if content.startswith("Write "):
            n = int(content.split()[1])
            payload = {
                "questions": [{"question": self.pool.pop(0), "topic": "deposits"} for _ in range(n)]
            }
        else:
            keys = re.findall(r"^(Q\d+) \[(\w+)\]: (.*)$", content, flags=re.MULTILINE)
            payload = {"results": [self.evidence_for(k, t, q) for k, t, q in keys]}
        message = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=json.dumps(payload)),
            ],
            model=kwargs["model"],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )
        return FakeStream(message)


def evidence_for(key, intended, question):
    if intended == "unanswerable":
        return {
            "id": key,
            "answerable": False,
            "evidence": [],
            "reference_answer": "The guidance does not cover this.",
            "comment": "",
        }
    quotes = [QUOTE_30, QUOTE_10] if intended == "multi_passage" else [QUOTE_30]
    return {
        "id": key,
        "answerable": True,
        "evidence": [{"doc_id": "tenancy-deposit-protection", "quote": q} for q in quotes],
        "reference_answer": "Within 30 days.",
        "comment": "Clear answer.",
    }


def fake_client(evidence=evidence_for):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeBetaMessages(evidence)))


MIX = {"factual": 3, "multi_passage": 2, "informal": 2, "unanswerable": 2}


def test_generates_drafts_of_each_type(tmp_path, docs):
    result = generate_test_set(
        fake_client(),
        LLMCache(tmp_path),
        "topics",
        docs,
        "claude-test",
        "high",
        MIX,
        progress=lambda _: None,
    )
    types = [d.question_type for d in result.drafts]
    assert types.count("factual") == 3 and types.count("multi_passage") == 2
    assert types.count("informal") == 2 and types.count("unanswerable") == 2
    assert all(d.author == "llm_draft" for d in result.drafts)
    assert [d.id for d in result.drafts] == [f"draft-{n:03d}" for n in range(1, 10)]
    assert all("not found verbatim" not in d.notes for d in result.drafts)


def test_step_one_never_sees_the_passages(tmp_path, docs):
    client = fake_client()
    generate_test_set(
        client,
        LLMCache(tmp_path),
        "Topic: Deposits",
        docs,
        "claude-test",
        "high",
        MIX,
        progress=lambda _: None,
    )
    question_calls = [
        r for r in client.beta.messages.requests if r["messages"][0]["content"].startswith("Write ")
    ]
    assert len(question_calls) == 4
    for request in question_calls:
        assert QUOTE_30 not in json.dumps(request)
        assert "Topic: Deposits" in request["system"]


def test_step_two_caches_the_corpus_and_uses_the_fallback(tmp_path, docs):
    client = fake_client()
    generate_test_set(
        client, LLMCache(tmp_path), "t", docs, "claude-test", "high", MIX, progress=lambda _: None
    )
    evidence_calls = [
        r
        for r in client.beta.messages.requests
        if not r["messages"][0]["content"].startswith("Write ")
    ]
    request = evidence_calls[0]
    corpus = request["system"][1]
    assert corpus["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert QUOTE_30 in corpus["text"]
    assert request["fallbacks"] == "default" and request["betas"]


def test_rerun_uses_the_cache(tmp_path, docs):
    cache = LLMCache(tmp_path)
    generate_test_set(
        fake_client(), cache, "t", docs, "claude-test", "high", MIX, progress=lambda _: None
    )
    client = fake_client()
    again = generate_test_set(
        client, cache, "t", docs, "claude-test", "high", MIX, progress=lambda _: None
    )
    assert client.beta.messages.requests == []
    assert len(again.drafts) == 9 and all(r.cached for r in again.replies)


def test_types_are_corrected_when_evidence_disagrees(tmp_path, docs):
    def swapped(key, intended, question):
        if intended == "unanswerable":  # the corpus does answer it after all
            return evidence_for(key, "factual", question)
        if intended == "multi_passage":  # one passage is enough
            return evidence_for(key, "factual", question)
        if intended == "informal":  # nothing found
            return evidence_for(key, "unanswerable", question)
        return {
            **evidence_for(key, intended, question),
            "evidence": [
                {
                    "doc_id": "tenancy-deposit-protection",
                    "quote": "A paraphrased quote that is not in the document.",
                }
            ],
        }

    result = generate_test_set(
        fake_client(swapped),
        LLMCache(tmp_path),
        "t",
        docs,
        "claude-test",
        "high",
        MIX,
        progress=lambda _: None,
    )
    notes = {d.question: d for d in result.drafts}
    assert all(d.question_type in ("factual", "unanswerable") for d in result.drafts)
    assert any("retyped as factual" in d.notes for d in notes.values())
    assert any("no evidence was found" in d.notes for d in notes.values())
    assert sum("not found verbatim" in d.notes for d in notes.values()) == 3


def test_requests_match_the_real_sdk_signature(docs):
    accepted = set(inspect.signature(BetaMessages.stream).parameters)
    for request in (
        question_request("m", "high", "topics", "factual", 5),
        evidence_request(
            "m", "high", docs, [{"key": "Q1", "intended_type": "factual", "question": "Deposit?"}]
        ),
    ):
        assert set(request) | {"betas", "fallbacks"} <= accepted


def test_topic_list_keeps_titles_and_headings_only():
    overview = (ROOT / "eval" / "testset" / "corpus_overview.md").read_text(encoding="utf-8")
    topics = topic_list(overview)
    assert "Topic: Tenancy deposit protection" in topics
    assert "https://" not in topics and "| " not in topics


def test_cost_estimate_is_a_sensible_range(docs):
    estimate = estimate_cost("claude-opus-5-5", "topics", docs, DEFAULT_MIX)
    assert 0 < estimate.low < estimate.high
    assert estimate.calls == 4 + 13
