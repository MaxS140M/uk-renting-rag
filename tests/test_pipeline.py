"""Tests for prompt building, generation and the end-to-end pipeline, with the LLM mocked."""

import inspect

import pytest
from anthropic.resources.messages import Messages

from rag.config import RAGConfig
from rag.generate import Generator, MissingAPIKeyError, create_client
from rag.pipeline import RAGPipeline, cited_sources, is_refusal
from rag.prompts import DISCLAIMER_TEXT, REFUSAL_TEXT, SYSTEM_PROMPT, build_user_prompt
from rag.retrieval import DenseRetriever, RetrievalResult


@pytest.fixture
def results(chunks) -> list[RetrievalResult]:
    return [
        RetrievalResult.from_chunk(c, 1.0 - i / 10, i + 1, "dense")
        for i, c in enumerate(chunks[:3])
    ]


def make_pipeline(loaded_index, embedder, client, **config_overrides) -> RAGPipeline:
    config = RAGConfig(**config_overrides)
    return RAGPipeline(
        config,
        retriever=DenseRetriever(loaded_index, embedder),
        generator=Generator(config, client=client),
    )


# --- Prompt ---------------------------------------------------------------------------------


def test_prompt_contains_every_passage_url_and_date(results):
    prompt = build_user_prompt("How long to protect my deposit?", results)
    for number, result in enumerate(results, start=1):
        assert f"[{number}] Title: {result.title}" in prompt
        assert result.text in prompt
        assert result.url in prompt
        assert f"Retrieved: {result.date_retrieved}" in prompt
    assert prompt.endswith("<question>\nHow long to protect my deposit?\n</question>")


def test_system_prompt_states_the_grounding_rules():
    assert REFUSAL_TEXT in SYSTEM_PROMPT
    assert DISCLAIMER_TEXT in SYSTEM_PROMPT
    assert "Sources:" in SYSTEM_PROMPT


# --- Generator ------------------------------------------------------------------------------


def test_generator_sends_configured_model_settings(results, fake_client_factory):
    client = fake_client_factory("Protect it within 30 days [1].")
    config = RAGConfig(llm_model="claude-test", llm_max_tokens=321, llm_temperature=0.0)
    generation = Generator(config, client=client).generate("Deposit?", results)

    call = client.messages.calls[0]
    assert call["model"] == "claude-test"
    assert call["max_tokens"] == 321
    assert call["extra_body"] == {"temperature": 0.0}
    assert "temperature" not in call  # SDK 1.x rejects it as a keyword argument
    assert call["system"] == SYSTEM_PROMPT
    assert results[0].url in call["messages"][0]["content"]
    assert generation.text == "Protect it within 30 days [1]."
    assert generation.input_tokens == 100


def test_request_only_uses_arguments_the_real_sdk_accepts(results, fake_client_factory):
    # The fake client accepts anything, so check the request against the real SDK signature.
    # (This catches changes like SDK 1.x removing the `temperature` keyword argument.)
    accepted = set(inspect.signature(Messages.create).parameters)
    client = fake_client_factory()
    Generator(RAGConfig(), client=client).generate("Deposit?", results)
    assert set(client.messages.calls[0]) <= accepted


def test_temperature_none_is_not_sent(results, fake_client_factory):
    client = fake_client_factory()
    Generator(RAGConfig(llm_temperature=None), client=client).generate("Deposit?", results)
    assert client.messages.calls[0]["extra_body"] is None


def test_missing_api_key_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr("rag.generate.load_dotenv", lambda *a, **k: None)  # ignore a real .env
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match=r"\.env"):
        create_client()


# --- Pipeline -------------------------------------------------------------------------------


def test_answer_returns_structured_result(loaded_index, embedder, fake_client_factory):
    client = fake_client_factory("Within 30 days [1]. Use an approved scheme [1][2].")
    pipeline = make_pipeline(loaded_index, embedder, client, final_k=3)

    result = pipeline.answer("How long does my landlord have to protect my deposit?")

    assert len(result.retrieved) == 3
    assert result.retrieved_chunk_ids == [r.chunk_id for r in result.retrieved]
    assert result.retrieved_chunk_ids[0] == "deposit-000"
    assert [s.number for s in result.sources] == [1, 2]
    assert result.sources[0].url == result.retrieved[0].url
    assert result.refused is False
    assert set(result.latency_ms) == {"dense", "retrieval", "generation", "total"}
    assert result.config["retrieval_mode"] == "dense"
    assert result.latency_ms["total"] >= result.latency_ms["retrieval"]
    assert result.prompt_version
    assert result.to_dict()["retrieved_chunk_ids"] == result.retrieved_chunk_ids


def test_refusal_is_detected_and_has_no_sources(loaded_index, embedder, fake_client_factory):
    reply = "I can’t find that in the guidance. Try Citizens Advice [1]."
    pipeline = make_pipeline(loaded_index, embedder, fake_client_factory(reply))
    result = pipeline.answer("What is the capital of France?")
    assert result.refused is True
    assert result.sources == []


def test_retrieval_only_never_calls_the_llm(loaded_index, embedder, fake_client_factory):
    client = fake_client_factory()
    pipeline = make_pipeline(loaded_index, embedder, client, final_k=2)
    assert len(pipeline.retrieve("deposit")) == 2
    assert client.messages.calls == []


def test_partial_answer_is_not_a_refusal():
    assert not is_refusal(f"Deposits must be protected [1]. {REFUSAL_TEXT} for the fee part.")
    assert is_refusal(f"  {REFUSAL_TEXT} You could ask Shelter.")


def test_citations_map_to_passages_and_ignore_invalid_numbers(results):
    sources = cited_sources("A [2]. B [1, 3]. C [9]. D [2].", results)
    assert [s.number for s in sources] == [2, 1, 3]
    assert [s.chunk_id for s in sources] == [
        results[1].chunk_id,
        results[0].chunk_id,
        results[2].chunk_id,
    ]
