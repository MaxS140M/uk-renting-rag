"""API tests with FastAPI's TestClient. The pipeline uses a tiny fake index and a mocked
LLM, so no models are loaded and no API calls are made."""

import logging
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.service import DemoService, Settings
from rag.config import RAGConfig
from rag.generate import Generator
from rag.pipeline import RAGPipeline
from rag.retrieval import DenseRetriever

QUESTION = "How long does my landlord have to protect my deposit?"


def make_service(loaded_index, embedder, client, **settings) -> DemoService:
    config = RAGConfig(final_k=3, candidate_k=3)
    pipeline = RAGPipeline(
        config,
        retriever=DenseRetriever(loaded_index, embedder),
        generator=Generator(config, client=client),
    )
    return DemoService(Settings(config_name="test_config", **settings), pipeline=pipeline)


@pytest.fixture
def api(loaded_index, embedder, fake_client_factory):
    def build(reply="Within 30 days [1].\n\nThis is general information.", client=None, **settings):
        client = client or fake_client_factory(reply)
        service = make_service(loaded_index, embedder, client, **settings)
        return TestClient(create_app(service, load=False), raise_server_exceptions=False)

    return build


def test_health_reports_loaded_components(api):
    response = api().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["index_loaded"] and body["llm_configured"]
    assert body["config_name"] == "test_config"


def test_health_is_degraded_before_loading():
    service = DemoService(Settings(config_name="x"))
    body = TestClient(create_app(service, load=False)).get("/health").json()
    assert body["status"] == "degraded" and body["index_loaded"] is False


def test_ask_returns_answer_sources_and_timings(api):
    response = api().post("/ask", json={"question": QUESTION})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"].startswith("Within 30 days")
    assert body["refused"] is False
    assert body["sources"] == [
        {
            "number": 1,
            "title": "Tenancy deposit protection",
            "url": "https://www.gov.uk/deposit",
            "date_retrieved": "2026-10-01",
        }
    ]
    assert {"retrieval", "generation", "total"} <= set(body["latency_ms"])
    assert body["config_name"] == "test_config" and body["prompt_version"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"question": ""}, "Please type a question."),
        ({"question": "   \n "}, "Please type a question."),
        ({"question": "x" * 501}, "500 characters or fewer"),
        ({}, "Send JSON like"),
        ({"question": 42}, "valid string"),
    ],
)
def test_invalid_input_is_rejected_with_a_clear_message(api, payload, message):
    response = api().post("/ask", json=payload)
    assert response.status_code == 422
    assert message in response.json()["error"]


def test_exactly_500_characters_is_allowed(api):
    assert api().post("/ask", json={"question": "deposit " * 62 + "rent"}).status_code == 200


def test_rate_limit_is_per_client(api):
    client = api(rate_limit_per_minute=2)
    first_ip = {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}
    for _ in range(2):
        assert client.post("/ask", json={"question": QUESTION}, headers=first_ip).status_code == 200
    limited = client.post("/ask", json={"question": QUESTION}, headers=first_ip)
    assert limited.status_code == 429
    assert "per minute" in limited.json()["error"] and int(limited.headers["Retry-After"]) > 0
    other_ip = {"X-Forwarded-For": "198.51.100.7"}
    assert client.post("/ask", json={"question": QUESTION}, headers=other_ip).status_code == 200


def test_daily_cap_returns_a_polite_message(api):
    client = api(daily_llm_cap=1)
    assert client.post("/ask", json={"question": QUESTION}).status_code == 200
    capped = client.post("/ask", json={"question": QUESTION})
    assert capped.status_code == 429 and "try again tomorrow" in capped.json()["error"]
    assert client.get("/health").json()["daily_llm_calls_remaining"] == 0


class FailingMessages:
    def __init__(self, error):
        self.error = error

    def create(self, **kwargs):
        raise self.error


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (anthropic.APITimeoutError(request=REQUEST), 504, "took too long"),
        (anthropic.APIConnectionError(request=REQUEST), 502, "unavailable right now"),
        (RuntimeError("secret internal detail"), 500, "Something went wrong"),
    ],
)
def test_llm_failures_return_friendly_errors_without_internals(api, error, status, message):
    client = api(client=SimpleNamespace(messages=FailingMessages(error)))
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == status
    assert message in response.json()["error"]
    assert "Traceback" not in response.text and "secret internal detail" not in response.text


def test_unconfigured_llm_is_reported_not_crashed(loaded_index, embedder, fake_client_factory):
    service = make_service(loaded_index, embedder, fake_client_factory())
    service.llm_configured = False
    response = TestClient(create_app(service, load=False)).post("/ask", json={"question": QUESTION})
    assert response.status_code == 503 and "isn't configured" in response.json()["error"]


def test_question_text_is_never_logged(api, caplog):
    secret = "my landlord John Smith at 12 Acacia Avenue won't return my deposit"
    with caplog.at_level(logging.DEBUG):
        api().post("/ask", json={"question": secret})
        api(rate_limit_per_minute=0).post("/ask", json={"question": secret})
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "John Smith" not in logged and "Acacia" not in logged
    assert '"outcome": "ok"' in logged and '"outcome": "rate_limited"' in logged


def test_rate_limiter_window_slides_with_time():
    from app.service import RateLimiter

    now = [0.0]
    limiter = RateLimiter(limit=2, window=60, clock=lambda: now[0])
    assert limiter.check("ip") is None and limiter.check("ip") is None
    assert limiter.check("ip") == 61  # oldest request expires in 60 s
    now[0] = 60.0
    assert limiter.check("ip") is None  # the first request has left the window
    assert RateLimiter(limit=0).check("ip") == 60


def test_daily_cap_resets_on_a_new_utc_day():
    from datetime import date

    from app.service import DailyCap

    day = [date(2026, 10, 2)]
    cap = DailyCap(limit=1, today=lambda: day[0])
    assert cap.try_use() and not cap.try_use()
    day[0] = date(2026, 10, 3)
    assert cap.remaining == 1 and cap.try_use()
