"""Tests for the Gradio demo: answer rendering, error display, and mounting on FastAPI."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import build_app
from app.service import DemoAnswer, DemoError, DemoService, Settings
from app.ui import answer_body, make_handler, readable_date, render_answer, render_details
from tests.test_api import make_service

MODEL_ANSWER = """Your landlord must protect your deposit within 30 days [1].

Guidance retrieved: 2026-10-01

Sources:
[1] Tenancy deposit protection - https://www.gov.uk/tenancy-deposit-protection

This is general information from GOV.UK guidance, not legal advice."""


def result(**overrides) -> DemoAnswer:
    data = dict(
        answer=MODEL_ANSWER,
        refused=False,
        sources=[
            {
                "number": 1,
                "title": "Tenancy deposit protection",
                "section": "Overview",
                "url": "https://www.gov.uk/tenancy-deposit-protection",
                "date_retrieved": "2026-10-01",
            }
        ],
        passages=[
            {
                "rank": 1,
                "title": "Tenancy deposit protection",
                "section": "Overview",
                "url": "https://www.gov.uk/tenancy-deposit-protection",
                "score": 7.6,
                "text": "Your landlord must put your deposit in a scheme within 30 days.",
            }
        ],
        latency_ms={"dense": 12.0, "rerank": 700.0, "generation": 2500.0, "total": 3300.0},
        config_name="hybrid_rerank_bge",
        prompt_version="v2",
        date_retrieved="2026-10-01",
    )
    data.update(overrides)
    return DemoAnswer(**data)


def test_answer_body_drops_the_models_own_trailer():
    assert (
        answer_body(MODEL_ANSWER) == "Your landlord must protect your deposit within 30 days [1]."
    )


def test_rendered_answer_has_numbered_links_date_and_disclaimer():
    md = render_answer(result())
    assert (
        "1. [Tenancy deposit protection > Overview](https://www.gov.uk/tenancy-deposit-protection)"
        in md
    )
    assert "retrieved from GOV.UK on 1 October 2026" in md
    assert "not legal advice" in md
    assert md.count("Sources") == 1  # the model's own list is not repeated


def test_refusal_renders_without_sources():
    refusal = (
        "I can't find that in the guidance. Try Shelter.\n\n"
        "This is general information from GOV.UK guidance, not legal advice."
    )
    md = render_answer(result(answer=refusal, refused=True, sources=[], date_retrieved=None))
    assert md.startswith("I can't find that in the guidance.") and "**Sources**" not in md


def test_details_show_timings_and_passages():
    md = render_details(result())
    assert "| rerank | 700 ms |" in md and "[1] Tenancy deposit protection" in md
    assert "hybrid_rerank_bge" in md and "RESULTS.md" in md


def test_readable_date():
    assert readable_date("2026-10-01") == "1 October 2026"


def test_handler_shows_friendly_errors():
    class Failing:
        def ask(self, question, client_id):
            raise DemoError(429, "Please wait 30 seconds.")

    request = SimpleNamespace(headers={"x-forwarded-for": "203.0.113.9"}, client=None)
    answer, details = make_handler(Failing())("Deposit?", request)
    assert answer == "⚠️ Please wait 30 seconds." and details == ""


def test_demo_page_and_api_are_served_by_one_app(loaded_index, embedder, fake_client_factory):
    service = make_service(loaded_index, embedder, fake_client_factory("Answer [1]."))
    service.guidance_date = "2026-10-01"
    client = TestClient(build_app(service, load=False))

    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert client.get("/health").json()["config_name"] == "test_config"  # API not shadowed
    assert client.post("/ask", json={"question": "Deposit deadline?"}).status_code == 200


def test_app_builds_without_loading_models():
    client = TestClient(
        build_app(DemoService(Settings(config_name="hybrid_rerank_bge")), load=False)
    )
    assert client.get("/health").json()["index_loaded"] is False
    assert {"/ask", "/health"} <= set(client.get("/openapi.json").json()["paths"])
