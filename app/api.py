"""FastAPI endpoints: POST /ask and GET /health.

Every error, including invalid input, comes back as JSON {"error": "..."} with a friendly
message and a suitable status code; stack traces are never sent to the client.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.service import MAX_QUESTION_CHARS, DemoError, DemoService, validate_question

log = logging.getLogger("uk_renting_rag")


class AskRequest(BaseModel):
    question: str = Field(
        description=f"A question about renting in England, up to {MAX_QUESTION_CHARS} characters"
    )

    @field_validator("question")
    @classmethod
    def _check(cls, value: str) -> str:
        try:
            return validate_question(value)
        except DemoError as err:
            raise ValueError(err.message) from err


class Source(BaseModel):
    number: int
    title: str
    url: str
    date_retrieved: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]
    latency_ms: dict[str, float]
    config_name: str
    prompt_version: str


class Health(BaseModel):
    status: str
    index_loaded: bool
    models_loaded: bool
    llm_configured: bool
    config_name: str
    embedding_model: str | None
    daily_llm_calls_remaining: int
    error: str | None


def client_id(request: Request) -> str:
    """The visitor's IP address. Behind a proxy (as on Hugging Face) it is the first address
    in X-Forwarded-For; it is only used, in memory, for the per-minute rate limit."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def build_router(service: DemoService) -> APIRouter:
    router = APIRouter()

    @router.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        # Retrieval and the LLM call are blocking, so run them off the event loop.
        result = await run_in_threadpool(service.ask, body.question, client_id(request))
        return AskResponse(
            answer=result.answer,
            refused=result.refused,
            sources=[Source(**{k: s[k] for k in Source.model_fields}) for s in result.sources],
            latency_ms=result.latency_ms,
            config_name=result.config_name,
            prompt_version=result.prompt_version,
        )

    @router.get("/health", response_model=Health)
    def health() -> Health:
        return Health(**service.health())

    return router


def add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DemoError)
    async def demo_error(request: Request, err: DemoError) -> JSONResponse:
        headers = {"Retry-After": str(err.retry_after)} if err.retry_after else None
        return JSONResponse({"error": err.message}, status_code=err.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, err: RequestValidationError) -> JSONResponse:
        first = err.errors()[0] if err.errors() else {}
        message = str(first.get("msg", "Invalid request.")).removeprefix("Value error, ")
        if first.get("type") == "missing":
            message = 'Send JSON like {"question": "..."}.'
        return JSONResponse({"error": message}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, err: Exception) -> JSONResponse:
        log.error("Unhandled error: %s", type(err).__name__)  # the type only, never user input
        return JSONResponse(
            {"error": "Something went wrong on our side. Please try again."}, status_code=500
        )


def create_app(service: DemoService, load: bool = True) -> FastAPI:
    """The API app. With ``load``, models and the index are loaded once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load:
            await run_in_threadpool(service.load)
        yield

    app = FastAPI(
        title="UK Renting Guidance Assistant",
        description="Answers questions about renting in England from GOV.UK guidance, with "
        "citations. Demo project: not legal advice.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(build_router(service))
    add_error_handlers(app)
    return app
