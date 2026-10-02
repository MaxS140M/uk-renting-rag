"""Server entry point: the FastAPI API with the Gradio demo mounted, served by one process.

    uvicorn app.main:app --host 0.0.0.0 --port 7860 --no-access-log

(--no-access-log: uvicorn's access log records visitors' IP addresses, which this demo
promises not to keep.)

API: POST /ask, GET /health, interactive docs at /docs. Demo page: /.
"""

from __future__ import annotations

import logging
import os

# Gradio checks for updates and sends usage analytics by default. A demo that promises not
# to log visitors' questions should not make calls nobody asked for.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402  (imported after the environment setting above)
from fastapi import FastAPI

from app.api import create_app
from app.service import DemoService
from app.ui import build_ui

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def build_app(service: DemoService | None = None, load: bool = True) -> FastAPI:
    service = service or DemoService()
    api = create_app(service, load=load)  # API routes are added first, so they take priority
    return gr.mount_gradio_app(api, build_ui(service), path="/")


app = build_app()
