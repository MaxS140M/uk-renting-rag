"""Gradio demo page: ask a question, see a cited answer and, for reviewers, how it was made."""

from __future__ import annotations

import os
import re
from datetime import date

import gradio as gr

from app.service import MAX_QUESTION_CHARS, DemoAnswer, DemoError, DemoService
from rag.config import GITHUB_URL

RESULTS_URL = f"{GITHUB_URL}/blob/main/eval/RESULTS.md"
EXAMPLES = [
    "Does my landlord have to protect my deposit in a scheme?",
    "Can my landlord evict me without giving a reason?",
    "How much notice do I get before my rent goes up?",
    "Who has to fix a broken boiler in my rented flat?",
    # Not covered by the guidance: shows the assistant refusing instead of guessing.
    "What is the average rent for a one-bedroom flat in Manchester?",
]

# Lines the model adds at the end of an answer; the page shows these itself, from the
# structured result, so links and dates are always accurate.
_TRAILER_RE = re.compile(
    r"^\s*(sources:|guidance retrieved:|this is general information from gov\.uk)",
    re.IGNORECASE,
)


def cold_start_note() -> str:
    """Hugging Face Spaces set SPACE_ID; free Spaces sleep when idle, so warn visitors there."""
    if not os.environ.get("SPACE_ID"):
        return ""
    return (
        "⏳ This runs on free hosting. If nobody has used it for a while, the first answer "
        "can take up to a minute while it wakes up.\n\n"
    )


def readable_date(iso: str | None) -> str:
    if not iso:
        return "the retrieval date shown with each source"
    day = date.fromisoformat(iso)
    return f"{day.day} {day.strftime('%B %Y')}"


def answer_body(answer: str) -> str:
    """The answer without the model's own sources list, retrieval line and disclaimer."""
    lines = answer.splitlines()
    for i, line in enumerate(lines):
        if _TRAILER_RE.match(line):
            lines = lines[:i]
            break
    return "\n".join(lines).strip()


def render_answer(result: DemoAnswer) -> str:
    parts = [answer_body(result.answer)]
    if result.sources:
        links = [
            f"{s['number']}. [{s['title']}{' > ' + s['section'] if s['section'] else ''}]"
            f"({s['url']})"
            for s in result.sources
        ]
        parts += ["**Sources**", "\n".join(links)]
    if result.date_retrieved:
        parts.append(f"*Guidance retrieved from GOV.UK on {readable_date(result.date_retrieved)}.*")
    parts.append("*General information from GOV.UK guidance, not legal advice.*")
    return "\n\n".join(parts)


def render_details(result: DemoAnswer) -> str:
    rows = "\n".join(f"| {stage} | {ms:,.0f} ms |" for stage, ms in result.latency_ms.items())
    passages = "\n\n".join(
        f"**[{p['rank']}] {p['title']}** > {p['section'] or '(no section)'} "
        f"(reranker score {p['score']})\n\n> "
        + (p["text"][:400].replace("\n", " ") + ("…" if len(p["text"]) > 400 else ""))
        for p in result.passages
    )
    return (
        f"Configuration `{result.config_name}`, prompt `{result.prompt_version}`: hybrid "
        "search (dense embeddings + BM25 fused with Reciprocal Rank Fusion), then a "
        f"cross-encoder reranker; the top passages go to Claude Haiku 4.5. "
        f"[Evaluation results]({RESULTS_URL}).\n\n"
        f"| Stage | Time |\n|---|---:|\n{rows}\n\n"
        f"**Passages given to the model**\n\n{passages}"
    )


def client_id(request: gr.Request | None) -> str:
    if request is None:
        return "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def make_handler(service: DemoService):
    def handle(question: str, request: gr.Request) -> tuple[str, str]:
        try:
            result = service.ask(question, client_id(request))
        except DemoError as err:
            return f"⚠️ {err.message}", ""
        except Exception:  # never show internals to a visitor
            return "⚠️ Something went wrong on our side. Please try again.", ""
        return render_answer(result), render_details(result)

    return handle


def build_ui(service: DemoService) -> gr.Blocks:
    guidance_date = readable_date(service.guidance_date)
    with gr.Blocks(title="UK Renting Guidance Assistant") as demo:
        gr.Markdown(
            "# UK Renting Guidance Assistant\n"
            "Ask a question about renting a home in England. Answers come only from official "
            "GOV.UK guidance, with numbered links to the sources."
        )
        gr.Markdown(
            f"**Demo project. Not legal advice. Covers GOV.UK guidance for England as of "
            f"{guidance_date}.** Renting law changed on 1 May 2026; check GOV.UK, Citizens "
            "Advice or Shelter before acting.\n\n"
            + cold_start_note()
            + "🔒 Your question is sent to Anthropic's API to write the answer. This app does "
            "not store it: only anonymous metrics (timings, errors, whether it refused) are "
            "logged. Please don't include names, addresses or other personal details."
        )
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g. Can my landlord keep my deposit for normal wear and tear?",
            max_length=MAX_QUESTION_CHARS,
            lines=2,
        )
        ask = gr.Button("Ask", variant="primary")
        gr.Markdown("**Try an example** (the last one isn't covered by the guidance):")
        with gr.Row():
            example_buttons = [gr.Button(text, size="sm") for text in EXAMPLES]
        answer = gr.Markdown(label="Answer")
        with gr.Accordion("How it works: retrieved passages and timings", open=False):
            details = gr.Markdown()

        handle = make_handler(service)
        ask.click(handle, inputs=question, outputs=[answer, details])
        question.submit(handle, inputs=question, outputs=[answer, details])
        for button, text in zip(example_buttons, EXAMPLES, strict=True):
            button.click(lambda t=text: t, outputs=question).then(
                handle, inputs=question, outputs=[answer, details]
            )

        gr.Markdown(
            f"[Code on GitHub]({GITHUB_URL}) · [Evaluation results]({RESULTS_URL}) · "
            "Code under the MIT licence. Contains public sector information licensed under "
            "the [Open Government Licence v3.0]"
            "(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)."
        )
    return demo
