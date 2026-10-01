"""Tests for LLM question drafting (LLM mocked) and reviewing a draft with add_question.py."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
from anthropic.resources.messages import Messages

from rag.drafting import draft_questions, parse_json_array
from rag.eval_schema import append_item, load_items
from rag.eval_utils import load_corpus

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"
QUOTE = (
    "Your landlord or letting agent must put your deposit in the scheme within 30 days of "
    "getting it."
)
SECOND_QUOTE = "Your landlord must return your deposit within 10 days of you both agreeing"


@pytest.fixture(scope="module")
def doc():
    return load_corpus(SAMPLE)["tenancy-deposit-protection"]


def reply(*proposals) -> str:
    return "Here are the drafts:\n" + json.dumps(list(proposals))


GOOD = {
    "question": "How long does my landlord have to protect my deposit?",
    "question_type": "factual",
    "quotes": [QUOTE],
    "reference_answer": "Within 30 days of getting it.",
}


def test_drafts_are_marked_llm_draft_with_draft_ids(doc, fake_client_factory):
    multi = {
        "question": "When must my deposit be protected and returned?",
        "question_type": "multi_passage",
        "quotes": [QUOTE, SECOND_QUOTE],
        "reference_answer": "Protected within 30 days; returned within 10 days of agreeing.",
    }
    client = fake_client_factory(reply(GOOD, multi))
    result = draft_questions(doc, client, "claude-test", n=2)

    assert [d.id for d in result.drafts] == ["draft-001", "draft-002"]
    assert all(d.author == "llm_draft" and d.split == "dev" for d in result.drafts)
    assert all("review" in d.notes and "not found" not in d.notes for d in result.drafts)
    assert result.rejected == []


def test_prompt_contains_the_document_and_request_is_valid_for_the_sdk(doc, fake_client_factory):
    client = fake_client_factory(reply(GOOD))
    draft_questions(doc, client, "claude-test", n=3)
    call = client.messages.calls[0]
    assert QUOTE in call["messages"][0]["content"]
    assert "Propose 3 questions" in call["messages"][0]["content"]
    assert set(call) <= set(inspect.signature(Messages.create).parameters)


def test_paraphrased_quotes_are_flagged_for_fixing(doc, fake_client_factory):
    paraphrase = {**GOOD, "quotes": ["Landlords have to protect deposits within thirty days."]}
    result = draft_questions(doc, fake_client_factory(reply(paraphrase)), "claude-test")
    (draft,) = result.drafts
    assert "not found verbatim" in draft.notes


def test_invalid_proposals_are_rejected_with_reasons(doc, fake_client_factory):
    proposals = [
        {**GOOD, "question_type": "trivia"},
        {**GOOD, "quotes": []},
        {**GOOD, "question_type": "unanswerable", "quotes": []},
        "not an object",
        GOOD,
    ]
    result = draft_questions(doc, fake_client_factory(reply(*proposals)), "claude-test")
    assert len(result.drafts) == 1 and len(result.rejected) == 4


def test_draft_ids_continue_after_existing_drafts(doc, fake_client_factory):
    first = draft_questions(doc, fake_client_factory(reply(GOOD)), "claude-test").drafts
    second = draft_questions(
        doc, fake_client_factory(reply(GOOD)), "claude-test", existing_drafts=first
    )
    assert second.drafts[0].id == "draft-002"


def test_truncated_reply_is_an_error(doc, fake_client_factory):
    client = fake_client_factory(reply(GOOD), stop_reason="max_tokens")
    with pytest.raises(ValueError, match="cut off"):
        draft_questions(doc, client, "claude-test")


def test_parse_json_array_rejects_replies_without_an_array():
    with pytest.raises(ValueError):
        parse_json_array("Sorry, I cannot help with that.")


def test_reviewing_a_draft_saves_an_llm_draft_reviewed_item(tmp_path, doc, fake_client_factory):
    (draft,) = draft_questions(doc, fake_client_factory(reply(GOOD)), "claude-test").drafts
    drafts_file, questions_file = tmp_path / "drafts.jsonl", tmp_path / "questions.jsonl"
    append_item(drafts_file, draft)

    answers = [
        "How long has my landlord got to put my deposit in a scheme?",  # edited question
        "",  # keep type
        "y",  # keep the draft quote
        "",  # finish searching
        "",  # keep reference answer
        "",  # keep default notes
        "y",
    ]
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "add_question.py"),
            "--questions",
            str(questions_file),
            "--corpus",
            str(SAMPLE),
            "--drafts",
            str(drafts_file),
            "--from-draft",
            "draft-001",
        ],
        input="\n".join(answers) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (item,) = load_items(questions_file)
    assert item.id == "q-001" and item.author == "llm_draft_reviewed"
    assert item.question.startswith("How long has my landlord got")
    assert item.evidence == draft.evidence
    assert item.notes == "Reviewed from draft-001"
