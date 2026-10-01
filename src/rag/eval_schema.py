"""Schema for evaluation test items, stored one per line in eval/questions.jsonl.

Gold evidence is stored as (doc_id, exact quote), never as chunk_ids: chunk_ids change
whenever the chunking settings change (an ablation in the evaluation), but a quote from the
document stays valid for any chunking, and is mapped to chunk_ids at evaluation time.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

QuestionType = Literal["factual", "multi_passage", "informal", "unanswerable"]
Split = Literal["dev", "heldout"]
# "max": written by me. "llm_draft_reviewed": started as an LLM draft, then reviewed and
# edited by me. "llm_draft": unreviewed draft, only allowed in eval/drafts.jsonl.
# "example": format examples, excluded from all metrics.
# "llm_generated": LLM-generated and automatically checked (quotes verified word for word),
# but NOT individually reviewed by a person.
Author = Literal["max", "llm_draft_reviewed", "llm_generated", "llm_draft", "example"]

QUESTION_TYPES: tuple[str, ...] = QuestionType.__args__
MIN_QUOTE_CHARS = 20  # long enough that a quote pins down one specific statement
ID_PREFIX = {
    "max": "q",
    "llm_draft_reviewed": "q",
    "llm_generated": "q",
    "example": "ex",
    "llm_draft": "draft",
}
_ID_RE = re.compile(r"^(q|ex|draft)-(\d{3,})$")


class Evidence(BaseModel):
    """A passage that supports the reference answer: a document and an exact quote from it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1)
    quote: str = Field(min_length=MIN_QUOTE_CHARS)


class EvalItem(BaseModel):
    """One evaluation question with its reference answer and gold evidence."""

    model_config = ConfigDict(extra="forbid")  # a typo in a field name is an error, not ignored

    id: str = Field(pattern=_ID_RE.pattern)
    question: str = Field(min_length=10)
    reference_answer: str = Field(min_length=1)
    question_type: QuestionType
    evidence: list[Evidence] = Field(default_factory=list)
    split: Split = "dev"
    author: Author
    notes: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> EvalItem:
        if self.question_type == "unanswerable":
            if self.evidence:
                raise ValueError("unanswerable items must have no evidence")
        elif not self.evidence:
            raise ValueError(f"{self.question_type} items need at least one evidence quote")
        if self.question_type == "multi_passage" and len({e.quote for e in self.evidence}) < 2:
            raise ValueError("multi_passage items need at least two different evidence quotes")
        prefix = _ID_RE.match(self.id).group(1)
        if prefix != ID_PREFIX[self.author]:
            raise ValueError(
                f"id '{self.id}' should start with '{ID_PREFIX[self.author]}-' "
                f"for author '{self.author}'"
            )
        return self

    @property
    def answerable(self) -> bool:
        return self.question_type != "unanswerable"

    @property
    def is_example(self) -> bool:
        return self.author == "example"


class ItemFileError(ValueError):
    """Raised when a JSONL file contains invalid items; lists every problem with its line."""

    def __init__(self, path: Path, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(f"{path}: {len(problems)} invalid line(s)\n" + "\n".join(problems))


def parse_items(path: Path) -> tuple[list[EvalItem], list[str]]:
    """Read a JSONL file, returning the valid items and a message for each invalid line."""
    items, problems = [], []
    if not path.exists():
        return items, problems
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(EvalItem.model_validate(json.loads(line)))
        except json.JSONDecodeError as err:
            problems.append(f"line {line_no}: not valid JSON ({err.msg})")
        except ValidationError as err:
            details = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'item'}: {e['msg']}" for e in err.errors()
            )
            problems.append(f"line {line_no}: {details}")
    return items, problems


def load_items(path: Path) -> list[EvalItem]:
    """Load every item from a JSONL file, raising ItemFileError if any line is invalid."""
    items, problems = parse_items(path)
    if problems:
        raise ItemFileError(path, problems)
    return items


def item_to_json(item: EvalItem) -> str:
    return json.dumps(item.model_dump(), ensure_ascii=False)


def write_items(path: Path, items: Iterable[EvalItem]) -> None:
    path.write_text("".join(item_to_json(i) + "\n" for i in items), encoding="utf-8")


def append_item(path: Path, item: EvalItem) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(item_to_json(item) + "\n")


def next_id(items: Iterable[EvalItem], author: str) -> str:
    """The next free id for an author, e.g. 'q-007' after 'q-006'."""
    prefix = ID_PREFIX[author]
    numbers = [int(m.group(2)) for i in items if (m := _ID_RE.match(i.id)) and m.group(1) == prefix]
    return f"{prefix}-{max(numbers, default=0) + 1:03d}"
