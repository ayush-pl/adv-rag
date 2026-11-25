"""Schema for benchmark questions."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ragbench.evaluation.gold import EvidenceGroup

QuestionType = Literal[
    "single_hop",
    "multi_hop",
    "config_code",
    "table",
    "version_specific",
    "unanswerable",
    "paraphrase_ambiguous",
]
Difficulty = Literal["easy", "medium", "hard"]
Split = Literal["development", "calibration", "test"]

# Spec section 4.1 targets, adjusted by 5 single-hop questions to reach the plan's 120.
TARGET_COUNTS: dict[str, int] = {
    "single_hop": 40,
    "multi_hop": 20,
    "config_code": 15,
    "table": 10,
    "version_specific": 10,
    "unanswerable": 15,
    "paraphrase_ambiguous": 10,
}


class Provenance(BaseModel):
    """Where a question came from, so results can be checked for drafting bias."""

    method: Literal["llm_drafted", "manual"]
    model: str | None = None
    model_digest: str | None = None
    # For a paraphrase, the question it rewrites. Both must land in the same split.
    source_question_id: str | None = None
    edited_by_reviewer: bool = False


class Question(BaseModel):
    question_id: str
    question: str = Field(min_length=1)
    question_type: QuestionType
    expected_answer: str = Field(min_length=1)
    evidence: list[EvidenceGroup] = Field(default_factory=list)
    answerable: bool
    document_version: str | None = None
    difficulty: Difficulty = "medium"
    human_verified: bool = False
    provenance: Provenance
    # Share of the question's content words that appear in its source passage.
    lexical_overlap: float | None = None
    # For an unanswerable draft, the closest passages found, to help the reviewer confirm it.
    nearest_passages: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_matches_type(self) -> "Question":
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.question_id}: an answerable question needs evidence")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.question_id}: an unanswerable question cannot have evidence")
        if self.answerable == (self.question_type == "unanswerable"):
            raise ValueError(f"{self.question_id}: answerable flag contradicts its type")
        if self.question_type == "multi_hop" and len(self.evidence) < 2:
            raise ValueError(f"{self.question_id}: a multi-hop question needs 2 evidence groups")
        return self


def read_questions(path: Path) -> list[Question]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [Question.model_validate(json.loads(line)) for line in lines if line.strip()]


def write_questions(path: Path, questions: list[Question]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for question in questions:
            out.write(question.model_dump_json() + "\n")


def append_question(path: Path, question: Question) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as out:
        out.write(question.model_dump_json() + "\n")
