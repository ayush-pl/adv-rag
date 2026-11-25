"""Interactive verification of drafted questions. Every decision is saved immediately."""

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from ragbench.dataset.drafting import build_evidence_group, locate_quote
from ragbench.dataset.models import TARGET_COUNTS, Question
from ragbench.evaluation.gold import GoldSpan
from ragbench.ingestion.models import Document

CONTEXT_CHARS = 350

HELP = """Commands:
  a  accept as shown       r  reject (asks for a reason)    s  skip for now
  q  rewrite the question  n  rewrite the expected answer   d  change difficulty
  e  replace evidence (paste text copied from the document)
  g  change evidence grade (2 = answers directly, 1 = supports)
  x  save and exit"""


# What to check for each type, beyond whether the question is clear and the answer is right.
TYPE_CHECKS = {
    "single_hop": "Is one fact enough, and does the evidence state it?",
    "multi_hop": "Is EVERY evidence passage truly needed? Reject if one alone answers it.",
    "config_code": "Does the answer depend on the code or configuration shown?",
    "table": "Is the answer actually in this table row?",
    "version_specific": "Would PostgreSQL's other version answer differently? Reject if not.",
    "unanswerable": "Is it a sensible question, and truly not answered? Search the docs if unsure.",
    "paraphrase_ambiguous": "Does it still mean the same thing, with the same answer?",
}


class ReviewDecision(BaseModel):
    question_id: str
    decision: str
    reason: str = ""
    question: Question | None = None


def read_decisions(path: Path) -> dict[str, ReviewDecision]:
    """Return the latest decision per question. The file is an append-only history."""
    decisions: dict[str, ReviewDecision] = {}
    if path.exists():
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    decision = ReviewDecision.model_validate(json.loads(line))
                    decisions[decision.question_id] = decision
    return decisions


def accepted_questions(path: Path) -> list[Question]:
    return [
        decision.question
        for decision in read_decisions(path).values()
        if decision.decision == "accept" and decision.question is not None
    ]


def append_decision(path: Path, record: ReviewDecision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as out:
        out.write(record.model_dump_json() + "\n")


def evidence_context(document: Document, span: GoldSpan) -> tuple[str, str]:
    """The document text just before and after a span, for judging it in place."""
    before = document.content[max(0, span.char_start - CONTEXT_CHARS) : span.char_start]
    after = document.content[span.char_end : span.char_end + CONTEXT_CHARS]
    return before, after


def replace_evidence(
    question: Question, index: int, text: str, documents: dict[str, Document]
) -> bool:
    """Point an evidence group at text pasted from its document. False if the text isn't there."""
    group = question.evidence[index]
    document = documents[group.spans[0].document_id]
    located = locate_quote(document.content, text)
    if located is None:
        return False
    question.evidence[index] = build_evidence_group(
        document,
        *located,
        documents,
        include_other_versions=question.document_version is None,
        grade=group.grade,
    )
    question.provenance.edited_by_reviewer = True
    return True


class ReviewSession:
    def __init__(
        self,
        drafts: list[Question],
        decisions_path: Path,
        documents: dict[str, Document],
        ask: Callable[[str], str] = input,
        show: Callable[[str], None] = print,
    ) -> None:
        self._drafts = drafts
        self._path = decisions_path
        self._documents = documents
        self._ask = ask
        self._show = show

    def run(self) -> None:
        decisions = read_decisions(self._path)
        pending = [d for d in self._drafts if d.question_id not in decisions]
        self._show(HELP)
        accepted = Counter(
            d.question.question_type
            for d in decisions.values()
            if d.decision == "accept" and d.question is not None
        )
        self._show(self._progress(accepted))

        for position, draft in enumerate(pending, start=1):
            if accepted[draft.question_type] >= TARGET_COUNTS[draft.question_type]:
                continue
            self._show(f"\n{'=' * 78}\nDraft {position} of {len(pending)}")
            outcome = self._review_one(draft)
            if outcome == "exit":
                break
            if outcome == "accept":
                accepted[draft.question_type] += 1
                self._show(self._progress(accepted))
        self._show("Progress saved.")

    def _review_one(self, draft: Question) -> str:
        question = draft.model_copy(deep=True)
        while True:
            self._show(self._render(question))
            command = self._ask("\n[a/r/s/q/n/e/g/d/x] > ").strip().lower()
            if command == "a":
                question.human_verified = True
                self._save(draft.question_id, "accept", question=question)
                return "accept"
            if command == "r":
                self._save(draft.question_id, "reject", reason=self._ask("Reason > ").strip())
                return "reject"
            if command == "s":
                return "skip"
            if command == "x":
                return "exit"
            if command == "q":
                question.question = self._edited(question, self._ask("New question > "), "question")
            elif command == "n":
                question.expected_answer = self._edited(
                    question, self._ask("New answer > "), "expected_answer"
                )
            elif command == "d":
                self._change_difficulty(question)
            elif command == "e":
                self._replace_evidence(question)
            elif command == "g":
                self._change_grade(question)
            else:
                self._show(HELP)

    def _edited(self, question: Question, new_value: str, field: str) -> str:
        cleaned = new_value.strip()
        if not cleaned:
            self._show("Empty input, nothing changed.")
            return getattr(question, field)
        question.provenance.edited_by_reviewer = True
        return cleaned

    def _change_difficulty(self, question: Question) -> None:
        value = self._ask("Difficulty (easy/medium/hard) > ").strip().lower()
        if value in ("easy", "medium", "hard"):
            question.difficulty = value
        else:
            self._show("Not a valid difficulty, nothing changed.")

    def _choose_group(self, question: Question) -> int | None:
        if not question.evidence:
            self._show("This question has no evidence.")
            return None
        if len(question.evidence) == 1:
            return 0
        value = self._ask(f"Which evidence (1-{len(question.evidence)}) > ").strip()
        if value.isdigit() and 1 <= int(value) <= len(question.evidence):
            return int(value) - 1
        self._show("Not a valid number, nothing changed.")
        return None

    def _change_grade(self, question: Question) -> None:
        index = self._choose_group(question)
        if index is None:
            return
        value = self._ask("Grade (1 or 2) > ").strip()
        if value in ("1", "2"):
            question.evidence[index].grade = int(value)
        else:
            self._show("Not a valid grade, nothing changed.")

    def _replace_evidence(self, question: Question) -> None:
        index = self._choose_group(question)
        if index is None:
            return
        document_id = question.evidence[index].spans[0].document_id
        text = self._ask("Paste exact text > ")
        if not replace_evidence(question, index, text, self._documents):
            self._show(f"That text is not in {document_id}, nothing changed.")

    def _save(self, question_id: str, decision: str, reason: str = "", question=None) -> None:
        record = ReviewDecision(
            question_id=question_id, decision=decision, reason=reason, question=question
        )
        append_decision(self._path, record)

    def _render(self, question: Question) -> str:
        lines = [
            f"Type: {question.question_type}    Difficulty: {question.difficulty}"
            + (f"    Version: {question.document_version}" if question.document_version else ""),
            f"\nQUESTION: {question.question}",
            f"\nEXPECTED ANSWER: {question.expected_answer}",
            f"\nCheck: {TYPE_CHECKS[question.question_type]}",
        ]
        if question.lexical_overlap is not None and question.lexical_overlap > 0.8:
            lines.append(
                f"\nNote: {question.lexical_overlap:.0%} of its key words appear in the source. "
                "Consider rewording with q."
            )
        for number, group in enumerate(question.evidence, start=1):
            lines.append(self._render_group(number, group))
        if question.nearest_passages:
            lines.append("\nCheck these closest passages do NOT answer it:")
            lines.extend(f"  - {passage}" for passage in question.nearest_passages)
        return "\n".join(lines)

    def _render_group(self, number: int, group) -> str:
        span = group.spans[0]
        document = self._documents[span.document_id]
        before, after = evidence_context(document, span)
        also = [other.document_id for other in group.spans[1:]]
        return (
            f"\nEVIDENCE {number} (grade {group.grade}): {document.title}\n"
            f"  {document.source_url}\n"
            + (f"  Also accepted in: {', '.join(also)}\n" if also else "")
            + f"\n...{before}\n>>>>> {span.quote} <<<<<\n{after}..."
        )

    def _progress(self, accepted: Counter) -> str:
        parts = [f"{name} {accepted[name]}/{target}" for name, target in TARGET_COUNTS.items()]
        total = sum(accepted[name] for name in TARGET_COUNTS)
        return f"\nAccepted {total}/{sum(TARGET_COUNTS.values())}: " + ", ".join(parts)
