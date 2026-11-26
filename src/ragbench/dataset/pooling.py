"""Pooled relevance review: find relevant passages the gold labels missed.

Labels come from one source passage per fact, but the docs often state a fact in several
places. Any such passage a system retrieves would count as a miss. Pooling, as in TREC,
gathers the top results of several systems, and a reviewer judges the unlabelled ones.
Passages judged relevant become alternative evidence, and every judgment is kept, so the same
passage is never shown twice.

Unanswerable questions get the same check from the other direction: a passage that answers
one means its label is wrong, and the question is withdrawn.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from pydantic import BaseModel

from ragbench.chunking.models import chunk_from_id, parse_chunk_id  # noqa: F401
from ragbench.dataset.drafting import locate_quote
from ragbench.dataset.models import Question
from ragbench.dataset.review import ReviewDecision, read_decisions
from ragbench.evaluation.gold import DEFAULT_MIN_OVERLAP, GoldSpan, chunk_matches_span
from ragbench.ingestion.models import Document

SHEET = "Missed evidence"
YES_VALUES = {"y", "yes"}
NO_VALUES = {"n", "no"}


class Judgment(BaseModel):
    question_id: str
    document_id: str
    char_start: int
    char_end: int
    relevant: bool
    quote: str = ""


@dataclass
class Candidate:
    question: Question
    document_id: str
    char_start: int
    char_end: int
    best_rank: int
    configs: list[str] = field(default_factory=list)


def _overlaps(a: tuple[str, int, int], b: tuple[str, int, int]) -> bool:
    """Same passage for review purposes: same document and mostly the same text."""
    if a[0] != b[0]:
        return False
    shared = min(a[2], b[2]) - max(a[1], b[1])
    return shared > 0 and shared >= DEFAULT_MIN_OVERLAP * min(a[2] - a[1], b[2] - b[1])


def build_pool(
    questions: list[Question],
    rankings: dict[str, dict[str, list[str]]],
    depth: int,
    judgments: list[Judgment],
) -> list[Candidate]:
    """Unlabelled, unjudged passages in the top `depth` of any ranking, one per region of text.

    `rankings` maps a configuration name to each question's ranked chunk IDs.
    """
    judged = {
        q: [(j.document_id, j.char_start, j.char_end) for j in judgments if j.question_id == q]
        for q in {j.question_id for j in judgments}
    }
    candidates: list[Candidate] = []
    for question in questions:
        found: list[Candidate] = []
        for name in sorted(rankings):
            for rank, chunk_id in enumerate(
                rankings[name].get(question.question_id, [])[:depth], 1
            ):
                passage = parse_chunk_id(chunk_id)
                if _is_labelled(question, passage):
                    continue
                if any(_overlaps(passage, seen) for seen in judged.get(question.question_id, [])):
                    continue
                existing = next(
                    (
                        c
                        for c in found
                        if _overlaps(passage, (c.document_id, c.char_start, c.char_end))
                    ),
                    None,
                )
                if existing is None:
                    existing = Candidate(question, *passage, best_rank=rank)
                    found.append(existing)
                existing.best_rank = min(existing.best_rank, rank)
                if name not in existing.configs:
                    existing.configs.append(name)
        candidates.extend(sorted(found, key=lambda c: (c.best_rank, c.document_id, c.char_start)))
    return candidates


def _is_labelled(question: Question, passage: tuple[str, int, int]) -> bool:
    stub = chunk_from_id(f"{passage[0]}::{passage[1]}-{passage[2]}")
    return any(chunk_matches_span(stub, span) for g in question.evidence for span in g.spans)


HEADERS = [
    ("ID", 15),
    ("Type", 14),
    ("Question", 40),
    ("Expected answer", 36),
    ("Labelled evidence", 40),
    ("Retrieved passage (not labelled)", 80),
    ("Source", 34),
    ("Answers it? (y / n)", 11),
    ("Supporting text (copy from the passage, needed for y)", 40),
    ("Evidence # (multi-hop only)", 11),
]
_DECISION, _QUOTE, _GROUP = 8, 9, 10
_EDITABLE_FILL = PatternFill("solid", start_color="FFF4CC")
_HEADER_FILL = PatternFill("solid", start_color="1F3A5F")


def export_pool(candidates: list[Candidate], documents: dict[str, Document], output: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET
    sheet.append([header for header, _ in HEADERS])
    for candidate in candidates:
        question = candidate.question
        document = documents[candidate.document_id]
        labelled = "\n".join(
            f"[{n}] {' '.join(group.spans[0].quote.split())}"
            for n, group in enumerate(question.evidence, start=1)
        )
        if not question.answerable:
            labelled = "UNANSWERABLE: answer y only if this passage answers the question."
        passage = document.content[candidate.char_start : candidate.char_end]
        sheet.append(
            [
                question.question_id,
                question.question_type,
                question.question,
                question.expected_answer,
                labelled,
                passage,
                f"{document.title}\n{document.source_url}\n{candidate.document_id}"
                f"@{candidate.char_start}-{candidate.char_end}",
                "",
                "",
                "",
            ]
        )

    last_row = len(candidates) + 1
    for index, (_, width) in enumerate(HEADERS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
        header = sheet.cell(row=1, column=index)
        header.font = Font(bold=True, color="FFFFFF")
        header.fill = _HEADER_FILL
        header.alignment = Alignment(wrap_text=True, vertical="center")
        for row in range(2, last_row + 1):
            cell = sheet.cell(row=row, column=index)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if index in (_DECISION, _QUOTE, _GROUP):
                cell.fill = _EDITABLE_FILL
    for row in range(2, last_row + 1):
        sheet.row_dimensions[row].height = 180
    sheet.freeze_panes = "D2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{last_row}"
    letter = get_column_letter(_DECISION)
    validation = DataValidation(type="list", formula1='"y,n"', allow_blank=True)
    validation.add(f"{letter}2:{letter}{last_row}")
    sheet.add_data_validation(validation)
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)


@dataclass
class PoolImport:
    judgments: list[Judgment]
    decisions: list[ReviewDecision]
    added_evidence: int
    withdrawn: list[str]
    undecided: int
    problems: list[str]


def read_pool(
    path: Path, decisions_path: Path, documents: dict[str, Document], known: list[Judgment]
) -> PoolImport:
    decided = read_decisions(decisions_path)
    already = {(j.question_id, j.document_id, j.char_start, j.char_end) for j in known}
    updated: dict[str, Question] = {}
    withdrawn: dict[str, str] = {}
    result = PoolImport([], [], 0, [], 0, [])

    sheet = load_workbook(path, read_only=True, data_only=True)[SHEET]
    rows = sheet.iter_rows(min_row=1, values_only=True)
    if [str(v or "").strip() for v in next(rows)][: len(HEADERS)] != [h for h, _ in HEADERS]:
        raise ValueError("The sheet's columns were changed. Export a fresh sheet.")

    for row_number, values in enumerate(rows, start=2):
        values = list(values) + [None] * (len(HEADERS) - len(values))
        question_id = str(values[0] or "").strip()
        if not question_id:
            continue
        where = f"Row {row_number} ({question_id})"
        choice = str(values[_DECISION - 1] or "").strip().lower()
        if not choice:
            result.undecided += 1
            continue
        if choice not in YES_VALUES | NO_VALUES:
            result.problems.append(f"{where}: answer must be y or n, not {choice!r}")
            continue

        location = str(values[6] or "").strip().splitlines()[-1]
        document_id, _, span = location.rpartition("@")
        start, end = (int(part) for part in span.split("-"))
        if (question_id, document_id, start, end) in already:
            continue
        decision = decided.get(question_id)
        if decision is None or decision.decision != "accept" or decision.question is None:
            result.problems.append(f"{where}: this question is no longer accepted")
            continue
        question = updated.get(question_id) or decision.question.model_copy(deep=True)

        if choice in NO_VALUES:
            result.judgments.append(
                Judgment(
                    question_id=question_id,
                    document_id=document_id,
                    char_start=start,
                    char_end=end,
                    relevant=False,
                )
            )
            continue

        if not question.answerable:
            withdrawn[question_id] = f"Answered by {document_id} in pooled review"
            result.judgments.append(
                Judgment(
                    question_id=question_id,
                    document_id=document_id,
                    char_start=start,
                    char_end=end,
                    relevant=True,
                )
            )
            continue

        quote_text = str(values[_QUOTE - 1] or "").strip()
        located = locate_quote(documents[document_id].content, quote_text, start, end)
        if not quote_text or located is None:
            result.problems.append(f"{where}: paste the supporting text, copied from the passage")
            continue
        group_text = str(values[_GROUP - 1] or "").strip()
        if len(question.evidence) > 1 and not group_text:
            result.problems.append(f"{where}: say which evidence # this passage supports")
            continue
        group_index = int(group_text) - 1 if group_text.isdigit() else 0
        if not 0 <= group_index < len(question.evidence):
            result.problems.append(f"{where}: this question has no evidence {group_index + 1}")
            continue

        quote = documents[document_id].content[located[0] : located[1]]
        question.evidence[group_index].spans.append(
            GoldSpan(
                document_id=document_id,
                char_start=located[0],
                char_end=located[1],
                quote=quote,
            )
        )
        updated[question_id] = question
        result.added_evidence += 1
        result.judgments.append(
            Judgment(
                question_id=question_id,
                document_id=document_id,
                char_start=start,
                char_end=end,
                relevant=True,
                quote=quote,
            )
        )

    for question_id, question in updated.items():
        if question_id in withdrawn:
            continue
        result.decisions.append(
            ReviewDecision(
                question_id=question_id,
                decision="accept",
                reason=decided[question_id].reason,
                question=question,
            )
        )
    for question_id, reason in withdrawn.items():
        result.decisions.append(
            ReviewDecision(question_id=question_id, decision="reject", reason=reason)
        )
        result.withdrawn.append(question_id)
    return result


def read_judgments(path: Path) -> list[Judgment]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [Judgment.model_validate(json.loads(line)) for line in lines if line.strip()]


def append_judgments(path: Path, judgments: list[Judgment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as out:
        for judgment in judgments:
            out.write(judgment.model_dump_json() + "\n")
