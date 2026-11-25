"""Find evidence quoted word for word on other pages, and let a reviewer accept those copies.

Docs repeat text, such as the same install step on every Linux distribution's page. A label
naming only one page marks retrieval of an equally good copy as a miss. A copy is not always
equivalent, though, since a sentence can mean something else on another page, so a reviewer
decides for each one.
"""

from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from ragbench.dataset.models import Question
from ragbench.dataset.review import ReviewDecision, evidence_context, read_decisions
from ragbench.evaluation.gold import GoldSpan
from ragbench.ingestion.models import Document

SHEET = "Repeated evidence"
YES_VALUES = {"y", "yes", "a", "accept"}
NO_VALUES = {"n", "no", "d", "discard"}

HEADERS = [
    ("ID", 15),
    ("Evidence #", 10),
    ("Type", 14),
    ("Other page", 40),
    ("Also answers it? (y / n)", 12),
    ("Question", 40),
    ("Expected answer", 40),
    ("Same text on the other page (between >>> and <<<)", 80),
    ("Labelled page", 40),
]
_DECISION_COLUMN = 5
_EDITABLE_FILL = PatternFill("solid", start_color="FFF4CC")
_HEADER_FILL = PatternFill("solid", start_color="1F3A5F")


@dataclass(frozen=True)
class RepeatedEvidence:
    question: Question
    group_index: int
    other: Document
    span: GoldSpan


def find_repeated_evidence(
    questions: list[Question], documents: dict[str, Document]
) -> list[RepeatedEvidence]:
    """Every page, outside a group's current spans, that contains one of its quotes verbatim."""
    found = []
    for question in questions:
        for index, group in enumerate(question.evidence):
            quote = group.spans[0].quote
            labelled = {span.document_id for span in group.spans}
            for document_id in sorted(documents):
                if document_id in labelled:
                    continue
                position = documents[document_id].content.find(quote)
                if position < 0:
                    continue
                span = GoldSpan(
                    document_id=document_id,
                    char_start=position,
                    char_end=position + len(quote),
                    quote=quote,
                )
                found.append(RepeatedEvidence(question, index, documents[document_id], span))
    return found


def export_repeated_evidence(
    repeats: list[RepeatedEvidence], documents: dict[str, Document], output: Path
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = SHEET
    sheet.append([header for header, _ in HEADERS])
    for repeat in repeats:
        before, after = evidence_context(repeat.other, repeat.span)
        labelled = documents[repeat.question.evidence[repeat.group_index].spans[0].document_id]
        sheet.append(
            [
                repeat.question.question_id,
                repeat.group_index + 1,
                repeat.question.question_type,
                f"{repeat.other.document_id}\n{repeat.other.source_url}",
                "",
                repeat.question.question,
                repeat.question.expected_answer,
                f"...{before} >>> {repeat.span.quote} <<< {after}...",
                f"{labelled.document_id}\n{labelled.source_url}",
            ]
        )

    last_row = len(repeats) + 1
    for index, (_, width) in enumerate(HEADERS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
        header = sheet.cell(row=1, column=index)
        header.font = Font(bold=True, color="FFFFFF")
        header.fill = _HEADER_FILL
        header.alignment = Alignment(wrap_text=True, vertical="center")
        for row in range(2, last_row + 1):
            sheet.cell(row=row, column=index).alignment = Alignment(wrap_text=True, vertical="top")
    for row in range(2, last_row + 1):
        sheet.cell(row=row, column=_DECISION_COLUMN).fill = _EDITABLE_FILL
        sheet.row_dimensions[row].height = 120
    sheet.freeze_panes = "F2"
    letter = get_column_letter(_DECISION_COLUMN)
    validation = DataValidation(type="list", formula1='"y,n"', allow_blank=True)
    validation.add(f"{letter}2:{letter}{last_row}")
    sheet.add_data_validation(validation)

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)


@dataclass
class RepeatImport:
    decisions: list[ReviewDecision]
    added: int
    declined: int
    undecided: int
    problems: list[str]


def read_repeated_evidence(
    path: Path, decisions_path: Path, documents: dict[str, Document]
) -> RepeatImport:
    """Add each copy marked y to its evidence group, as one new decision per question."""
    decided = read_decisions(decisions_path)
    updated: dict[str, Question] = {}
    result = RepeatImport(decisions=[], added=0, declined=0, undecided=0, problems=[])

    sheet = load_workbook(path, read_only=True, data_only=True)[SHEET]
    rows = sheet.iter_rows(min_row=1, values_only=True)
    if [str(value or "").strip() for value in next(rows)][: len(HEADERS)] != [
        header for header, _ in HEADERS
    ]:
        raise ValueError("The sheet's columns were changed. Export a fresh sheet.")

    for row_number, values in enumerate(rows, start=2):
        question_id = str(values[0] or "").strip()
        if not question_id:
            continue
        where = f"Row {row_number} ({question_id})"
        choice = str(values[_DECISION_COLUMN - 1] or "").strip().lower()
        if not choice:
            result.undecided += 1
            continue
        if choice in NO_VALUES:
            result.declined += 1
            continue
        if choice not in YES_VALUES:
            result.problems.append(f"{where}: answer must be y or n, not {choice!r}")
            continue

        decision = decided.get(question_id)
        if decision is None or decision.decision != "accept" or decision.question is None:
            result.problems.append(f"{where}: this question is no longer accepted")
            continue
        question = updated.get(question_id) or decision.question.model_copy(deep=True)
        group_index = int(values[1]) - 1
        if not 0 <= group_index < len(question.evidence):
            result.problems.append(f"{where}: this question has no evidence {group_index + 1}")
            continue
        group = question.evidence[group_index]
        other_id = str(values[3] or "").split("\n")[0].strip()
        other = documents.get(other_id)
        quote = group.spans[0].quote
        position = other.content.find(quote) if other else -1
        if position < 0:
            result.problems.append(f"{where}: the quote is no longer on {other_id}")
            continue
        if any(span.document_id == other_id for span in group.spans):
            continue
        group.spans.append(
            GoldSpan(
                document_id=other_id,
                char_start=position,
                char_end=position + len(quote),
                quote=quote,
            )
        )
        updated[question_id] = question
        result.added += 1

    for question_id, question in updated.items():
        result.decisions.append(
            ReviewDecision(
                question_id=question_id,
                decision="accept",
                reason=decided[question_id].reason,
                question=question,
            )
        )
    return result
