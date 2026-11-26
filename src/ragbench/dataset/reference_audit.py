"""Audit reference answers that look wrong, in a spreadsheet.

Questions come here when a human scorer flagged the reference answer, or when every model got
the question wrong even with the gold passages. Each one is kept, given a corrected reference
answer, or withdrawn. Decisions go into the same append-only history as the original review,
so the dataset is rebuilt from it with stable splits.
"""

from dataclasses import dataclass, field
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from ragbench.dataset.models import Question
from ragbench.dataset.review import ReviewDecision

AUDIT_SHEET = "Audit"
GUIDE_SHEET = "Guide"
ACTIONS = ("keep", "fix", "withdraw")

HEADERS = [
    ("ID", 15, False),
    ("Why it is here", 30, False),
    ("Question", 40, False),
    ("Current reference answer", 45, False),
    ("Gold evidence", 70, False),
    ("What models answered with the gold passages", 60, False),
    ("Decision (keep / fix / withdraw)", 12, True),
    ("Corrected reference answer (for fix)", 45, True),
    ("Reworded question (optional, for fix)", 35, True),
    ("Reason", 30, True),
]
_DECISION, _NEW_ANSWER, _NEW_QUESTION, _REASON = 7, 8, 9, 10

GUIDE = [
    "Reference answer audit",
    "",
    "For each row, decide whether the reference answer is right, using the gold evidence.",
    "",
    "  keep      the reference answer is right. The models were wrong, or the flag was mistaken.",
    "  fix       the evidence supports a different answer. Write the corrected reference answer.",
    "            You can also reword the question if it was ambiguous.",
    "  withdraw  the question is unclear, or its evidence doesn't answer it. It leaves the",
    "            dataset, and every metric is recomputed without it.",
    "",
    "The model answers are there to show what the evidence led to. They can be wrong too.",
    "If the gold evidence itself is wrong, choose withdraw and say so in the reason.",
    "A reason is required for fix and withdraw.",
]

_EDITABLE_FILL = PatternFill("solid", start_color="FFF4CC")
_HEADER_FILL = PatternFill("solid", start_color="1F3A5F")


@dataclass
class AuditCase:
    question: Question
    flagged_because: str
    # (label, answer) pairs, such as ("llama3.1:8b", "...").
    model_answers: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class AuditDecision:
    question_id: str
    action: str
    new_answer: str = ""
    new_question: str = ""
    reason: str = ""


@dataclass
class AuditImport:
    decisions: list[AuditDecision]
    undecided: int
    problems: list[str]


def export_audit_sheet(cases: list[AuditCase], output: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = AUDIT_SHEET
    sheet.append([header for header, _, _ in HEADERS])
    for case in cases:
        question = case.question
        evidence = "\n\n".join(
            f"Evidence {number} (any one of these):\n"
            + "\n".join(f"[{span.document_id}]\n{span.quote}" for span in group.spans)
            for number, group in enumerate(question.evidence, start=1)
        )
        answers = "\n\n".join(f"{label}:\n{text}" for label, text in case.model_answers)
        sheet.append(
            [
                question.question_id,
                case.flagged_because,
                question.question,
                question.expected_answer,
                evidence[:32_000],
                answers[:32_000],
                "",
                "",
                "",
                "",
            ]
        )

    last_row = len(cases) + 1
    for index, (_, width, editable) in enumerate(HEADERS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
        header = sheet.cell(row=1, column=index)
        header.font = Font(bold=True, color="FFFFFF")
        header.fill = _HEADER_FILL
        header.alignment = Alignment(wrap_text=True, vertical="center")
        for row in range(2, last_row + 1):
            cell = sheet.cell(row=row, column=index)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if editable:
                cell.fill = _EDITABLE_FILL
    for row in range(2, last_row + 1):
        sheet.row_dimensions[row].height = 260
    sheet.freeze_panes = "D2"
    letter = get_column_letter(_DECISION)
    validation = DataValidation(type="list", formula1=f'"{",".join(ACTIONS)}"', allow_blank=True)
    validation.add(f"{letter}2:{letter}{last_row}")
    sheet.add_data_validation(validation)

    guide = workbook.create_sheet(GUIDE_SHEET)
    for line in GUIDE:
        guide.append([line])
    guide["A1"].font = Font(bold=True, size=14)
    guide.column_dimensions["A"].width = 100
    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)


def read_audit_sheet(path: Path, known: set[str]) -> AuditImport:
    sheet = load_workbook(path, read_only=True, data_only=True)[AUDIT_SHEET]
    rows = sheet.iter_rows(min_row=1, values_only=True)
    if [_text(v) for v in next(rows)][: len(HEADERS)] != [h for h, _, _ in HEADERS]:
        raise ValueError("The sheet's columns were changed. Export a fresh sheet.")

    result = AuditImport([], 0, [])
    for row_number, values in enumerate(rows, start=2):
        values = list(values) + [None] * (len(HEADERS) - len(values))
        question_id = _text(values[0])
        if not question_id:
            continue
        if question_id not in known:
            result.problems.append(f"Row {row_number}: unknown question {question_id}")
            continue
        action = _text(values[_DECISION - 1]).lower()
        new_answer = _text(values[_NEW_ANSWER - 1])
        new_question = _text(values[_NEW_QUESTION - 1])
        reason = _text(values[_REASON - 1])
        problem = None
        if not action:
            result.undecided += 1
            continue
        if action not in ACTIONS:
            problem = f"decision must be keep, fix or withdraw, not {action!r}"
        elif action == "fix" and not (new_answer or new_question):
            problem = "fix needs a corrected answer or a reworded question"
        elif action in ("fix", "withdraw") and not reason:
            problem = f"{action} needs a reason"
        elif action != "fix" and (new_answer or new_question):
            problem = f"a corrected answer or question only applies to fix, not {action}"
        if problem:
            result.problems.append(f"Row {row_number} ({question_id}): {problem}")
            continue
        result.decisions.append(
            AuditDecision(question_id, action, new_answer, new_question, reason)
        )
    return result


def audit_records(
    decisions: list[AuditDecision], questions: dict[str, Question]
) -> list[ReviewDecision]:
    """Review history records for the decisions that change the dataset. Keep changes nothing."""
    records = []
    for decision in decisions:
        question = questions[decision.question_id]
        reason = f"Reference audit: {decision.reason}"
        if decision.action == "withdraw":
            records.append(
                ReviewDecision(question_id=question.question_id, decision="reject", reason=reason)
            )
        elif decision.action == "fix":
            provenance = question.provenance.model_copy(update={"edited_by_reviewer": True})
            updated = question.model_copy(
                update={
                    "expected_answer": decision.new_answer or question.expected_answer,
                    "question": decision.new_question or question.question,
                    "provenance": provenance,
                }
            )
            records.append(
                ReviewDecision(
                    question_id=question.question_id,
                    decision="accept",
                    reason=reason,
                    question=Question.model_validate(updated.model_dump()),
                )
            )
    return records


def _text(value) -> str:
    return "" if value is None else str(value).strip()
