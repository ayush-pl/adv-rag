"""Review drafts in a spreadsheet: one row per question, decisions imported in one step.

This is an alternative to the one-at-a-time terminal review. Both write the same decision
history, and both apply the same checks to edited evidence.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from ragbench.dataset.models import TARGET_COUNTS, Question
from ragbench.dataset.review import (
    TYPE_CHECKS,
    ReviewDecision,
    evidence_context,
    read_decisions,
    replace_evidence,
)
from ragbench.ingestion.models import Document

REVIEW_SHEET = "Review"
GUIDE_SHEET = "Guide"
HIGH_OVERLAP = 0.8

ACCEPT_VALUES = {"a", "accept", "y", "yes"}
DISCARD_VALUES = {"d", "discard", "n", "no", "r", "reject"}


@dataclass(frozen=True)
class Column:
    header: str
    width: int
    editable: bool = False


# Columns you fill in are highlighted. All others are for reading and are ignored on import.
COLUMNS = {
    "id": Column("ID", 15),
    "type": Column("Type", 14),
    "decision": Column("Decision (a / d)", 11, editable=True),
    "question": Column("Question", 45),
    "answer": Column("Expected answer", 45),
    "check": Column("What to check", 28),
    "evidence": Column("Evidence (the quote is between >>> and <<<)", 80),
    "sources": Column("Source", 32),
    "notes": Column("Warnings", 22),
    "difficulty": Column("Difficulty", 11, editable=True),
    "new_question": Column("Reworded question (optional)", 35, editable=True),
    "new_answer": Column("Corrected answer (optional)", 35, editable=True),
    "new_evidence_1": Column("Replacement evidence 1 (optional)", 30, editable=True),
    "new_evidence_2": Column("Replacement evidence 2 (optional)", 30, editable=True),
    "reason": Column("Reason or comment (optional)", 30, editable=True),
}
_KEYS = list(COLUMNS)
_EDITABLE_FILL = PatternFill("solid", start_color="FFF4CC")
_HEADER_FILL = PatternFill("solid", start_color="1F3A5F")

GUIDE = [
    "How to review",
    "",
    "1. Open the Review sheet. Each row is one drafted question.",
    "2. For each row, read the question, the expected answer and the evidence, then type",
    "   a in the Decision column to accept it, or d to discard it. Leave it empty to decide later.",
    "3. Optional: fill the highlighted columns to fix a question before accepting it:",
    "   - Reworded question: write it in your own words if it copies the source too closely.",
    "   - Corrected answer: if the expected answer is wrong or incomplete.",
    "   - Replacement evidence: paste text copied exactly from the same document page.",
    "   - Difficulty: easy, medium or hard.",
    "   - Reason: why you discarded it. Helpful but not required.",
    "4. Save the file (keep it as .xlsx), close it, and tell Claude, or run:",
    "   .venv/Scripts/python.exe scripts/import_review_sheet.py",
    "",
    "Rules of thumb",
    "- Discard freely. A small trustworthy set beats a large questionable one.",
    "- Only accept if the evidence really proves the expected answer.",
    "- The 'What to check' column gives the main thing to check for that question type.",
    "- Accepting more than a type's target is fine. The counts below update as you go.",
    "",
    "Progress (accepted / target)",
]


def export_review_sheet(
    drafts: list[Question],
    decisions_path: Path,
    documents: dict[str, Document],
    output: Path,
    question_ids: set[str] | None = None,
) -> int:
    """Write every undecided draft to a workbook. Returns the number of rows.

    With `question_ids`, write exactly those questions instead, decided or not, to review them
    again. Accepted questions appear as accepted, including earlier edits. A row left without a
    decision keeps its earlier one.
    """
    decided = read_decisions(decisions_path)
    if question_ids is None:
        pending = [draft for draft in drafts if draft.question_id not in decided]
    else:
        unknown = question_ids - {draft.question_id for draft in drafts}
        if unknown:
            raise ValueError(f"Unknown question IDs: {', '.join(sorted(unknown))}")
        pending = [
            _current_version(draft, decided)
            for draft in drafts
            if draft.question_id in question_ids
        ]
    type_order = {name: index for index, name in enumerate(TARGET_COUNTS)}
    pending.sort(key=lambda draft: type_order[draft.question_type])

    # A paraphrase is judged against the question it rewrites, so show that question too.
    originals = {draft.question_id: _current_version(draft, decided) for draft in drafts}

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = REVIEW_SHEET
    sheet.append([column.header for column in COLUMNS.values()])
    for draft in pending:
        sheet.append([_cell_values(draft, documents, originals)[key] for key in _KEYS])
    _format_review_sheet(sheet, len(pending))

    # Rows in this sheet are counted by the sheet's own formulas, so leave them out here.
    exported = {draft.question_id for draft in pending}
    already_accepted = Counter(
        d.question.question_type
        for d in decided.values()
        if d.decision == "accept" and d.question is not None and d.question_id not in exported
    )
    _write_guide(workbook.create_sheet(GUIDE_SHEET, 0), len(pending), already_accepted)
    workbook.active = 1

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    return len(pending)


@dataclass
class ImportResult:
    decisions: list[ReviewDecision]
    problems: list[str]
    undecided: int
    unchanged: int


def read_review_sheet(
    path: Path,
    drafts: list[Question],
    decisions_path: Path,
    documents: dict[str, Document],
) -> ImportResult:
    """Turn filled-in rows into decisions. Rows with problems are reported and left pending."""
    by_id = {draft.question_id: draft for draft in drafts}
    previous = read_decisions(decisions_path)
    sheet = load_workbook(path, read_only=True, data_only=True)[REVIEW_SHEET]
    rows = sheet.iter_rows(min_row=1, values_only=True)
    header = [str(value or "").strip() for value in next(rows)]
    expected = [column.header for column in COLUMNS.values()]
    if header[: len(expected)] != expected:
        raise ValueError("The Review sheet's columns were changed. Export a fresh sheet.")

    result = ImportResult(decisions=[], problems=[], undecided=0, unchanged=0)
    seen: set[str] = set()
    for row_number, values in enumerate(rows, start=2):
        row = {key: _text(values[i] if i < len(values) else None) for i, key in enumerate(_KEYS)}
        if not row["id"]:
            continue
        where = f"Row {row_number} ({row['id']})"
        draft = by_id.get(row["id"])
        if draft is None:
            result.problems.append(f"{where}: unknown question ID")
            continue
        if row["id"] in seen:
            result.problems.append(f"{where}: this question appears twice")
            continue
        seen.add(row["id"])

        choice = row["decision"].lower()
        if not choice:
            result.undecided += 1
            continue
        if choice in DISCARD_VALUES:
            decision = ReviewDecision(
                question_id=draft.question_id, decision="reject", reason=row["reason"]
            )
        elif choice in ACCEPT_VALUES:
            question, problems = _accepted(_current_version(draft, previous), row, documents)
            if problems:
                result.problems.extend(f"{where}: {problem}" for problem in problems)
                continue
            decision = ReviewDecision(
                question_id=draft.question_id,
                decision="accept",
                reason=row["reason"],
                question=question,
            )
        else:
            result.problems.append(f"{where}: decision must be a or d, not {row['decision']!r}")
            continue

        if previous.get(draft.question_id) == decision:
            result.unchanged += 1
        else:
            result.decisions.append(decision)
    return result


def _current_version(draft: Question, decided: dict[str, ReviewDecision]) -> Question:
    """The accepted version if there is one, so earlier edits are kept, otherwise the draft."""
    decision = decided.get(draft.question_id)
    if decision is not None and decision.decision == "accept" and decision.question is not None:
        return decision.question
    return draft


def _accepted(
    draft: Question, row: dict[str, str], documents: dict[str, Document]
) -> tuple[Question, list[str]]:
    question = draft.model_copy(deep=True)
    problems: list[str] = []
    if row["new_question"]:
        question.question = row["new_question"]
        question.provenance.edited_by_reviewer = True
    if row["new_answer"]:
        question.expected_answer = row["new_answer"]
        question.provenance.edited_by_reviewer = True

    difficulty = row["difficulty"].lower()
    if difficulty in ("easy", "medium", "hard"):
        question.difficulty = difficulty
    elif difficulty:
        problems.append(f"difficulty must be easy, medium or hard, not {row['difficulty']!r}")

    for index, key in enumerate(("new_evidence_1", "new_evidence_2")):
        if not row[key]:
            continue
        if index >= len(question.evidence):
            problems.append(f"this question has no evidence {index + 1} to replace")
        elif not replace_evidence(question, index, row[key], documents):
            document_id = question.evidence[index].spans[0].document_id
            problems.append(f"replacement evidence {index + 1} is not in {document_id}")

    question.human_verified = True
    return question, problems


def _cell_values(
    draft: Question, documents: dict[str, Document], originals: dict[str, Question]
) -> dict[str, str]:
    evidence_parts = []
    sources = []
    for number, group in enumerate(draft.evidence, start=1):
        span = group.spans[0]
        document = documents[span.document_id]
        before, after = evidence_context(document, span)
        label = f"[{number}] " if len(draft.evidence) > 1 else ""
        evidence_parts.append(f"{label}...{before} >>> {span.quote} <<< {after}...")
        also = f" (also {len(group.spans) - 1} other version)" if len(group.spans) > 1 else ""
        sources.append(f"{label}{document.source_url}{also}")
    if draft.nearest_passages:
        evidence_parts.append(
            "No evidence: this should NOT be answerable. Closest passages found:\n"
            + "\n".join(f"- {passage}" for passage in draft.nearest_passages)
        )

    notes = []
    source = originals.get(draft.provenance.source_question_id or "")
    if source is not None:
        notes.append(f"Rewords: {source.question}")
    if draft.lexical_overlap is not None and draft.lexical_overlap > HIGH_OVERLAP:
        notes.append(f"{draft.lexical_overlap:.0%} of key words copied. Reword if you can.")
    if draft.document_version:
        notes.append(f"PostgreSQL {draft.document_version} only")

    return {
        "id": draft.question_id,
        "type": draft.question_type,
        "decision": "",
        "question": draft.question,
        "answer": draft.expected_answer,
        "check": TYPE_CHECKS[draft.question_type],
        "evidence": _excel_safe("\n\n".join(evidence_parts)),
        "sources": "\n".join(sources),
        "notes": " ".join(notes),
        "difficulty": draft.difficulty,
        "new_question": "",
        "new_answer": "",
        "new_evidence_1": "",
        "new_evidence_2": "",
        "reason": "",
    }


def _format_review_sheet(sheet, rows: int) -> None:
    last_row = rows + 1
    for index, column in enumerate(COLUMNS.values(), start=1):
        letter = get_column_letter(index)
        sheet.column_dimensions[letter].width = column.width
        header = sheet.cell(row=1, column=index)
        header.font = Font(bold=True, color="FFFFFF")
        header.fill = _HEADER_FILL
        header.alignment = Alignment(wrap_text=True, vertical="center")
        for row in range(2, last_row + 1):
            cell = sheet.cell(row=row, column=index)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if column.editable:
                cell.fill = _EDITABLE_FILL
    for row in range(2, last_row + 1):
        sheet.row_dimensions[row].height = 150
    sheet.row_dimensions[1].height = 32
    sheet.freeze_panes = "D2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{last_row}"

    decision_column = get_column_letter(_KEYS.index("decision") + 1)
    difficulty_column = get_column_letter(_KEYS.index("difficulty") + 1)
    for letter, options in ((decision_column, "a,d"), (difficulty_column, "easy,medium,hard")):
        validation = DataValidation(type="list", formula1=f'"{options}"', allow_blank=True)
        validation.add(f"{letter}2:{letter}{last_row}")
        sheet.add_data_validation(validation)


def _write_guide(sheet, rows: int, already_accepted: Counter) -> None:
    for line in GUIDE:
        sheet.append([line])
    sheet["A1"].font = Font(bold=True, size=14)
    for row in (15, len(GUIDE)):
        sheet.cell(row=row, column=1).font = Font(bold=True)

    type_column = get_column_letter(_KEYS.index("type") + 1)
    decision_column = get_column_letter(_KEYS.index("decision") + 1)
    last_row = rows + 1
    sheet.append(["Type", "Accepted", "Target", "Still to review"])
    for cell in sheet[sheet.max_row]:
        cell.font = Font(bold=True)
    for name, target in TARGET_COUNTS.items():
        types = f"{REVIEW_SHEET}!{type_column}2:{type_column}{last_row}"
        choices = f"{REVIEW_SHEET}!{decision_column}2:{decision_column}{last_row}"
        accepted = f'=COUNTIFS({types},"{name}",{choices},"a")+{already_accepted[name]}'
        undecided = f'=COUNTIFS({types},"{name}",{choices},"")'
        sheet.append([name, accepted, target, undecided])
    sheet.column_dimensions["A"].width = 95
    for letter in "BCD":
        sheet.column_dimensions[letter].width = 14


def _excel_safe(text: str) -> str:
    """Excel cells hold at most 32,767 characters."""
    limit = 32_000
    return text if len(text) <= limit else text[:limit] + " [cut]"


def _text(value) -> str:
    return "" if value is None else str(value).strip()
