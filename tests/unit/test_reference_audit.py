import pytest
from openpyxl import load_workbook

from ragbench.dataset.models import Provenance, Question
from ragbench.dataset.reference_audit import (
    AUDIT_SHEET,
    AuditCase,
    AuditDecision,
    audit_records,
    export_audit_sheet,
    read_audit_sheet,
)
from ragbench.evaluation.gold import EvidenceGroup, GoldSpan


def question(question_id: str) -> Question:
    span = GoldSpan(document_id="doc", char_start=0, char_end=10, quote="the quote")
    return Question(
        question_id=question_id,
        question=f"What about {question_id}?",
        question_type="single_hop",
        expected_answer="Old answer.",
        evidence=[EvidenceGroup(spans=[span])],
        answerable=True,
        human_verified=True,
        provenance=Provenance(method="manual"),
    )


QUESTIONS = {q.question_id: q for q in (question("q1"), question("q2"), question("q3"))}


def filled_sheet(tmp_path, rows: dict[str, tuple]) -> tuple:
    path = tmp_path / "audit.xlsx"
    cases = [
        AuditCase(q, "Flagged while scoring.", [("model", "An answer.")])
        for q in QUESTIONS.values()
    ]
    export_audit_sheet(cases, path)
    workbook = load_workbook(path)
    sheet = workbook[AUDIT_SHEET]
    for row in range(2, sheet.max_row + 1):
        values = rows.get(sheet.cell(row=row, column=1).value)
        if values:
            for offset, value in enumerate(values):
                sheet.cell(row=row, column=7 + offset, value=value)
    workbook.save(path)
    return read_audit_sheet(path, set(QUESTIONS))


def test_export_shows_evidence_and_model_answers(tmp_path):
    path = tmp_path / "audit.xlsx"
    export_audit_sheet([AuditCase(question("q1"), "why", [("llama", "It is X.")])], path)
    row = [cell.value for cell in load_workbook(path)[AUDIT_SHEET][2]]
    assert row[0] == "q1" and row[3] == "Old answer."
    assert "the quote" in row[4] and "llama:\nIt is X." in row[5]


def test_reads_each_decision(tmp_path):
    result = filled_sheet(
        tmp_path,
        {
            "q1": ("keep",),
            "q2": ("Fix", "New answer.", "", "Evidence says otherwise."),
            "q3": ("withdraw", "", "", "Ambiguous."),
        },
    )
    assert result.problems == [] and result.undecided == 0
    assert [(d.question_id, d.action) for d in result.decisions] == [
        ("q1", "keep"),
        ("q2", "fix"),
        ("q3", "withdraw"),
    ]


@pytest.mark.parametrize(
    "values, problem",
    [
        (("maybe",), "keep, fix or withdraw"),
        (("fix", "", "", "reason"), "needs a corrected answer"),
        (("fix", "New.", "", ""), "fix needs a reason"),
        (("withdraw",), "withdraw needs a reason"),
        (("keep", "New."), "only applies to fix"),
    ],
)
def test_incomplete_decisions_are_problems(tmp_path, values, problem):
    result = filled_sheet(tmp_path, {"q1": values})
    assert len(result.problems) == 1 and problem in result.problems[0]
    assert result.undecided == 2


def test_records_fix_and_withdraw_but_not_keep():
    decisions = [
        AuditDecision("q1", "keep"),
        AuditDecision("q2", "fix", new_answer="New answer.", reason="wrong"),
        AuditDecision("q3", "withdraw", reason="ambiguous"),
    ]
    records = {r.question_id: r for r in audit_records(decisions, QUESTIONS)}
    assert set(records) == {"q2", "q3"}
    fixed = records["q2"].question
    assert records["q2"].decision == "accept"
    assert fixed.expected_answer == "New answer."
    assert fixed.question == QUESTIONS["q2"].question
    assert fixed.provenance.edited_by_reviewer is True
    assert fixed.evidence == QUESTIONS["q2"].evidence
    assert records["q3"].decision == "reject" and records["q3"].question is None
    assert records["q3"].reason == "Reference audit: ambiguous"
    # The originals are untouched.
    assert QUESTIONS["q2"].expected_answer == "Old answer."
