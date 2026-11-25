import pytest
from openpyxl import load_workbook

from ragbench.dataset.models import Provenance, Question
from ragbench.dataset.repeated_evidence import (
    SHEET,
    export_repeated_evidence,
    find_repeated_evidence,
    read_repeated_evidence,
)
from ragbench.dataset.review import ReviewDecision, append_decision, read_decisions
from ragbench.evaluation.gold import EvidenceGroup, GoldSpan
from ragbench.ingestion.cleaner import content_hash, extract_sections
from ragbench.ingestion.models import Document

STEP = "Set up the repository and install from it for easy upgrades."


def document(document_id: str, content: str) -> Document:
    return Document(
        document_id=document_id,
        source="Docker Documentation",
        source_url=f"https://example.org/{document_id}",
        title=document_id,
        version="1",
        content=content,
        content_hash=content_hash(content),
        sections=extract_sections(content),
        ingestion_date="2026-11-01",
    )


DOCUMENTS = {
    "docker/rhel": document("docker/rhel", f"# RHEL\n\n{STEP}\n"),
    "docker/ubuntu": document("docker/ubuntu", f"# Ubuntu\n\nIntro text.\n\n{STEP}\n"),
    "docker/fedora": document("docker/fedora", f"# Fedora\n\n{STEP}\n"),
    "docker/other": document("docker/other", "# Other\n\nNothing shared here at all.\n"),
}


def question() -> Question:
    start = DOCUMENTS["docker/rhel"].content.index(STEP)
    span = GoldSpan(
        document_id="docker/rhel", char_start=start, char_end=start + len(STEP), quote=STEP
    )
    return Question(
        question_id="q-1",
        question="How should Docker be installed?",
        question_type="single_hop",
        expected_answer="From the repository.",
        evidence=[EvidenceGroup(spans=[span])],
        answerable=True,
        human_verified=True,
        provenance=Provenance(method="manual"),
    )


def accepted(tmp_path):
    path = tmp_path / "decisions.jsonl"
    append_decision(path, ReviewDecision(question_id="q-1", decision="accept", question=question()))
    return path


def answer(sheet_path, choices: dict[str, str]) -> None:
    workbook = load_workbook(sheet_path)
    sheet = workbook[SHEET]
    for row in range(2, sheet.max_row + 1):
        other = sheet.cell(row=row, column=4).value.split("\n")[0]
        if other in choices:
            sheet.cell(row=row, column=5).value = choices[other]
    workbook.save(sheet_path)


def test_finds_every_other_page_with_the_quote():
    repeats = find_repeated_evidence([question()], DOCUMENTS)
    assert [r.other.document_id for r in repeats] == ["docker/fedora", "docker/ubuntu"]
    ubuntu = repeats[1]
    assert ubuntu.span.quote == STEP
    content = DOCUMENTS["docker/ubuntu"].content
    assert content[ubuntu.span.char_start : ubuntu.span.char_end] == STEP


def test_only_pages_marked_yes_are_added(tmp_path):
    decisions_path = accepted(tmp_path)
    sheet_path = tmp_path / "sheet.xlsx"
    export_repeated_evidence(find_repeated_evidence([question()], DOCUMENTS), DOCUMENTS, sheet_path)
    answer(sheet_path, {"docker/ubuntu": "y", "docker/fedora": "n"})

    result = read_repeated_evidence(sheet_path, decisions_path, DOCUMENTS)
    for decision in result.decisions:
        append_decision(decisions_path, decision)

    assert (result.added, result.declined, result.undecided) == (1, 1, 0)
    spans = read_decisions(decisions_path)["q-1"].question.evidence[0].spans
    assert [s.document_id for s in spans] == ["docker/rhel", "docker/ubuntu"]
    content = DOCUMENTS["docker/ubuntu"].content
    assert content[spans[1].char_start : spans[1].char_end] == STEP


def test_reimport_adds_nothing_new(tmp_path):
    decisions_path = accepted(tmp_path)
    sheet_path = tmp_path / "sheet.xlsx"
    export_repeated_evidence(find_repeated_evidence([question()], DOCUMENTS), DOCUMENTS, sheet_path)
    answer(sheet_path, {"docker/ubuntu": "y"})
    for _ in range(2):
        result = read_repeated_evidence(sheet_path, decisions_path, DOCUMENTS)
        for decision in result.decisions:
            append_decision(decisions_path, decision)
    assert result.added == 0
    assert len(read_decisions(decisions_path)["q-1"].question.evidence[0].spans) == 2


def test_bad_answers_and_rejected_questions_are_reported(tmp_path):
    decisions_path = accepted(tmp_path)
    sheet_path = tmp_path / "sheet.xlsx"
    export_repeated_evidence(find_repeated_evidence([question()], DOCUMENTS), DOCUMENTS, sheet_path)
    answer(sheet_path, {"docker/ubuntu": "maybe", "docker/fedora": "y"})
    append_decision(decisions_path, ReviewDecision(question_id="q-1", decision="reject"))

    result = read_repeated_evidence(sheet_path, decisions_path, DOCUMENTS)
    problems = "\n".join(result.problems)
    assert "answer must be y or n, not 'maybe'" in problems
    assert "no longer accepted" in problems
    assert result.decisions == []


def test_changed_columns_are_refused(tmp_path):
    decisions_path = accepted(tmp_path)
    sheet_path = tmp_path / "sheet.xlsx"
    export_repeated_evidence(find_repeated_evidence([question()], DOCUMENTS), DOCUMENTS, sheet_path)
    workbook = load_workbook(sheet_path)
    workbook[SHEET].delete_cols(2)
    workbook.save(sheet_path)
    with pytest.raises(ValueError, match="columns were changed"):
        read_repeated_evidence(sheet_path, decisions_path, DOCUMENTS)
