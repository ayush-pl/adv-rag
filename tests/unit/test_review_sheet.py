import pytest
from openpyxl import load_workbook

from ragbench.dataset.models import Provenance, Question
from ragbench.dataset.review import accepted_questions, append_decision, read_decisions
from ragbench.dataset.review_sheet import (
    COLUMNS,
    GUIDE_SHEET,
    REVIEW_SHEET,
    export_review_sheet,
    read_review_sheet,
)
from ragbench.evaluation.gold import EvidenceGroup, GoldSpan
from ragbench.ingestion.cleaner import content_hash, extract_sections
from ragbench.ingestion.models import Document

CONTENT = "# Pods\n\nA Pod restarts failed containers according to its restart policy setting.\n"
OTHER = "# Docker\n\nDocker uses the restart flag to control container restarts on exit.\n"
DOCUMENTS = {
    "k8s/pods": Document(
        document_id="k8s/pods",
        source="Kubernetes Documentation",
        source_url="https://example.org/pods",
        title="Pods",
        version="1",
        content=CONTENT,
        content_hash=content_hash(CONTENT),
        sections=extract_sections(CONTENT),
        ingestion_date="2026-11-01",
    ),
    "docker/r": Document(
        document_id="docker/r",
        source="Docker Documentation",
        source_url="https://example.org/docker",
        title="Restart",
        version="1",
        content=OTHER,
        content_hash=content_hash(OTHER),
        sections=extract_sections(OTHER),
        ingestion_date="2026-11-01",
    ),
}
KEYS = list(COLUMNS)


def group(document_id: str, content: str, quote: str) -> EvidenceGroup:
    start = content.index(quote)
    span = GoldSpan(
        document_id=document_id, char_start=start, char_end=start + len(quote), quote=quote
    )
    return EvidenceGroup(spans=[span])


def draft(question_id: str, question_type="single_hop", **overrides) -> Question:
    fields = {
        "question_id": question_id,
        "question": f"Kubernetes question {question_id}?",
        "question_type": question_type,
        "expected_answer": "Restarts happen.",
        "evidence": [group("k8s/pods", CONTENT, "A Pod restarts failed containers")],
        "answerable": True,
        "provenance": Provenance(method="llm_drafted", model="m"),
    }
    return Question(**{**fields, **overrides})


DRAFTS = [
    draft("q-1"),
    draft("q-2", lexical_overlap=0.95),
    draft(
        "q-3",
        "multi_hop",
        evidence=[
            group("k8s/pods", CONTENT, "A Pod restarts failed containers"),
            group("docker/r", OTHER, "Docker uses the restart flag"),
        ],
    ),
    draft(
        "q-4",
        "unanswerable",
        evidence=[],
        answerable=False,
        nearest_passages=["https://example.org :: near text"],
    ),
]


def export(tmp_path, drafts=DRAFTS):
    sheet_path = tmp_path / "sheet.xlsx"
    decisions_path = tmp_path / "decisions.jsonl"
    rows = export_review_sheet(drafts, decisions_path, DOCUMENTS, sheet_path)
    return sheet_path, decisions_path, rows


def fill(sheet_path, edits: dict[str, dict[str, str]]) -> None:
    """Simulate a reviewer typing into the sheet in Excel."""
    workbook = load_workbook(sheet_path)
    sheet = workbook[REVIEW_SHEET]
    for row in range(2, sheet.max_row + 1):
        question_id = sheet.cell(row=row, column=1).value
        for key, value in edits.get(question_id, {}).items():
            sheet.cell(row=row, column=KEYS.index(key) + 1).value = value
    workbook.save(sheet_path)


def import_sheet(sheet_path, decisions_path, drafts=DRAFTS):
    result = read_review_sheet(sheet_path, drafts, decisions_path, DOCUMENTS)
    for decision in result.decisions:
        append_decision(decisions_path, decision)
    return result


class TestExport:
    def test_writes_one_row_per_undecided_draft_with_context(self, tmp_path):
        sheet_path, _, rows = export(tmp_path)
        assert rows == 4
        workbook = load_workbook(sheet_path)
        assert workbook.sheetnames == [GUIDE_SHEET, REVIEW_SHEET]
        sheet = workbook[REVIEW_SHEET]
        assert [cell.value for cell in sheet[1]] == [c.header for c in COLUMNS.values()]

        values = {
            sheet.cell(row=r, column=1).value: {
                key: sheet.cell(row=r, column=i + 1).value for i, key in enumerate(KEYS)
            }
            for r in range(2, sheet.max_row + 1)
        }
        assert ">>> A Pod restarts failed containers <<<" in values["q-1"]["evidence"]
        assert values["q-1"]["sources"] == "https://example.org/pods"
        assert "95% of key words copied" in values["q-2"]["notes"]
        assert "[2]" in values["q-3"]["evidence"]
        assert "should NOT be answerable" in values["q-4"]["evidence"]
        assert values["q-1"]["decision"] is None

    def test_rows_are_grouped_by_type(self, tmp_path):
        drafts = [draft("q-u", "unanswerable", evidence=[], answerable=False), draft("q-s")]
        sheet_path, _, _ = export(tmp_path, drafts)
        sheet = load_workbook(sheet_path)[REVIEW_SHEET]
        assert [sheet.cell(row=r, column=1).value for r in (2, 3)] == ["q-s", "q-u"]

    def test_decision_column_offers_only_a_or_d(self, tmp_path):
        sheet_path, _, _ = export(tmp_path)
        sheet = load_workbook(sheet_path)[REVIEW_SHEET]
        options = {v.formula1 for v in sheet.data_validations.dataValidation}
        assert '"a,d"' in options
        assert '"easy,medium,hard"' in options

    def test_guide_counts_progress_including_earlier_decisions(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(sheet_path, {"q-1": {"decision": "a"}})
        import_sheet(sheet_path, decisions_path)

        sheet_path, _, rows = export(tmp_path)
        assert rows == 3
        guide = load_workbook(sheet_path)[GUIDE_SHEET]
        formulas = [row[1].value for row in guide.iter_rows() if row[0].value == "single_hop"]
        assert formulas[0].startswith("=COUNTIFS(") and formulas[0].endswith("+1")


class TestImport:
    def test_accept_and_discard_with_any_common_spelling(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(
            sheet_path,
            {
                "q-1": {"decision": "A"},
                "q-2": {"decision": "yes"},
                "q-3": {"decision": "d", "reason": "second passage not needed"},
            },
        )
        result = import_sheet(sheet_path, decisions_path)

        assert result.problems == []
        assert result.undecided == 1
        decisions = read_decisions(decisions_path)
        assert decisions["q-1"].decision == "accept"
        assert decisions["q-1"].question.human_verified is True
        assert decisions["q-1"].question.provenance.edited_by_reviewer is False
        assert decisions["q-2"].decision == "accept"
        assert decisions["q-3"].decision == "reject"
        assert decisions["q-3"].reason == "second passage not needed"
        assert "q-4" not in decisions

    def test_edits_are_applied(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(
            sheet_path,
            {
                "q-3": {
                    "decision": "a",
                    "new_question": "How do Kubernetes and Docker restarts differ?",
                    "new_answer": "Pods use a policy, Docker uses a flag.",
                    "difficulty": "hard",
                    "new_evidence_2": "control  container\nrestarts",
                }
            },
        )
        import_sheet(sheet_path, decisions_path)

        question = accepted_questions(decisions_path)[0]
        assert question.question == "How do Kubernetes and Docker restarts differ?"
        assert question.expected_answer == "Pods use a policy, Docker uses a flag."
        assert question.difficulty == "hard"
        assert question.provenance.edited_by_reviewer is True
        replaced = question.evidence[1].spans[0]
        assert replaced.quote == "control container restarts"
        assert OTHER[replaced.char_start : replaced.char_end] == replaced.quote
        assert question.evidence[0] == DRAFTS[2].evidence[0]

    def test_bad_rows_are_reported_and_left_pending(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(
            sheet_path,
            {
                "q-1": {"decision": "maybe"},
                "q-2": {"decision": "a", "new_evidence_1": "text that is not in the page"},
                "q-3": {"decision": "a", "difficulty": "impossible"},
                "q-4": {"decision": "a", "new_evidence_1": "anything"},
            },
        )
        result = import_sheet(sheet_path, decisions_path)

        assert result.decisions == []
        assert read_decisions(decisions_path) == {}
        problems = "\n".join(result.problems)
        assert "Row 2 (q-1): decision must be a or d, not 'maybe'" in problems
        assert "replacement evidence 1 is not in k8s/pods" in problems
        assert "difficulty must be easy, medium or hard" in problems
        assert "no evidence 1 to replace" in problems

    def test_reimporting_the_same_sheet_adds_nothing(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(sheet_path, {"q-1": {"decision": "a"}, "q-2": {"decision": "d"}})
        import_sheet(sheet_path, decisions_path)
        lines_before = decisions_path.read_text(encoding="utf-8")

        result = import_sheet(sheet_path, decisions_path)
        assert result.decisions == []
        assert result.unchanged == 2
        assert decisions_path.read_text(encoding="utf-8") == lines_before

    def test_changed_mind_is_recorded_as_latest_decision(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(sheet_path, {"q-1": {"decision": "a"}})
        import_sheet(sheet_path, decisions_path)
        fill(sheet_path, {"q-1": {"decision": "d", "reason": "on reflection, ambiguous"}})
        import_sheet(sheet_path, decisions_path)

        assert read_decisions(decisions_path)["q-1"].decision == "reject"
        assert accepted_questions(decisions_path) == []

    def test_unknown_and_repeated_ids_are_reported(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        workbook = load_workbook(sheet_path)
        sheet = workbook[REVIEW_SHEET]
        sheet.append(["q-unknown", "single_hop", "a"])
        sheet.append(["q-1", "single_hop", "a"])
        workbook.save(sheet_path)

        problems = "\n".join(import_sheet(sheet_path, decisions_path).problems)
        assert "q-unknown): unknown question ID" in problems
        assert "q-1): this question appears twice" in problems

    def test_changed_columns_are_refused(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        workbook = load_workbook(sheet_path)
        workbook[REVIEW_SHEET].delete_cols(3)
        workbook.save(sheet_path)
        with pytest.raises(ValueError, match="columns were changed"):
            import_sheet(sheet_path, decisions_path)


class TestRecheck:
    def test_exports_chosen_decided_questions_as_accepted(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(sheet_path, {"q-3": {"decision": "a", "new_question": "Edited multi-hop?"}})
        import_sheet(sheet_path, decisions_path)

        recheck = tmp_path / "recheck.xlsx"
        rows = export_review_sheet(DRAFTS, decisions_path, DOCUMENTS, recheck, {"q-3"})
        assert rows == 1
        sheet = load_workbook(recheck)[REVIEW_SHEET]
        assert sheet.cell(row=2, column=KEYS.index("question") + 1).value == "Edited multi-hop?"
        guide = load_workbook(recheck)[GUIDE_SHEET]
        formula = next(r[1].value for r in guide.iter_rows() if r[0].value == "multi_hop")
        assert formula.endswith("+0")

    def test_blank_decision_keeps_the_earlier_one(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(sheet_path, {"q-3": {"decision": "a"}})
        import_sheet(sheet_path, decisions_path)

        recheck = tmp_path / "recheck.xlsx"
        export_review_sheet(DRAFTS, decisions_path, DOCUMENTS, recheck, {"q-3"})
        result = import_sheet(recheck, decisions_path)
        assert (result.undecided, result.decisions) == (1, [])
        assert read_decisions(decisions_path)["q-3"].decision == "accept"

    def test_discard_on_recheck_replaces_accept(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(sheet_path, {"q-3": {"decision": "a"}})
        import_sheet(sheet_path, decisions_path)

        recheck = tmp_path / "recheck.xlsx"
        export_review_sheet(DRAFTS, decisions_path, DOCUMENTS, recheck, {"q-3"})
        fill(recheck, {"q-3": {"decision": "d", "reason": "really single-hop"}})
        import_sheet(recheck, decisions_path)
        assert read_decisions(decisions_path)["q-3"].decision == "reject"

    def test_accept_on_recheck_keeps_earlier_edits(self, tmp_path):
        sheet_path, decisions_path, _ = export(tmp_path)
        fill(sheet_path, {"q-3": {"decision": "a", "new_question": "Edited multi-hop?"}})
        import_sheet(sheet_path, decisions_path)

        recheck = tmp_path / "recheck.xlsx"
        export_review_sheet(DRAFTS, decisions_path, DOCUMENTS, recheck, {"q-3"})
        fill(recheck, {"q-3": {"decision": "a", "difficulty": "hard"}})
        import_sheet(recheck, decisions_path)
        question = read_decisions(decisions_path)["q-3"].question
        assert question.question == "Edited multi-hop?"
        assert question.difficulty == "hard"

    def test_unknown_ids_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown question IDs: q-nope"):
            export_review_sheet(
                DRAFTS, tmp_path / "d.jsonl", DOCUMENTS, tmp_path / "x.xlsx", {"q-nope"}
            )


def test_paraphrase_row_shows_the_question_it_rewords(tmp_path):
    paraphrase = draft(
        "q-p",
        "paraphrase_ambiguous",
        question="Why do Kubernetes things come back?",
        provenance=Provenance(method="llm_drafted", source_question_id="q-1"),
    )
    sheet_path, _, _ = export(tmp_path, [*DRAFTS, paraphrase])
    sheet = load_workbook(sheet_path)[REVIEW_SHEET]
    notes = {
        sheet.cell(row=r, column=1).value: sheet.cell(row=r, column=KEYS.index("notes") + 1).value
        for r in range(2, sheet.max_row + 1)
    }
    assert notes["q-p"] == "Rewords: Kubernetes question q-1?"
