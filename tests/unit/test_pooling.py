import pytest
from openpyxl import load_workbook

from ragbench.dataset.models import Provenance, Question
from ragbench.dataset.pooling import (
    SHEET,
    Judgment,
    append_judgments,
    build_pool,
    export_pool,
    parse_chunk_id,
    read_judgments,
    read_pool,
)
from ragbench.dataset.review import ReviewDecision, append_decision, read_decisions
from ragbench.evaluation.gold import EvidenceGroup, GoldSpan
from ragbench.ingestion.cleaner import content_hash, extract_sections
from ragbench.ingestion.models import Document

GOLD = "Pods restart failed containers by policy."
ALSO = "The kubelet restarts containers that exit."
TEXT_A = f"# A\n\n{GOLD}\n\n" + "filler " * 40 + "\n"
TEXT_B = f"# B\n\n{ALSO}\n\n" + "other " * 40 + "\n"


def document(document_id: str, content: str) -> Document:
    return Document(
        document_id=document_id,
        source="Kubernetes Documentation",
        source_url=f"https://example.org/{document_id}",
        title=document_id,
        version="1",
        content=content,
        content_hash=content_hash(content),
        sections=extract_sections(content),
        ingestion_date="2026-11-01",
    )


DOCUMENTS = {"k/a": document("k/a", TEXT_A), "k/b": document("k/b", TEXT_B)}


def gold_group(document_id="k/a", content=TEXT_A, quote=GOLD) -> EvidenceGroup:
    start = content.index(quote)
    span = GoldSpan(
        document_id=document_id, char_start=start, char_end=start + len(quote), quote=quote
    )
    return EvidenceGroup(spans=[span])


def question(question_id="q-1", groups=None, answerable=True) -> Question:
    return Question(
        question_id=question_id,
        question="How do Kubernetes pods restart?",
        question_type=("multi_hop" if groups and len(groups) > 1 else "single_hop")
        if answerable
        else "unanswerable",
        expected_answer="By policy.",
        evidence=(groups or [gold_group()]) if answerable else [],
        answerable=answerable,
        human_verified=True,
        provenance=Provenance(method="manual"),
    )


def chunk_id(document_id: str, start: int, end: int) -> str:
    return f"{document_id}::{start}-{end}"


def gold_chunk() -> str:
    start = TEXT_A.index(GOLD)
    return chunk_id("k/a", start, start + len(GOLD))


B_PASSAGE = chunk_id("k/b", 0, 60)


class TestBuildPool:
    def test_parse_chunk_id(self):
        assert parse_chunk_id("postgres-16/sql::12-340") == ("postgres-16/sql", 12, 340)

    def test_skips_labelled_and_merges_overlapping_passages(self):
        rankings = {
            "dense": {"q-1": [gold_chunk(), B_PASSAGE, chunk_id("k/b", 100, 160)]},
            "bm25": {"q-1": [chunk_id("k/b", 5, 60)]},
        }
        pool = build_pool([question()], rankings, depth=3, judgments=[])
        assert [(c.document_id, c.char_start) for c in pool] == [("k/b", 5), ("k/b", 100)]
        assert sorted(pool[0].configs) == ["bm25", "dense"]
        assert pool[0].best_rank == 1

    def test_depth_limits_each_ranking(self):
        rankings = {"dense": {"q-1": [gold_chunk(), gold_chunk(), B_PASSAGE]}}
        assert build_pool([question()], rankings, depth=2, judgments=[]) == []

    def test_already_judged_passages_are_not_shown_again(self):
        judged = Judgment(
            question_id="q-1", document_id="k/b", char_start=0, char_end=60, relevant=False
        )
        rankings = {"dense": {"q-1": [B_PASSAGE]}}
        assert build_pool([question()], rankings, depth=3, judgments=[judged]) == []

    def test_judgment_for_another_question_does_not_hide_passage(self):
        judged = Judgment(
            question_id="q-9", document_id="k/b", char_start=0, char_end=60, relevant=False
        )
        rankings = {"dense": {"q-1": [B_PASSAGE]}}
        assert len(build_pool([question()], rankings, depth=3, judgments=[judged])) == 1


def setup(tmp_path, questions):
    decisions_path = tmp_path / "decisions.jsonl"
    for q in questions:
        append_decision(
            decisions_path, ReviewDecision(question_id=q.question_id, decision="accept", question=q)
        )
    sheet_path = tmp_path / "pool.xlsx"
    rankings = {"dense": {q.question_id: [B_PASSAGE] for q in questions}}
    export_pool(build_pool(questions, rankings, 3, []), DOCUMENTS, sheet_path)
    return decisions_path, sheet_path


def fill(sheet_path, values: dict[str, dict[int, str]]) -> None:
    workbook = load_workbook(sheet_path)
    sheet = workbook[SHEET]
    for row in range(2, sheet.max_row + 1):
        for column, value in values.get(sheet.cell(row=row, column=1).value, {}).items():
            sheet.cell(row=row, column=column).value = value
    workbook.save(sheet_path)


class TestImport:
    def test_yes_with_quote_adds_an_alternative_span(self, tmp_path):
        decisions_path, sheet_path = setup(tmp_path, [question()])
        fill(sheet_path, {"q-1": {8: "y", 9: "kubelet  restarts\ncontainers"}})
        result = read_pool(sheet_path, decisions_path, DOCUMENTS, [])
        for decision in result.decisions:
            append_decision(decisions_path, decision)

        assert result.added_evidence == 1
        spans = read_decisions(decisions_path)["q-1"].question.evidence[0].spans
        assert [s.document_id for s in spans] == ["k/a", "k/b"]
        assert spans[1].quote == "kubelet restarts containers"
        assert TEXT_B[spans[1].char_start : spans[1].char_end] == spans[1].quote
        assert result.judgments[0].relevant is True

    def test_no_is_recorded_without_changing_the_question(self, tmp_path):
        decisions_path, sheet_path = setup(tmp_path, [question()])
        fill(sheet_path, {"q-1": {8: "n"}})
        result = read_pool(sheet_path, decisions_path, DOCUMENTS, [])
        assert result.decisions == []
        assert result.judgments == [
            Judgment(
                question_id="q-1", document_id="k/b", char_start=0, char_end=60, relevant=False
            )
        ]

    def test_yes_needs_text_from_the_passage(self, tmp_path):
        decisions_path, sheet_path = setup(tmp_path, [question()])
        fill(sheet_path, {"q-1": {8: "y", 9: "text found nowhere"}})
        result = read_pool(sheet_path, decisions_path, DOCUMENTS, [])
        assert "paste the supporting text" in result.problems[0]
        assert result.judgments == []

    def test_multi_hop_needs_the_evidence_number(self, tmp_path):
        two = [gold_group(), gold_group()]
        decisions_path, sheet_path = setup(tmp_path, [question(groups=two)])
        fill(sheet_path, {"q-1": {8: "y", 9: ALSO}})
        assert (
            "which evidence #" in read_pool(sheet_path, decisions_path, DOCUMENTS, []).problems[0]
        )

        fill(sheet_path, {"q-1": {10: "2"}})
        result = read_pool(sheet_path, decisions_path, DOCUMENTS, [])
        assert result.problems == []
        assert len(result.decisions[0].question.evidence[1].spans) == 2

    def test_answered_unanswerable_question_is_withdrawn(self, tmp_path):
        decisions_path, sheet_path = setup(tmp_path, [question("q-u", answerable=False)])
        fill(sheet_path, {"q-u": {8: "y"}})
        result = read_pool(sheet_path, decisions_path, DOCUMENTS, [])
        for decision in result.decisions:
            append_decision(decisions_path, decision)
        assert result.withdrawn == ["q-u"]
        assert read_decisions(decisions_path)["q-u"].decision == "reject"

    def test_judged_rows_are_skipped_on_reimport(self, tmp_path):
        decisions_path, sheet_path = setup(tmp_path, [question()])
        fill(sheet_path, {"q-1": {8: "n"}})
        first = read_pool(sheet_path, decisions_path, DOCUMENTS, [])
        append_judgments(tmp_path / "judgments.jsonl", first.judgments)
        second = read_pool(
            sheet_path, decisions_path, DOCUMENTS, read_judgments(tmp_path / "judgments.jsonl")
        )
        assert second.judgments == []

    def test_invalid_answer_and_changed_columns(self, tmp_path):
        decisions_path, sheet_path = setup(tmp_path, [question()])
        fill(sheet_path, {"q-1": {8: "maybe"}})
        assert "must be y or n" in read_pool(sheet_path, decisions_path, DOCUMENTS, []).problems[0]

        workbook = load_workbook(sheet_path)
        workbook[SHEET].delete_cols(2)
        workbook.save(sheet_path)
        with pytest.raises(ValueError, match="columns were changed"):
            read_pool(sheet_path, decisions_path, DOCUMENTS, [])
