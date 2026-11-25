import json
import random

import httpx
import pytest
from pydantic import ValidationError

from ragbench.chunking.models import Chunk
from ragbench.dataset.drafting import (
    MAX_DRAFT_TOKENS,
    DraftRejected,
    QuestionDrafter,
    VersionPassage,
    build_evidence_group,
    eligible_passages,
    lexical_overlap,
    locate_quote,
    passages_for_type,
    quote_relevance,
    require_product,
    round_robin_by_product,
    version_only_passages,
)
from ragbench.dataset.models import Provenance, Question, read_questions, write_questions
from ragbench.dataset.review import ReviewSession, accepted_questions, read_decisions
from ragbench.dataset.splits import assign_splits
from ragbench.dataset.validation import validate_questions, validate_splits
from ragbench.evaluation.gold import EvidenceGroup, GoldSpan
from ragbench.generation.ollama_client import OllamaClient
from ragbench.ingestion.cleaner import content_hash, extract_sections
from ragbench.ingestion.models import Document

MANUAL = Provenance(method="manual")
CONTENT = "# Pods\n\nA Pod restarts failed containers according to its restart policy setting.\n"
QUOTE = "A Pod restarts failed containers"


def make_document(document_id="k8s/pods", content=CONTENT, version="1") -> Document:
    return Document(
        document_id=document_id,
        source="PostgreSQL" if document_id.startswith("postgres") else "Kubernetes",
        source_url="https://example.org/pods",
        title="Pods",
        version=version,
        content=content,
        content_hash=content_hash(content),
        sections=extract_sections(content),
        ingestion_date="2026-11-01",
    )


def span(document_id="k8s/pods", quote=QUOTE, content=CONTENT) -> GoldSpan:
    start = content.index(quote)
    return GoldSpan(
        document_id=document_id, char_start=start, char_end=start + len(quote), quote=quote
    )


def make_question(question_id="q-1", text="What restarts containers?", **overrides) -> Question:
    fields = {
        "question_id": question_id,
        "question": text,
        "question_type": "single_hop",
        "expected_answer": "The Pod does.",
        "evidence": [EvidenceGroup(spans=[span()])],
        "answerable": True,
        "human_verified": True,
        "provenance": MANUAL,
    }
    return Question(**{**fields, **overrides})


def make_chunk(
    document_id="k8s/pods", text=CONTENT, source="Kubernetes", tokens=100, start=0
) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}::{start}-{start + len(text)}",
        document_id=document_id,
        char_start=start,
        char_end=start + len(text),
        text=text,
        token_count=tokens,
        section="",
        source=source,
        source_url="https://example.org",
        title="T",
        version="1",
    )


class TestQuestionSchema:
    def test_answerable_needs_evidence(self):
        with pytest.raises(ValidationError, match="needs evidence"):
            make_question(evidence=[])

    def test_unanswerable_cannot_have_evidence(self):
        with pytest.raises(ValidationError, match="cannot have evidence"):
            make_question(question_type="unanswerable", answerable=False)

    def test_answerable_flag_must_match_type(self):
        with pytest.raises(ValidationError, match="contradicts"):
            make_question(question_type="unanswerable", answerable=True)

    def test_multi_hop_needs_two_groups(self):
        with pytest.raises(ValidationError, match="2 evidence groups"):
            make_question(question_type="multi_hop")

    def test_round_trips_through_file(self, tmp_path):
        path = tmp_path / "questions.jsonl"
        questions = [make_question()]
        write_questions(path, questions)
        assert read_questions(path) == questions

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert read_questions(tmp_path / "missing.jsonl") == []


class TestValidation:
    documents = {"k8s/pods": make_document()}

    def test_valid_dataset_has_no_problems(self):
        assert validate_questions([make_question()], self.documents) == []

    def test_unverified_question_is_flagged(self):
        problems = validate_questions([make_question(human_verified=False)], self.documents)
        assert problems == ["q-1: not human verified"]

    def test_duplicate_id_and_text_are_flagged(self):
        problems = validate_questions(
            [make_question(), make_question(text="what RESTARTS containers")], self.documents
        )
        assert any("duplicate question ID" in p for p in problems)
        assert any("same text as" in p for p in problems)

    def test_changed_corpus_is_detected(self):
        changed = {"k8s/pods": make_document(content=CONTENT.replace("restarts", "reboots"))}
        problems = validate_questions([make_question()], changed)
        assert problems == [f"q-1: quote no longer matches k8s/pods at {CONTENT.index(QUOTE)}"]

    def test_unknown_document_and_out_of_range_span_are_detected(self):
        assert "unknown document" in validate_questions([make_question()], {})[0]
        short = {"k8s/pods": make_document(content="# Pods\n")}
        assert "past the end" in validate_questions([make_question()], short)[0]

    def test_missing_paraphrase_source_is_detected(self):
        provenance = Provenance(method="manual", source_question_id="q-gone")
        problems = validate_questions([make_question(provenance=provenance)], self.documents)
        assert problems == ["q-1: source question q-gone is missing"]

    def test_paraphrase_in_other_split_is_leakage(self):
        source = make_question("q-1")
        follower = make_question(
            "q-2", "Other?", provenance=Provenance(method="manual", source_question_id="q-1")
        )
        problems = validate_splits({"test": [source], "development": [follower]})
        assert "q-2: in development but its source q-1 is in test" in problems

    def test_question_in_two_splits_is_leakage(self):
        question = make_question()
        problems = validate_splits({"test": [question], "development": [question]})
        assert len(problems) == 1 and "in both test" in problems[0]


class TestSplits:
    def many(self, count: int, question_type="single_hop") -> list[Question]:
        kwargs = {}
        if question_type == "unanswerable":
            kwargs = {"evidence": [], "answerable": False, "question_type": question_type}
        return [
            make_question(f"{question_type}-{i:03}", f"{question_type} {i}?", **kwargs)
            for i in range(count)
        ]

    def test_proportions_and_every_question_used_once(self):
        questions = self.many(120)
        splits = assign_splits(questions)
        assert {name: len(values) for name, values in splits.items()} == {
            "development": 25,
            "calibration": 25,
            "test": 70,
        }
        all_ids = [q.question_id for values in splits.values() for q in values]
        assert sorted(all_ids) == sorted(q.question_id for q in questions)

    def test_each_type_is_stratified(self):
        splits = assign_splits(self.many(60) + self.many(24, "unanswerable"))
        test_types = [q.question_type for q in splits["test"]]
        assert test_types.count("single_hop") == 35
        assert test_types.count("unanswerable") == 14

    def test_is_deterministic_and_seed_changes_it(self):
        questions = self.many(30)
        first = assign_splits(questions, seed=1)
        assert first == assign_splits(list(reversed(questions)), seed=1)
        assert first != assign_splits(questions, seed=2)

    def test_paraphrase_follows_its_source(self):
        questions = self.many(30)
        followers = [
            make_question(
                f"p-{i}",
                f"Paraphrase {i}?",
                question_type="paraphrase_ambiguous",
                provenance=Provenance(method="manual", source_question_id=questions[i].question_id),
            )
            for i in range(10)
        ]
        splits = assign_splits(questions + followers)
        assert validate_splits(splits) == []
        assert sum(len(values) for values in splits.values()) == 40


class TestDraftingHelpers:
    def test_product_must_be_named(self):
        require_product("How do Kubernetes pods restart?", ["Kubernetes"])
        with pytest.raises(DraftRejected, match="does not name Docker"):
            require_product("How do pods restart?", ["Docker"])
        with pytest.raises(DraftRejected):
            require_product("Docker limits for the product?", ["Docker"])

    def test_quote_relevance_measures_use_in_question_and_answer(self):
        assert quote_relevance("restart policy", "What restart setting?", "The policy.") == 1.0
        assert quote_relevance("nftables iptables", "What restart setting?", "None.") == 0.0
        assert quote_relevance("the", "x", "y") == 0.0

    def test_locate_quote_tolerates_whitespace_only(self):
        content = "alpha beta\n  gamma delta"
        assert locate_quote(content, "beta gamma") == (6, 18)
        assert locate_quote(content, "beta  delta") is None

    def test_locate_quote_respects_range_and_empty_input(self):
        content = "same text here, same text there"
        assert locate_quote(content, "same text", 5) == (16, 25)
        assert locate_quote(content, "same text", 0, 5) is None
        assert locate_quote(content, "   ") is None

    def test_lexical_overlap_ignores_stopwords(self):
        assert lexical_overlap("what is the restart policy", "restart policy") == 1.0
        assert lexical_overlap("how to reboot", "restart policy") == 0.0
        assert lexical_overlap("what is the", "anything") == 0.0

    def test_evidence_group_adds_other_postgres_version(self):
        documents = {
            "postgres-16/a": make_document("postgres-16/a", version="16"),
            "postgres-15/a": make_document("postgres-15/a", version="15"),
            "postgres-15/b": make_document("postgres-15/b", version="15"),
        }
        start = CONTENT.index(QUOTE)
        group = build_evidence_group(
            documents["postgres-16/a"], start, start + len(QUOTE), documents, True
        )
        assert [s.document_id for s in group.spans] == ["postgres-16/a", "postgres-15/a"]

    def test_evidence_group_can_stay_on_one_version(self):
        documents = {
            "postgres-16/a": make_document("postgres-16/a"),
            "postgres-15/a": make_document("postgres-15/a"),
        }
        group = build_evidence_group(documents["postgres-16/a"], 8, 20, documents, False)
        assert len(group.spans) == 1

    def test_eligible_passages_drop_short_and_release_notes(self):
        chunks = [
            make_chunk("k8s/ok"),
            make_chunk("k8s/short", tokens=10),
            make_chunk("postgres-16/release-16"),
            make_chunk("docker/desktop/previous-versions/x"),
        ]
        assert [c.document_id for c in eligible_passages(chunks)] == ["k8s/ok"]

    def test_passage_filter_by_type(self):
        code = make_chunk("a", "Intro\n```\nx\n```\n")
        table = make_chunk("b", "| a |\n| --- |\n| 1 |\n")
        prose = make_chunk("c", "Just words.\n")
        chunks = [code, table, prose]
        assert passages_for_type(chunks, "config_code") == [code]
        assert passages_for_type(chunks, "table") == [table]
        assert code not in passages_for_type(chunks, "single_hop")

    def test_round_robin_balances_products_and_skips_repeats_and_pg15(self):
        chunks = [
            make_chunk("k8s/a", source="K", start=0),
            make_chunk("k8s/a", source="K", start=500),
            make_chunk("k8s/b", source="K"),
            make_chunk("docker/a", source="D"),
            make_chunk("postgres-15/a", source="P"),
            make_chunk("postgres-16/a", source="P"),
        ]
        order = [c.document_id for c in round_robin_by_product(chunks, random.Random(0))]
        assert len(order) == len(set(order))
        assert "postgres-15/a" not in order
        assert order[0] == "docker/a"
        assert order[1].startswith("k8s/")
        assert order[2] == "postgres-16/a"
        assert sorted(order) == ["docker/a", "k8s/a", "k8s/b", "postgres-16/a"]

    def test_version_only_passages_find_truly_new_paragraphs(self):
        shared = "This paragraph appears in both versions and is long enough to count here. " * 3
        new = "Brand new feature description about incremental backups in the server. " * 3
        reworded = shared.replace("both versions", "either release")
        newer = make_document("postgres-16/a", f"{reworded}\n\n{new}\n", "16")
        older = make_document("postgres-15/a", f"{shared}\n", "15")
        found = version_only_passages(
            {newer.document_id: newer, older.document_id: older}, random.Random(0)
        )
        assert len(found) == 1
        assert newer.content[found[0].new_start : found[0].new_end].startswith("Brand new")

    def test_version_only_passages_skip_identical_pages(self):
        same = make_document("postgres-16/a"), make_document("postgres-15/a")
        documents = {d.document_id: d for d in same}
        assert version_only_passages(documents, random.Random(0)) == []


def scripted_client(*outputs: dict) -> OllamaClient:
    """An Ollama client that returns these JSON outputs in turn."""
    remaining = list(outputs)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m", "digest": "sha"}]})
        body = json.dumps(remaining.pop(0))
        return httpx.Response(200, json={"model": "m", "response": body})

    return OllamaClient(transport=httpx.MockTransport(handler))


class TestQuestionDrafter:
    documents = {"k8s/pods": make_document()}

    def output(self, **overrides) -> dict:
        fields = {
            "suitable": True,
            "question": "In Kubernetes, what happens to crashed containers?",
            "answer": "They are restarted.",
            "evidence_quote": QUOTE,
            "difficulty": "easy",
        }
        return {**fields, **overrides}

    def draft(self, **overrides) -> Question:
        drafter = QuestionDrafter(scripted_client(self.output(**overrides)), "m", self.documents)
        return drafter.draft_single(make_chunk(), "single_hop")

    def test_valid_output_becomes_unverified_question_with_exact_span(self):
        question = self.draft()
        located = question.evidence[0].spans[0]
        assert CONTENT[located.char_start : located.char_end] == QUOTE == located.quote
        assert question.human_verified is False
        assert question.provenance.model == "m"
        assert question.provenance.model_digest == "sha"
        assert 0 < question.lexical_overlap < 1

    def test_invented_quote_is_rejected(self):
        with pytest.raises(DraftRejected, match="not in the passage"):
            self.draft(evidence_quote="Pods never restart anything at all")

    def test_tiny_quote_is_rejected(self):
        with pytest.raises(DraftRejected, match="too short"):
            self.draft(evidence_quote="Pod")

    def test_unsuitable_passage_is_rejected(self):
        with pytest.raises(DraftRejected, match="unsuitable"):
            self.draft(suitable=False)

    def test_invalid_json_is_rejected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": []})
            return httpx.Response(200, json={"model": "m", "response": "not json"})

        client = OllamaClient(transport=httpx.MockTransport(handler))
        with pytest.raises(DraftRejected, match="not valid JSON"):
            QuestionDrafter(client, "m", self.documents).draft_single(make_chunk(), "single_hop")

    def test_multi_hop_needs_both_quotes_found(self):
        second_text = "Docker uses the restart flag to control container restarts."
        documents = {
            "k8s/pods": make_document(),
            "docker/r": make_document("docker/r", second_text + "\n"),
        }
        output = {
            "suitable": True,
            "question": "How do Kubernetes and Docker differ?",
            "answer": "A Pod restarts failed containers, and Docker uses the restart flag.",
            "evidence_quote_1": QUOTE,
            "evidence_quote_2": "Docker uses the restart flag",
            "difficulty": "hard",
        }
        second = make_chunk("docker/r", second_text)
        question = QuestionDrafter(scripted_client(output), "m", documents).draft_multi_hop(
            make_chunk(), second
        )
        assert [g.spans[0].document_id for g in question.evidence] == ["k8s/pods", "docker/r"]

        unrelated = {
            **output,
            "question": "How does Kubernetes restart?",
            "answer": "A Pod restarts failed containers.",
        }
        unrelated["evidence_quote_2"] = "flag to control"
        with pytest.raises(DraftRejected, match="plays no part"):
            QuestionDrafter(scripted_client(unrelated), "m", documents).draft_multi_hop(
                make_chunk(), second
            )

        output["evidence_quote_2"] = "Invented text that is not in the document"
        with pytest.raises(DraftRejected):
            QuestionDrafter(scripted_client(output), "m", documents).draft_multi_hop(
                make_chunk(), second
            )

    def test_version_question_must_name_the_version(self):
        text = "New paragraph about the feature that only exists in this version of it."
        document = make_document("postgres-16/a", text + "\n", "16")
        documents = {document.document_id: document}
        passage = VersionPassage(document, 0, len(text), 0, len(text))
        output = self.output(
            question="What is new?", evidence_quote="New paragraph about the feature"
        )

        with pytest.raises(DraftRejected, match="PostgreSQL 16"):
            QuestionDrafter(scripted_client(output), "m", documents).draft_version_specific(passage)

        output["question"] = "What is new in PostgreSQL 16?"
        question = QuestionDrafter(scripted_client(output), "m", documents).draft_version_specific(
            passage
        )
        assert question.document_version == "16"

    def test_unanswerable_has_no_evidence_and_keeps_nearest_passages(self):
        output = {"suitable": True, "question": "What is the Kubernetes pod limit?"}
        question = QuestionDrafter(scripted_client(output), "m", self.documents).draft_unanswerable(
            make_chunk(), ["near passage"]
        )
        assert question.answerable is False
        assert question.evidence == []
        assert question.nearest_passages == ["near passage"]

    def test_every_draft_has_an_output_limit(self):
        limits = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": []})
            limits.append(json.loads(request.content)["options"].get("num_predict"))
            return httpx.Response(
                200, json={"model": "m", "response": '{"question": "Kubernetes?"}'}
            )

        client = OllamaClient(transport=httpx.MockTransport(handler))
        drafter = QuestionDrafter(client, "m", self.documents)
        with pytest.raises(KeyError):
            drafter.draft_single(make_chunk(), "single_hop")
        drafter.draft_paraphrase(make_question())
        assert limits == [MAX_DRAFT_TOKENS, MAX_DRAFT_TOKENS]

    def test_output_cut_off_mid_json_is_rejected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": []})
            return httpx.Response(200, json={"model": "m", "response": '{"question": "unfinis'})

        client = OllamaClient(transport=httpx.MockTransport(handler))
        with pytest.raises(DraftRejected, match="not valid JSON"):
            QuestionDrafter(client, "m", self.documents).draft_single(make_chunk(), "single_hop")

    def test_questions_not_naming_the_product_are_rejected(self):
        with pytest.raises(DraftRejected, match="does not name"):
            self.draft(question="What happens to crashed containers?")
        output = {"suitable": True, "question": "What is the limit?"}
        with pytest.raises(DraftRejected, match="does not name"):
            QuestionDrafter(scripted_client(output), "m", self.documents).draft_unanswerable(
                make_chunk(), []
            )

    def test_paraphrase_keeps_evidence_and_links_source(self):
        source = make_question()
        output = {"question": "Why do my Kubernetes things come back after dying?"}
        question = QuestionDrafter(scripted_client(output), "m", self.documents).draft_paraphrase(
            source
        )
        assert question.evidence == source.evidence
        assert question.expected_answer == source.expected_answer
        assert question.provenance.source_question_id == source.question_id


class TestReviewSession:
    documents = {"k8s/pods": make_document()}

    def run(self, tmp_path, answers: list[str], drafts=None) -> list[str]:
        shown: list[str] = []
        replies = iter(answers)
        ReviewSession(
            drafts or [make_question(human_verified=False)],
            tmp_path / "decisions.jsonl",
            self.documents,
            ask=lambda prompt: next(replies),
            show=shown.append,
        ).run()
        return shown

    def test_accept_marks_verified_and_saves(self, tmp_path):
        self.run(tmp_path, ["a"])
        accepted = accepted_questions(tmp_path / "decisions.jsonl")
        assert [q.question_id for q in accepted] == ["q-1"]
        assert accepted[0].human_verified is True
        assert accepted[0].provenance.edited_by_reviewer is False

    def test_reject_records_reason_and_is_not_accepted(self, tmp_path):
        self.run(tmp_path, ["r", "wrong answer"])
        decisions = read_decisions(tmp_path / "decisions.jsonl")
        assert decisions["q-1"].decision == "reject"
        assert decisions["q-1"].reason == "wrong answer"
        assert accepted_questions(tmp_path / "decisions.jsonl") == []

    def test_skip_and_exit_save_nothing(self, tmp_path):
        self.run(tmp_path, ["s"])
        self.run(tmp_path, ["x"])
        assert read_decisions(tmp_path / "decisions.jsonl") == {}

    def test_resume_continues_after_decided_questions(self, tmp_path):
        drafts = [
            make_question("q-1", human_verified=False),
            make_question("q-2", "Second?", human_verified=False),
        ]
        self.run(tmp_path, ["a", "x"], drafts)
        shown = self.run(tmp_path, ["a"], drafts)
        assert any("Second?" in line for line in shown)
        assert not any("What restarts containers?" in line for line in shown)
        assert len(accepted_questions(tmp_path / "decisions.jsonl")) == 2

    def test_edits_are_applied_and_recorded_as_edited(self, tmp_path):
        self.run(tmp_path, ["q", "Better question?", "n", "Better answer.", "d", "hard", "a"])
        question = accepted_questions(tmp_path / "decisions.jsonl")[0]
        assert question.question == "Better question?"
        assert question.expected_answer == "Better answer."
        assert question.difficulty == "hard"
        assert question.provenance.edited_by_reviewer is True

    def test_empty_edit_changes_nothing(self, tmp_path):
        self.run(tmp_path, ["q", "  ", "a"])
        question = accepted_questions(tmp_path / "decisions.jsonl")[0]
        assert question.question == "What restarts containers?"
        assert question.provenance.edited_by_reviewer is False

    def test_replacing_evidence_finds_exact_position(self, tmp_path):
        self.run(tmp_path, ["e", "its restart   policy", "a"])
        located = accepted_questions(tmp_path / "decisions.jsonl")[0].evidence[0].spans[0]
        assert located.quote == "its restart policy"
        assert CONTENT[located.char_start : located.char_end] == located.quote

    def test_evidence_not_in_document_is_refused(self, tmp_path):
        shown = self.run(tmp_path, ["e", "text that does not exist", "a"])
        assert any("nothing changed" in line for line in shown)
        assert (
            accepted_questions(tmp_path / "decisions.jsonl")[0].evidence[0].spans[0].quote == QUOTE
        )

    def test_grade_change_and_invalid_grade(self, tmp_path):
        self.run(tmp_path, ["g", "9", "g", "1", "a"])
        assert accepted_questions(tmp_path / "decisions.jsonl")[0].evidence[0].grade == 1

    def test_unknown_command_shows_help_and_reprompts(self, tmp_path):
        shown = self.run(tmp_path, ["?", "a"])
        assert sum("Commands:" in line for line in shown) == 2

    def test_stops_offering_a_type_once_its_target_is_met(self, tmp_path):
        drafts = [
            make_question(f"t-{i}", f"Table {i}?", question_type="table", human_verified=False)
            for i in range(12)
        ]
        self.run(tmp_path, ["a"] * 10, drafts)
        assert len(accepted_questions(tmp_path / "decisions.jsonl")) == 10

    def test_shows_evidence_context_and_copy_warning(self, tmp_path):
        draft = make_question(human_verified=False, lexical_overlap=0.95)
        shown = "\n".join(self.run(tmp_path, ["a"], [draft]))
        assert f">>>>> {QUOTE} <<<<<" in shown
        assert "https://example.org/pods" in shown
        assert "95% of its key words" in shown
        assert "Check: Is one fact enough" in shown
