import pytest
from pydantic import ValidationError

from ragbench.chunking.models import Chunk
from ragbench.dataset.models import Provenance, Question
from ragbench.evaluation.gold import EvidenceGroup, GoldSpan
from ragbench.experiments.config import ChunkingChoice, RetrievalConfig
from ragbench.experiments.runner import (
    QuestionResult,
    load_run,
    rescore,
    run_retrieval,
    save_run,
    summarize,
)
from ragbench.experiments.sweep import (
    SweepSpec,
    expand_sweep,
    paired_difference,
    render_report,
)
from ragbench.experiments.tracker import Environment, capture_environment
from ragbench.retrieval.base import Retriever, ScoredChunk

CHUNKING = ChunkingChoice(strategy="recursive", chunk_size=512, overlap=50)
ENVIRONMENT = Environment(
    git_commit="abc1234567",
    git_dirty=False,
    python="3.12",
    platform="test",
    torch=None,
    cuda_device=None,
)


def config(**overrides) -> RetrievalConfig:
    fields = {"name": "run", "chunking": CHUNKING, "method": "bm25"}
    return RetrievalConfig(**{**fields, **overrides})


def chunk(document_id: str, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}::{start}-{end}",
        document_id=document_id,
        char_start=start,
        char_end=end,
        text="x" * (end - start),
        token_count=1,
        section="",
        source="S",
        source_url="https://example.org",
        title="T",
        version="1",
    )


def question(question_id: str, answerable: bool = True) -> Question:
    span = GoldSpan(document_id="d", char_start=10, char_end=20, quote="q")
    return Question(
        question_id=question_id,
        question=f"Question {question_id}?",
        question_type="single_hop" if answerable else "unanswerable",
        expected_answer="A",
        evidence=[EvidenceGroup(spans=[span])] if answerable else [],
        answerable=answerable,
        human_verified=True,
        provenance=Provenance(method="manual"),
    )


class ScriptedRetriever(Retriever):
    """Returns the gold chunk at a fixed rank for each question."""

    def __init__(self, rank_of_gold: int) -> None:
        self.rank_of_gold = rank_of_gold
        self.calls = 0

    def retrieve(self, query, top_k, metadata_filter=None):
        self.calls += 1
        ranked = [chunk("other", i * 100, i * 100 + 50) for i in range(top_k)]
        ranked.insert(self.rank_of_gold - 1, chunk("d", 0, 50))
        return [ScoredChunk(c, 1.0 / (i + 1)) for i, c in enumerate(ranked[:top_k])]


def result(question_id: str, recall: float, ndcg: float) -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
        question_type="single_hop",
        chunk_ids=[],
        scores=[],
        latency_ms=1.0,
        metrics={"recall_at_5": recall, "mrr": recall, "ndcg_at_10": ndcg},
    )


class TestConfig:
    def test_dense_and_hybrid_need_an_embedder(self):
        with pytest.raises(ValidationError, match="needs an embedder"):
            config(method="dense")
        with pytest.raises(ValidationError, match="does not use an embedder"):
            config(embedder="m")

    def test_id_ignores_name_but_tracks_every_setting(self):
        assert config(name="a").config_id == config(name="b").config_id
        assert config().config_id != config(top_k=10).config_id
        assert config().run_name == f"run-{config().config_id}"


class TestRunner:
    def test_scores_answerable_questions_only_and_warms_up_first(self):
        retriever = ScriptedRetriever(rank_of_gold=2)
        questions = [question("q1"), question("q2", answerable=False), question("q3")]
        results = run_retrieval(config(), questions, retriever)
        assert [r.question_id for r in results] == ["q1", "q3"]
        assert retriever.calls == 3
        assert results[0].metrics["mrr"] == 0.5
        assert results[0].metrics["recall_at_1"] == 0.0
        assert results[0].chunk_ids[1] == "d::0-50"

    def test_summary_and_save_round_trip(self, tmp_path):
        questions = [question("q1"), question("q2", answerable=False)]
        results = run_retrieval(config(), questions, ScriptedRetriever(1))
        record = summarize(config(), results, questions, "development", "dh", "ch", ENVIRONMENT)
        assert (record.questions_scored, record.questions_skipped) == (1, 1)
        assert record.metrics["mrr"] == 1.0

        save_run(tmp_path / "run", record, results)
        assert load_run(tmp_path / "run") == (record, results)

    def test_summary_without_answerable_questions_is_an_error(self):
        with pytest.raises(ValueError, match="no answerable"):
            summarize(config(), [], [], "development", "dh", "ch", ENVIRONMENT)

    def test_environment_records_git_state(self):
        environment = capture_environment()
        assert environment.python
        assert environment.git_commit is None or len(environment.git_commit) == 40


SPEC = {
    "chunking": {"r512": CHUNKING.model_dump()},
    "embedders": {"bge": "org/bge"},
    "runs": [
        {"chunking": ["r512"], "method": "bm25"},
        {"chunking": ["r512"], "method": "hybrid", "embedder": "bge", "rrf_k": [10, 60]},
        {"chunking": ["r512"], "method": "hybrid", "embedder": "bge", "rerank": True},
    ],
    "comparisons": [
        {"experiment": "R4", "baseline": "r512-hybrid-bge-k60", "treatment": "r512-bm25"}
    ],
    "top_k_configs": ["r512-bm25"],
}


class TestSweep:
    def test_expands_names_and_settings(self):
        configs = {c.name: c for c in expand_sweep(SweepSpec.model_validate(SPEC))}
        assert sorted(configs) == [
            "r512-bm25",
            "r512-hybrid-bge-k10",
            "r512-hybrid-bge-k60",
            "r512-hybrid-bge-k60-rerank",
        ]
        assert configs["r512-hybrid-bge-k10"].rrf_k == 10
        assert configs["r512-hybrid-bge-k60"].embedder == "org/bge"
        assert configs["r512-hybrid-bge-k60-rerank"].rerank is True

    def test_unknown_comparison_names_are_rejected(self):
        spec = {**SPEC, "comparisons": [{"experiment": "R1", "baseline": "x", "treatment": "y"}]}
        with pytest.raises(ValueError, match="unknown run x"):
            expand_sweep(SweepSpec.model_validate(spec))

    def test_duplicate_runs_are_rejected(self):
        spec = {**SPEC, "runs": SPEC["runs"] + [{"chunking": ["r512"], "method": "bm25"}]}
        with pytest.raises(ValueError, match="twice"):
            expand_sweep(SweepSpec.model_validate(spec))

    def test_paired_difference_verdicts(self):
        baseline = [result(f"q{i}", 0.0, i / 40) for i in range(40)]
        better = [result(f"q{i}", 1.0, i / 40 + 0.1) for i in range(40)]
        noisy = [result(f"q{i}", 0.0, (i / 40) + (0.2 if i % 2 else -0.2)) for i in range(40)]
        assert paired_difference(baseline, better, "ndcg_at_10").verdict == "better"
        assert paired_difference(better, baseline, "ndcg_at_10").verdict == "worse"
        assert paired_difference(baseline, noisy, "ndcg_at_10").verdict == "inconclusive"
        assert paired_difference(baseline, better, "recall_at_5").estimate == pytest.approx(1.0)

    def test_paired_difference_needs_the_same_questions(self):
        with pytest.raises(ValueError, match="different questions"):
            paired_difference([result("q1", 1, 1)], [result("q2", 1, 1)], "mrr")

    def test_report_contains_every_section(self):
        spec = SweepSpec.model_validate(SPEC)
        questions = [question(f"q{i}") for i in range(6)] + [question("u", answerable=False)]
        runs = {}
        for item, rank in (("r512-bm25", 1), ("r512-hybrid-bge-k60", 3)):
            cfg = config(name=item)
            results = run_retrieval(cfg, questions, ScriptedRetriever(rank))
            record = summarize(cfg, results, questions, "development", "dh", "ch", ENVIRONMENT)
            runs[item] = (record, results)

        report = render_report(spec, runs, "Sweep")
        assert report.startswith("# Sweep")
        assert "6 answerable questions scored (1 unanswerable skipped)" in report
        assert report.index("| r512-bm25 |") < report.index("| r512-hybrid-bge-k60 |")
        assert "| R4 | r512-hybrid-bge-k60 | r512-bm25 |" in report
        assert "## R7: Recall by cutoff" in report
        assert "| single_hop | 6 |" in report


class TestRescore:
    def test_uses_new_labels_without_retrieving_again(self):
        questions = [question("q1")]
        results = run_retrieval(config(), questions, ScriptedRetriever(rank_of_gold=2))
        assert results[0].metrics["mrr"] == 0.5

        # A reviewer finds that the passage ranked first also answers the question.
        first = results[0].chunk_ids[0]
        document_id, _, span = first.rpartition("::")
        start, end = (int(x) for x in span.split("-"))
        extra = GoldSpan(document_id=document_id, char_start=start, char_end=end, quote="q")
        relabelled = question("q1").model_copy(deep=True)
        relabelled.evidence[0].spans.append(extra)

        rescored = rescore(results, [relabelled])
        assert rescored[0].metrics["mrr"] == 1.0
        assert rescored[0].chunk_ids == results[0].chunk_ids
        assert rescored[0].latency_ms == results[0].latency_ms

    def test_different_questions_need_a_real_run(self):
        results = run_retrieval(config(), [question("q1")], ScriptedRetriever(1))
        assert rescore(results, [question("q1"), question("q2")]) is None
        assert rescore(results, [question("q1"), question("u", answerable=False)]) is not None
