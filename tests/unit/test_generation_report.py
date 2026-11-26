import pytest

from ragbench.evaluation.generation_report import (
    Scored,
    audit_candidates,
    compare_runs,
    diagnose,
    join,
    summarize_run,
)
from ragbench.evaluation.judge import AnswerEvaluation, DimensionScore, Judgement
from ragbench.generation.rag_pipeline import GeneratedAnswer


def scored(
    question_id: str,
    correctness: int | None = 5,
    faithfulness: int = 5,
    answerable=True,
    abstained=False,
    parsed=True,
    total_ms=1000.0,
    load_ms=0.0,
    citation_precision=None,
) -> Scored:
    answer = GeneratedAnswer(
        question_id=question_id,
        model="m",
        mode="retrieved",
        prompt_version="v",
        context_chunk_ids=[],
        answer="a",
        citations=[],
        invalid_citations=[],
        abstained=abstained,
        parsed=parsed,
        raw_output="",
        prompt_tokens=1,
        completion_tokens=1,
        possibly_truncated=False,
        load_ms=load_ms,
        generation_ms=1,
        total_ms=total_ms,
    )
    judged = answerable and not abstained and parsed and correctness is not None
    scores = (
        {
            "correctness": DimensionScore(score=correctness, reason=""),
            "faithfulness": DimensionScore(score=faithfulness, reason=""),
            "relevance": DimensionScore(score=5, reason=""),
        }
        if judged
        else None
    )
    evaluation = AnswerEvaluation(
        question_id=question_id,
        question_type="single_hop" if answerable else "unanswerable",
        answerable=answerable,
        abstained=abstained,
        abstention_correct=abstained != answerable,
        citation_precision=citation_precision,
        citation_recall=None,
        judgement=Judgement(
            question_id=question_id,
            judge_model="j",
            judge_prompt_version="v",
            scores=scores,
            valid=parsed,
        ),
    )
    return Scored(answer, evaluation)


class TestSummary:
    def test_abstaining_on_answerable_counts_as_incorrect(self):
        run = [scored("a"), scored("b", abstained=True), scored("c", correctness=2)]
        summary = summarize_run(run)
        assert summary.answerable == 3
        assert summary.correct_rate == pytest.approx(1 / 3)
        assert summary.mean_correctness == pytest.approx((5 + 1 + 2) / 3)
        assert summary.false_abstention_rate == pytest.approx(1 / 3)

    def test_faithfulness_and_hallucination_cover_given_answers_only(self):
        run = [
            scored("a", faithfulness=5),
            scored("b", faithfulness=1),
            scored("c", abstained=True),
        ]
        summary = summarize_run(run)
        assert summary.mean_faithfulness == pytest.approx(3.0)
        assert summary.hallucination_rate == pytest.approx(0.5)

    def test_refusals_failures_citations_and_latency(self):
        run = [
            scored("a", total_ms=3000, load_ms=2000, citation_precision=0.5),
            scored("b", parsed=False, total_ms=500),
            scored("u1", answerable=False, abstained=True),
            scored("u2", answerable=False, abstained=False),
        ]
        summary = summarize_run(run)
        assert summary.refusal_rate == pytest.approx(0.5)
        assert summary.failed_outputs == 1
        assert summary.citation_precision == pytest.approx(0.5)
        assert summary.latency_p50_ms == pytest.approx(1000)

    def test_unanswerable_has_no_correctness(self):
        with pytest.raises(ValueError):
            _ = scored("u", answerable=False).correctness


class TestPairs:
    def test_join_requires_an_evaluation_for_every_answer(self):
        item = scored("a")
        assert join([item.answer], [item.evaluation]) == [item]
        with pytest.raises(ValueError, match="no evaluation"):
            join([item.answer], [])

    def test_compare_counts_discordant_pairs(self):
        baseline = [scored(f"q{i}", correctness=2) for i in range(10)]
        treatment = [scored(f"q{i}", correctness=5 if i < 8 else 2) for i in range(10)]
        result = compare_runs(baseline, treatment)
        assert (result.baseline_only, result.treatment_only) == (0, 8)
        assert result.correct_difference == pytest.approx(0.8)
        assert result.mcnemar_p < 0.01

    def test_compare_rejects_different_questions(self):
        with pytest.raises(ValueError, match="different questions"):
            compare_runs([scored("a")], [scored("b")])


class TestOracle:
    def test_diagnosis_separates_retrieval_and_generation_failures(self):
        retrieved = [
            scored("ok"),
            scored("ret", correctness=1),
            scored("gen", correctness=1),
            scored("odd"),
        ]
        oracle = [
            scored("ok"),
            scored("ret"),
            scored("gen", correctness=2),
            scored("odd", correctness=1),
        ]
        result = diagnose(retrieved, oracle)
        assert result.questions == 4
        assert (result.both_correct, result.retrieval_failures) == (1, 1)
        assert (result.generation_failures, result.oracle_only_wrong) == (2, 1)
        assert result.oracle_correct_rate == pytest.approx(0.5)

    def test_audit_flags_questions_every_oracle_run_failed(self):
        first = [scored("bad", correctness=1), scored("mixed", correctness=1), scored("good")]
        second = [scored("bad", correctness=2), scored("mixed", correctness=5), scored("good")]
        assert audit_candidates([first, second]) == ["bad"]

    def test_abstaining_with_gold_passages_counts_as_failure(self):
        assert audit_candidates([[scored("q", abstained=True)], [scored("q", correctness=1)]]) == [
            "q"
        ]
