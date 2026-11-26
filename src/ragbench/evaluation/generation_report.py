"""Generation metrics, paired comparisons and oracle-context diagnostics.

Answer correctness is counted over every answerable question, with abstaining or failing to
answer scored as incorrect, so a system cannot look accurate by refusing hard questions.
Faithfulness and relevance describe the answers actually given.
"""

from dataclasses import dataclass

import numpy as np

from ragbench.evaluation.judge import AnswerEvaluation
from ragbench.evaluation.significance import bootstrap_ci, mcnemar_exact, paired_bootstrap
from ragbench.generation.rag_pipeline import GeneratedAnswer

# A correctness score of 4 or 5 counts as a correct answer.
CORRECT_THRESHOLD = 4
# A faithfulness score of 1 or 2 counts as a hallucinated answer.
HALLUCINATION_THRESHOLD = 2


@dataclass(frozen=True)
class Scored:
    """One answer with its evaluation, joined on question ID."""

    answer: GeneratedAnswer
    evaluation: AnswerEvaluation

    def score(self, dimension: str) -> int | None:
        scores = self.evaluation.judgement.scores
        return scores[dimension].score if scores else None

    @property
    def answered(self) -> bool:
        return self.evaluation.answerable and not self.answer.abstained and self.answer.parsed

    @property
    def correctness(self) -> int:
        """Judge score, or 1 when an answerable question got no usable answer."""
        if not self.evaluation.answerable:
            raise ValueError("Unanswerable questions have no correctness score")
        return self.score("correctness") or 1

    @property
    def correct(self) -> bool:
        return self.correctness >= CORRECT_THRESHOLD


def join(answers: list[GeneratedAnswer], evaluations: list[AnswerEvaluation]) -> list[Scored]:
    by_id = {evaluation.question_id: evaluation for evaluation in evaluations}
    missing = [a.question_id for a in answers if a.question_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} answers have no evaluation, such as {missing[0]}")
    return [Scored(answer, by_id[answer.question_id]) for answer in answers]


@dataclass(frozen=True)
class RunSummary:
    answerable: int
    unanswerable: int
    correct_rate: float
    correct_low: float
    correct_high: float
    mean_correctness: float
    mean_faithfulness: float | None
    mean_relevance: float | None
    hallucination_rate: float | None
    false_abstention_rate: float
    # Share of unanswerable questions correctly refused. None when the run had none.
    refusal_rate: float | None
    failed_outputs: int
    citation_precision: float | None
    citation_recall: float | None
    latency_p50_ms: float
    latency_p95_ms: float


def summarize_run(scored: list[Scored]) -> RunSummary:
    answerable = [s for s in scored if s.evaluation.answerable]
    unanswerable = [s for s in scored if not s.evaluation.answerable]
    answered = [s for s in answerable if s.answered and s.score("faithfulness") is not None]
    correct = bootstrap_ci([float(s.correct) for s in answerable])
    precision = [
        s.evaluation.citation_precision
        for s in scored
        if s.evaluation.citation_precision is not None
    ]
    recall = [
        s.evaluation.citation_recall for s in scored if s.evaluation.citation_recall is not None
    ]
    # Steady-state latency: model loading is reported separately, not as query time.
    latency = [s.answer.total_ms - s.answer.load_ms for s in scored if not s.answer.error]

    def mean(values):
        return float(np.mean(values)) if values else None

    return RunSummary(
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        correct_rate=correct.estimate,
        correct_low=correct.low,
        correct_high=correct.high,
        mean_correctness=float(np.mean([s.correctness for s in answerable])),
        mean_faithfulness=mean([s.score("faithfulness") for s in answered]),
        mean_relevance=mean([s.score("relevance") for s in answered]),
        hallucination_rate=mean(
            [float(s.score("faithfulness") <= HALLUCINATION_THRESHOLD) for s in answered]
        ),
        false_abstention_rate=float(np.mean([s.answer.abstained for s in answerable])),
        refusal_rate=mean([float(s.answer.abstained) for s in unanswerable]),
        failed_outputs=sum(not s.answer.parsed for s in scored),
        citation_precision=mean(precision),
        citation_recall=mean(recall),
        latency_p50_ms=float(np.percentile(latency, 50)) if latency else 0.0,
        latency_p95_ms=float(np.percentile(latency, 95)) if latency else 0.0,
    )


@dataclass(frozen=True)
class PairedGeneration:
    correct_difference: float
    correct_low: float
    correct_high: float
    mcnemar_p: float
    baseline_only: int
    treatment_only: int


def compare_runs(baseline: list[Scored], treatment: list[Scored]) -> PairedGeneration:
    """Treatment minus baseline on the answerable questions both runs answered."""
    base = {s.answer.question_id: s for s in baseline if s.evaluation.answerable}
    treat = {s.answer.question_id: s for s in treatment if s.evaluation.answerable}
    if set(base) != set(treat):
        raise ValueError("Runs cover different questions, so they cannot be paired")
    ids = sorted(base)
    b = [base[i].correct for i in ids]
    t = [treat[i].correct for i in ids]
    difference = paired_bootstrap([float(x) for x in b], [float(x) for x in t]).difference
    test = mcnemar_exact(b, t)
    return PairedGeneration(
        difference.estimate,
        difference.low,
        difference.high,
        test.p_value,
        test.baseline_only,
        test.treatment_only,
    )


@dataclass(frozen=True)
class OracleDiagnosis:
    """Where answers fail, from running the same questions with gold passages."""

    questions: int
    both_correct: int
    # Correct with gold passages but not with retrieved ones: retrieval was the bottleneck.
    retrieval_failures: int
    # Wrong even with gold passages: generation, or the reference answer, is the problem.
    generation_failures: int
    # Correct with retrieved passages but not gold ones: usually noise in judging.
    oracle_only_wrong: int
    retrieved_correct_rate: float
    oracle_correct_rate: float


def diagnose(retrieved: list[Scored], oracle: list[Scored]) -> OracleDiagnosis:
    real = {s.answer.question_id: s for s in retrieved if s.evaluation.answerable}
    gold = {s.answer.question_id: s for s in oracle if s.evaluation.answerable}
    ids = sorted(set(real) & set(gold))
    pairs = [(real[i].correct, gold[i].correct) for i in ids]
    return OracleDiagnosis(
        questions=len(ids),
        both_correct=sum(r and g for r, g in pairs),
        retrieval_failures=sum(g and not r for r, g in pairs),
        generation_failures=sum(not g for _, g in pairs),
        oracle_only_wrong=sum(r and not g for r, g in pairs),
        retrieved_correct_rate=float(np.mean([r for r, _ in pairs])) if pairs else 0.0,
        oracle_correct_rate=float(np.mean([g for _, g in pairs])) if pairs else 0.0,
    )


def audit_candidates(oracle_runs: list[list[Scored]], max_correctness: int = 2) -> list[str]:
    """Questions every oracle run got badly wrong, despite having the gold passages.

    When several different models all fail with the right evidence in hand, the reference
    answer or the question is the more likely problem, so these go to a human to check.
    """
    by_question: dict[str, list[int]] = {}
    for run in oracle_runs:
        for scored in run:
            if scored.evaluation.answerable:
                by_question.setdefault(scored.answer.question_id, []).append(scored.correctness)
    return sorted(
        question_id
        for question_id, scores in by_question.items()
        if len(scores) == len(oracle_runs) and max(scores) <= max_correctness
    )
