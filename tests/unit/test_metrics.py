import math

import pytest
from pydantic import ValidationError

from ragbench.chunking.models import Chunk
from ragbench.evaluation.gold import (
    EvidenceGroup,
    GoldSpan,
    chunk_matches_span,
    match_ranking,
)
from ragbench.evaluation.retrieval_metrics import (
    average_precision,
    evaluate_ranking,
    full_coverage_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragbench.evaluation.significance import bootstrap_ci, mcnemar_exact, paired_bootstrap


def make_chunk(document_id: str, start: int, end: int) -> Chunk:
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


def span(document_id: str, start: int, end: int) -> GoldSpan:
    return GoldSpan(document_id=document_id, char_start=start, char_end=end, quote="q")


def group(document_id: str, start: int, end: int, grade: int = 2) -> EvidenceGroup:
    return EvidenceGroup(spans=[span(document_id, start, end)], grade=grade)


class TestSpanMatching:
    def test_empty_span_is_rejected(self):
        with pytest.raises(ValidationError):
            span("d", 5, 5)

    def test_grade_must_be_one_or_two(self):
        with pytest.raises(ValidationError):
            EvidenceGroup(spans=[span("d", 0, 1)], grade=3)

    def test_group_needs_a_span(self):
        with pytest.raises(ValidationError):
            EvidenceGroup(spans=[])

    def test_different_document_never_matches(self):
        assert not chunk_matches_span(make_chunk("a", 0, 100), span("b", 0, 100))

    @pytest.mark.parametrize(
        ("chunk_range", "span_range", "expected"),
        [
            ((0, 100), (10, 20), True),  # chunk contains span
            ((10, 20), (0, 100), True),  # span contains chunk
            ((0, 100), (90, 110), True),  # exactly half of the 20-char span
            ((0, 100), (91, 111), False),  # just under half
            ((0, 100), (100, 110), False),  # touching but no shared characters
        ],
    )
    def test_overlap_is_measured_against_the_shorter(self, chunk_range, span_range, expected):
        assert chunk_matches_span(make_chunk("d", *chunk_range), span("d", *span_range)) is expected

    def test_threshold_is_configurable(self):
        chunk = make_chunk("d", 0, 100)
        assert chunk_matches_span(chunk, span("d", 95, 115), min_overlap=0.25)
        assert not chunk_matches_span(chunk, span("d", 95, 115), min_overlap=0.5)

    def test_any_alternative_span_satisfies_a_group(self):
        alternatives = EvidenceGroup(spans=[span("pg-15/x", 0, 10), span("pg-16/x", 0, 10)])
        ranked = [make_chunk("pg-16/x", 0, 50), make_chunk("other", 0, 50)]
        assert match_ranking(ranked, [alternatives]) == [{0}, set()]

    def test_one_chunk_can_satisfy_several_groups(self):
        groups = [group("d", 0, 10), group("d", 20, 30)]
        assert match_ranking([make_chunk("d", 0, 100)], groups) == [{0, 1}]


# Ranks 1-5: miss, evidence 0, miss, evidence 1, repeat of evidence 0.
MATCHES = [set(), {0}, set(), {1}, {0}]


class TestRankingMetrics:
    def test_recall_counts_evidence_not_chunks(self):
        assert recall_at_k(MATCHES, 2, 1) == 0.0
        assert recall_at_k(MATCHES, 2, 2) == 0.5
        assert recall_at_k(MATCHES, 2, 5) == 1.0

    def test_recall_with_no_results_is_zero(self):
        assert recall_at_k([], 2, 5) == 0.0

    def test_full_coverage_needs_every_group(self):
        assert not full_coverage_at_k(MATCHES, 2, 3)
        assert full_coverage_at_k(MATCHES, 2, 4)

    def test_precision_divides_by_k_even_with_few_results(self):
        assert precision_at_k(MATCHES, 2) == 0.5
        assert precision_at_k(MATCHES, 5) == 3 / 5
        assert precision_at_k([{0}], 5) == 1 / 5

    def test_reciprocal_rank_uses_first_relevant(self):
        assert reciprocal_rank(MATCHES) == 0.5
        assert reciprocal_rank(MATCHES, k=1) == 0.0
        assert reciprocal_rank([set(), set()]) == 0.0

    def test_average_precision_hand_computed(self):
        # Evidence 0 first found at rank 2 (P=1/2), evidence 1 at rank 4 (P=2/4).
        assert average_precision(MATCHES, 2) == pytest.approx((1 / 2 + 2 / 4) / 2)

    def test_average_precision_counts_missing_evidence_as_zero(self):
        assert average_precision([{0}], 2) == pytest.approx(0.5)

    def test_ndcg_hand_computed(self):
        # Grade 2 at rank 2, grade 1 at rank 3.
        matches = [set(), {0}, {1}]
        dcg = 3 / math.log2(3) + 1 / math.log2(4)
        ideal = 3 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(matches, [2, 1], 3) == pytest.approx(dcg / ideal)

    def test_ndcg_is_one_for_ideal_order_and_zero_for_no_evidence(self):
        assert ndcg_at_k([{0}, {1}], [2, 1], 5) == pytest.approx(1.0)
        assert ndcg_at_k([set(), set()], [2, 1], 5) == 0.0

    def test_ndcg_ideal_is_limited_to_k(self):
        # Top 1 holds the best possible single result, so nDCG@1 is perfect.
        assert ndcg_at_k([{0}], [2, 1], 1) == pytest.approx(1.0)

    def test_ndcg_is_capped_when_one_chunk_satisfies_several_groups(self):
        assert ndcg_at_k([{0, 1}], [2, 2], 5) == 1.0

    def test_repeated_evidence_earns_no_extra_credit(self):
        once = [{0}, set()]
        repeated = [{0}, {0}]
        assert ndcg_at_k(once, [2, 2], 2) == ndcg_at_k(repeated, [2, 2], 2)
        assert average_precision(once, 2) == average_precision(repeated, 2)
        assert recall_at_k(once, 2, 2) == recall_at_k(repeated, 2, 2)


class TestEvaluateRanking:
    def test_produces_all_metrics_from_chunks(self):
        ranked = [make_chunk("x", 0, 10), make_chunk("d", 0, 100)]
        metrics = evaluate_ranking(ranked, [group("d", 10, 20)], cutoffs=(1, 2))
        assert metrics["mrr"] == 0.5
        assert metrics["recall_at_1"] == 0.0
        assert metrics["recall_at_2"] == 1.0
        assert metrics["full_coverage_at_2"] == 1.0
        assert metrics["precision_at_2"] == 0.5

    def test_same_evidence_scores_alike_under_different_chunking(self):
        groups = [group("d", 100, 200)]
        coarse = [make_chunk("d", 0, 1000)]
        fine = [make_chunk("d", 100, 150), make_chunk("d", 150, 200)]
        assert evaluate_ranking(coarse, groups, (5,))["recall_at_5"] == 1.0
        assert evaluate_ranking(fine, groups, (5,))["recall_at_5"] == 1.0
        assert evaluate_ranking(coarse, groups, (5,))["ndcg_at_5"] == pytest.approx(1.0)
        assert evaluate_ranking(fine, groups, (5,))["ndcg_at_5"] == pytest.approx(1.0)

    def test_unanswerable_question_is_rejected(self):
        with pytest.raises(ValueError, match="no gold evidence"):
            evaluate_ranking([make_chunk("d", 0, 1)], [])


@pytest.mark.reference
class TestAgainstReferenceLibrary:
    """Our metrics must equal an established implementation when evidence maps 1:1 to chunks."""

    def test_matches_ranx(self):
        ranx = pytest.importorskip("ranx")
        ranked_grades = {
            "q1": [0, 2, 0, 1, 0, 0],
            "q2": [1, 0, 0, 0, 2, 1],
            "q3": [0, 0, 0, 0, 0, 2],
            "q4": [2, 1, 1, 0, 0, 0],
        }
        qrels = {
            query: {f"d{i}": grade for i, grade in enumerate(grades) if grade}
            for query, grades in ranked_grades.items()
        }
        run = {
            query: {f"d{i}": float(100 - i) for i in range(len(grades))}
            for query, grades in ranked_grades.items()
        }

        ours: dict[str, list[float]] = {}
        for grades in ranked_grades.values():
            relevant = [grade for grade in grades if grade]
            order = iter(range(len(relevant)))
            matches = [{next(order)} if grade else set() for grade in grades]
            for name, value in {
                "recall@5": recall_at_k(matches, len(relevant), 5),
                "precision@5": precision_at_k(matches, 5),
                "mrr": reciprocal_rank(matches),
                "map": average_precision(matches, len(relevant)),
                "ndcg_burges@5": ndcg_at_k(matches, relevant, 5),
            }.items():
                ours.setdefault(name, []).append(value)

        expected = ranx.evaluate(ranx.Qrels(qrels), ranx.Run(run), list(ours))
        for name, values in ours.items():
            assert sum(values) / len(values) == pytest.approx(expected[name]), name


class TestSignificance:
    def test_bootstrap_contains_mean_and_is_reproducible(self):
        values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        first = bootstrap_ci(values, seed=3)
        assert first.estimate == pytest.approx(0.7)
        assert first.low < 0.7 < first.high
        assert first.sample_size == 10
        assert bootstrap_ci(values, seed=3) == first

    def test_bootstrap_of_constant_has_zero_width(self):
        interval = bootstrap_ci([0.5] * 20)
        assert (interval.low, interval.high) == (0.5, 0.5)

    def test_bootstrap_width_is_near_normal_approximation(self):
        values = [1.0] * 50 + [0.0] * 50
        interval = bootstrap_ci(values)
        assert interval.high - interval.low == pytest.approx(2 * 1.96 * 0.05, abs=0.01)

    def test_bootstrap_rejects_empty_input(self):
        with pytest.raises(ValueError, match="no values"):
            bootstrap_ci([])

    def test_paired_detects_consistent_small_gain(self):
        # Every question improves by a little. An unpaired view would hide this in the spread.
        baseline = [i / 40 for i in range(40)]
        treatment = [value + 0.02 for value in baseline]
        comparison = paired_bootstrap(baseline, treatment)
        assert comparison.difference.estimate == pytest.approx(0.02)
        assert comparison.is_conclusive

    def test_paired_is_inconclusive_for_noise(self):
        baseline = [0, 1] * 20
        treatment = [1, 0] * 20
        assert not paired_bootstrap(baseline, treatment).is_conclusive

    def test_paired_rejects_mismatched_sizes(self):
        with pytest.raises(ValueError, match="differ in size"):
            paired_bootstrap([1.0], [1.0, 2.0])

    def test_mcnemar_hand_computed(self):
        # 1 baseline-only win and 9 treatment-only wins: 2 * (1 + 10 + ... ) / 2^10 = 0.021484375
        baseline = [True] + [False] * 9 + [True] * 5
        treatment = [False] + [True] * 9 + [True] * 5
        result = mcnemar_exact(baseline, treatment)
        assert (result.baseline_only, result.treatment_only) == (1, 9)
        assert result.p_value == pytest.approx(22 / 1024)

    def test_mcnemar_with_no_disagreement_is_not_significant(self):
        assert mcnemar_exact([True, False], [True, False]).p_value == 1.0
