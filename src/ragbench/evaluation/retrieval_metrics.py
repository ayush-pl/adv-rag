"""Deterministic ranking metrics.

Inputs come from `match_ranking`: for each rank, the evidence groups that chunk satisfies.
Recall, nDCG and MAP credit each evidence group once, at the first rank that satisfies it.
Overlapping chunks that repeat the same evidence therefore cannot inflate a score, which keeps
these metrics comparable across chunking configurations. Precision and MRR are per chunk.
"""

import math
from collections.abc import Sequence

from ragbench.chunking.models import Chunk
from ragbench.evaluation.gold import DEFAULT_MIN_OVERLAP, EvidenceGroup, match_ranking

DEFAULT_CUTOFFS = (1, 3, 5, 10, 20)


def recall_at_k(matches: Sequence[set[int]], group_count: int, k: int) -> float:
    """Share of evidence groups satisfied in the top k. Also the spec's evidence coverage."""
    covered = set().union(*matches[:k]) if matches[:k] else set()
    return len(covered) / group_count


def full_coverage_at_k(matches: Sequence[set[int]], group_count: int, k: int) -> bool:
    """Whether all evidence is in the top k. A multi-hop question needs every hop."""
    return recall_at_k(matches, group_count, k) == 1.0


def precision_at_k(matches: Sequence[set[int]], k: int) -> float:
    """Share of the k result slots holding a chunk with any evidence."""
    return sum(1 for groups in matches[:k] if groups) / k


def reciprocal_rank(matches: Sequence[set[int]], k: int | None = None) -> float:
    for rank, groups in enumerate(matches[:k], start=1):
        if groups:
            return 1.0 / rank
    return 0.0


def average_precision(matches: Sequence[set[int]], group_count: int, k: int | None = None) -> float:
    """Mean over evidence groups of the precision at the rank that first satisfied them."""
    seen: set[int] = set()
    relevant_so_far = 0
    total = 0.0
    for rank, groups in enumerate(matches[:k], start=1):
        if groups:
            relevant_so_far += 1
        new_groups = groups - seen
        seen |= new_groups
        total += len(new_groups) * relevant_so_far / rank
    return total / group_count


def ndcg_at_k(matches: Sequence[set[int]], grades: Sequence[int], k: int) -> float:
    """Normalized DCG with gain 2^grade - 1.

    One large chunk can satisfy several groups at a single rank, which can beat the ideal
    ordering of one group per rank. The result is capped at 1.0 for that case.
    """
    seen: set[int] = set()
    dcg = 0.0
    for rank, groups in enumerate(matches[:k], start=1):
        for group in groups - seen:
            dcg += _gain(grades[group]) / math.log2(rank + 1)
        seen |= groups

    ideal_grades = sorted(grades, reverse=True)[:k]
    ideal = sum(_gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, 1))
    return min(dcg / ideal, 1.0) if ideal else 0.0


def _gain(grade: int) -> float:
    return 2.0**grade - 1.0


def evaluate_ranking(
    ranked: list[Chunk],
    groups: list[EvidenceGroup],
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> dict[str, float]:
    """All retrieval metrics for one question. Unanswerable questions have no retrieval score."""
    if not groups:
        raise ValueError("Retrieval metrics are undefined for a question with no gold evidence")

    matches = match_ranking(ranked, groups, min_overlap)
    grades = [group.grade for group in groups]
    metrics = {
        "mrr": reciprocal_rank(matches),
        "map": average_precision(matches, len(groups)),
    }
    for k in cutoffs:
        metrics[f"recall_at_{k}"] = recall_at_k(matches, len(groups), k)
        metrics[f"full_coverage_at_{k}"] = float(full_coverage_at_k(matches, len(groups), k))
        metrics[f"precision_at_{k}"] = precision_at_k(matches, k)
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(matches, grades, k)
    return metrics
