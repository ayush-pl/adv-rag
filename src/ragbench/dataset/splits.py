"""Deterministic, stratified dataset splits that keep paraphrases with their source."""

import random
from collections import defaultdict

from ragbench.dataset.models import Question

# The plan's split of 120 questions: development 25, calibration 25, test 70.
DEFAULT_SHARES = {"development": 25 / 120, "calibration": 25 / 120, "test": 70 / 120}


def assign_splits(
    questions: list[Question],
    shares: dict[str, float] | None = None,
    seed: int = 0,
    previous: dict[str, str] | None = None,
) -> dict[str, list[Question]]:
    """Split each question type in proportion, using largest remainders for rounding.

    A paraphrase follows its source question into the same split, which prevents leakage.
    `previous` maps question IDs to the split they were already in. Those questions stay put,
    because moving a question that was tuned on into the test split would leak. Only new
    questions are assigned, stratified by type as before.
    """
    shares = shares or DEFAULT_SHARES
    previous = previous or {}
    by_id = {question.question_id: question for question in questions}
    followers: dict[str, list[Question]] = defaultdict(list)
    strata: dict[str, list[Question]] = defaultdict(list)
    splits: dict[str, list[Question]] = {name: [] for name in shares}
    for question in questions:
        source = question.provenance.source_question_id
        if source in by_id:
            followers[source].append(question)
        elif previous.get(question.question_id) in splits:
            splits[previous[question.question_id]].append(question)
        else:
            strata[question.question_type].append(question)

    for name in list(splits):
        for question in list(splits[name]):
            splits[name].extend(followers[question.question_id])

    rng = random.Random(seed)
    for question_type in sorted(strata):
        members = sorted(strata[question_type], key=lambda q: q.question_id)
        rng.shuffle(members)
        position = 0
        for name, count in _allocate(len(members), shares).items():
            for question in members[position : position + count]:
                splits[name].append(question)
                splits[name].extend(sorted(followers[question.question_id], key=_identifier))
            position += count

    return {name: sorted(values, key=_identifier) for name, values in splits.items()}


def _identifier(question: Question) -> str:
    return question.question_id


def _allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    exact = {name: total * share for name, share in shares.items()}
    counts = {name: int(value) for name, value in exact.items()}
    leftover = total - sum(counts.values())
    by_remainder = sorted(shares, key=lambda name: (counts[name] - exact[name], name))
    for name in by_remainder[:leftover]:
        counts[name] += 1
    return counts
