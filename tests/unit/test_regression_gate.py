import math

import numpy as np
import pytest

from ragbench.regression.fixture import CachedQueryEmbedder
from ragbench.regression.gate import RETRIEVAL_METRICS, Baseline, compare, overall, render


def values(per_question: list[float]) -> dict:
    return {f"q{i}": {m: v for m in RETRIEVAL_METRICS} for i, v in enumerate(per_question)}


def generation(correct: list[bool | None]) -> dict:
    return {
        f"q{i}": {
            "correct": c,
            "abstention_correct": True,
            "citation_precision": None if c is None else 1.0,
            "citation_recall": None if c is None else 0.5,
        }
        for i, c in enumerate(correct)
    }


BEFORE = [0.0, 0.5, 1.0, 1.0] * 10


def baseline(**overrides) -> Baseline:
    fields = {
        "dataset_hash": "h",
        "tolerance": 0.01,
        "retrieval": {"hybrid": values(BEFORE)},
        "generation": {"run": generation([True, False, None])},
    }
    return Baseline(**{**fields, **overrides})


def verdicts(retrieval_after: list[float], generation_after=None) -> list[str]:
    checks = compare(
        baseline(),
        {"hybrid": values(retrieval_after)},
        {"run": generation_after or generation([True, False, None])},
    )
    return [c.verdict for c in checks]


def test_identical_results_pass():
    assert set(verdicts(BEFORE)) == {"pass"}


def test_drop_within_tolerance_passes():
    after = list(BEFORE)
    after[1] = 0.4  # mean drops by 0.0025
    assert set(verdicts(after)) == {"pass"}


def test_consistent_drop_fails():
    after = [max(0.0, v - 0.2) for v in BEFORE]
    checks = compare(
        baseline(), {"hybrid": values(after)}, {"run": generation([True, False, None])}
    )
    assert overall(checks) == "fail"
    failed = [c for c in checks if c.verdict == "fail"]
    assert {c.metric for c in failed} == set(RETRIEVAL_METRICS)
    assert all(c.high < 0 for c in failed)


def test_mixed_change_is_inconclusive():
    # Some questions gain and others lose: the mean drops, but not consistently.
    after = [1.0, 0.5, 0.0, 0.75] * 10
    checks = compare(
        baseline(), {"hybrid": values(after)}, {"run": generation([True, False, None])}
    )
    retrieval = [c for c in checks if c.group == "retrieval"]
    assert all(c.delta < -0.01 for c in retrieval)
    assert {c.verdict for c in retrieval} == {"inconclusive"}
    assert overall(checks) == "inconclusive"


def test_improvement_passes_with_a_note():
    after = [min(1.0, v + 0.2) for v in BEFORE]
    checks = compare(
        baseline(), {"hybrid": values(after)}, {"run": generation([True, False, None])}
    )
    assert overall(checks) == "pass"
    assert all("update the baseline" in c.note for c in checks if c.group == "retrieval")


def test_changed_questions_fail():
    checks = compare(baseline(), {"hybrid": values(BEFORE[:-1])}, {"run": generation([True])})
    assert overall(checks) == "fail"
    assert all(math.isnan(c.baseline) for c in checks)


def test_any_change_in_an_evaluation_value_is_drift():
    checks = compare(
        baseline(), {"hybrid": values(BEFORE)}, {"run": generation([False, False, None])}
    )
    drift = [c for c in checks if c.verdict == "fail"]
    assert [(c.metric, c.note) for c in drift] == [("correct", "1 questions changed, e.g. q0")]


def test_report_lists_failures_first():
    after = [max(0.0, v - 0.2) for v in BEFORE]
    checks = compare(
        baseline(), {"hybrid": values(after)}, {"run": generation([True, False, None])}
    )
    report = render(checks, baseline())
    assert report.startswith("## Regression gate: ❌ fail")
    rows = [line for line in report.splitlines() if line.startswith("| ")][2:]
    assert rows[0].startswith("| ❌") and rows[-1].startswith("| ✅")


def test_cached_embedder_returns_stored_vectors_only():
    vectors = np.eye(3, dtype=np.float32)
    embedder = CachedQueryEmbedder(["a", "b", "c"], vectors)
    assert embedder.dimension == 3
    np.testing.assert_array_equal(embedder.embed_queries(["c", "a"]), vectors[[2, 0]])
    with pytest.raises(KeyError, match="No stored vector"):
        embedder.embed_queries(["unknown"])
