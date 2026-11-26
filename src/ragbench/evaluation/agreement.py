"""Agreement between human and judge scores on an ordinal 1-5 scale.

Several statistics, because each hides something: exact agreement ignores near misses, mean
absolute difference ignores systematic bias, and kappa corrects for chance agreement.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ragbench.evaluation.significance import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES

SCALE = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class AgreementReport:
    pairs: int
    exact: float
    within_one: float
    mean_absolute_difference: float
    # Mean of judge minus human. Positive means the judge scores more generously.
    mean_bias: float
    weighted_kappa: float
    kappa_low: float
    kappa_high: float


def quadratic_weighted_kappa(
    human: Sequence[int], judge: Sequence[int], scale: Sequence[int] = SCALE
) -> float:
    """Cohen's kappa with quadratic weights, so a 1-point miss costs less than a 3-point one."""
    if len(human) != len(judge) or not human:
        raise ValueError("Need equal, non-empty score lists")
    index = {value: i for i, value in enumerate(scale)}
    size = len(scale)
    observed = np.zeros((size, size))
    for h, j in zip(human, judge, strict=True):
        observed[index[h], index[j]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = np.array([[(a - b) ** 2 for b in range(size)] for a in range(size)]) / (size - 1) ** 2
    disagreement_expected = (weights * expected).sum()
    if disagreement_expected == 0:
        # Both raters used one identical score throughout: agreement is perfect but undefined
        # by the formula. Report 1.0 only if they also matched.
        return 1.0 if (weights * observed).sum() == 0 else 0.0
    return float(1 - (weights * observed).sum() / disagreement_expected)


def agreement(
    human: Sequence[int],
    judge: Sequence[int],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> AgreementReport:
    h = np.asarray(human)
    j = np.asarray(judge)
    if h.shape != j.shape or h.size == 0:
        raise ValueError("Need equal, non-empty score lists")

    rng = np.random.default_rng(seed)
    kappas = []
    for _ in range(resamples):
        sample = rng.integers(0, h.size, size=h.size)
        kappas.append(quadratic_weighted_kappa(h[sample].tolist(), j[sample].tolist()))
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(kappas, [tail, 100.0 - tail])
    return AgreementReport(
        pairs=int(h.size),
        exact=float(np.mean(h == j)),
        within_one=float(np.mean(np.abs(h - j) <= 1)),
        mean_absolute_difference=float(np.mean(np.abs(h - j))),
        mean_bias=float(np.mean(j - h)),
        weighted_kappa=quadratic_weighted_kappa(h.tolist(), j.tolist()),
        kappa_low=float(low),
        kappa_high=float(high),
    )
