"""Uncertainty and paired comparisons, so small differences aren't reported as improvements."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    sample_size: int


@dataclass(frozen=True)
class PairedComparison:
    """Difference `treatment - baseline` on identical questions."""

    difference: ConfidenceInterval

    @property
    def is_conclusive(self) -> bool:
        """Whether the interval excludes zero."""
        return self.difference.low > 0 or self.difference.high < 0


@dataclass(frozen=True)
class McNemarResult:
    baseline_only: int
    treatment_only: int
    p_value: float


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for a mean. Seeded, so reports are reproducible."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("Cannot compute a confidence interval with no values")

    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return ConfidenceInterval(float(data.mean()), float(low), float(high), int(data.size))


def paired_bootstrap(
    baseline: Sequence[float],
    treatment: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> PairedComparison:
    """Bootstrap the per-question differences, which keeps each question's pairing intact."""
    if len(baseline) != len(treatment):
        raise ValueError(f"Paired samples differ in size: {len(baseline)}, {len(treatment)}")
    differences = np.asarray(treatment, dtype=float) - np.asarray(baseline, dtype=float)
    return PairedComparison(bootstrap_ci(differences, resamples, confidence, seed))


def mcnemar_exact(baseline: Sequence[bool], treatment: Sequence[bool]) -> McNemarResult:
    """Exact McNemar test for paired binary outcomes, such as correct or not per question."""
    if len(baseline) != len(treatment):
        raise ValueError(f"Paired samples differ in size: {len(baseline)}, {len(treatment)}")
    baseline_only = sum(1 for b, t in zip(baseline, treatment, strict=True) if b and not t)
    treatment_only = sum(1 for b, t in zip(baseline, treatment, strict=True) if t and not b)
    discordant = baseline_only + treatment_only
    p_value = binomtest(baseline_only, discordant, 0.5).pvalue if discordant else 1.0
    return McNemarResult(baseline_only, treatment_only, float(p_value))
