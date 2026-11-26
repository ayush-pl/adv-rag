"""Compare the current code's results on the fixture with a committed baseline.

Retrieval checks have three outcomes:
- pass: the mean did not drop by more than the tolerance;
- fail: it dropped by more than the tolerance, and the paired bootstrap interval of the
  per-question differences lies entirely below zero;
- inconclusive: it dropped by more than the tolerance, but the interval includes zero, so
  some questions got better and others worse. A human should look.

Evaluation checks are exact. The fixture's answers and judgements are recorded, so the same
code must reproduce the same per-question values. Any change is metric drift: a deliberate
change to a metric's definition needs a new baseline, which makes it visible in review.
"""

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from ragbench.evaluation.generation_report import join
from ragbench.evaluation.judge import abstention_correct, citation_scores
from ragbench.evaluation.significance import paired_bootstrap
from ragbench.experiments.runner import run_retrieval
from ragbench.regression.fixture import Fixture

RETRIEVAL_METRICS = ["recall_at_5", "mrr", "ndcg_at_10"]
GENERATION_FIELDS = ["correct", "abstention_correct", "citation_precision", "citation_recall"]
# The largest drop in a mean that still passes. Retrieval on stored vectors is deterministic,
# so this only has to absorb floating-point differences between platforms, which can reorder
# near-tied passages. It is set well above the difference measured between Windows and Linux
# (recorded in the baseline), and well below any change worth reporting.
DEFAULT_TOLERANCE = 0.01

Verdict = Literal["pass", "fail", "inconclusive"]
Values = dict[str, dict[str, float | bool | None]]


class Baseline(BaseModel):
    dataset_hash: str
    tolerance: float
    # Largest difference in any checked mean between platforms when the baseline was made.
    measured_platform_noise: float | None = None
    # Configuration name to question ID to metric values.
    retrieval: dict[str, Values]
    # Generation run name to question ID to evaluation values.
    generation: dict[str, Values]


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    metric: str
    baseline: float
    current: float
    low: float | None
    high: float | None
    verdict: Verdict
    note: str = ""

    @property
    def delta(self) -> float:
        return self.current - self.baseline


def measure(fixture: Fixture) -> tuple[dict[str, Values], dict[str, Values]]:
    """Run every gate configuration and recompute every recorded evaluation."""
    retrieval = {}
    for config in fixture.manifest.configs:
        results = run_retrieval(config, fixture.questions, fixture.retriever(config))
        retrieval[config.name] = {
            r.question_id: {m: r.metrics[m] for m in RETRIEVAL_METRICS} for r in results
        }

    questions = {q.question_id: q for q in fixture.questions}
    generation = {}
    for run in fixture.generation:
        values: Values = {}
        for scored in join(run.answers, run.evaluations):
            question = questions[scored.answer.question_id]
            precision, recall = citation_scores(question, scored.answer)
            values[question.question_id] = {
                "correct": scored.correct if question.answerable else None,
                "abstention_correct": abstention_correct(question, scored.answer),
                "citation_precision": precision,
                "citation_recall": recall,
            }
        generation[run.name] = values
    return retrieval, generation


def compare(
    baseline: Baseline, retrieval: dict[str, Values], generation: dict[str, Values]
) -> list[Check]:
    checks = []
    for name, expected in baseline.retrieval.items():
        found = retrieval.get(name)
        if found is None or set(found) != set(expected):
            checks.append(_missing("retrieval", name, "questions or configuration changed"))
            continue
        ids = sorted(expected)
        for metric in RETRIEVAL_METRICS:
            before = [float(expected[i][metric]) for i in ids]
            after = [float(found[i][metric]) for i in ids]
            checks.append(_retrieval_check(name, metric, before, after, baseline.tolerance))

    for name, expected in baseline.generation.items():
        found = generation.get(name)
        if found is None or set(found) != set(expected):
            checks.append(_missing("generation", name, "recorded answers changed"))
            continue
        for field in GENERATION_FIELDS:
            changed = sorted(i for i in expected if not _same(expected[i][field], found[i][field]))
            before, after = _mean(expected, field), _mean(found, field)
            checks.append(
                Check(
                    "generation",
                    name,
                    field,
                    before,
                    after,
                    None,
                    None,
                    "fail" if changed else "pass",
                    f"{len(changed)} questions changed, e.g. {', '.join(changed[:3])}"
                    if changed
                    else "",
                )
            )
    return checks


def overall(checks: list[Check]) -> Verdict:
    verdicts = {check.verdict for check in checks}
    if "fail" in verdicts:
        return "fail"
    return "inconclusive" if "inconclusive" in verdicts else "pass"


def render(checks: list[Check], baseline: Baseline) -> str:
    result = overall(checks)
    icon = {"pass": "✅", "fail": "❌", "inconclusive": "⚠️"}
    lines = [
        f"## Regression gate: {icon[result]} {result}",
        "",
        f"Tolerance: a mean may drop by up to {baseline.tolerance}. Retrieval fails only when "
        "the drop is larger and its 95% paired bootstrap interval excludes zero; evaluation "
        "values must match exactly.",
        "",
        "| | Group | Run | Metric | Baseline | Current | Δ [95% CI] | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = {"fail": 0, "inconclusive": 1, "pass": 2}
    for check in sorted(checks, key=lambda c: order[c.verdict]):
        interval = "" if check.low is None else f" [{check.low:+.3f}, {check.high:+.3f}]"
        lines.append(
            f"| {icon[check.verdict]} | {check.group} | {check.name} | {check.metric} | "
            f"{_fmt(check.baseline)} | {_fmt(check.current)} | {check.delta:+.3f}{interval} | "
            f"{check.note} |"
        )
    return "\n".join(lines) + "\n"


def _retrieval_check(
    name: str, metric: str, before: list[float], after: list[float], tolerance: float
) -> Check:
    difference = paired_bootstrap(before, after).difference
    delta = difference.estimate
    note = ""
    if delta >= -tolerance:
        verdict: Verdict = "pass"
        if delta > tolerance and difference.low > 0:
            note = "improved: update the baseline to keep the gain"
    elif difference.high < 0:
        verdict = "fail"
    else:
        verdict = "inconclusive"
        note = "dropped, but some questions improved: review the change"
    return Check(
        "retrieval",
        name,
        metric,
        sum(before) / len(before),
        sum(after) / len(after),
        difference.low,
        difference.high,
        verdict,
        note,
    )


def _missing(group: str, name: str, why: str) -> Check:
    return Check(group, name, "-", math.nan, math.nan, None, None, "fail", why)


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), abs_tol=1e-9)


def _mean(values: Values, field: str) -> float:
    numbers = [float(v[field]) for v in values.values() if v[field] is not None]
    return sum(numbers) / len(numbers) if numbers else math.nan


def _fmt(value: float) -> str:
    return "–" if math.isnan(value) else f"{value:.3f}"
