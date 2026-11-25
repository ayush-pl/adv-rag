"""Expand a sweep specification into configurations, and report paired comparisons."""

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from ragbench.evaluation.significance import bootstrap_ci, paired_bootstrap
from ragbench.experiments.config import ChunkingChoice, RetrievalConfig
from ragbench.experiments.runner import QuestionResult, RunRecord


class RunGroup(BaseModel):
    chunking: list[str]
    method: str
    embedder: str | None = None
    rrf_k: list[int] = [60]
    rerank: bool = False


class Comparison(BaseModel):
    experiment: str
    baseline: str
    treatment: str


class SweepSpec(BaseModel):
    chunking: dict[str, ChunkingChoice]
    embedders: dict[str, str]
    runs: list[RunGroup]
    comparisons: list[Comparison]
    top_k_configs: list[str] = []


def expand_sweep(spec: SweepSpec) -> list[RetrievalConfig]:
    """One config per combination. Names read like `r512-hybrid-bge-k60-rerank`."""
    configs: dict[str, RetrievalConfig] = {}
    for group in spec.runs:
        for chunk_name in group.chunking:
            for rrf_k in group.rrf_k if group.method == "hybrid" else [60]:
                parts = [chunk_name, group.method]
                if group.embedder:
                    parts.append(group.embedder)
                if group.method == "hybrid":
                    parts.append(f"k{rrf_k}")
                if group.rerank:
                    parts.append("rerank")
                name = "-".join(parts)
                if name in configs:
                    raise ValueError(f"The sweep defines {name} twice")
                configs[name] = RetrievalConfig(
                    name=name,
                    chunking=spec.chunking[chunk_name],
                    method=group.method,
                    embedder=spec.embedders[group.embedder] if group.embedder else None,
                    rrf_k=rrf_k,
                    rerank=group.rerank,
                )

    names = set(configs)
    for comparison in spec.comparisons:
        for name in (comparison.baseline, comparison.treatment):
            if name not in names:
                raise ValueError(f"Comparison {comparison.experiment} names unknown run {name}")
    for name in spec.top_k_configs:
        if name not in names:
            raise ValueError(f"top_k_configs names unknown run {name}")
    return list(configs.values())


@dataclass(frozen=True)
class MetricDifference:
    estimate: float
    low: float
    high: float

    @property
    def verdict(self) -> str:
        if self.low > 0:
            return "better"
        if self.high < 0:
            return "worse"
        return "inconclusive"


def paired_difference(
    baseline: list[QuestionResult], treatment: list[QuestionResult], metric: str
) -> MetricDifference:
    """Treatment minus baseline on the questions both runs scored."""
    by_id = {result.question_id: result for result in treatment}
    pairs = [(b, by_id[b.question_id]) for b in baseline if b.question_id in by_id]
    if len(pairs) != len(baseline) or len(pairs) != len(treatment):
        raise ValueError("Runs scored different questions, so they cannot be paired")
    comparison = paired_bootstrap(
        [b.metrics[metric] for b, _ in pairs], [t.metrics[metric] for _, t in pairs]
    )
    interval = comparison.difference
    return MetricDifference(interval.estimate, interval.low, interval.high)


HEADLINE = [
    ("recall_at_5", "Recall@5"),
    ("full_coverage_at_5", "All evidence@5"),
    ("recall_at_20", "Recall@20"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "nDCG@10"),
    ("map", "MAP"),
]
COMPARED = [("recall_at_5", "Recall@5"), ("mrr", "MRR"), ("ndcg_at_10", "nDCG@10")]
TOP_K_CUTOFFS = (1, 3, 5, 10, 20)


def render_report(
    spec: SweepSpec,
    runs: dict[str, tuple[RunRecord, list[QuestionResult]]],
    title: str,
) -> str:
    any_record = next(iter(runs.values()))[0]
    lines = [
        f"# {title}",
        "",
        f"- Split: **{any_record.split}**, {any_record.questions_scored} answerable questions "
        f"scored ({any_record.questions_skipped} unanswerable skipped).",
        f"- Dataset `{any_record.dataset_hash[:19]}`, corpus `{any_record.corpus_hash[:19]}`, "
        f"code `{(any_record.environment.git_commit or 'unknown')[:10]}`"
        + (" with uncommitted changes" if any_record.environment.git_dirty else "")
        + ".",
        f"- Hardware: {any_record.environment.cuda_device or 'CPU only'}.",
        "- Brackets are 95% bootstrap intervals. Latency is warm, per query, and excludes "
        "model loading.",
        "- Recall counts evidence found, not chunks, so it is comparable across chunking.",
        "",
        "## All configurations",
        "",
        "| Configuration | " + " | ".join(label for _, label in HEADLINE) + " | p50 ms |",
        "|---|" + "---|" * (len(HEADLINE) + 1),
    ]
    order = sorted(runs, key=lambda name: -runs[name][0].metrics["ndcg_at_10"])
    for name in order:
        record, results = runs[name]
        cells = []
        for metric, _ in HEADLINE:
            if metric == "recall_at_5":
                interval = bootstrap_ci([r.metrics[metric] for r in results])
                cells.append(f"{interval.estimate:.3f} [{interval.low:.2f}, {interval.high:.2f}]")
            else:
                cells.append(f"{record.metrics[metric]:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {record.latency_p50_ms:.0f} |")

    lines += [
        "",
        "## Paired comparisons",
        "",
        "Differences are treatment minus baseline on identical questions. A result is only "
        "called better or worse when its interval excludes zero.",
        "",
        "| Exp | Baseline | Treatment | "
        + " | ".join(f"Δ {label}" for _, label in COMPARED)
        + " | Verdict (nDCG@10) |",
        "|---|---|---|" + "---|" * (len(COMPARED) + 1),
    ]
    for comparison in spec.comparisons:
        baseline = runs[comparison.baseline][1]
        treatment = runs[comparison.treatment][1]
        differences = [paired_difference(baseline, treatment, m) for m, _ in COMPARED]
        cells = [f"{d.estimate:+.3f} [{d.low:+.2f}, {d.high:+.2f}]" for d in differences]
        lines.append(
            f"| {comparison.experiment} | {comparison.baseline} | {comparison.treatment} | "
            + " | ".join(cells)
            + f" | {differences[-1].verdict} |"
        )

    if spec.top_k_configs:
        lines += [
            "",
            "## R7: Recall by cutoff",
            "",
            "| Configuration | " + " | ".join(f"@{k}" for k in TOP_K_CUTOFFS) + " |",
            "|---|" + "---|" * len(TOP_K_CUTOFFS),
        ]
        for name in spec.top_k_configs:
            record = runs[name][0]
            values = [f"{record.metrics[f'recall_at_{k}']:.3f}" for k in TOP_K_CUTOFFS]
            lines.append(f"| {name} | " + " | ".join(values) + " |")

    best = order[0]
    lines += [
        "",
        f"## By question type, for {best}",
        "",
        "Small groups, so these are exploratory only.",
        "",
        "| Type | Questions | Recall@5 | nDCG@10 |",
        "|---|---|---|---|",
    ]
    by_type: dict[str, list[QuestionResult]] = {}
    for result in runs[best][1]:
        by_type.setdefault(result.question_type, []).append(result)
    for question_type in sorted(by_type):
        group = by_type[question_type]
        recall = np.mean([r.metrics["recall_at_5"] for r in group])
        ndcg = np.mean([r.metrics["ndcg_at_10"] for r in group])
        lines.append(f"| {question_type} | {len(group)} | {recall:.3f} | {ndcg:.3f} |")
    return "\n".join(lines) + "\n"
