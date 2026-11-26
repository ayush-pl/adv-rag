"""Draw the README charts from the saved test runs, in light and dark versions.

Writes reports/charts/<name>-light.svg and <name>-dark.svg. Every number is recomputed from
data/runs, so the charts follow the reports.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from ragbench.dataset.models import read_questions  # noqa: E402
from ragbench.evaluation.generation_report import join, summarize_run  # noqa: E402
from ragbench.evaluation.judge import JUDGE_PROMPT_VERSION, AnswerEvaluation  # noqa: E402
from ragbench.evaluation.significance import bootstrap_ci  # noqa: E402
from ragbench.experiments.runner import load_run  # noqa: E402
from ragbench.experiments.sweep import SweepSpec, expand_sweep, paired_difference  # noqa: E402
from ragbench.generation.rag_pipeline import GeneratedAnswer  # noqa: E402

DATA_DIR = Path("data")
OUTPUT_DIR = Path("reports/charts")
SPLIT = "test"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str]


# The reference data-viz palette: categorical slots 1 (blue) and 2 (orange), validated for
# colour-vision deficiency and contrast against each surface.
THEMES = [
    Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
          ("#2a78d6", "#eb6834")),
    Theme("dark", "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
          ("#3987e5", "#d95926")),
]  # fmt: skip


def configs(sweep: str) -> dict:
    spec = SweepSpec.model_validate(yaml.safe_load(Path(sweep).read_text(encoding="utf-8")))
    return {c.name: c for c in expand_sweep(spec)}


def run(split: str, config) -> list:
    return load_run(DATA_DIR / "runs" / "retrieval" / split / config.run_name)[1]


def retrieval_data() -> list[tuple[str, float, float, float]]:
    """nDCG@10 with its bootstrap interval for the headline configurations."""
    found = configs("configs/retrieval_sweep.yaml") | configs("configs/retrieval_improvement.yaml")
    rows = [
        ("Dense (BGE-small)", "r512-dense-bge"),
        ("BM25", "r512-bm25"),
        ("Hybrid (RRF)", "r512-hybrid-bge-k60"),
        ("Hybrid + rerank", "r512-hybrid-bge-k60-rerank"),
        ("Hybrid + dedup + rerank", "r512-hybrid-bge-k60-dedup-rerank"),
    ]
    out = []
    for label, name in rows:
        interval = bootstrap_ci([r.metrics["ndcg_at_10"] for r in run(SPLIT, found[name])])
        out.append((label, interval.estimate, interval.low, interval.high))
    return out


def dedup_data() -> list[tuple[str, tuple, tuple]]:
    """Paired nDCG@10 gain from duplicate removal: on the selection splits and on test."""
    found = configs("configs/retrieval_improvement.yaml")
    rows = [
        ("Dense", "r512-dense-bge"),
        ("Hybrid, 256-token chunks", "r256-hybrid-bge-k60"),
        ("Hybrid", "r512-hybrid-bge-k60"),
        ("Hybrid + rerank, 256-token", "r256-hybrid-bge-k60-rerank"),
        ("Hybrid + rerank", "r512-hybrid-bge-k60-rerank"),
    ]
    out = []
    for label, name in rows:
        # Sweep names put "dedup" before "rerank": r512-hybrid-bge-k60-dedup-rerank.
        if name.endswith("-rerank"):
            treatment = name.removesuffix("-rerank") + "-dedup-rerank"
        else:
            treatment = name + "-dedup"
        estimates = []
        for splits in (["development", "calibration"], [SPLIT]):
            before = [r for s in splits for r in run(s, found[name])]
            after = [r for s in splits for r in run(s, found[treatment])]
            d = paired_difference(before, after, "ndcg_at_10")
            estimates.append((d.estimate, d.low, d.high))
        out.append((label, *estimates))
    return out


def answer_data() -> dict[str, list[tuple[str, float, float, float]]]:
    """Correct rate with its interval for each generator and context."""
    spec = yaml.safe_load(Path("configs/generation.yaml").read_text(encoding="utf-8"))
    judge = spec["judge"].replace(":", "-")
    current = {q.question_id for q in read_questions(DATA_DIR / "datasets/v1" / f"{SPLIT}.jsonl")}
    runs_dir = DATA_DIR / "runs" / "generation" / SPLIT
    contexts = [
        ("Dense", "r512-dense-bge-*__{m}__retrieved"),
        ("Hybrid", "r512-hybrid-bge-k60-[0-9a-f]*__{m}__retrieved"),
        ("Hybrid + rerank", "r256-hybrid-bge-k60-rerank-*__{m}__retrieved"),
        ("Gold passages", "r512-overlap50__{m}__oracle"),
    ]
    out = {}
    for model in spec["models"]:
        model_name = model.replace(":", "-")
        rows = []
        for label, pattern in contexts:
            path = next(runs_dir.glob(pattern.format(m=model_name) + "__*.jsonl"))
            judged = (
                DATA_DIR / "runs" / "judgements" / SPLIT
                / f"{path.stem}__{JUDGE_PROMPT_VERSION}__{judge}.jsonl"
            )  # fmt: skip
            answers = [GeneratedAnswer.model_validate(json.loads(x)) for x in _lines(path)]
            evaluations = [AnswerEvaluation.model_validate(json.loads(x)) for x in _lines(judged)]
            answers = [a for a in answers if a.question_id in current]
            summary = summarize_run(join(answers, evaluations))
            rows.append((label, summary.correct_rate, summary.correct_low, summary.correct_high))
        out[model] = rows
    return out


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _axes(theme: Theme, width: float, height: float):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "path",
            "axes.edgecolor": theme.axis,
            "axes.labelcolor": theme.secondary,
            "xtick.color": theme.muted,
            "ytick.color": theme.secondary,
            "text.color": theme.ink,
        }
    )
    figure, axes = plt.subplots(figsize=(width, height), dpi=100)
    figure.patch.set_facecolor(theme.surface)
    axes.set_facecolor(theme.surface)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(theme.axis)
    axes.tick_params(axis="y", length=0, labelsize=10, colors=theme.secondary)
    axes.tick_params(axis="x", length=0, labelsize=9, colors=theme.muted)
    axes.grid(axis="x", color=theme.grid, linewidth=1, linestyle="-")
    axes.set_axisbelow(True)
    return figure, axes


def _title(figure, theme: Theme, title: str, subtitle: str) -> None:
    figure.text(0.012, 0.965, title, ha="left", va="top", fontsize=13, weight="bold",
                color=theme.ink)  # fmt: skip
    figure.text(0.012, 0.885, subtitle, ha="left", va="top", fontsize=9.5,
                color=theme.secondary)  # fmt: skip


def retrieval_chart(theme: Theme, rows) -> plt.Figure:
    figure, axes = _axes(theme, 7.6, 3.5)
    _title(
        figure,
        theme,
        "Retrieval quality on held-out questions",
        "nDCG@10 on 101 test questions, 512-token chunks. Lines are 95% bootstrap intervals.",
    )
    for index, (_, value, low, high) in enumerate(reversed(rows)):
        color = theme.series[0]
        axes.barh(index, value, height=0.5, color=color, linewidth=0, zorder=2)
        axes.plot([low, high], [index, index], color=theme.ink, linewidth=1.2, alpha=0.55,
                  zorder=3, solid_capstyle="round")  # fmt: skip
        axes.text(high + 0.012, index, f"{value:.3f}", va="center", ha="left", fontsize=10,
                  color=theme.ink)  # fmt: skip
    axes.set_yticks(range(len(rows)), [label for label, *_ in reversed(rows)])
    axes.set_xlim(0, 1.0)
    axes.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    figure.subplots_adjust(left=0.25, right=0.97, top=0.76, bottom=0.1)
    return figure


def answers_chart(theme: Theme, data) -> plt.Figure:
    figure, axes = _axes(theme, 7.6, 4.0)
    _title(
        figure,
        theme,
        "Retrieval limits answer accuracy",
        "Share of 101 answerable test questions answered correctly. Gold passages show the "
        "ceiling.\nHybrid + rerank uses 256-token chunks, the others 512. Lines are 95% "
        "intervals.",
    )
    models = list(data)
    labels = [label for label, *_ in data[models[0]]]
    height = 0.34
    for m_index, model in enumerate(models):
        color = theme.series[m_index]
        for c_index, (_, value, low, high) in enumerate(reversed(data[model])):
            y = c_index + (0.5 - m_index) * (height + 0.04)
            axes.barh(y, value, height=height, color=color, linewidth=0, zorder=2,
                      label=model if c_index == 0 else None)  # fmt: skip
            axes.plot([low, high], [y, y], color=theme.ink, linewidth=1.2, alpha=0.55,
                      zorder=3, solid_capstyle="round")  # fmt: skip
            axes.text(high + 0.012, y, f"{value:.0%}", va="center", ha="left", fontsize=9.5,
                      color=theme.ink)  # fmt: skip
    axes.set_yticks(range(len(labels)), list(reversed(labels)))
    axes.set_xlim(0, 1.0)
    axes.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0%", "20%", "40%", "60%", "80%", "100%"])
    legend = axes.legend(loc="lower right", frameon=False, fontsize=9.5, handlelength=1.2,
                         handleheight=1.0, labelcolor=theme.secondary,
                         bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)  # fmt: skip
    for handle in legend.legend_handles:
        handle.set_height(8)
    figure.subplots_adjust(left=0.19, right=0.97, top=0.72, bottom=0.09)
    return figure


def dedup_chart(theme: Theme, rows) -> plt.Figure:
    figure, axes = _axes(theme, 7.6, 3.6)
    _title(
        figure,
        theme,
        "Duplicate removal: chosen on 70 questions, confirmed on 101",
        "Paired gain in nDCG@10 from dropping repeated passages, with 95% bootstrap "
        "intervals.\nThe test gain is real but smaller than the selection estimate.",
    )
    axes.axvline(0, color=theme.axis, linewidth=1, zorder=1)
    names = ("Selection: dev + calibration", "Test (held out)")
    for index, (_, selection, test) in enumerate(reversed(rows)):
        for s_index, (estimate, low, high) in enumerate((selection, test)):
            y = index + (0.17 if s_index == 0 else -0.17)
            color = theme.series[1 - s_index]
            axes.plot([low, high], [y, y], color=color, linewidth=2, zorder=2,
                      solid_capstyle="round")  # fmt: skip
            axes.scatter([estimate], [y], s=64, color=color, edgecolors=theme.surface,
                         linewidths=2, zorder=3,
                         label=names[s_index] if index == 0 else None)  # fmt: skip
    axes.set_yticks(range(len(rows)), [label for label, *_ in reversed(rows)])
    axes.set_xlim(-0.02, 0.12)
    axes.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.10],
                    ["0", "+0.02", "+0.04", "+0.06", "+0.08", "+0.10"])  # fmt: skip
    axes.set_ylim(-0.6, len(rows) - 0.4)
    axes.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=theme.secondary,
                bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2, handletextpad=0.3,
                markerscale=0.9)  # fmt: skip
    figure.subplots_adjust(left=0.27, right=0.97, top=0.72, bottom=0.1)
    return figure


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "retrieval": (retrieval_chart, retrieval_data()),
        "answers": (answers_chart, answer_data()),
        "dedup": (dedup_chart, dedup_data()),
    }
    for name, (draw, values) in data.items():
        for theme in THEMES:
            figure = draw(theme, values)
            path = OUTPUT_DIR / f"{name}-{theme.name}.svg"
            figure.savefig(path, format="svg", facecolor=theme.surface, metadata={"Date": None})
            plt.close(figure)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
