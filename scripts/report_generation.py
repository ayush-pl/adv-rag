"""Write the generation report for a split: answer quality, comparisons and oracle diagnostics.

Run after scripts/judge_answers.py. Read alongside reports/judge_calibration.md, which says how
far the judge's scores can be trusted.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from ragbench.dataset.models import Question, read_questions
from ragbench.evaluation.generation_report import (
    audit_candidates,
    compare_runs,
    diagnose,
    join,
    summarize_run,
)
from ragbench.evaluation.judge import JUDGE_PROMPT_VERSION, AnswerEvaluation, citation_scores
from ragbench.generation.prompts import PROMPT_VERSION
from ragbench.generation.rag_pipeline import GeneratedAnswer

DATA_DIR = Path("data")


def load(split: str, stem: str, judge_name: str, current: dict[str, Question]):
    """Answers and evaluations, for questions still in the dataset. Withdrawn ones are left out.

    Citation scores are recomputed from the current gold labels, because labels can gain
    evidence after judging. They need no judge, so nothing is judged again.
    """
    generation = DATA_DIR / "runs" / "generation" / split / f"{stem}.jsonl"
    judgements = (
        DATA_DIR
        / "runs"
        / "judgements"
        / split
        / f"{stem}__{JUDGE_PROMPT_VERSION}__{judge_name}.jsonl"
    )
    with generation.open(encoding="utf-8") as lines:
        answers = [GeneratedAnswer.model_validate(json.loads(line)) for line in lines]
    with judgements.open(encoding="utf-8") as lines:
        evaluations = [AnswerEvaluation.model_validate(json.loads(line)) for line in lines]
    answers = [answer for answer in answers if answer.question_id in current]
    by_id = {answer.question_id: answer for answer in answers}
    for evaluation in evaluations:
        if evaluation.question_id in by_id:
            precision, recall = citation_scores(
                current[evaluation.question_id], by_id[evaluation.question_id]
            )
            evaluation.citation_precision, evaluation.citation_recall = precision, recall
    return join(answers, evaluations)


def pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def num(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="development", choices=["development", "calibration", "test"]
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    spec = yaml.safe_load(Path("configs/generation.yaml").read_text(encoding="utf-8"))
    judge_name = spec["judge"].replace(":", "-")
    runs_dir = DATA_DIR / "runs" / "generation" / args.split
    dataset = DATA_DIR / "datasets" / "v1" / f"{args.split}.jsonl"
    current = {question.question_id: question for question in read_questions(dataset)}
    model_names = {m: m.replace(":", "-").replace("/", "-") for m in spec["models"]}
    retrieval_names = {
        name: next(runs_dir.glob(f"{name}-*__*__retrieved__{PROMPT_VERSION}.jsonl")).name.split(
            "__"
        )[0]
        for name in spec["retrieval_configs"]
    }

    runs = {}
    for model, model_name in model_names.items():
        for name, run_name in retrieval_names.items():
            runs[(model, name)] = load(
                args.split,
                f"{run_name}__{model_name}__retrieved__{PROMPT_VERSION}",
                judge_name,
                current,
            )
        for path in sorted(runs_dir.glob(f"*__{model_name}__oracle__{PROMPT_VERSION}.jsonl")):
            runs[(model, f"oracle {path.name.split('__')[0]}")] = load(
                args.split, path.stem, judge_name, current
            )

    any_run = next(iter(runs.values()))
    lines = [
        f"# Generation results on the {args.split} split",
        "",
        f"- Judge `{spec['judge']}` with prompt `{JUDGE_PROMPT_VERSION}`; generation prompt "
        f"`{PROMPT_VERSION}`, temperature {spec['settings']['temperature']}, "
        f"top {spec['top_n']} passages.",
        "- **Read with reports/judge_calibration.md**, which measures how well the judge agrees "
        "with human scores. Every score here is the judge's.",
        "- On held-out human scores the judge was harsher than the human and almost never called "
        "a wrong answer correct, so **correct rates are conservative**. Comparisons between runs "
        "share that bias, so they are the more reliable reading.",
        "- The judge's faithfulness scores did **not** agree with human scores: it tends to mark "
        "answers that miss the question as unsupported. Treat the faithfulness and "
        "low-faithfulness columns as indicative only.",
        "- *Correct* means a correctness score of 4 or 5. Abstaining on, or failing to answer, an "
        "answerable question counts as incorrect. *Low faithfulness* means a score of 1 or 2, "
        "among answers given.",
        f"- {any_run and sum(s.evaluation.answerable for s in any_run)} answerable and "
        f"{sum(not s.evaluation.answerable for s in any_run)} unanswerable questions. Oracle runs "
        "give the gold passages and cover answerable questions only.",
        "- Latency is steady-state generation time per answer, excluding model loading.",
        "",
        "## All runs",
        "",
        "| Model | Context | Correct [95% CI] | Correctness | Faithfulness | Low faithfulness | "
        "Relevance | False abstain | Refused unanswerable | Failed output | Cite precision | "
        "p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    summaries = {}
    for (model, context), scored in runs.items():
        s = summaries[(model, context)] = summarize_run(scored)
        lines.append(
            f"| {model} | {context} | {s.correct_rate:.0%} "
            f"[{s.correct_low:.0%}, {s.correct_high:.0%}] | {s.mean_correctness:.2f} | "
            f"{num(s.mean_faithfulness)} | {pct(s.hallucination_rate)} | "
            f"{num(s.mean_relevance)} | {pct(s.false_abstention_rate)} | {pct(s.refusal_rate)} | "
            f"{s.failed_outputs} | {pct(s.citation_precision)} | {s.latency_p50_ms:.0f} |"
        )

    lines += [
        "",
        "## Paired comparisons of correct answers",
        "",
        "Same model, same questions. McNemar's exact test on correct or not, with a paired "
        "bootstrap interval for the difference in correct rate.",
        "",
        "| Model | Baseline | Treatment | Δ correct [95% CI] | Only baseline right | "
        "Only treatment right | p |",
        "|---|---|---|---|---|---|---|",
    ]
    configs = spec["retrieval_configs"]
    comparisons = [(configs[0], configs[1]), (configs[1], configs[2]), (configs[0], configs[2])]
    for model in model_names:
        for baseline, treatment in comparisons:
            c = compare_runs(runs[(model, baseline)], runs[(model, treatment)])
            lines.append(
                f"| {model} | {baseline} | {treatment} | {c.correct_difference:+.0%} "
                f"[{c.correct_low:+.0%}, {c.correct_high:+.0%}] | {c.baseline_only} | "
                f"{c.treatment_only} | {c.mcnemar_p:.3f} |"
            )
    models = list(model_names)
    for name in configs:
        c = compare_runs(runs[(models[0], name)], runs[(models[1], name)])
        lines.append(
            f"| {models[1]} vs {models[0]} | {name} | same | {c.correct_difference:+.0%} "
            f"[{c.correct_low:+.0%}, {c.correct_high:+.0%}] | {c.baseline_only} | "
            f"{c.treatment_only} | {c.mcnemar_p:.3f} |"
        )

    lines += [
        "",
        "## Oracle-context diagnosis",
        "",
        "Each retrieved run against the same model given the gold passages, at the same "
        "chunking. *Retrieval failures* were right with gold passages but wrong with retrieved "
        "ones. *Generation failures* were wrong even with gold passages.",
        "",
        "| Model | Retrieval | Questions | Correct, retrieved | Correct, oracle | Both right | "
        "Retrieval failures | Generation failures | Right only when retrieved |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model in model_names:
        for name in configs:
            chunking = retrieval_names[name].split("-")[0]
            oracle_key = next(
                k for k in runs if k[0] == model and k[1].startswith(f"oracle {chunking}")
            )
            d = diagnose(runs[(model, name)], runs[oracle_key])
            lines.append(
                f"| {model} | {name} | {d.questions} | {d.retrieved_correct_rate:.0%} | "
                f"{d.oracle_correct_rate:.0%} | {d.both_correct} | {d.retrieval_failures} | "
                f"{d.generation_failures} | {d.oracle_only_wrong} |"
            )

    oracle_runs = [
        scored for (model, context), scored in runs.items() if context.startswith("oracle")
    ]
    flagged = audit_candidates(oracle_runs)
    lines += [
        "",
        "## Reference answers to check",
        "",
        f"{len(flagged)} questions were answered badly (correctness 2 or lower) by every oracle "
        "run, despite having the gold passages. The reference answer or the question is the likely "
        "problem, so these need a human check before the results are final.",
        "",
        ", ".join(f"`{q}`" for q in flagged) or "None.",
    ]
    output = Path("reports") / f"generation_{args.split}.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Report written to {output}. {len(flagged)} reference answers flagged for checking.")


if __name__ == "__main__":
    main()
