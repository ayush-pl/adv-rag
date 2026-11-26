"""Calibrate the LLM judge against your own scores.

  export  sample answers into a sheet for blind scoring
  import  save your scores from the filled-in sheet
  report  compare your scores with every judge version and write reports/judge_calibration.md

Two scoring sets keep the measurement honest. The calibration set was used to find flaws and
design the judge's rubric, so agreement on it is optimistic. The validation set, from another
split, is never used to change the judge, so its agreement is the one to trust.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from ragbench.dataset.models import read_questions
from ragbench.evaluation.agreement import agreement
from ragbench.evaluation.calibration import (
    CalibrationItem,
    HumanScore,
    export_sheet,
    read_items,
    read_scores,
    read_sheet,
    sample_items,
    write_items,
    write_scores,
)
from ragbench.evaluation.judge import DIMENSIONS, JUDGE_PROMPT_VERSION
from ragbench.generation.contexts import passages_for
from ragbench.generation.rag_pipeline import GeneratedAnswer
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
CALIBRATION_DIR = DATASET_DIR / "calibration"
JUDGE_VERSIONS = ["judge-v1", "judge-v2"]


@dataclass(frozen=True)
class ScoringSet:
    split: str
    size: int
    prefix: str
    items: Path
    scores: Path
    sheet: Path
    role: str


SETS = {
    "calibration": ScoringSet(
        split="calibration",
        size=50,
        prefix="s",
        items=CALIBRATION_DIR / "items.jsonl",
        scores=CALIBRATION_DIR / "human_scores.jsonl",
        sheet=DATASET_DIR / "review_sheet_calibration.xlsx",
        role="used to design judge-v2, so its agreement is optimistic",
    ),
    "validation": ScoringSet(
        split="development",
        size=25,
        prefix="v",
        items=CALIBRATION_DIR / "validation_items.jsonl",
        scores=CALIBRATION_DIR / "validation_human_scores.jsonl",
        sheet=DATASET_DIR / "review_sheet_validation.xlsx",
        role="never used to change the judge",
    ),
}


def load_answers(split: str) -> dict[str, list[GeneratedAnswer]]:
    answers = {}
    for path in sorted((DATA_DIR / "runs" / "generation" / split).glob("*.jsonl")):
        with path.open(encoding="utf-8") as lines:
            answers[path.stem] = [GeneratedAnswer.model_validate(json.loads(row)) for row in lines]
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["export", "import", "report"])
    parser.add_argument("--set", default="calibration", choices=sorted(SETS))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.action == "report":
        write_report()
        return

    scoring = SETS[args.set]
    questions = {q.question_id: q for q in read_questions(DATASET_DIR / f"{scoring.split}.jsonl")}
    if args.action == "export":
        answers = load_answers(scoring.split)
        if scoring.items.exists():
            items = read_items(scoring.items)
            print(f"Reusing the existing sample of {len(items)} answers.")
        else:
            items = sample_items(answers, questions, scoring.size, prefix=scoring.prefix)
            write_items(scoring.items, items)
        documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}
        by_key = {(run, a.question_id): a for run, list_ in answers.items() for a in list_}
        chosen = {(i.run, i.question_id): by_key[(i.run, i.question_id)] for i in items}
        passages = {key: passages_for(a.context_chunk_ids, documents) for key, a in chosen.items()}
        try:
            export_sheet(items, chosen, passages, questions, scoring.sheet)
        except PermissionError as exc:
            raise SystemExit(f"Cannot write {scoring.sheet}. Close it in Excel first.") from exc
        print(f"Wrote {len(items)} answers to score into {scoring.sheet}")
        return

    result = read_sheet(scoring.sheet, read_items(scoring.items))
    if result.problems:
        print("Not saved. Fix these rows and import again:")
        print("\n".join(f"  {problem}" for problem in result.problems))
        raise SystemExit(1)
    write_scores(scoring.scores, result.scores)
    print(f"Saved {len(result.scores)} scored answers, {result.incomplete} not scored yet.")


def judge_scores(
    scoring: ScoringSet, items: list[CalibrationItem], version: str, judge_name: str
) -> dict[str, dict] | None:
    """The judge's judgement for each sample, or None if this version hasn't been run."""
    found = {}
    for item in items:
        path = (
            DATA_DIR
            / "runs"
            / "judgements"
            / scoring.split
            / f"{item.run}__{version}__{judge_name}.jsonl"
        )
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                record = json.loads(line)
                if record["question_id"] == item.question_id:
                    found[item.sample_id] = record["judgement"]
    return found


def write_report() -> None:
    spec = yaml.safe_load(Path("configs/generation.yaml").read_text(encoding="utf-8"))
    judge_model = spec["judge"]
    judge_name = judge_model.replace(":", "-")
    lines = [
        "# Judge calibration",
        "",
        f"- Judge model `{judge_model}`. The current rubric is `{JUDGE_PROMPT_VERSION}`.",
        "- A human scored each answer blind, without the judge's scores or the system's identity.",
        "- Kappa is quadratic-weighted, with a 95% bootstrap interval. Bias is the judge's mean "
        "score minus the human's: negative means the judge is harsher.",
        "- *Correct agreement* is agreement on correct (4 or 5) versus not, which is how "
        "correctness is used in the generation report.",
        "",
    ]
    for name, scoring in SETS.items():
        lines.append(f"- **{name}** set: {scoring.split} split, {scoring.role}.")
    lines += [
        "",
        "| Rubric | Set | Dimension | Pairs | Exact | Within 1 | Bias | Weighted kappa | "
        "Correct agreement |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    # Every version is compared on the same answers: those every version judged. An answer can
    # be missing from one version when its question's reference answer was later corrected.
    loaded = {}
    excluded: list[str] = []
    for name, scoring in SETS.items():
        if not scoring.scores.exists():
            continue
        items = read_items(scoring.items)
        human: dict[str, HumanScore] = {s.sample_id: s for s in read_scores(scoring.scores)}
        machines = {
            version: found
            for version in JUDGE_VERSIONS
            if (found := judge_scores(scoring, items, version, judge_name)) is not None
        }
        common = [s for s in human if all(m.get(s, {}).get("scores") for m in machines.values())]
        excluded += [f"{s} ({name})" for s in human if s not in common]
        loaded[name] = (human, machines, common)

    disagreements: list[tuple] = []
    for version in JUDGE_VERSIONS:
        for name, (human, machines, paired) in loaded.items():
            if version not in machines:
                continue
            machine = machines[version]
            for dimension in DIMENSIONS:
                h = [human[s].scores[dimension] for s in paired]
                j = [machine[s]["scores"][dimension]["score"] for s in paired]
                report = agreement(h, j)
                binary = ""
                if dimension == "correctness":
                    matches = sum((a >= 4) == (b >= 4) for a, b in zip(h, j, strict=True))
                    binary = f"{matches / len(h):.0%}"
                lines.append(
                    f"| {version} | {name} | {dimension} | {report.pairs} | {report.exact:.2f} | "
                    f"{report.within_one:.2f} | {report.mean_bias:+.2f} | "
                    f"{report.weighted_kappa:.2f} [{report.kappa_low:.2f}, {report.kappa_high:.2f}]"
                    f" | {binary} |"
                )
                if version == JUDGE_PROMPT_VERSION:
                    for sample_id, hs, js in zip(paired, h, j, strict=True):
                        if abs(hs - js) >= 2:
                            reason = machine[sample_id]["scores"][dimension]["reason"]
                            note = human[sample_id].note
                            disagreements.append((name, sample_id, dimension, hs, js, reason, note))

    lines += [
        "",
        f"## Largest disagreements for {JUDGE_PROMPT_VERSION}",
        "",
        "Differences of 2 or more points.",
        "",
        "| Set | Sample | Dimension | Human | Judge | Judge's reason | Human note |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, sample_id, dimension, hs, js, reason, note in sorted(
        disagreements, key=lambda d: (-abs(d[3] - d[4]), d[0], d[1])
    )[:30]:
        reason = reason.replace("|", "/")[:150]
        note = note.replace("|", "/")[:80]
        lines.append(f"| {name} | {sample_id} | {dimension} | {hs} | {js} | {reason} | {note} |")

    flagged = []
    for scoring in SETS.values():
        if scoring.scores.exists():
            items = {i.sample_id: i for i in read_items(scoring.items)}
            for score in read_scores(scoring.scores):
                if "reference wrong" in score.note.lower():
                    flagged.append(f"{score.sample_id} ({items[score.sample_id].question_id})")
    lines += [
        "",
        f"Reference answers flagged as wrong by the reviewer: {', '.join(flagged) or 'none'}.",
        "",
        "Left out of every row, because not every rubric version judged them (their reference "
        f"answer was corrected after scoring): {', '.join(excluded) or 'none'}.",
        "",
        "## Known limitations",
        "",
        "- One human rater, so human-to-human agreement is not measured.",
        "- Small samples, so intervals are wide and per-category agreement is not reported.",
        "- Most answers were good, so scores cluster at 5. That makes kappa unstable, so read it "
        "alongside exact agreement and bias.",
        "- Abstentions and unanswerable questions are scored deterministically, not by the "
        "judge, so this calibration does not cover them.",
    ]
    output = Path("reports") / "judge_calibration.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
