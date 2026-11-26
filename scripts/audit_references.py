"""Audit reference answers that look wrong.

  export  write the questions to check into data/datasets/v1/review_sheet_references*.xlsx
  import  apply your decisions to the review history, and clear judgements of fixed questions

A question is checked when a human scorer wrote "reference wrong" about it, or when every
oracle run, which had the gold passages, scored 2 or lower on it. After importing, rebuild the
dataset with scripts/finalize_dataset.py, then re-judge with scripts/judge_answers.py.

The test split is audited on its own sheet with --split test, after its missed-evidence review,
so that development decisions are not made while looking at test answers.
"""

import argparse
import json
import sys
from pathlib import Path

from ragbench.dataset.models import Question, read_questions
from ragbench.dataset.reference_audit import (
    AuditCase,
    audit_records,
    export_audit_sheet,
    read_audit_sheet,
)
from ragbench.dataset.review import append_decision
from ragbench.evaluation.calibration import read_items, read_scores
from ragbench.evaluation.generation_report import Scored, audit_candidates, join
from ragbench.evaluation.judge import JUDGE_PROMPT_VERSION, AnswerEvaluation
from ragbench.generation.rag_pipeline import GeneratedAnswer

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
CALIBRATION_DIR = DATASET_DIR / "calibration"
# Each audit: the splits it covers, and its sheet.
AUDITS = {
    "development": (["development", "calibration"], DATASET_DIR / "review_sheet_references.xlsx"),
    "test": (["test"], DATASET_DIR / "review_sheet_references_test.xlsx"),
}
SCORED_SETS = [
    ("items.jsonl", "human_scores.jsonl"),
    ("validation_items.jsonl", "validation_human_scores.jsonl"),
]


def read_jsonl(path: Path, model):
    with path.open(encoding="utf-8") as lines:
        return [model.model_validate(json.loads(line)) for line in lines if line.strip()]


def oracle_runs(split: str) -> dict[str, list[Scored]]:
    runs = {}
    for path in sorted((DATA_DIR / "runs" / "generation" / split).glob("*__oracle__*.jsonl")):
        judged = next(
            (DATA_DIR / "runs" / "judgements" / split).glob(
                f"{path.stem}__{JUDGE_PROMPT_VERSION}__*.jsonl"
            )
        )
        runs[path.stem] = join(
            read_jsonl(path, GeneratedAnswer), read_jsonl(judged, AnswerEvaluation)
        )
    return runs


def flagged_by_scorers() -> dict[str, str]:
    """Question ID to the scorer's note, for answers whose reference was called wrong."""
    flagged = {}
    for items_name, scores_name in SCORED_SETS:
        scores_path = CALIBRATION_DIR / scores_name
        if not scores_path.exists():
            continue
        items = {i.sample_id: i for i in read_items(CALIBRATION_DIR / items_name)}
        for score in read_scores(scores_path):
            if "reference wrong" in score.note.lower():
                flagged[items[score.sample_id].question_id] = score.note
    return flagged


def export(splits: list[str], sheet: Path) -> None:
    questions: dict[str, Question] = {}
    split_of: dict[str, str] = {}
    for split in splits:
        for question in read_questions(DATASET_DIR / f"{split}.jsonl"):
            questions[question.question_id] = question
            split_of[question.question_id] = split

    reasons: dict[str, list[str]] = {}
    for question_id, note in flagged_by_scorers().items():
        if question_id in questions:
            reasons.setdefault(question_id, []).append(f"Flagged while scoring: {note}")
    runs_by_split = {split: oracle_runs(split) for split in splits}
    for runs in runs_by_split.values():
        for question_id in audit_candidates(list(runs.values())):
            reasons.setdefault(question_id, []).append(
                f"All {len(runs)} runs that were given the gold passages got it wrong."
            )

    cases = []
    for question_id in sorted(reasons):
        runs = runs_by_split[split_of[question_id]]
        answers = [
            (
                f"{name.split('__')[1]} ({name.split('__')[0]} passages)",
                scored.answer.answer or "(no answer)",
            )
            for name, run in runs.items()
            # One chunking is enough to see what the evidence leads to.
            if name.startswith("r512")
            for scored in run
            if scored.answer.question_id == question_id
        ]
        cases.append(AuditCase(questions[question_id], "\n".join(reasons[question_id]), answers))
    try:
        export_audit_sheet(cases, sheet)
    except PermissionError as exc:
        raise SystemExit(f"Cannot write {sheet}. Close it in Excel first.") from exc
    print(f"Wrote {len(cases)} questions to check into {sheet}")


def drop_judgements(question_ids: set[str]) -> int:
    """Remove the current judge version's judgements of these questions, so the next judging
    run redoes them. Older versions are history, kept for the calibration comparison."""
    removed = 0
    pattern = f"*/*__{JUDGE_PROMPT_VERSION}__*.jsonl"
    for path in (DATA_DIR / "runs" / "judgements").glob(pattern):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if json.loads(line)["question_id"] not in question_ids]
        if len(kept) != len(lines):
            removed += len(lines) - len(kept)
            path.write_text("".join(kept), encoding="utf-8", newline="\n")
    return removed


def import_sheet(splits: list[str], sheet: Path) -> None:
    questions = {
        q.question_id: q for split in splits for q in read_questions(DATASET_DIR / f"{split}.jsonl")
    }
    result = read_audit_sheet(sheet, set(questions))
    if result.problems:
        print("Nothing saved. Fix these rows and import again:")
        print("\n".join(f"  {problem}" for problem in result.problems))
        raise SystemExit(1)
    if result.undecided:
        raise SystemExit(f"{result.undecided} rows have no decision yet. Nothing saved.")

    for record in audit_records(result.decisions, questions):
        append_decision(DATASET_DIR / "review_decisions.jsonl", record)
    fixed = {d.question_id for d in result.decisions if d.action == "fix"}
    removed = drop_judgements(fixed)
    for action in ("keep", "fix", "withdraw"):
        chosen = [d.question_id for d in result.decisions if d.action == action]
        print(f"{action}: {len(chosen)} {' '.join(chosen)}")
    print(f"Cleared {removed} judgements of fixed questions, to be judged again.")
    print("Next: scripts/finalize_dataset.py, then scripts/judge_answers.py.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument(
        "--split",
        default="development",
        choices=sorted(AUDITS),
        help="development covers the development and calibration splits",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    splits, sheet = AUDITS[args.split]
    export(splits, sheet) if args.action == "export" else import_sheet(splits, sheet)


if __name__ == "__main__":
    main()
