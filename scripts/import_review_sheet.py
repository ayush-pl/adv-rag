"""Save the decisions from a filled-in review spreadsheet. Safe to run more than once."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from ragbench.dataset.models import TARGET_COUNTS, read_questions
from ragbench.dataset.review import accepted_questions, append_decision
from ragbench.dataset.review_sheet import read_review_sheet
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
DECISIONS_PATH = DATASET_DIR / "review_decisions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=DATASET_DIR / "review_sheet.xlsx")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    drafts = read_questions(DATASET_DIR / "drafts.jsonl")
    documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}
    result = read_review_sheet(args.sheet, drafts, DECISIONS_PATH, documents)
    for decision in result.decisions:
        append_decision(DECISIONS_PATH, decision)

    made = Counter(decision.decision for decision in result.decisions)
    print(
        f"Saved {made['accept']} accepted and {made['reject']} discarded. "
        f"{result.unchanged} already saved, {result.undecided} still undecided."
    )
    if result.problems:
        count = len(result.problems)
        print(f"\n{count} rows were NOT saved. Fix them in the sheet and import again:")
        print("\n".join(f"  {problem}" for problem in result.problems))

    accepted = Counter(q.question_type for q in accepted_questions(DECISIONS_PATH))
    print("\nAccepted so far (accepted / target):")
    for name, target in TARGET_COUNTS.items():
        print(f"  {name}: {accepted[name]} / {target}")


if __name__ == "__main__":
    main()
