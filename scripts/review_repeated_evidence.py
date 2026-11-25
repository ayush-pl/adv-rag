"""Review evidence that also appears word for word on other pages.

export  write a sheet listing each copy, to mark y (also answers it) or n
import  add the copies marked y to the accepted questions' evidence
"""

import argparse
import sys
from pathlib import Path

from ragbench.dataset.repeated_evidence import (
    export_repeated_evidence,
    find_repeated_evidence,
    read_repeated_evidence,
)
from ragbench.dataset.review import accepted_questions, append_decision
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
DECISIONS_PATH = DATASET_DIR / "review_decisions.jsonl"
SHEET_PATH = DATASET_DIR / "review_sheet_repeated_evidence.xlsx"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("--sheet", type=Path, default=SHEET_PATH)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}

    if args.action == "export":
        repeats = find_repeated_evidence(accepted_questions(DECISIONS_PATH), documents)
        try:
            export_repeated_evidence(repeats, documents, args.sheet)
        except PermissionError as exc:
            raise SystemExit(f"Cannot write {args.sheet}. Close it in Excel first.") from exc
        print(f"Wrote {len(repeats)} copies to review into {args.sheet}")
        return

    result = read_repeated_evidence(args.sheet, DECISIONS_PATH, documents)
    for decision in result.decisions:
        append_decision(DECISIONS_PATH, decision)
    print(
        f"Added {result.added} pages to {len(result.decisions)} questions. "
        f"{result.declined} marked n, {result.undecided} left blank."
    )
    if result.problems:
        print(f"\n{len(result.problems)} rows were NOT applied:")
        print("\n".join(f"  {problem}" for problem in result.problems))


if __name__ == "__main__":
    main()
