"""Export all undecided drafts to a spreadsheet for review in one go."""

import argparse
from pathlib import Path

from ragbench.dataset.models import read_questions
from ragbench.dataset.review_sheet import export_review_sheet
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATASET_DIR / "review_sheet.xlsx")
    parser.add_argument(
        "--ids", nargs="+", help="Export exactly these questions, even if already decided"
    )
    args = parser.parse_args()

    drafts = read_questions(DATASET_DIR / "drafts.jsonl")
    if not drafts:
        raise SystemExit("No drafts found. Run scripts/draft_questions.py first.")
    documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}
    try:
        rows = export_review_sheet(
            drafts,
            DATASET_DIR / "review_decisions.jsonl",
            documents,
            args.output,
            set(args.ids) if args.ids else None,
        )
    except PermissionError as exc:
        raise SystemExit(f"Cannot write {args.output}. Close it in Excel and try again.") from exc
    print(f"Wrote {rows} questions to review into {args.output}")


if __name__ == "__main__":
    main()
