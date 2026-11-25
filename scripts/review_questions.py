"""Verify drafted questions one at a time. Stop whenever you like, and rerun to continue."""

import sys
from pathlib import Path

from ragbench.dataset.models import read_questions
from ragbench.dataset.review import ReviewSession
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    drafts = read_questions(DATASET_DIR / "drafts.jsonl")
    if not drafts:
        raise SystemExit("No drafts found. Run scripts/draft_questions.py first.")
    documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}
    ReviewSession(drafts, DATASET_DIR / "review_decisions.jsonl", documents).run()


if __name__ == "__main__":
    main()
