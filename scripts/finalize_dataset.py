"""Validate accepted questions, split them, and write the versioned benchmark dataset."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from ragbench.dataset.models import TARGET_COUNTS, write_questions
from ragbench.dataset.review import accepted_questions
from ragbench.dataset.splits import assign_splits
from ragbench.dataset.validation import validate_questions, validate_splits
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"


def main() -> None:
    questions = accepted_questions(DATASET_DIR / "review_decisions.jsonl")
    if not questions:
        raise SystemExit("No accepted questions yet. Run scripts/review_questions.py first.")

    documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}
    splits = assign_splits(questions)
    problems = validate_questions(questions, documents) + validate_splits(splits)
    if problems:
        print(f"{len(problems)} problems found, dataset not written:")
        print("\n".join(f"  {problem}" for problem in problems))
        sys.exit(1)

    counts = Counter(question.question_type for question in questions)
    print(f"{len(questions)} valid questions:")
    for name, target in TARGET_COUNTS.items():
        status = "" if counts[name] >= target else f"  (short of {target})"
        print(f"  {name}: {counts[name]}{status}")

    digest = hashlib.sha256()
    for name in sorted(splits):
        write_questions(DATASET_DIR / f"{name}.jsonl", splits[name])
        print(f"  split {name}: {len(splits[name])}")
        for question in splits[name]:
            digest.update(f"{name}\0{question.model_dump_json()}\n".encode())

    corpus_lock = json.loads(
        (DATA_DIR / "manifests" / "corpus_v1.lock.json").read_text(encoding="utf-8")
    )
    lock = {
        "dataset_version": "v1",
        "dataset_hash": "sha256-" + digest.hexdigest(),
        "corpus_hash": corpus_lock["corpus_hash"],
        "questions": len(questions),
        "splits": {name: len(values) for name, values in sorted(splits.items())},
        "types": dict(sorted(counts.items())),
    }
    lock_path = DATASET_DIR / "dataset.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Dataset hash: {lock['dataset_hash']}")


if __name__ == "__main__":
    main()
