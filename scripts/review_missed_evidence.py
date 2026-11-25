"""Review relevant passages that retrieval found but the gold labels missed.

export  pool the top results of several configurations into a sheet. Retrieving for the
        unanswerable questions needs the GPU environment, so in WSL run:
        wsl.exe -e bash scripts/wsl.sh python scripts/review_missed_evidence.py export
import  add passages marked y as evidence, withdraw unanswerable questions that turn out
        to be answered, and record every judgment
"""

import argparse
import sys
from pathlib import Path

import yaml

from ragbench.dataset.models import read_questions
from ragbench.dataset.pooling import (
    append_judgments,
    build_pool,
    export_pool,
    read_judgments,
    read_pool,
)
from ragbench.dataset.review import append_decision
from ragbench.experiments.runner import load_run
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
DECISIONS_PATH = DATASET_DIR / "review_decisions.jsonl"
JUDGMENTS_PATH = DATASET_DIR / "pool_judgments.jsonl"

# One configuration per family, so the pool covers what different methods find.
POOL_CONFIGS = [
    "r512-dense-bge",
    "r512-dense-minilm",
    "r512-bm25",
    "f256-bm25",
    "r512-hybrid-bge-k60",
    "r256-hybrid-bge-k60-rerank",
    "r512-hybrid-bge-k60-rerank",
]
# The best development configuration, used to search for answers to unanswerable questions.
UNANSWERABLE_CONFIG = "r256-hybrid-bge-k60-rerank"
DEPTH = 3


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("--split", default="development")
    parser.add_argument("--sheet", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sheet = args.sheet or DATASET_DIR / f"review_sheet_missed_evidence_{args.split}.xlsx"
    documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}
    judgments = read_judgments(JUDGMENTS_PATH)

    if args.action == "import":
        result = read_pool(sheet, DECISIONS_PATH, documents, judgments)
        for decision in result.decisions:
            append_decision(DECISIONS_PATH, decision)
        append_judgments(JUDGMENTS_PATH, result.judgments)
        print(
            f"Recorded {len(result.judgments)} judgments. Added {result.added_evidence} "
            f"evidence passages, withdrew {len(result.withdrawn)} unanswerable questions, "
            f"{result.undecided} rows left blank."
        )
        if result.withdrawn:
            print("Withdrawn: " + ", ".join(result.withdrawn))
        if result.problems:
            print(f"\n{len(result.problems)} rows were NOT applied:")
            print("\n".join(f"  {problem}" for problem in result.problems))
        print("\nRun scripts/finalize_dataset.py to rebuild the splits with these changes.")
        return

    questions = read_questions(DATASET_DIR / f"{args.split}.jsonl")
    runs_root = DATA_DIR / "runs" / "retrieval" / args.split
    rankings = {}
    for directory in sorted(runs_root.iterdir()):
        record, results = load_run(directory)
        if record.config.name in POOL_CONFIGS:
            rankings[record.config.name] = {r.question_id: r.chunk_ids for r in results}
    missing = set(POOL_CONFIGS) - set(rankings)
    if missing:
        raise SystemExit(f"Missing runs: {', '.join(sorted(missing))}. Run the sweep first.")

    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    if unanswerable:
        rankings[UNANSWERABLE_CONFIG].update(_retrieve_unanswerable(unanswerable))

    candidates = build_pool(answerable, rankings, DEPTH, judgments)
    candidates += build_pool(
        unanswerable, {UNANSWERABLE_CONFIG: rankings[UNANSWERABLE_CONFIG]}, DEPTH, judgments
    )
    try:
        export_pool(candidates, documents, sheet)
    except PermissionError as exc:
        raise SystemExit(f"Cannot write {sheet}. Close it in Excel first.") from exc
    print(f"Wrote {len(candidates)} passages for {len(questions)} questions into {sheet}")


def _retrieve_unanswerable(questions) -> dict[str, list[str]]:
    from ragbench.experiments.resources import Resources
    from ragbench.experiments.sweep import SweepSpec, expand_sweep

    sweep = Path("configs/retrieval_sweep.yaml").read_text(encoding="utf-8")
    spec = SweepSpec.model_validate(yaml.safe_load(sweep))
    config = next(c for c in expand_sweep(spec) if c.name == UNANSWERABLE_CONFIG)
    retriever = Resources(DATA_DIR).retriever(config)
    return {
        q.question_id: [r.chunk.chunk_id for r in retriever.retrieve(q.question, DEPTH)]
        for q in questions
    }


if __name__ == "__main__":
    main()
