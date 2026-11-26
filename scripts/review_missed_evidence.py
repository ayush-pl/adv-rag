"""Review relevant passages that retrieval found but the gold labels missed.

export  pool the top results of several configurations into a sheet. Retrieving for the
        unanswerable questions needs the GPU environment, so in WSL run:
        wsl.exe -e bash scripts/wsl.sh python scripts/review_missed_evidence.py export
import  add passages marked y as evidence, withdraw unanswerable questions that turn out
        to be answered, and record every judgment

export --compare BASELINE TREATMENT pools only the passages in one configuration's top
--depth results but not the other's, to judge before comparing a new configuration with an
old one on labels that were pooled without it. Rows go to a separate sheet, and the main pool
settings in pooling.json are left as they were.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from ragbench.chunking.models import parse_chunk_id
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
from ragbench.retrieval.dedup import duplicate_key

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
DECISIONS_PATH = DATASET_DIR / "review_decisions.jsonl"
JUDGMENTS_PATH = DATASET_DIR / "pool_judgments.jsonl"
# How each split was pooled, for reports to state.
POOLING_PATH = DATASET_DIR / "pooling.json"

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
# Development was pooled at depth 3. Test was pooled at depth 2, which on development would have
# found 36 of the 39 relevant passages from two-thirds of the rows.
DEPTH = 3


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("--split", default="development")
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--depth", type=int, default=DEPTH, help="results pooled per config")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "TREATMENT"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    suffix = f"_{args.compare[1]}" if args.compare else ""
    sheet = args.sheet or DATASET_DIR / f"review_sheet_missed_evidence_{args.split}{suffix}.xlsx"
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
    if args.compare:
        _export_comparison(args, questions, runs_root, documents, judgments, sheet)
        return
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
        rankings[UNANSWERABLE_CONFIG].update(_retrieve_unanswerable(unanswerable, args.depth))

    candidates = build_pool(answerable, rankings, args.depth, judgments)
    candidates += build_pool(
        unanswerable, {UNANSWERABLE_CONFIG: rankings[UNANSWERABLE_CONFIG]}, args.depth, judgments
    )
    try:
        export_pool(candidates, documents, sheet)
    except PermissionError as exc:
        raise SystemExit(f"Cannot write {sheet}. Close it in Excel first.") from exc
    pooling = json.loads(POOLING_PATH.read_text(encoding="utf-8")) if POOLING_PATH.exists() else {}
    pooling[args.split] = {
        "depth": args.depth,
        "configs": POOL_CONFIGS,
        "unanswerable_config": UNANSWERABLE_CONFIG,
        "passages": len(candidates),
    }
    POOLING_PATH.write_text(json.dumps(pooling, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(candidates)} passages for {len(questions)} questions into {sheet}")


def _export_comparison(args, questions, runs_root, documents, judgments, sheet: Path) -> None:
    """Passages in one configuration's top results but not the other's.

    Passages in both lists count the same for both configurations, so only the differences can
    change the comparison. Text identical to a passage in the other list (a PostgreSQL 15 and 16
    twin) is not a difference.
    """
    rankings = {}
    for name in args.compare:
        directories = [d for d in runs_root.iterdir() if d.name.rsplit("-", 1)[0] == name]
        if len(directories) != 1:
            raise SystemExit(f"Expected one {args.split} run of {name}, found {len(directories)}")
        _, results = load_run(directories[0])
        rankings[name] = {r.question_id: r.chunk_ids[: args.depth] for r in results}

    def text(chunk_id: str) -> str:
        document_id, start, end = parse_chunk_id(chunk_id)
        return duplicate_key(documents[document_id].content[start:end])

    differing: dict[str, dict[str, list[str]]] = {name: {} for name in args.compare}
    for name, other in (args.compare, args.compare[::-1]):
        for question_id, chunk_ids in rankings[name].items():
            other_texts = {text(c) for c in rankings[other].get(question_id, [])}
            differing[name][question_id] = [c for c in chunk_ids if text(c) not in other_texts]

    answerable = [q for q in questions if q.answerable]
    candidates = build_pool(answerable, differing, args.depth, judgments)
    try:
        export_pool(candidates, documents, sheet)
    except PermissionError as exc:
        raise SystemExit(f"Cannot write {sheet}. Close it in Excel first.") from exc
    pooling = json.loads(POOLING_PATH.read_text(encoding="utf-8")) if POOLING_PATH.exists() else {}
    pooling.setdefault("comparisons", {})[f"{args.split}: {' vs '.join(args.compare)}"] = {
        "depth": args.depth,
        "passages": len(candidates),
        "questions": len({c.question.question_id for c in candidates}),
    }
    POOLING_PATH.write_text(json.dumps(pooling, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"Wrote {len(candidates)} passages for "
        f"{len({c.question.question_id for c in candidates})} questions into {sheet}"
    )


def _retrieve_unanswerable(questions, depth: int) -> dict[str, list[str]]:
    from ragbench.experiments.resources import Resources
    from ragbench.experiments.sweep import SweepSpec, expand_sweep

    sweep = Path("configs/retrieval_sweep.yaml").read_text(encoding="utf-8")
    spec = SweepSpec.model_validate(yaml.safe_load(sweep))
    config = next(c for c in expand_sweep(spec) if c.name == UNANSWERABLE_CONFIG)
    retriever = Resources(DATA_DIR).retriever(config)
    return {
        q.question_id: [r.chunk.chunk_id for r in retriever.retrieve(q.question, depth)]
        for q in questions
    }


if __name__ == "__main__":
    main()
