"""Run the retrieval experiments R1-R7 on a dataset split and write a comparison report.

Finished runs are reused, so rerunning only computes what is missing. Use --force to redo all.
"""

import argparse
import sys
from pathlib import Path

import yaml

from ragbench.dataset.models import read_questions
from ragbench.experiments.resources import Resources
from ragbench.experiments.runner import load_run, rescore, run_retrieval, save_run, summarize
from ragbench.experiments.sweep import SweepSpec, expand_sweep, render_report
from ragbench.experiments.tracker import capture_environment, read_lock

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="development", choices=["development", "test"])
    parser.add_argument("--sweep", type=Path, default=Path("configs/retrieval_sweep.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    spec = SweepSpec.model_validate(yaml.safe_load(args.sweep.read_text(encoding="utf-8")))
    configs = expand_sweep(spec)
    questions = read_questions(DATASET_DIR / f"{args.split}.jsonl")
    dataset_hash = read_lock(DATASET_DIR / "dataset.lock.json")["dataset_hash"]
    resources = Resources(DATA_DIR)
    environment = capture_environment()
    runs_root = DATA_DIR / "runs" / "retrieval" / args.split

    runs = {}
    for number, config in enumerate(configs, start=1):
        directory = runs_root / config.run_name
        if (directory / "run.json").exists() and not args.force:
            record, results = load_run(directory)
            if record.dataset_hash == dataset_hash:
                runs[config.name] = (record, results)
                print(f"[{number}/{len(configs)}] {config.name}: reused")
                continue
            # Labels changed but the questions didn't: score the saved rankings again. The
            # rankings, latency and environment are still those of the original retrieval.
            rescored = rescore(results, questions)
            if rescored is not None:
                record = summarize(
                    config,
                    rescored,
                    questions,
                    args.split,
                    dataset_hash,
                    resources.corpus_hash,
                    record.environment,
                ).model_copy(
                    update={
                        "latency_p50_ms": record.latency_p50_ms,
                        "latency_p95_ms": record.latency_p95_ms,
                    }
                )
                save_run(directory, record, rescored)
                runs[config.name] = (record, rescored)
                print(f"[{number}/{len(configs)}] {config.name}: rescored with updated labels")
                continue

        results = run_retrieval(config, questions, resources.retriever(config))
        record = summarize(
            config, results, questions, args.split, dataset_hash, resources.corpus_hash, environment
        )
        save_run(directory, record, results)
        runs[config.name] = (record, results)
        metrics = record.metrics
        print(
            f"[{number}/{len(configs)}] {config.name}: recall@5 {metrics['recall_at_5']:.3f}, "
            f"nDCG@10 {metrics['ndcg_at_10']:.3f}, p50 {record.latency_p50_ms:.0f} ms",
            flush=True,
        )

    report = render_report(spec, runs, f"Retrieval sweep on the {args.split} split")
    output = Path("reports") / f"retrieval_sweep_{args.split}.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(report, encoding="utf-8", newline="\n")
    print(f"\nReport written to {output}")


if __name__ == "__main__":
    main()
