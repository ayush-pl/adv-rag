"""Save the top passages per question for each generation configuration. Needs the GPU.

wsl.exe -e bash scripts/wsl.sh python scripts/retrieve_contexts.py
"""

import argparse
import sys
from pathlib import Path

import yaml

from ragbench.dataset.models import read_questions
from ragbench.experiments.resources import Resources
from ragbench.experiments.sweep import SweepSpec, expand_sweep
from ragbench.generation.contexts import retrieve_contexts, write_contexts

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
SPLITS = ["development", "calibration", "test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    generation = yaml.safe_load(Path("configs/generation.yaml").read_text(encoding="utf-8"))
    sweep = Path("configs/retrieval_sweep.yaml").read_text(encoding="utf-8")
    configs = {c.name: c for c in expand_sweep(SweepSpec.model_validate(yaml.safe_load(sweep)))}
    resources = Resources(DATA_DIR)

    for name in generation["retrieval_configs"]:
        config = configs[name]
        retriever = resources.retriever(config)
        for split in args.splits:
            questions = read_questions(DATASET_DIR / f"{split}.jsonl")
            contexts = retrieve_contexts(questions, retriever, generation["top_n"])
            path = DATA_DIR / "runs" / "contexts" / split / f"{config.run_name}.jsonl"
            write_contexts(path, contexts)
            print(f"{name} / {split}: {len(contexts)} contexts", flush=True)


if __name__ == "__main__":
    main()
