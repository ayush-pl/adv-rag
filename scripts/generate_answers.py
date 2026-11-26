"""Generate grounded answers with each model, from retrieved and from oracle (gold) contexts.

Runs on Windows next to Ollama, from the contexts that scripts/retrieve_contexts.py saved.
Each answer is saved as soon as it is generated, so the script can be stopped and rerun.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from ragbench.chunking.models import Chunk
from ragbench.dataset.models import Question, read_questions
from ragbench.experiments.resources import Resources
from ragbench.experiments.sweep import SweepSpec, expand_sweep
from ragbench.generation.contexts import oracle_chunk_ids, passages_for, read_contexts
from ragbench.generation.ollama_client import OllamaClient, OllamaError
from ragbench.generation.prompts import PROMPT_VERSION
from ragbench.generation.rag_pipeline import (
    GeneratedAnswer,
    GenerationSettings,
    failed_answer,
    generate_answer,
    is_model_failure,
)
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
SPLITS = ["development", "calibration", "test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--models", nargs="+", help="Default: all models in the config")
    parser.add_argument("--limit", type=int, help="At most this many questions per run")
    parser.add_argument("--no-oracle", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    spec = yaml.safe_load(Path("configs/generation.yaml").read_text(encoding="utf-8"))
    settings = GenerationSettings(**spec["settings"])
    sweep = Path("configs/retrieval_sweep.yaml").read_text(encoding="utf-8")
    configs = {c.name: c for c in expand_sweep(SweepSpec.model_validate(yaml.safe_load(sweep)))}
    documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}
    resources = Resources(DATA_DIR)
    client = OllamaClient()

    for model in args.models or spec["models"]:
        for name in spec["retrieval_configs"]:
            config = configs[name]
            by_document = _chunks_by_document(resources.chunks(config.chunking))
            for split in args.splits:
                questions = read_questions(DATASET_DIR / f"{split}.jsonl")[: args.limit]
                contexts = read_contexts(
                    DATA_DIR / "runs" / "contexts" / split / f"{config.run_name}.jsonl"
                )
                runs = [("retrieved", lambda q, found=contexts: found[q.question_id])]
                if not args.no_oracle:
                    runs.append(
                        ("oracle", lambda q, chunks=by_document: oracle_chunk_ids(q, chunks))
                    )
                for mode, chunk_ids_for in runs:
                    targets = [q for q in questions if q.answerable or mode == "retrieved"]
                    # Oracle context depends only on the chunking, not on retrieval, so
                    # configurations sharing a chunking share one oracle run.
                    run_name = config.run_name if mode == "retrieved" else _oracle_name(config)
                    path = _output_path(split, run_name, model, mode)
                    _generate(
                        client, model, mode, targets, chunk_ids_for, documents, settings, path
                    )


def _generate(
    client, model, mode, questions: list[Question], chunk_ids_for, documents, settings, path
):
    done = _done_ids(path)
    pending = [q for q in questions if q.question_id not in done]
    print(f"{path.parent.name}/{path.name}: {len(done)} done, {len(pending)} to go", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    for question in pending:
        passages = passages_for(chunk_ids_for(question), documents)
        try:
            answer = generate_answer(
                client, model, question.question_id, question.question, passages, mode, settings
            )
        except OllamaError as exc:
            if not is_model_failure(str(exc)):
                raise SystemExit(f"Ollama failed, stopping. Rerun to continue: {exc}") from exc
            # Retrying cannot help at temperature 0 with a fixed seed, so record the failure.
            answer = failed_answer(question.question_id, model, passages, mode, str(exc))
            print(f"  FAILED {question.question_id}: {exc}", flush=True)
        with path.open("a", encoding="utf-8", newline="\n") as out:
            out.write(answer.model_dump_json() + "\n")
        if answer.possibly_truncated:
            print(f"  WARNING {question.question_id}: prompt may have been truncated", flush=True)


def _oracle_name(config) -> str:
    chunking = config.chunking
    return f"{chunking.strategy[0]}{chunking.chunk_size}-overlap{chunking.overlap}"


def _output_path(split: str, run_name: str, model: str, mode: str) -> Path:
    model_name = model.replace(":", "-").replace("/", "-")
    file_name = f"{run_name}__{model_name}__{mode}__{PROMPT_VERSION}.jsonl"
    return DATA_DIR / "runs" / "generation" / split / file_name


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as lines:
        return {GeneratedAnswer.model_validate(json.loads(line)).question_id for line in lines}


def _chunks_by_document(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)
    return grouped


if __name__ == "__main__":
    main()
