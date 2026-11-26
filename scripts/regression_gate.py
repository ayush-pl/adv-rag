"""Regression gate for CI: rerun retrieval and evaluation code on a small committed fixture.

  check     run the gate and compare with tests/regression/baseline.json. Exits 1 on fail.
            Needs no GPU, models, Ollama or corpus, so it runs in CI.
  build     rebuild the fixture from the full pipeline outputs. Needs the corpus, indexes,
            the embedding model and saved generation runs.
  baseline  record the current results as the new baseline. Do this only for a change you
            mean to keep, such as a better retriever or a corrected metric, and say why in
            the commit, because the baseline is what CI defends.

The fixture holds the development and calibration questions only, so the test split stays out
of anything tuned against it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from ragbench.dataset.models import read_questions
from ragbench.evaluation.gold import chunk_matches_span
from ragbench.evaluation.judge import JUDGE_PROMPT_VERSION, AnswerEvaluation
from ragbench.experiments.sweep import SweepSpec, expand_sweep
from ragbench.experiments.tracker import read_lock
from ragbench.generation.rag_pipeline import GeneratedAnswer
from ragbench.regression.fixture import (
    Fixture,
    FixtureManifest,
    GenerationRun,
    load_fixture,
    save_fixture,
)
from ragbench.regression.gate import (
    DEFAULT_TOLERANCE,
    Baseline,
    compare,
    measure,
    overall,
    render,
)

FIXTURE_DIR = Path("tests/regression/fixture")
BASELINE_PATH = Path("tests/regression/baseline.json")
DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
SPLITS = ["development", "calibration"]
GATE_CONFIGS = ["r512-dense-bge", "r512-bm25", "r512-hybrid-bge-k60"]
# Each gate configuration's top results for every question go into the fixture, with the gold
# passages. Twenty is the deepest cutoff any metric reads.
POOL_DEPTH = 20
# Recorded development runs: a retrieved run per model and an oracle run.
GENERATION_RUNS = [
    "r512-hybrid-bge-k60-83e17a5e__llama3.1-8b__retrieved__grounded-v1",
    "r256-hybrid-bge-k60-rerank-d354ef52__gemma3-4b__retrieved__grounded-v1",
    "r512-overlap50__llama3.1-8b__oracle__grounded-v1",
]
JUDGE_NAME = "qwen2.5-7b"
LICENSES = (
    "Chunk text is from the Kubernetes documentation (CC BY 4.0), the Docker documentation "
    "(Apache License 2.0) and the PostgreSQL documentation (PostgreSQL License). Each chunk "
    "keeps its source URL."
)


def gate_configs():
    sweep = yaml.safe_load(Path("configs/retrieval_sweep.yaml").read_text(encoding="utf-8"))
    configs = {c.name: c for c in expand_sweep(SweepSpec.model_validate(sweep))}
    return [configs[name] for name in GATE_CONFIGS]


def build() -> None:
    from ragbench.embeddings.index_store import index_directory, load_vectors
    from ragbench.embeddings.sentence_transformer import SentenceTransformerEmbedder
    from ragbench.experiments.resources import Resources

    configs = gate_configs()
    resources = Resources(DATA_DIR)
    questions = [q for split in SPLITS for q in read_questions(DATASET_DIR / f"{split}.jsonl")]
    chunking = configs[0].chunking
    chunks = resources.chunks(chunking)
    row_of = {chunk.chunk_id: row for row, chunk in enumerate(chunks)}

    pooled: set[str] = set()
    for config in configs:
        retriever = resources.retriever(config)
        for question in questions:
            if question.answerable:
                found = retriever.retrieve(question.question, POOL_DEPTH)
                pooled.update(result.chunk.chunk_id for result in found)
    spans = [span for q in questions for group in q.evidence for span in group.spans]
    pooled.update(c.chunk_id for c in chunks if any(chunk_matches_span(c, s) for s in spans))
    # Keep the corpus order, so the fixture is the same on every rebuild.
    rows = sorted(row_of[chunk_id] for chunk_id in pooled)

    model = configs[0].embedder
    embedder_config = resources.embedder_config(model)
    vectors_dir = index_directory(
        DATA_DIR / "indexes", resources.chunks_dir(chunking), embedder_config
    )
    document_vectors = load_vectors(vectors_dir)[rows]
    embedder = SentenceTransformerEmbedder(embedder_config)
    query_vectors = embedder.embed_queries([q.question for q in questions])

    current = {q.question_id for q in questions}
    generation = []
    for name in GENERATION_RUNS:
        answers_path = DATA_DIR / "runs" / "generation" / "development" / f"{name}.jsonl"
        judged_path = (
            DATA_DIR
            / "runs"
            / "judgements"
            / "development"
            / f"{name}__{JUDGE_PROMPT_VERSION}__{JUDGE_NAME}.jsonl"
        )
        answers = [GeneratedAnswer.model_validate_json(line) for line in _lines(answers_path)]
        evaluations = [AnswerEvaluation.model_validate_json(line) for line in _lines(judged_path)]
        generation.append(
            GenerationRun(
                name,
                [a for a in answers if a.question_id in current],
                [e for e in evaluations if e.question_id in current],
            )
        )

    manifest = FixtureManifest(
        description="Development and calibration questions with the passages real retrievers "
        "return for them. See scripts/regression_gate.py.",
        splits=SPLITS,
        dataset_hash=read_lock(DATASET_DIR / "dataset.lock.json")["dataset_hash"],
        corpus_hash=resources.corpus_hash,
        chunking=resources.chunks_dir(chunking).name,
        embedder=model,
        embedder_revision=embedder_config.revision,
        pooled_from=GATE_CONFIGS,
        pool_depth=POOL_DEPTH,
        configs=configs,
        generation_runs=GENERATION_RUNS,
        licenses=LICENSES,
    )
    fixture = Fixture(
        manifest=manifest,
        questions=questions,
        chunks=[chunks[row] for row in rows],
        document_vectors=np.asarray(document_vectors, dtype=np.float32),
        query_vectors=np.asarray(query_vectors, dtype=np.float32),
        generation=generation,
    )
    save_fixture(FIXTURE_DIR, fixture)
    size = sum(p.stat().st_size for p in FIXTURE_DIR.rglob("*") if p.is_file())
    print(
        f"Fixture: {len(questions)} questions, {len(rows)} of {len(chunks)} chunks, "
        f"{len(generation)} generation runs, {size / 1e6:.1f} MB in {FIXTURE_DIR}"
    )


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_baseline(tolerance: float) -> None:
    fixture = load_fixture(FIXTURE_DIR)
    retrieval, generation = measure(fixture)
    baseline = Baseline(
        dataset_hash=fixture.manifest.dataset_hash,
        tolerance=tolerance,
        retrieval=retrieval,
        generation=generation,
    )
    _write_baseline(baseline)
    print(f"Baseline written to {BASELINE_PATH}")


def _write_baseline(baseline: Baseline) -> None:
    text = json.dumps(baseline.model_dump(), indent=1, sort_keys=True)
    BASELINE_PATH.write_text(text + "\n", encoding="utf-8", newline="\n")


def check(record_noise: bool) -> None:
    baseline = Baseline.model_validate_json(BASELINE_PATH.read_text(encoding="utf-8"))
    retrieval, generation = measure(load_fixture(FIXTURE_DIR))
    checks = compare(baseline, retrieval, generation)
    report = render(checks, baseline)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as out:
            out.write(report)

    if record_noise:
        noise = max(abs(c.delta) for c in checks if c.group == "retrieval")
        baseline.measured_platform_noise = max(noise, baseline.measured_platform_noise or 0.0)
        _write_baseline(baseline)
        print(f"Recorded platform noise {noise:.6f} in {BASELINE_PATH}")

    result = overall(checks)
    if result == "fail":
        raise SystemExit(1)
    if result == "inconclusive":
        # Shown as an annotation on the GitHub run. The job passes, so a human decides.
        print(
            "::warning title=Regression gate inconclusive::A metric dropped, but not "
            "consistently. Review the gate report before merging."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["check", "build", "baseline"])
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--record-noise",
        action="store_true",
        help="with check: save the largest difference from the baseline as platform noise",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.action == "build":
        build()
    elif args.action == "baseline":
        record_baseline(args.tolerance)
    else:
        check(args.record_noise)


if __name__ == "__main__":
    main()
