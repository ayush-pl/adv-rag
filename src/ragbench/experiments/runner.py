"""Run a retrieval configuration over a dataset split and record every result."""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from ragbench.chunking.models import chunk_from_id
from ragbench.dataset.models import Question
from ragbench.evaluation.retrieval_metrics import evaluate_ranking
from ragbench.experiments.config import RetrievalConfig
from ragbench.experiments.tracker import Environment
from ragbench.performance.latency import timed
from ragbench.retrieval.base import Retriever


class QuestionResult(BaseModel):
    question_id: str
    question_type: str
    chunk_ids: list[str]
    scores: list[float]
    latency_ms: float
    metrics: dict[str, float]


class RunRecord(BaseModel):
    config: RetrievalConfig
    config_id: str
    split: str
    dataset_hash: str
    corpus_hash: str
    environment: Environment
    # Unanswerable questions have no gold evidence, so they are not scored for retrieval.
    questions_scored: int
    questions_skipped: int
    latency_p50_ms: float
    latency_p95_ms: float
    # Means over scored questions. Uncertainty is computed from the per-question results.
    metrics: dict[str, float]


def run_retrieval(
    config: RetrievalConfig, questions: list[Question], retriever: Retriever
) -> list[QuestionResult]:
    """Retrieve for every answerable question. Latency excludes model loading (warm queries)."""
    scored = [question for question in questions if question.answerable]
    if scored:
        retriever.retrieve(scored[0].question, config.top_k)

    results = []
    for question in scored:
        ranked, latency_ms = timed(lambda q=question: retriever.retrieve(q.question, config.top_k))
        results.append(
            QuestionResult(
                question_id=question.question_id,
                question_type=question.question_type,
                chunk_ids=[result.chunk.chunk_id for result in ranked],
                scores=[round(result.score, 6) for result in ranked],
                latency_ms=round(latency_ms, 2),
                metrics=evaluate_ranking([result.chunk for result in ranked], question.evidence),
            )
        )
    return results


def rescore(
    results: list[QuestionResult], questions: list[Question]
) -> list[QuestionResult] | None:
    """Recompute metrics for saved rankings after labels change, without retrieving again.

    Chunk IDs encode each passage's document and offsets, which is all that matching gold
    spans needs. Withdrawn questions are dropped. Returns None when there are new answerable
    questions, since they need real retrieval.
    """
    by_id = {question.question_id: question for question in questions if question.answerable}
    if not set(by_id) <= {result.question_id for result in results}:
        return None
    rescored = []
    for result in (r for r in results if r.question_id in by_id):
        question = by_id[result.question_id]
        ranked = [chunk_from_id(chunk_id) for chunk_id in result.chunk_ids]
        rescored.append(
            result.model_copy(
                update={
                    "question_type": question.question_type,
                    "metrics": evaluate_ranking(ranked, question.evidence),
                }
            )
        )
    return rescored


def summarize(
    config: RetrievalConfig,
    results: list[QuestionResult],
    questions: list[Question],
    split: str,
    dataset_hash: str,
    corpus_hash: str,
    environment: Environment,
) -> RunRecord:
    if not results:
        raise ValueError(f"{config.name}: no answerable questions to score")
    latencies = [result.latency_ms for result in results]
    names = results[0].metrics.keys()
    return RunRecord(
        config=config,
        config_id=config.config_id,
        split=split,
        dataset_hash=dataset_hash,
        corpus_hash=corpus_hash,
        environment=environment,
        questions_scored=len(results),
        questions_skipped=len(questions) - len(results),
        latency_p50_ms=round(float(np.percentile(latencies, 50)), 2),
        latency_p95_ms=round(float(np.percentile(latencies, 95)), 2),
        metrics={
            name: round(float(np.mean([r.metrics[name] for r in results])), 4) for name in names
        },
    )


def save_run(directory: Path, record: RunRecord, results: list[QuestionResult]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "questions.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for result in results:
            out.write(result.model_dump_json() + "\n")
    # Written last, so an interrupted run is never mistaken for a finished one.
    (directory / "run.json").write_text(
        record.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def load_run(directory: Path) -> tuple[RunRecord, list[QuestionResult]]:
    record = RunRecord.model_validate_json((directory / "run.json").read_text(encoding="utf-8"))
    with (directory / "questions.jsonl").open(encoding="utf-8") as lines:
        results = [QuestionResult.model_validate(json.loads(line)) for line in lines]
    return record, results
