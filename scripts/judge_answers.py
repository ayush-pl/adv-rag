"""Evaluate every generated answer: deterministic checks, then the LLM judge where needed.

Runs on Windows next to Ollama. Results are saved per answer, so the script can be stopped and
rerun. Run the generation first: scripts/generate_answers.py.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from ragbench.dataset.models import read_questions
from ragbench.evaluation.judge import (
    JUDGE_PROMPT_VERSION,
    AnswerEvaluation,
    abstention_correct,
    citation_scores,
    judge_answer,
)
from ragbench.generation.contexts import passages_for
from ragbench.generation.ollama_client import OllamaClient, OllamaError
from ragbench.generation.rag_pipeline import GeneratedAnswer
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
SPLITS = ["development", "calibration", "test"]


def judgement_path(generation_path: Path, judge_model: str) -> Path:
    split = generation_path.parent.name
    judge_name = judge_model.replace(":", "-").replace("/", "-")
    name = f"{generation_path.stem}__{JUDGE_PROMPT_VERSION}__{judge_name}.jsonl"
    return DATA_DIR / "runs" / "judgements" / split / name


def read_evaluations(path: Path) -> list[AnswerEvaluation]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [AnswerEvaluation.model_validate(json.loads(line)) for line in lines if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    spec = yaml.safe_load(Path("configs/generation.yaml").read_text(encoding="utf-8"))
    judge_model = spec["judge"]
    documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}
    client = OllamaClient()

    for split in args.splits:
        questions = {q.question_id: q for q in read_questions(DATASET_DIR / f"{split}.jsonl")}
        for generation_path in sorted((DATA_DIR / "runs" / "generation" / split).glob("*.jsonl")):
            output = judgement_path(generation_path, judge_model)
            done = {e.question_id for e in read_evaluations(output)}
            with generation_path.open(encoding="utf-8") as lines:
                answers = [GeneratedAnswer.model_validate(json.loads(line)) for line in lines]
            pending = [
                a for a in answers if a.question_id not in done and a.question_id in questions
            ]
            print(f"{split}/{generation_path.stem}: {len(pending)} to judge", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            for answer in pending:
                question = questions[answer.question_id]
                passages = passages_for(answer.context_chunk_ids, documents)
                try:
                    judgement = judge_answer(client, judge_model, question, answer, passages)
                except OllamaError as exc:
                    raise SystemExit(f"Ollama failed, stopping. Rerun to continue: {exc}") from exc
                precision, recall = citation_scores(question, answer)
                evaluation = AnswerEvaluation(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    answerable=question.answerable,
                    abstained=answer.abstained,
                    abstention_correct=abstention_correct(question, answer),
                    citation_precision=precision,
                    citation_recall=recall,
                    judgement=judgement,
                )
                with output.open("a", encoding="utf-8", newline="\n") as out:
                    out.write(evaluation.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
