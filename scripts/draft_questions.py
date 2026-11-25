"""Draft benchmark questions with a local LLM. Safe to stop and rerun: it tops up each type."""

import argparse
import glob
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ragbench.chunking.chunk_manager import load_chunks
from ragbench.chunking.models import Chunk
from ragbench.dataset.drafting import (
    DraftRejected,
    QuestionDrafter,
    eligible_passages,
    passages_for_type,
    round_robin_by_product,
    version_only_passages,
)
from ragbench.dataset.models import Question, append_question, read_questions
from ragbench.dataset.review import accepted_questions
from ragbench.generation.ollama_client import OllamaClient, OllamaError
from ragbench.ingestion.corpus_manager import load_documents
from ragbench.retrieval.bm25 import BM25Retriever

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "datasets" / "v1"
DRAFTS_PATH = DATASET_DIR / "drafts.jsonl"
DECISIONS_PATH = DATASET_DIR / "review_decisions.jsonl"

# About twice each target, because many drafts will be rejected in review.
DEFAULT_COUNTS = {
    "single_hop": 80,
    "multi_hop": 60,
    "config_code": 30,
    "table": 20,
    "version_specific": 20,
    "unanswerable": 30,
}
MAX_ATTEMPTS_PER_DRAFT = 6
MAX_CONSECUTIVE_FAILURES = 3
# Passages from different products must be at least this similar to make a real comparison.
MIN_CROSS_PRODUCT_SIMILARITY = 0.7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Draft at most this many of each type")
    parser.add_argument("--paraphrases", type=int, default=0, help="Paraphrase accepted questions")
    parser.add_argument("--output", type=Path, default=DRAFTS_PATH)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    documents = {d.document_id: d for d in load_documents(DATA_DIR / "processed" / "v1")}
    chunks_dir = Path(glob.glob(str(DATA_DIR / "chunks" / "*" / "recursive-512-*"))[0])
    chunks = load_chunks(chunks_dir)
    passages = eligible_passages(chunks)
    drafter = QuestionDrafter(OllamaClient(), args.model, documents, args.seed)
    rng = random.Random(args.seed)

    existing = read_questions(args.output)
    known_ids = {question.question_id for question in existing}
    counts = Counter(question.question_type for question in existing)

    def save(question: Question) -> bool:
        if question.question_id in known_ids:
            return False
        known_ids.add(question.question_id)
        counts[question.question_type] += 1
        append_question(args.output, question)
        print(f"[{question.question_type}] {question.question}", flush=True)
        return True

    def target(question_type: str) -> int:
        full = DEFAULT_COUNTS[question_type]
        return full if args.limit is None else min(full, args.limit)

    def attempts(question_type: str, candidates, make) -> None:
        rejected: Counter[str] = Counter()
        budget = (target(question_type) - counts[question_type]) * MAX_ATTEMPTS_PER_DRAFT
        failures = 0
        for candidate in candidates:
            if counts[question_type] >= target(question_type) or budget <= 0:
                break
            budget -= 1
            try:
                save(make(candidate))
                failures = 0
            except DraftRejected as exc:
                rejected[str(exc).split(":")[0]] += 1
                failures = 0
            except OllamaError as exc:
                rejected["ollama error"] += 1
                failures += 1
                print(f"Ollama failed on one draft: {exc}", flush=True)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    raise SystemExit("Stopping after repeated Ollama failures.") from exc
        print(f"{question_type}: {counts[question_type]} drafts. Rejected: {dict(rejected)}")

    if args.paraphrases:
        sources = [q for q in accepted_questions(DECISIONS_PATH) if q.question_type == "single_hop"]
        rng.shuffle(sources)
        for source in sources[: args.paraphrases]:
            try:
                save(drafter.draft_paraphrase(source))
            except DraftRejected as exc:
                print(f"Rejected: {exc}")
        return

    for question_type in ("single_hop", "config_code", "table"):
        candidates = round_robin_by_product(passages_for_type(passages, question_type), rng)
        attempts(
            question_type,
            candidates,
            lambda chunk, kind=question_type: drafter.draft_single(chunk, kind),
        )

    attempts(
        "version_specific",
        version_only_passages(documents, rng),
        drafter.draft_version_specific,
    )

    prose = passages_for_type(passages, "single_hop")
    bm25 = BM25Retriever(prose)
    attempts(
        "unanswerable",
        round_robin_by_product(prose, rng),
        lambda chunk: drafter.draft_unanswerable(chunk, []),
    )
    _add_nearest_passages(args.output, bm25)

    vectors = _load_vectors(chunks_dir)
    position = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
    attempts(
        "multi_hop",
        _related_pairs(prose, chunks, position, vectors, rng),
        lambda pair: drafter.draft_multi_hop(*pair),
    )


def _add_nearest_passages(path: Path, bm25: BM25Retriever) -> None:
    """Show the reviewer what keyword search finds for each question it must confirm."""
    questions = read_questions(path)
    for question in questions:
        if question.question_type == "unanswerable" and not question.nearest_passages:
            question.nearest_passages = [
                f"{hit.chunk.source_url} :: {' '.join(hit.chunk.text.split())[:240]}"
                for hit in bm25.retrieve(question.question, 3)
            ]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(q.model_dump_json() + "\n" for q in questions), encoding="utf-8", newline="\n"
    )
    tmp.replace(path)


def _load_vectors(chunks_dir: Path) -> np.ndarray:
    corpus_key, chunking_id = chunks_dir.parts[-2:]
    pattern = DATA_DIR / "indexes" / corpus_key / chunking_id / "BAAI--bge-small*" / "vectors.npy"
    return np.load(glob.glob(str(pattern))[0])


def _related_pairs(
    prose: list[Chunk],
    chunks: list[Chunk],
    position: dict[str, int],
    vectors: np.ndarray,
    rng: random.Random,
):
    """Pair passages with a similar one from another document, favouring other products."""
    eligible = {chunk.chunk_id for chunk in prose}
    for first in round_robin_by_product(prose, rng):
        similarity = vectors @ vectors[position[first.chunk_id]]
        order = np.argsort(-similarity)[:400]
        same_product = None
        for index in order:
            candidate = chunks[int(index)]
            if candidate.chunk_id not in eligible:
                continue
            if candidate.document_id.startswith("postgres-15/"):
                continue
            if candidate.document_id == first.document_id:
                continue
            if candidate.document_id.split("/", 1)[1] == first.document_id.split("/", 1)[1]:
                continue
            if candidate.source != first.source:
                if similarity[index] >= MIN_CROSS_PRODUCT_SIMILARITY:
                    yield first, candidate
                    break
                continue
            if same_product is None:
                same_product = candidate
        else:
            if same_product is not None:
                yield first, same_product


if __name__ == "__main__":
    main()
