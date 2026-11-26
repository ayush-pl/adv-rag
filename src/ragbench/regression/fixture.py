"""A small, self-contained copy of the benchmark that runs without models, GPU or corpus.

It holds real questions, the chunks real retrievers return for them, the chunks' stored
embedding vectors and each question's query vector. Retrieval code runs on it unchanged:
BM25 indexes the fixture's chunks, and dense retrieval searches the stored vectors with a
query vector looked up instead of computed. Recorded answers and judgements let the
evaluation code be rerun without Ollama.
"""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from ragbench.chunking.models import Chunk
from ragbench.dataset.models import Question
from ragbench.embeddings.base import Embedder, TruncationStats
from ragbench.evaluation.judge import AnswerEvaluation
from ragbench.experiments.config import RetrievalConfig
from ragbench.generation.rag_pipeline import GeneratedAnswer
from ragbench.retrieval.base import Retriever
from ragbench.retrieval.bm25 import BM25Retriever
from ragbench.retrieval.dedup import DeduplicatingRetriever
from ragbench.retrieval.dense import DenseRetriever
from ragbench.retrieval.hybrid import HybridRetriever

CHUNKS = "chunks.jsonl.gz"
QUESTIONS = "questions.jsonl"
DOCUMENT_VECTORS = "document_vectors.npy"
QUERY_VECTORS = "query_vectors.npy"
MANIFEST = "manifest.json"
GENERATION_DIR = "generation"


class FixtureManifest(BaseModel):
    description: str
    splits: list[str]
    dataset_hash: str
    corpus_hash: str
    chunking: str
    embedder: str
    embedder_revision: str
    # The configurations whose top results were pooled into the fixture.
    pooled_from: list[str]
    pool_depth: int
    # The gate's configurations. Dense and hybrid need the fixture's embedder.
    configs: list[RetrievalConfig]
    # Recorded generation runs, by name. Each has an answers and a judgements file.
    generation_runs: list[str]
    licenses: str


class CachedQueryEmbedder(Embedder):
    """Returns stored query vectors. A query that wasn't stored is an error, never a guess."""

    def __init__(self, queries: list[str], vectors: np.ndarray) -> None:
        if len(queries) != len(vectors):
            raise ValueError(f"{len(queries)} queries but {len(vectors)} vectors")
        self._rows = {query: row for row, query in enumerate(queries)}
        self._vectors = vectors

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        missing = [text for text in texts if text not in self._rows]
        if missing:
            raise KeyError(f"No stored vector for query {missing[0]!r}")
        return self._vectors[[self._rows[text] for text in texts]]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError("Document vectors are stored in the fixture")

    def truncation_stats(self, texts: list[str]) -> TruncationStats:
        raise NotImplementedError("Truncation is measured when the real index is built")


@dataclass
class GenerationRun:
    name: str
    answers: list[GeneratedAnswer]
    evaluations: list[AnswerEvaluation]


@dataclass
class Fixture:
    manifest: FixtureManifest
    questions: list[Question]
    chunks: list[Chunk]
    document_vectors: np.ndarray
    query_vectors: np.ndarray
    generation: list[GenerationRun]

    def retriever(self, config: RetrievalConfig) -> Retriever:
        if config.rerank:
            raise ValueError("The fixture has no reranker scores, so it cannot rerank")
        retriever = self._base_retriever(config)
        return DeduplicatingRetriever(retriever) if config.dedup else retriever

    def _base_retriever(self, config: RetrievalConfig) -> Retriever:
        if config.method == "bm25":
            return BM25Retriever(self.chunks)
        if config.embedder != self.manifest.embedder:
            raise ValueError(f"{config.name}: the fixture stores {self.manifest.embedder} only")
        embedder = CachedQueryEmbedder([q.question for q in self.questions], self.query_vectors)
        dense = DenseRetriever(self.chunks, self.document_vectors, embedder)
        if config.method == "dense":
            return dense
        return HybridRetriever(
            [dense, BM25Retriever(self.chunks)], rrf_k=config.rrf_k, depth=config.hybrid_depth
        )


def _read_jsonl(path: Path, model) -> list:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as lines:
        return [model.model_validate(json.loads(line)) for line in lines if line.strip()]


def _write_jsonl(path: Path, records: list[BaseModel]) -> None:
    text = "".join(record.model_dump_json() + "\n" for record in records)
    if path.suffix == ".gz":
        # mtime=0 keeps the file byte-identical across rebuilds.
        with gzip.GzipFile(path, "wb", mtime=0) as out:
            out.write(text.encode("utf-8"))
    else:
        path.write_text(text, encoding="utf-8", newline="\n")


def save_fixture(directory: Path, fixture: Fixture) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST).write_text(
        fixture.manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    _write_jsonl(directory / QUESTIONS, fixture.questions)
    _write_jsonl(directory / CHUNKS, fixture.chunks)
    # Half precision halves the committed size. The gate's baseline is computed from these
    # same stored values, so the precision loss cannot show up as a regression.
    np.save(directory / DOCUMENT_VECTORS, fixture.document_vectors.astype(np.float16))
    np.save(directory / QUERY_VECTORS, fixture.query_vectors.astype(np.float16))
    generation_dir = directory / GENERATION_DIR
    generation_dir.mkdir(exist_ok=True)
    for run in fixture.generation:
        _write_jsonl(generation_dir / f"{run.name}.answers.jsonl", run.answers)
        _write_jsonl(generation_dir / f"{run.name}.judgements.jsonl", run.evaluations)


def load_fixture(directory: Path) -> Fixture:
    manifest = FixtureManifest.model_validate_json(
        (directory / MANIFEST).read_text(encoding="utf-8")
    )
    generation = [
        GenerationRun(
            name,
            _read_jsonl(directory / GENERATION_DIR / f"{name}.answers.jsonl", GeneratedAnswer),
            _read_jsonl(directory / GENERATION_DIR / f"{name}.judgements.jsonl", AnswerEvaluation),
        )
        for name in manifest.generation_runs
    ]
    return Fixture(
        manifest=manifest,
        questions=_read_jsonl(directory / QUESTIONS, Question),
        chunks=_read_jsonl(directory / CHUNKS, Chunk),
        document_vectors=np.load(directory / DOCUMENT_VECTORS).astype(np.float32),
        query_vectors=np.load(directory / QUERY_VECTORS).astype(np.float32),
        generation=generation,
    )
