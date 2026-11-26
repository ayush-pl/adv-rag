"""Contexts for generation: retrieved passages, or gold passages for oracle runs.

Retrieval runs in the GPU environment and generation runs next to Ollama, so retrieved
contexts are saved to files in between.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from ragbench.chunking.models import Chunk, parse_chunk_id
from ragbench.dataset.models import Question
from ragbench.evaluation.gold import chunk_matches_span
from ragbench.generation.prompts import Passage
from ragbench.ingestion.models import Document
from ragbench.retrieval.base import Retriever


class RetrievedContext(BaseModel):
    question_id: str
    chunk_ids: list[str]


def retrieve_contexts(
    questions: list[Question], retriever: Retriever, top_n: int
) -> list[RetrievedContext]:
    """Top passages for every question, unanswerable ones included, to test abstention."""
    return [
        RetrievedContext(
            question_id=question.question_id,
            chunk_ids=[r.chunk.chunk_id for r in retriever.retrieve(question.question, top_n)],
        )
        for question in questions
    ]


def oracle_chunk_ids(question: Question, chunks_by_document: dict[str, list[Chunk]]) -> list[str]:
    """For each evidence group, the chunk that best covers its first span.

    Using chunks from the same chunking as the retrieved run keeps both contexts at the same
    granularity, so the comparison isolates retrieval rather than passage length.
    """
    selected: list[str] = []
    for group in question.evidence:
        span = group.spans[0]
        candidates = [
            chunk
            for chunk in chunks_by_document.get(span.document_id, [])
            if chunk_matches_span(chunk, span)
        ]
        if not candidates:
            raise ValueError(f"{question.question_id}: no chunk covers {span.document_id}")
        best = max(
            candidates,
            key=lambda c: (
                min(c.char_end, span.char_end) - max(c.char_start, span.char_start),
                -c.char_start,
            ),
        )
        if best.chunk_id not in selected:
            selected.append(best.chunk_id)
    return selected


def passages_for(chunk_ids: list[str], documents: dict[str, Document]) -> list[Passage]:
    passages = []
    for chunk_id in chunk_ids:
        document_id, start, end = parse_chunk_id(chunk_id)
        document = documents[document_id]
        passages.append(Passage(chunk_id, document.title, document.content[start:end]))
    return passages


def write_contexts(path: Path, contexts: list[RetrievedContext]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for context in contexts:
            out.write(context.model_dump_json() + "\n")


def read_contexts(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as lines:
        contexts = [RetrievedContext.model_validate(json.loads(line)) for line in lines]
    return {context.question_id: context.chunk_ids for context in contexts}
