"""Sparse keyword retrieval with BM25."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ragbench.chunking.models import Chunk
from ragbench.retrieval.base import MetadataFilter, Retriever, ScoredChunk, sort_by_score

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever(Retriever):
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25 needs at least one chunk")
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def retrieve(
        self, query: str, top_k: int, metadata_filter: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        if metadata_filter is not None:
            scores = np.where(metadata_filter.mask(self._chunks), scores, 0.0)

        # A zero score means no query term matched, which is not a retrieval result.
        matching = np.flatnonzero(scores > 0)
        if len(matching) > top_k:
            # Keep every chunk tied with the k-th score, so the tie-break decides the cut.
            kth_score = np.partition(scores[matching], -top_k)[-top_k]
            matching = matching[scores[matching] >= kth_score]
        results = [ScoredChunk(self._chunks[i], float(scores[i])) for i in matching]
        return sort_by_score(results)[:top_k]
