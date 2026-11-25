"""Common retriever and reranker interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ragbench.chunking.models import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved passage. `chunk` carries its identifiers and source metadata."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class MetadataFilter:
    """Restrict retrieval to these sources and versions. None means no restriction."""

    sources: frozenset[str] | None = None
    versions: frozenset[str] | None = None

    def matches(self, chunk: Chunk) -> bool:
        return (self.sources is None or chunk.source in self.sources) and (
            self.versions is None or chunk.version in self.versions
        )

    def mask(self, chunks: list[Chunk]) -> np.ndarray:
        return np.fromiter((self.matches(c) for c in chunks), dtype=bool, count=len(chunks))


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self, query: str, top_k: int, metadata_filter: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        """Return up to `top_k` passages, best first, in a deterministic order."""


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        """Return the best `top_n` candidates, scored by this reranker."""


def sort_by_score(results: list[ScoredChunk]) -> list[ScoredChunk]:
    """Order by score, breaking ties by ID. PostgreSQL 15 and 16 share pages, so ties are real."""
    return sorted(results, key=lambda result: (-result.score, result.chunk.chunk_id))
