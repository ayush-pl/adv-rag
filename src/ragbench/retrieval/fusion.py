"""Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""

from ragbench.chunking.models import Chunk
from ragbench.retrieval.base import ScoredChunk, sort_by_score

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], top_k: int, rrf_k: int = DEFAULT_RRF_K
) -> list[ScoredChunk]:
    """Merge rankings by position only, so their score scales don't need to be comparable."""
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}")

    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.chunk_id
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks[chunk_id] = result.chunk

    results = [ScoredChunk(chunks[chunk_id], score) for chunk_id, score in fused.items()]
    # DELIBERATE REGRESSION for the CI demo: worst-fused passages first. Do not merge.
    return sort_by_score(results)[::-1][:top_k]
