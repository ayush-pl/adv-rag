"""Composite retrievers: hybrid fusion and reranking on top of any base retriever."""

from ragbench.retrieval.base import MetadataFilter, Reranker, Retriever, ScoredChunk
from ragbench.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


class HybridRetriever(Retriever):
    def __init__(
        self, retrievers: list[Retriever], rrf_k: int = DEFAULT_RRF_K, depth: int = 100
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        self._depth = depth

    def retrieve(
        self, query: str, top_k: int, metadata_filter: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        depth = max(self._depth, top_k)
        rankings = [r.retrieve(query, depth, metadata_filter) for r in self._retrievers]
        return reciprocal_rank_fusion(rankings, top_k, self._rrf_k)


class RerankingRetriever(Retriever):
    def __init__(self, base: Retriever, reranker: Reranker, candidates: int = 20) -> None:
        self._base = base
        self._reranker = reranker
        self._candidates = candidates

    def retrieve(
        self, query: str, top_k: int, metadata_filter: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        candidates = self._base.retrieve(query, max(self._candidates, top_k), metadata_filter)
        return self._reranker.rerank(query, candidates, top_k)
