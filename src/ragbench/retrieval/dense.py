"""Dense retrieval with an exact FAISS inner-product index."""

import faiss
import numpy as np

from ragbench.chunking.models import Chunk
from ragbench.embeddings.base import Embedder
from ragbench.retrieval.base import MetadataFilter, Retriever, ScoredChunk, sort_by_score


class DenseRetriever(Retriever):
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self._chunks = chunks
        self._embedder = embedder
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)

    def retrieve(
        self, query: str, top_k: int, metadata_filter: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        query_vector = self._embedder.embed_queries([query])
        params = None
        if metadata_filter is not None:
            allowed = np.flatnonzero(metadata_filter.mask(self._chunks))
            if len(allowed) == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed))

        scores, indices = self._index.search(
            query_vector, min(top_k, len(self._chunks)), params=params
        )
        return sort_by_score(
            [
                ScoredChunk(self._chunks[index], float(score))
                for score, index in zip(scores[0], indices[0], strict=True)
                if index != -1
            ]
        )
