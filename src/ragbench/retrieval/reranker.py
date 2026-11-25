"""Cross-encoder reranking. Requires the `ml` extra."""

from ragbench.retrieval.base import Reranker, ScoredChunk, sort_by_score


class CrossEncoderReranker(Reranker):
    def __init__(
        self, model: str, revision: str, device: str | None = None, batch_size: int = 16
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model, revision=revision, device=device)
        self._batch_size = batch_size

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._model.predict(
            [(query, candidate.chunk.text) for candidate in candidates],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        rescored = [
            ScoredChunk(candidate.chunk, float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        return sort_by_score(rescored)[:top_n]
