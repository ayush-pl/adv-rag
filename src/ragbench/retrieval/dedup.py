"""Drop passages that repeat a higher-ranked passage word for word.

PostgreSQL 15 and 16 share most of their documentation, so 42% of 512-token chunks have an exact
twin. Left alone, the twins take result slots: on development and calibration, 15% of top-5
passages from the best configuration repeated one ranked above them. Removing them gives those
slots to new evidence, which matters most for questions that need several passages.
"""

import re

from ragbench.retrieval.base import MetadataFilter, Retriever, ScoredChunk

_SPACE = re.compile(r"\s+")


def duplicate_key(text: str) -> str:
    """Text that differs only in whitespace or case counts as the same passage."""
    return _SPACE.sub(" ", text).strip().lower()


def named_versions(query: str, versions: set[str]) -> set[str]:
    """Versions the query mentions as whole words, such as 16 in "PostgreSQL 16"."""
    return {v for v in versions if re.search(rf"(?<![\w.]){re.escape(v)}(?![\w.])", query)}


class DeduplicatingRetriever(Retriever):
    """Keeps the first copy of each passage, in the base ranking's order.

    When the query names a version and a later copy comes from that version, the later copy takes
    the first one's place. The text is identical, but the citation then points at the version the
    question asked about.
    """

    def __init__(self, base: Retriever, overfetch: int = 2) -> None:
        if overfetch < 1:
            raise ValueError(f"overfetch must be at least 1, got {overfetch}")
        self._base = base
        self._overfetch = overfetch

    def retrieve(
        self, query: str, top_k: int, metadata_filter: MetadataFilter | None = None
    ) -> list[ScoredChunk]:
        ranked = self._base.retrieve(query, top_k * self._overfetch, metadata_filter)
        wanted = named_versions(query, {result.chunk.version for result in ranked})
        kept: list[ScoredChunk] = []
        position: dict[str, int] = {}
        for result in ranked:
            key = duplicate_key(result.chunk.text)
            if key not in position:
                position[key] = len(kept)
                kept.append(result)
                continue
            first = kept[position[key]]
            if result.chunk.version in wanted and first.chunk.version not in wanted:
                kept[position[key]] = ScoredChunk(result.chunk, first.score)
        return kept[:top_k]
