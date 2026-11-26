"""Experiment configurations. A config's ID changes whenever any setting changes."""

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChunkingChoice(BaseModel):
    strategy: Literal["fixed", "recursive"]
    chunk_size: int
    overlap: int


class RetrievalConfig(BaseModel):
    name: str
    chunking: ChunkingChoice
    method: Literal["dense", "bm25", "hybrid"]
    # Model ID of the dense embedder, for the dense and hybrid methods.
    embedder: str | None = None
    rrf_k: int = 60
    # How many results each retriever passes to fusion.
    hybrid_depth: int = 100
    rerank: bool = False
    rerank_candidates: int = 20
    # Drop passages that repeat a higher-ranked one word for word, before any reranking.
    dedup: bool = False
    top_k: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def embedder_matches_method(self) -> "RetrievalConfig":
        if self.method in ("dense", "hybrid") and self.embedder is None:
            raise ValueError(f"{self.name}: the {self.method} method needs an embedder")
        if self.method == "bm25" and self.embedder is not None:
            raise ValueError(f"{self.name}: BM25 does not use an embedder")
        return self

    @property
    def config_id(self) -> str:
        # Settings added later are left out at their default, so existing IDs don't change.
        excluded = {"name"} | ({"dedup"} if not self.dedup else set())
        settings = self.model_dump_json(exclude=excluded)
        return hashlib.sha256(settings.encode()).hexdigest()[:8]

    @property
    def run_name(self) -> str:
        return f"{self.name}-{self.config_id}"
