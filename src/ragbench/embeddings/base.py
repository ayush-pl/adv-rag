"""Common embedder interface."""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel


class EmbedderConfig(BaseModel):
    model: str
    revision: str
    # Some models are trained with instruction prefixes and lose quality without them.
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 64

    @property
    def slug(self) -> str:
        return f"{self.model.replace('/', '--')}-{self.revision[:8]}"


class TruncationStats(BaseModel):
    """How much text an embedder cannot read, which otherwise silently skews comparisons."""

    max_tokens: int
    texts: int
    truncated_texts: int
    truncated_rate: float
    longest_text_tokens: int


class Embedder(ABC):
    """Produces L2-normalized float32 vectors, so inner product equals cosine similarity."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def truncation_stats(self, texts: list[str]) -> TruncationStats: ...
