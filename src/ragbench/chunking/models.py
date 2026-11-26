"""Schemas for chunks and chunking configurations."""

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "recursive"]
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)
    # One tokenizer defines "tokens" for every config, so chunks don't vary with the embedder.
    tokenizer: str
    tokenizer_revision: str

    @model_validator(mode="after")
    def overlap_is_smaller_than_size(self) -> "ChunkingConfig":
        if self.overlap >= self.chunk_size:
            raise ValueError(f"overlap {self.overlap} must be below size {self.chunk_size}")
        return self

    @property
    def config_id(self) -> str:
        digest = hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:8]
        return f"{self.strategy}-{self.chunk_size}-{self.overlap}-{digest}"


def parse_chunk_id(chunk_id: str) -> tuple[str, int, int]:
    """Chunk IDs are `document_id::start-end`, so a ranking identifies passages by itself."""
    document_id, _, span = chunk_id.rpartition("::")
    start, end = span.split("-")
    return document_id, int(start), int(end)


def chunk_from_id(chunk_id: str) -> "Chunk":
    """A chunk with only its location filled in, enough to match it against gold spans."""
    document_id, start, end = parse_chunk_id(chunk_id)
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        char_start=start,
        char_end=end,
        text="",
        token_count=0,
        section="",
        source="",
        source_url="",
        title="",
        version="",
    )


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    # `text` is always exactly document.content[char_start:char_end].
    char_start: int
    char_end: int
    text: str
    token_count: int
    section: str
    source: str
    source_url: str
    title: str
    version: str
