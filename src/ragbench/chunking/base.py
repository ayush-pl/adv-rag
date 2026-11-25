"""Common chunker interface and chunk construction."""

from abc import ABC, abstractmethod
from bisect import bisect_right

from ragbench.chunking.models import Chunk
from ragbench.chunking.tokenizer import TokenCounter, TokenOffsets
from ragbench.ingestion.models import Document


class Chunker(ABC):
    def __init__(self, chunk_size: int, overlap: int, counter: TokenCounter) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap {overlap} must be below size {chunk_size}")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.counter = counter

    def chunk(self, document: Document) -> list[Chunk]:
        offsets = self.counter.offsets(document.content)
        chunks = []
        for start, end in self._spans(document, offsets):
            chunk = _build_chunk(document, start, end, offsets)
            if _is_valid(chunk.text):
                chunks.append(chunk)
        return chunks

    @abstractmethod
    def _spans(self, document: Document, offsets: TokenOffsets) -> list[tuple[int, int]]:
        """Return the character spans of the chunks, in document order."""


def _is_valid(text: str) -> bool:
    """Reject empty chunks and chunks of pure markup, such as a lone code fence."""
    return any(character.isalnum() for character in text)


def _build_chunk(document: Document, start: int, end: int, offsets: TokenOffsets) -> Chunk:
    return Chunk(
        # Offsets identify the text, so the same span has the same ID in every run.
        chunk_id=f"{document.document_id}::{start}-{end}",
        document_id=document.document_id,
        char_start=start,
        char_end=end,
        text=document.content[start:end],
        token_count=offsets.count(start, end),
        section=_section_path(document, start),
        source=document.source,
        source_url=document.source_url,
        title=document.title,
        version=document.version,
    )


def _section_path(document: Document, char_start: int) -> str:
    """Heading path that applies at a position, such as `Pod Lifecycle > Pod phase`."""
    starts = [section.char_start for section in document.sections]
    stack: list[tuple[int, str]] = []
    for section in document.sections[: bisect_right(starts, char_start)]:
        while stack and stack[-1][0] >= section.level:
            stack.pop()
        stack.append((section.level, section.heading))
    return " > ".join(heading for _, heading in stack)
