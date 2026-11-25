"""Fixed-size chunking: sliding token windows that ignore document structure.

This is the naive baseline, so it may cut through code blocks and tables by design.
"""

from ragbench.chunking.base import Chunker
from ragbench.chunking.tokenizer import TokenOffsets
from ragbench.ingestion.models import Document


class FixedChunker(Chunker):
    def _spans(self, document: Document, offsets: TokenOffsets) -> list[tuple[int, int]]:
        return token_windows(offsets, 0, len(offsets), self.chunk_size, self.overlap)


def token_windows(
    offsets: TokenOffsets, first: int, last: int, size: int, overlap: int
) -> list[tuple[int, int]]:
    """Character spans of windows over tokens [first, last). The last window ends at `last`."""
    spans = []
    start = first
    while start < last:
        end = min(start + size, last)
        spans.append((offsets.starts[start], offsets.ends[end - 1]))
        if end == last:
            break
        start = end - overlap
    return spans
