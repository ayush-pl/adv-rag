"""Structure-aware chunking: split by heading, then block, then line or sentence, then token.

Code blocks and tables stay whole unless one is larger than a chunk by itself. Overlap uses
whole units from the end of the previous chunk, so it can be smaller than configured.
"""

import re
from dataclasses import dataclass
from typing import Literal

from ragbench.chunking.base import Chunker
from ragbench.chunking.fixed import token_windows
from ragbench.chunking.tokenizer import TokenOffsets
from ragbench.ingestion.cleaner import split_fences
from ragbench.ingestion.models import Document

_HEADING = re.compile(r"#{1,6}\s")
_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")
_LINE_END = re.compile(r"\n+")

BlockKind = Literal["heading", "prose", "code", "table"]


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    tokens: int
    kind: BlockKind


class RecursiveChunker(Chunker):
    def _spans(self, document: Document, offsets: TokenOffsets) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        current: list[_Unit] = []

        def flush() -> None:
            if current:
                spans.append((current[0].start, current[-1].end))

        for unit in self._units(document.content, offsets):
            if unit.kind == "heading" and any(u.kind != "heading" for u in current):
                # A new section starts a new chunk. Consecutive headings stay together, so a
                # heading is never separated from the first text below it.
                flush()
                current = []
            elif current and _total(current) + unit.tokens > self.chunk_size:
                flush()
                current = self._overlap_units(current, unit)
            current.append(unit)
        flush()
        return spans

    def _overlap_units(self, previous: list[_Unit], next_unit: _Unit) -> list[_Unit]:
        carried: list[_Unit] = []
        budget = min(self.overlap, self.chunk_size - next_unit.tokens)
        for unit in reversed(previous):
            if unit.tokens > budget:
                break
            carried.insert(0, unit)
            budget -= unit.tokens
        return carried

    def _units(self, content: str, offsets: TokenOffsets) -> list[_Unit]:
        units: list[_Unit] = []
        for start, end, kind in _blocks(content):
            tokens = offsets.count(start, end)
            if tokens == 0:
                continue
            if tokens <= self.chunk_size:
                units.append(_Unit(start, end, tokens, kind))
            else:
                units.extend(self._split_oversized(content, start, end, kind, offsets))
        return units

    def _split_oversized(
        self, content: str, start: int, end: int, kind: BlockKind, offsets: TokenOffsets
    ) -> list[_Unit]:
        separator = _SENTENCE_END if kind in ("prose", "heading") else _LINE_END
        units: list[_Unit] = []
        for piece_start, piece_end in _split_on(content, start, end, separator):
            tokens = offsets.count(piece_start, piece_end)
            if tokens == 0:
                continue
            if tokens <= self.chunk_size:
                units.append(_Unit(piece_start, piece_end, tokens, kind))
            else:
                units.extend(self._hard_split(piece_start, piece_end, kind, offsets))
        return units

    def _hard_split(
        self, start: int, end: int, kind: BlockKind, offsets: TokenOffsets
    ) -> list[_Unit]:
        """Last resort for a single line or sentence longer than a chunk."""
        first = offsets.count(0, start)
        last = first + offsets.count(start, end)
        windows = token_windows(offsets, first, last, self.chunk_size, 0)
        return [_Unit(s, e, offsets.count(s, e), kind) for s, e in windows]


def _total(units: list[_Unit]) -> int:
    return sum(unit.tokens for unit in units)


def _split_on(content: str, start: int, end: int, separator: re.Pattern) -> list[tuple[int, int]]:
    pieces = []
    position = start
    for match in separator.finditer(content, start, end):
        pieces.append((position, match.start()))
        position = match.end()
    pieces.append((position, end))
    return [(s, e) for s, e in pieces if e > s]


def _blocks(content: str) -> list[tuple[int, int, BlockKind]]:
    """Split a document into headings, paragraphs, tables and code blocks, with offsets."""
    blocks: list[tuple[int, int, BlockKind]] = []
    offset = 0
    for is_code, segment in split_fences(content):
        if is_code:
            blocks.append((offset, offset + len(segment.rstrip()), "code"))
        else:
            blocks.extend(_prose_blocks(segment, offset))
        offset += len(segment)
    return blocks


def _prose_blocks(segment: str, offset: int) -> list[tuple[int, int, BlockKind]]:
    blocks: list[tuple[int, int, BlockKind]] = []
    block_start: int | None = None
    block_end = 0
    block_kind: BlockKind = "prose"

    def close() -> None:
        nonlocal block_start
        if block_start is not None:
            blocks.append((block_start, block_end, block_kind))
        block_start = None

    position = offset
    for line in segment.splitlines(keepends=True):
        stripped = line.strip()
        line_end = position + len(line.rstrip())
        if not stripped:
            close()
        elif _HEADING.match(stripped):
            close()
            blocks.append((position, line_end, "heading"))
        else:
            kind: BlockKind = "table" if stripped.startswith("|") else "prose"
            if block_start is not None and kind != block_kind:
                close()
            if block_start is None:
                block_start = position
                block_kind = kind
            block_end = line_end
        position += len(line)
    close()
    return blocks
