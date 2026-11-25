"""Text normalization shared by all loaders. Fenced code blocks are never altered."""

import hashlib
import re
import unicodedata
from collections.abc import Callable

from ragbench.ingestion.models import Section

_FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def split_fences(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_code, segment) pairs. An unclosed fence runs to the end of the text."""
    segments: list[tuple[bool, str]] = []
    current: list[str] = []
    fence: str | None = None

    for line in text.splitlines(keepends=True):
        match = _FENCE_OPEN.match(line)
        if fence is None and match:
            if current:
                segments.append((False, "".join(current)))
            current = [line]
            fence = match.group(1)
        elif fence is not None and _closes_fence(line, fence):
            current.append(line)
            segments.append((True, "".join(current)))
            current = []
            fence = None
        else:
            current.append(line)

    if current:
        segments.append((fence is not None, "".join(current)))
    return segments


def _closes_fence(line: str, fence: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(fence) and stripped[0] == fence[0] and set(stripped) == {fence[0]}


def map_prose(text: str, transform: Callable[[str], str]) -> str:
    """Apply `transform` to everything outside fenced code blocks."""
    return "".join(
        segment if is_code else transform(segment) for is_code, segment in split_fences(text)
    )


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace, so the same page always hashes the same."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))

    def tidy(prose: str) -> str:
        prose = re.sub(r"[ \t]+$", "", prose, flags=re.MULTILINE)
        prose = re.sub(r"^\s*\|.*$", _tidy_table_row, prose, flags=re.MULTILINE)
        return re.sub(r"\n{3,}", "\n\n", prose)

    return map_prose(text, tidy).strip() + "\n"


def _tidy_table_row(match: re.Match[str]) -> str:
    """Remove column-alignment padding, which only adds noise for retrieval."""
    return re.sub(r"-{4,}", "---", re.sub(r"[ \t]{2,}", " ", match.group(0)))


def extract_sections(content: str) -> list[Section]:
    """Find headings outside code blocks, with their character offsets in `content`."""
    sections: list[Section] = []
    offset = 0
    for is_code, segment in split_fences(content):
        if not is_code:
            line_start = offset
            for line in segment.splitlines(keepends=True):
                match = _HEADING.match(line)
                if match:
                    sections.append(
                        Section(
                            heading=match.group(2),
                            level=len(match.group(1)),
                            char_start=line_start,
                        )
                    )
                line_start += len(line)
        offset += len(segment)
    return sections


def content_hash(content: str) -> str:
    return "sha256-" + hashlib.sha256(content.encode("utf-8")).hexdigest()
