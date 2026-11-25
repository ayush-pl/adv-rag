"""Common interface for source-format loaders."""

from abc import ABC, abstractmethod
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path

from ragbench.ingestion.cleaner import content_hash, extract_sections, normalize_text
from ragbench.ingestion.models import Document, SourceConfig

# Pages with less text than this are section stubs, such as a title-only Hugo _index.md.
MIN_CONTENT_CHARS = 200


class SkipDocument(Exception):
    """The page is valid but has no content worth indexing."""


class DocumentLoader(ABC):
    suffix: str

    def __init__(self, source: SourceConfig, source_root: Path, ingestion_date: str) -> None:
        self.source = source
        self.source_root = source_root
        self.content_root = source_root / source.content_dir
        self.ingestion_date = ingestion_date
        self.dropped_markup: Counter[str] = Counter()

    def discover(self) -> list[Path]:
        """Return included pages in a stable order."""
        paths = []
        for path in self.content_root.rglob(f"*{self.suffix}"):
            relative = path.relative_to(self.content_root).as_posix()
            if not any(fnmatch(relative, pattern) for pattern in self.source.exclude):
                paths.append(path)
        return sorted(paths, key=lambda p: p.relative_to(self.content_root).as_posix())

    @abstractmethod
    def load(self, path: Path) -> Document:
        """Parse one page. Raises SkipDocument for stubs, and other exceptions on failure."""

    def _build_document(self, slug: str, url_path: str, title: str, content: str) -> Document:
        content = normalize_text(content)
        if len(content) < MIN_CONTENT_CHARS:
            raise SkipDocument(f"only {len(content)} characters of content")
        return Document(
            document_id=f"{self.source.id}/{slug}",
            source=self.source.name,
            source_url=self.source.url_base + url_path,
            title=title,
            version=self.source.version,
            content=content,
            content_hash=content_hash(content),
            sections=extract_sections(content),
            ingestion_date=self.ingestion_date,
        )
