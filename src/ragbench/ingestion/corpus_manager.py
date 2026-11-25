"""Build a versioned corpus: load, dedupe and record every page's outcome."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from ragbench.ingestion.loaders.base import DocumentLoader, SkipDocument
from ragbench.ingestion.loaders.html_loader import HtmlLoader
from ragbench.ingestion.loaders.markdown_loader import MarkdownLoader
from ragbench.ingestion.models import CorpusManifest, Document, IngestionIssue, SourceConfig

_LOADERS: dict[str, type[DocumentLoader]] = {"markdown": MarkdownLoader, "html": HtmlLoader}


class SourceStats(BaseModel):
    source_id: str
    version: str
    pinned: str
    documents: int = 0
    characters: int = 0
    failed: int = 0
    skipped: int = 0
    duplicates: int = 0
    dropped_markup: dict[str, int] = {}


class CorpusLock(BaseModel):
    """Identifies a corpus build. Contains no timestamps, so rebuilds compare byte for byte."""

    corpus_version: str
    corpus_hash: str
    snapshot_date: str
    documents: int
    sources: list[SourceStats]


def ingest_source(
    source: SourceConfig, source_root: Path, ingestion_date: str
) -> tuple[list[Document], list[IngestionIssue], SourceStats]:
    loader = _LOADERS[source.format](source, source_root, ingestion_date)
    documents: dict[str, Document] = {}
    first_with_hash: dict[str, str] = {}
    issues: list[IngestionIssue] = []
    stats = SourceStats(
        source_id=source.id, version=source.version, pinned=source.commit or source.sha256 or ""
    )

    def record(path: Path, kind: str, reason: str) -> None:
        relative = path.relative_to(loader.content_root).as_posix()
        issues.append(IngestionIssue(source_id=source.id, path=relative, kind=kind, reason=reason))

    for path in loader.discover():
        try:
            document = loader.load(path)
        except SkipDocument as exc:
            record(path, "skipped", str(exc))
            stats.skipped += 1
            continue
        except Exception as exc:  # One bad page must not stop a corpus build; it is logged.
            record(path, "failed", f"{type(exc).__name__}: {exc}")
            stats.failed += 1
            continue

        if document.document_id in documents:
            record(path, "failed", f"duplicate document id {document.document_id}")
            stats.failed += 1
        elif document.content_hash in first_with_hash:
            record(path, "duplicate", f"same content as {first_with_hash[document.content_hash]}")
            stats.duplicates += 1
        else:
            documents[document.document_id] = document
            first_with_hash[document.content_hash] = document.document_id
            stats.documents += 1
            stats.characters += len(document.content)

    stats.dropped_markup = dict(sorted(Counter(loader.dropped_markup).items()))
    return list(documents.values()), issues, stats


def build_corpus(
    manifest: CorpusManifest, source_roots: dict[str, Path], output_dir: Path, lock_path: Path
) -> CorpusLock:
    """Ingest all sources and write the documents, the issue log and the lock file.

    Duplicates are removed within a source only. PostgreSQL 16 and 17 share many identical
    pages, and each version must stay retrievable for version-specific questions.
    """
    documents: list[Document] = []
    issues: list[IngestionIssue] = []
    stats: list[SourceStats] = []
    for source in manifest.sources:
        source_docs, source_issues, source_stats = ingest_source(
            source, source_roots[source.id], manifest.snapshot_date
        )
        if not source_docs:
            raise ValueError(f"Source {source.id} produced no documents")
        documents.extend(source_docs)
        issues.extend(source_issues)
        stats.append(source_stats)

    documents.sort(key=lambda d: d.document_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "documents.jsonl", documents)
    _write_jsonl(output_dir / "issues.jsonl", issues)

    lock = CorpusLock(
        corpus_version=manifest.corpus_version,
        corpus_hash=_corpus_hash(documents),
        snapshot_date=manifest.snapshot_date,
        documents=len(documents),
        sources=stats,
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(lock.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return lock


def load_documents(corpus_dir: Path) -> list[Document]:
    with (corpus_dir / "documents.jsonl").open(encoding="utf-8") as lines:
        return [Document.model_validate(json.loads(line)) for line in lines]


def _corpus_hash(documents: list[Document]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(f"{document.document_id}\0{document.content_hash}\n".encode())
    return "sha256-" + digest.hexdigest()


def _write_jsonl(path: Path, records: list[BaseModel]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(record.model_dump_json() + "\n")
