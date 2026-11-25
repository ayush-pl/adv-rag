"""Schemas for corpus manifests and normalized documents."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    """One pinned documentation source. Fetch fields depend on `fetch`."""

    id: str
    name: str
    version: str
    format: Literal["markdown", "html"]
    url_base: str
    content_dir: str
    exclude: list[str] = Field(default_factory=list)

    fetch: Literal["git", "tarball"]
    repo: str | None = None
    commit: str | None = None
    sparse_paths: list[str] = Field(default_factory=list)
    url: str | None = None
    sha256: str | None = None

    # Directory of example files that Kubernetes `code_sample` shortcodes point at.
    examples_dir: str | None = None


class CorpusManifest(BaseModel):
    corpus_version: str
    snapshot_date: str
    sources: list[SourceConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> "CorpusManifest":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class Section(BaseModel):
    heading: str
    level: int
    char_start: int


class Document(BaseModel):
    document_id: str
    source: str
    source_url: str
    title: str
    version: str
    content: str
    content_hash: str
    sections: list[Section]
    # The manifest's snapshot date, so re-ingesting a snapshot gives identical output.
    ingestion_date: str


class IngestionIssue(BaseModel):
    source_id: str
    path: str
    kind: Literal["failed", "skipped", "duplicate"]
    reason: str
