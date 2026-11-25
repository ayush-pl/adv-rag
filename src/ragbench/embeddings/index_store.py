"""Build and cache chunk vectors, with the measurements FR-024 to FR-028 require."""

import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from ragbench.chunking.models import Chunk
from ragbench.embeddings.base import Embedder, EmbedderConfig, TruncationStats


class IndexStats(BaseModel):
    embedder: EmbedderConfig
    chunks_dir: str
    chunks: int
    dimension: int
    device: str
    embedding_seconds: float
    chunks_per_second: float
    vector_bytes: int
    truncation: TruncationStats


def index_directory(indexes_root: Path, chunks_dir: Path, config: EmbedderConfig) -> Path:
    corpus_key, chunking_id = chunks_dir.parts[-2:]
    return indexes_root / corpus_key / chunking_id / config.slug


def build_vectors(
    chunks: list[Chunk],
    embedder: Embedder,
    config: EmbedderConfig,
    chunks_dir: Path,
    indexes_root: Path,
    device: str = "unknown",
) -> IndexStats:
    """Embed chunks unless a finished build exists. Vectors follow the order of `chunks`."""
    output_dir = index_directory(indexes_root, chunks_dir, config)
    stats_path = output_dir / "stats.json"
    if stats_path.exists():
        return IndexStats.model_validate_json(stats_path.read_text(encoding="utf-8"))

    texts = [chunk.text for chunk in chunks]
    truncation = embedder.truncation_stats(texts)
    started = time.perf_counter()
    vectors = embedder.embed_documents(texts)
    elapsed = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "vectors.npy", vectors)
    stats = IndexStats(
        embedder=config,
        chunks_dir=chunks_dir.as_posix(),
        chunks=len(chunks),
        dimension=embedder.dimension,
        device=device,
        embedding_seconds=round(elapsed, 2),
        chunks_per_second=round(len(chunks) / elapsed, 1) if elapsed else 0.0,
        vector_bytes=int(vectors.nbytes),
        truncation=truncation,
    )
    # Written last, so an interrupted build is never mistaken for a finished one.
    stats_path.write_text(stats.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return stats


def load_vectors(directory: Path) -> np.ndarray:
    return np.load(directory / "vectors.npy")
