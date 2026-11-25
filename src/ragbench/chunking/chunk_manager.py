"""Build and cache one chunked representation of a corpus per configuration."""

import json
from pathlib import Path
from statistics import median

from pydantic import BaseModel

from ragbench.chunking.base import Chunker
from ragbench.chunking.fixed import FixedChunker
from ragbench.chunking.models import Chunk, ChunkingConfig
from ragbench.chunking.recursive import RecursiveChunker
from ragbench.chunking.tokenizer import HuggingFaceTokenCounter, TokenCounter
from ragbench.ingestion.models import Document

_CHUNKERS: dict[str, type[Chunker]] = {"fixed": FixedChunker, "recursive": RecursiveChunker}


class ChunkingStats(BaseModel):
    config: ChunkingConfig
    config_id: str
    corpus_hash: str
    documents: int
    chunks: int
    tokens_min: int
    tokens_median: float
    tokens_max: int
    chars_total: int


def make_chunker(config: ChunkingConfig, counter: TokenCounter | None = None) -> Chunker:
    counter = counter or HuggingFaceTokenCounter(config.tokenizer, config.tokenizer_revision)
    return _CHUNKERS[config.strategy](config.chunk_size, config.overlap, counter)


def build_chunks(
    documents: list[Document],
    config: ChunkingConfig,
    corpus_hash: str,
    chunks_root: Path,
    counter: TokenCounter | None = None,
) -> ChunkingStats:
    """Chunk a corpus into `chunks_root/<corpus_hash>/<config_id>`, reusing a previous build."""
    output_dir = chunks_directory(chunks_root, corpus_hash, config)
    stats_path = output_dir / "stats.json"
    if stats_path.exists():
        return ChunkingStats.model_validate_json(stats_path.read_text(encoding="utf-8"))

    chunker = make_chunker(config, counter)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_counts: list[int] = []
    chars_total = 0
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for document in documents:
            for chunk in chunker.chunk(document):
                out.write(chunk.model_dump_json() + "\n")
                token_counts.append(chunk.token_count)
                chars_total += len(chunk.text)

    if not token_counts:
        raise ValueError(f"Config {config.config_id} produced no chunks")
    stats = ChunkingStats(
        config=config,
        config_id=config.config_id,
        corpus_hash=corpus_hash,
        documents=len(documents),
        chunks=len(token_counts),
        tokens_min=min(token_counts),
        tokens_median=median(token_counts),
        tokens_max=max(token_counts),
        chars_total=chars_total,
    )
    # Written last, so an interrupted build is never mistaken for a finished one.
    stats_path.write_text(stats.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return stats


def chunks_directory(chunks_root: Path, corpus_hash: str, config: ChunkingConfig) -> Path:
    return chunks_root / corpus_hash.removeprefix("sha256-")[:12] / config.config_id


def load_chunks(directory: Path) -> list[Chunk]:
    with (directory / "chunks.jsonl").open(encoding="utf-8") as lines:
        return [Chunk.model_validate(json.loads(line)) for line in lines]
