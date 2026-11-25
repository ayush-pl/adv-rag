"""Embed every chunking variant with every configured embedding model."""

import argparse
import json
from pathlib import Path

import yaml

from ragbench.chunking.chunk_manager import chunks_directory, load_chunks
from ragbench.chunking.models import ChunkingConfig
from ragbench.embeddings.base import EmbedderConfig
from ragbench.embeddings.index_store import build_vectors, index_directory
from ragbench.embeddings.sentence_transformer import SentenceTransformerEmbedder

DATA_DIR = Path("data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-version", default="v1")
    parser.add_argument("--chunking", type=Path, default=Path("configs/chunking.yaml"))
    parser.add_argument("--embeddings", type=Path, default=Path("configs/embeddings.yaml"))
    args = parser.parse_args()

    chunking = yaml.safe_load(args.chunking.read_text(encoding="utf-8"))
    embeddings = yaml.safe_load(args.embeddings.read_text(encoding="utf-8"))
    lock_path = DATA_DIR / "manifests" / f"corpus_{args.corpus_version}.lock.json"
    corpus_hash = json.loads(lock_path.read_text(encoding="utf-8"))["corpus_hash"]

    for entry in embeddings["embedders"]:
        config = EmbedderConfig(**entry)
        embedder = None
        for variant in chunking["variants"]:
            chunking_config = ChunkingConfig(**chunking["tokenizer"], **variant)
            chunks_dir = chunks_directory(DATA_DIR / "chunks", corpus_hash, chunking_config)
            if (index_directory(DATA_DIR / "indexes", chunks_dir, config) / "stats.json").exists():
                print(f"{config.model} / {chunking_config.config_id}: already built")
                continue

            embedder = embedder or SentenceTransformerEmbedder(config)
            stats = build_vectors(
                load_chunks(chunks_dir),
                embedder,
                config,
                chunks_dir,
                DATA_DIR / "indexes",
                device=embedder.device,
            )
            print(
                f"{config.model} / {chunking_config.config_id}: {stats.chunks:,} chunks in "
                f"{stats.embedding_seconds:.0f}s on {stats.device}, "
                f"{stats.truncation.truncated_rate:.1%} truncated "
                f"(limit {stats.truncation.max_tokens} tokens)",
                flush=True,
            )


if __name__ == "__main__":
    main()
