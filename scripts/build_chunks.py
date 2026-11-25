"""Build the chunked representations of a corpus for the configs in a YAML file."""

import argparse
import json
from pathlib import Path

import yaml

from ragbench.chunking.chunk_manager import build_chunks
from ragbench.chunking.models import ChunkingConfig
from ragbench.ingestion.corpus_manager import load_documents

DATA_DIR = Path("data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-version", default="v1")
    parser.add_argument("--configs", type=Path, default=Path("configs/chunking.yaml"))
    args = parser.parse_args()

    spec = yaml.safe_load(args.configs.read_text(encoding="utf-8"))
    lock_path = DATA_DIR / "manifests" / f"corpus_{args.corpus_version}.lock.json"
    corpus_hash = json.loads(lock_path.read_text(encoding="utf-8"))["corpus_hash"]
    documents = load_documents(DATA_DIR / "processed" / args.corpus_version)

    for variant in spec["variants"]:
        config = ChunkingConfig(**spec["tokenizer"], **variant)
        stats = build_chunks(documents, config, corpus_hash, DATA_DIR / "chunks")
        print(
            f"{stats.config_id}: {stats.chunks:,} chunks, tokens "
            f"min {stats.tokens_min} / median {stats.tokens_median:.0f} / max {stats.tokens_max}"
        )


if __name__ == "__main__":
    main()
