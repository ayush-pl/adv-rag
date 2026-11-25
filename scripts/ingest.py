"""Fetch the pinned snapshots in a corpus manifest and build the normalized corpus."""

import argparse
from pathlib import Path

from ragbench.ingestion.corpus_manager import build_corpus
from ragbench.ingestion.models import CorpusManifest
from ragbench.ingestion.snapshot import fetch_source

DATA_DIR = Path("data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-version", default="v1")
    args = parser.parse_args()

    manifest_path = DATA_DIR / "manifests" / f"corpus_{args.corpus_version}.yaml"
    manifest = CorpusManifest.from_yaml(manifest_path)

    source_roots = {}
    for source in manifest.sources:
        print(f"Fetching {source.id}...")
        source_roots[source.id] = fetch_source(source, DATA_DIR / "raw")

    lock = build_corpus(
        manifest,
        source_roots,
        output_dir=DATA_DIR / "processed" / args.corpus_version,
        lock_path=DATA_DIR / "manifests" / f"corpus_{args.corpus_version}.lock.json",
    )

    print(f"\nCorpus {lock.corpus_version}: {lock.documents} documents, {lock.corpus_hash}")
    for stats in lock.sources:
        print(
            f"  {stats.source_id}: {stats.documents} docs, {stats.characters:,} chars, "
            f"{stats.failed} failed, {stats.skipped} skipped, {stats.duplicates} duplicates"
        )


if __name__ == "__main__":
    main()
