"""Fetch pinned documentation snapshots into the raw data directory."""

import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path

import httpx

from ragbench.ingestion.models import SourceConfig


def fetch_source(source: SourceConfig, raw_root: Path) -> Path:
    """Fetch a source if needed and return its root directory. Verifies the pinned version."""
    source_root = raw_root / source.id
    if source.fetch == "git":
        _fetch_git(source, source_root)
    else:
        _fetch_tarball(source, source_root)
    return source_root


def _fetch_git(source: SourceConfig, root: Path) -> None:
    if (root / ".git").exists() and _git(root, "rev-parse", "HEAD") == source.commit:
        return

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _git(root, "init", "--quiet")
    _git(root, "remote", "add", "origin", source.repo)
    _git(root, "fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", source.commit)
    _git(root, "sparse-checkout", "set", "--cone", *source.sparse_paths)
    _git(root, "checkout", "--quiet", "FETCH_HEAD")

    actual = _git(root, "rev-parse", "HEAD")
    if actual != source.commit:
        raise RuntimeError(f"{source.id}: expected commit {source.commit}, got {actual}")


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _fetch_tarball(source: SourceConfig, root: Path) -> None:
    if (root / source.content_dir).exists():
        return

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "archive.tar.bz2"
    with httpx.stream("GET", source.url, follow_redirects=True, timeout=300.0) as response:
        response.raise_for_status()
        with archive.open("wb") as out:
            for block in response.iter_bytes():
                out.write(block)

    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != source.sha256:
        archive.unlink()
        raise RuntimeError(f"{source.id}: expected sha256 {source.sha256}, got {actual}")

    extract_content(archive, root, source.content_dir)
    archive.unlink()


def extract_content(archive: Path, root: Path, content_dir: str) -> None:
    """Extract `content_dir` from a release tarball whose first path component is a folder."""
    with tarfile.open(archive) as tar:
        wanted = [
            member
            for member in tar.getmembers()
            if member.isfile() and member.name.partition("/")[2].startswith(content_dir + "/")
        ]
        if not wanted:
            raise RuntimeError(f"{archive.name} has no files under {content_dir}")
        for member in wanted:
            member.name = member.name.partition("/")[2]
        tar.extractall(root, members=wanted, filter="data")
