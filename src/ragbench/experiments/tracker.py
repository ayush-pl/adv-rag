"""Run records: everything needed to tell exactly what produced a result."""

import json
import platform
import subprocess
from pathlib import Path

from pydantic import BaseModel


class Environment(BaseModel):
    git_commit: str | None
    # True when the code had uncommitted changes, so the commit alone can't reproduce the run.
    git_dirty: bool
    python: str
    platform: str
    torch: str | None
    cuda_device: str | None


def capture_environment(repo: Path = Path(".")) -> Environment:
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    torch_version = device = None
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        # A broken PyTorch install fails in many ways: application control blocking its DLLs
        # (OSError), or missing CUDA libraries (ValueError). The record then shows no torch.
        pass
    return Environment(
        git_commit=commit,
        git_dirty=bool(status),
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch_version,
        cuda_device=device,
    )


def read_lock(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(repo: Path, *args: str) -> str | None:
    # From WSL, the repository on the Windows drive has a different owner, and git refuses to
    # read it. Trusting it for this one read-only call avoids changing anyone's git config.
    trust = f"safe.directory={repo.resolve().as_posix()}"
    try:
        result = subprocess.run(
            ["git", "-c", trust, *args], cwd=repo, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
