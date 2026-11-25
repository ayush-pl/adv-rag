#!/usr/bin/env bash
# Run a command in the WSL GPU environment, from the project folder on the Windows drive.
#
# Windows Smart App Control blocks PyTorch's DLLs, so model work runs in WSL instead.
# The Linux virtual environment lives in the Linux home folder, apart from the Windows .venv.
#
# Usage, from Windows:  wsl.exe -e bash scripts/wsl.sh python scripts/run_retrieval_sweep.py
set -euo pipefail

cd "$(dirname "$0")/.."
# uv's installer puts it here, which a non-login shell started by wsl.exe doesn't have on PATH.
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/advanced-rag-bench}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
# Reach the Ollama server running on Windows. With mirrored networking this is localhost.
export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

exec uv run --extra ml --extra cu130 "$@"
