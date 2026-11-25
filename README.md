# Advanced RAG-Bench

An evaluation-first benchmarking framework for conventional and agentic RAG systems.

The full specification is in [docs/spec](docs/spec/).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a running [Ollama](https://ollama.com/) server.

```bash
uv sync --extra ml
uv run python scripts/check_env.py
uv run pytest
```

Omit `--extra ml` to skip PyTorch and model inference. Unit tests do not need it.
