# Advanced RAG-Bench

An evaluation-first benchmarking framework for conventional and agentic RAG systems.

Results so far are in [reports](reports/). A full README with reproduction steps is in progress.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a running [Ollama](https://ollama.com/) server.

```bash
uv sync --extra ml
uv run python scripts/check_env.py
uv run pytest
```

Omit `--extra ml` to skip PyTorch and model inference. Unit tests do not need it.

## Regression gate

CI runs lint, the unit tests and a regression gate on every push and pull request. The gate
reruns the real retrieval and evaluation code on a small committed fixture, so it needs no GPU,
models or corpus:

```bash
uv run python scripts/regression_gate.py check
```

It fails when a retrieval metric drops beyond its tolerance with a 95% paired bootstrap
interval below zero, and warns when a drop is not consistent across questions.

## License

Code is under the [MIT License](LICENSE). The documentation excerpts in
`tests/regression/fixture` keep their own licenses, listed in its `manifest.json`.
