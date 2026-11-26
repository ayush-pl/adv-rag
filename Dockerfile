# The query API in a container. CPU PyTorch only: published performance numbers come from
# native GPU runs, not from this image.
#
# The pipeline outputs (chunks, indexes, runs) are too large to bake in, so data/ is mounted
# at runtime. See docker-compose.yml.

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    RAGBENCH_ROOT=/app \
    OLLAMA_HOST=http://host.docker.internal:11434

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ml --extra cpu --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ml --extra cpu

COPY configs ./configs
COPY reports ./reports

RUN useradd --create-home --uid 1000 ragbench \
    && mkdir -p /cache/huggingface /app/data \
    && chown -R ragbench /cache
USER ragbench

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "ragbench.api", "--host", "0.0.0.0", "--port", "8000"]
