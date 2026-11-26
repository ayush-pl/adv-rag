"""Serve the API: `python -m ragbench.api [--host HOST] [--port PORT]`.

Environment: OLLAMA_HOST (default http://localhost:11434), RAGBENCH_ROOT (the project folder,
default the current folder), plus the RAGBENCH_* settings read by ApiSettings.from_env.
"""

import argparse
import logging
import os
from pathlib import Path

import uvicorn

from .app import ApiSettings, create_app
from .service import RagService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    service = RagService.from_project(
        Path(os.environ.get("RAGBENCH_ROOT", ".")),
        ollama_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
    uvicorn.run(create_app(service, ApiSettings.from_env()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
