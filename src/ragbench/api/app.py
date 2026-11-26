"""FastAPI application: conventional RAG queries, evaluation jobs and stored benchmark results.

Run locally with `python -m ragbench.api`. Settings come from environment variables, listed in
`ApiSettings`.
"""

import asyncio
import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import ragbench

from .jobs import JobQueue
from .schemas import (
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    EvaluationRequest,
    ExperimentSummary,
    HealthResponse,
    Job,
    QueryRequest,
    QueryResponse,
)
from .service import ApiError, RagService

log = logging.getLogger("ragbench.api")
EXECUTION_HEADER = "X-Execution-ID"
API_KEY_HEADER = "X-API-Key"
VERSION = getattr(ragbench, "__version__", "0.1.0")


@dataclass(frozen=True)
class ApiSettings:
    # When set, every endpoint except /health needs this key in the X-API-Key header.
    api_key: str | None = None
    # A query that takes longer gets a 504. The model may still finish in the background.
    query_timeout_s: float = 180.0
    # Queries share one GPU and one Ollama server, so they run a few at a time.
    max_concurrent_queries: int = 1
    max_pending_jobs: int = 5
    cors_origins: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            api_key=os.environ.get("RAGBENCH_API_KEY") or None,
            query_timeout_s=float(os.environ.get("RAGBENCH_QUERY_TIMEOUT_S", 180)),
            max_concurrent_queries=int(os.environ.get("RAGBENCH_MAX_CONCURRENT_QUERIES", 1)),
            max_pending_jobs=int(os.environ.get("RAGBENCH_MAX_PENDING_JOBS", 5)),
        )


def _error(status: int, code: str, message: str, execution_id: str, details=None) -> JSONResponse:
    body = ErrorResponse.model_validate(
        {
            "error": {"code": code, "message": message, "details": details},
            "execution_id": execution_id,
        }
    )
    return JSONResponse(
        body.model_dump(), status_code=status, headers={EXECUTION_HEADER: execution_id}
    )


def _execution_id(request: Request) -> str:
    return getattr(request.state, "execution_id", "unknown")


def create_app(service: RagService, settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    jobs = JobQueue(service.evaluate, settings.max_pending_jobs)
    query_slots = asyncio.Semaphore(settings.max_concurrent_queries)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()

    errors = {status: {"model": ErrorResponse} for status in (401, 404, 422, 429, 502, 503, 504)}
    app = FastAPI(
        title="Advanced RAG-Bench",
        version=VERSION,
        description=__doc__,
        lifespan=lifespan,
        responses=errors,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        request.state.execution_id = execution_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers[EXECUTION_HEADER] = execution_id
        log.info(
            "%s %s %s %d %.0fms",
            execution_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message, _execution_id(request), exc.details)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        details = [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return _error(
            422, "invalid_request", "The request is not valid.", _execution_id(request), details
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail), _execution_id(request))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        execution_id = _execution_id(request)
        log.exception("%s failed", execution_id)
        # Details stay in the server log, so internals never leak to clients.
        return _error(500, "internal_error", "Something went wrong.", execution_id)

    def require_key(request: Request) -> None:
        if settings.api_key is None:
            return
        given = request.headers.get(API_KEY_HEADER, "")
        if not hmac.compare_digest(given.encode(), settings.api_key.encode()):
            raise ApiError(401, "unauthorized", f"A valid {API_KEY_HEADER} header is required.")

    secured = [Depends(require_key)]

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    async def health() -> HealthResponse:
        ollama = await run_in_threadpool(service.ollama_status)
        return HealthResponse(
            status="ok" if ollama.startswith("ok") else "degraded",
            version=VERSION,
            ollama=ollama,
            configs=sorted(service.configs),
            default_config=service.default_config,
            models=service.models,
        )

    @app.post("/query", response_model=QueryResponse, dependencies=secured, tags=["rag"])
    async def query(body: QueryRequest, request: Request) -> QueryResponse:
        """Answer a question from the documentation, with citations, or abstain."""
        execution_id = _execution_id(request)
        try:
            async with asyncio.timeout(settings.query_timeout_s):
                async with query_slots:
                    return await run_in_threadpool(service.query, body, execution_id)
        except TimeoutError as exc:
            raise ApiError(
                504, "timeout", f"The query took longer than {settings.query_timeout_s:.0f}s."
            ) from exc

    @app.post(
        "/evaluate", response_model=Job, status_code=202, dependencies=secured, tags=["evaluation"]
    )
    async def evaluate(body: EvaluationRequest) -> Job:
        """Start a retrieval evaluation on a dataset split. Poll /jobs/{job_id} for the result."""
        service.config(body.config)
        return jobs.submit(body)

    @app.get("/jobs", response_model=list[Job], dependencies=secured, tags=["evaluation"])
    async def list_jobs() -> list[Job]:
        return jobs.list()

    @app.get("/jobs/{job_id}", response_model=Job, dependencies=secured, tags=["evaluation"])
    async def get_job(job_id: str) -> Job:
        return jobs.get(job_id)

    @app.get(
        "/experiments",
        response_model=list[ExperimentSummary],
        dependencies=secured,
        tags=["results"],
    )
    async def list_experiments(split: str | None = None) -> list[ExperimentSummary]:
        """Stored retrieval runs, as produced by scripts/run_retrieval_sweep.py."""
        return await run_in_threadpool(service.experiments, split)

    @app.get("/experiments/{experiment_id}", dependencies=secured, tags=["results"])
    async def get_experiment(experiment_id: str, questions: bool = False) -> dict:
        record, results = await run_in_threadpool(service.experiment, experiment_id)
        body = {"experiment_id": experiment_id, "record": record.model_dump(mode="json")}
        if questions:
            body["questions"] = [result.model_dump() for result in results]
        return body

    @app.post(
        "/experiments/compare",
        response_model=CompareResponse,
        dependencies=secured,
        tags=["results"],
    )
    async def compare(body: CompareRequest) -> CompareResponse:
        """Paired bootstrap comparison of two stored runs on the same questions."""
        return await run_in_threadpool(service.compare, body)

    @app.get("/reports", response_model=list[str], dependencies=secured, tags=["results"])
    async def list_reports() -> list[str]:
        return service.reports()

    @app.get(
        "/reports/{report_id}",
        response_class=PlainTextResponse,
        dependencies=secured,
        tags=["results"],
    )
    async def get_report(report_id: str) -> PlainTextResponse:
        return PlainTextResponse(service.report(report_id), media_type="text/markdown")

    return app
