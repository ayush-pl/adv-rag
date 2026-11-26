"""Background jobs for evaluations, which take too long to hold a request open.

Jobs run one at a time, because they share one GPU with queries, and are kept in memory:
a restart forgets them, which is acceptable for a single-user benchmark service.
"""

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from .schemas import ErrorBody, EvaluationRequest, EvaluationResult, Job
from .service import ApiError


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobQueue:
    def __init__(
        self, run: Callable[[EvaluationRequest], EvaluationResult], max_pending: int = 5
    ) -> None:
        self._run = run
        self._max_pending = max_pending
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evaluation")

    def submit(self, request: EvaluationRequest) -> Job:
        with self._lock:
            pending = sum(job.status in ("queued", "running") for job in self._jobs.values())
            if pending >= self._max_pending:
                raise ApiError(
                    429, "too_many_jobs", f"{pending} evaluations are already waiting. Retry later."
                )
            job = Job(
                job_id=f"job-{uuid.uuid4().hex[:12]}",
                status="queued",
                request=request,
                created_at=_now(),
            )
            self._jobs[job.job_id] = job
        self._executor.submit(self._execute, job.job_id)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ApiError(404, "unknown_job", f"No job {job_id!r}.")
        return job

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _update(self, job_id: str, **changes) -> Job:
        with self._lock:
            job = self._jobs[job_id] = self._jobs[job_id].model_copy(update=changes)
        return job

    def _execute(self, job_id: str) -> None:
        job = self._update(job_id, status="running", started_at=_now())
        try:
            result = self._run(job.request)
        except ApiError as exc:
            error = ErrorBody(code=exc.code, message=exc.message, details=exc.details)
            self._update(job_id, status="failed", finished_at=_now(), error=error)
        except Exception as exc:  # noqa: BLE001
            # The job records the failure; the worker thread must survive to run later jobs.
            error = ErrorBody(code="evaluation_failed", message=f"{type(exc).__name__}: {exc}")
            self._update(job_id, status="failed", finished_at=_now(), error=error)
        else:
            self._update(job_id, status="succeeded", finished_at=_now(), result=result)
