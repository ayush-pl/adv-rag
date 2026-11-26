"""Request and response schemas. Every field is validated before any work starts."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# The test split is kept for final reports, made by scripts, so the API can't be used to tune
# against it.
EvaluationSplit = Literal["development", "calibration"]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody
    execution_id: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    # A retrieval configuration name from configs/retrieval_sweep.yaml. Defaults to the
    # service's configured default.
    config: str | None = None
    # An Ollama generator model. Defaults to the first model in configs/generation.yaml.
    model: str | None = None
    top_n: int = Field(default=5, ge=1, le=20)
    sources: list[str] | None = Field(default=None, max_length=10)
    versions: list[str] | None = Field(default=None, max_length=10)
    include_passages: bool = True


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    source: str
    source_url: str
    version: str
    char_start: int
    char_end: int


class RetrievedPassage(Citation):
    rank: int
    score: float
    text: str


class QueryLatency(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    abstained: bool
    citations: list[Citation]
    # False when the model did not return the requested JSON. The raw text is then the answer.
    parsed: bool
    config: str
    config_id: str
    model: str
    prompt_version: str
    latency: QueryLatency
    passages: list[RetrievedPassage] | None = None


class EvaluationRequest(BaseModel):
    config: str
    split: EvaluationSplit = "development"


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class EvaluationResult(BaseModel):
    config: str
    config_id: str
    split: str
    dataset_hash: str
    corpus_hash: str
    questions_scored: int
    questions_skipped: int
    latency_p50_ms: float
    latency_p95_ms: float
    metrics: dict[str, float]


class Job(BaseModel):
    job_id: str
    status: JobStatus
    request: EvaluationRequest
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: EvaluationResult | None = None
    error: ErrorBody | None = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    split: str
    config_id: str
    questions_scored: int
    latency_p50_ms: float
    metrics: dict[str, float]
    git_commit: str | None


class CompareRequest(BaseModel):
    baseline: str
    treatment: str
    metrics: list[str] = Field(
        default=["recall_at_5", "mrr", "ndcg_at_10"], min_length=1, max_length=20
    )


class MetricComparison(BaseModel):
    metric: str
    baseline: float
    treatment: float
    # Treatment minus baseline, with a 95% paired bootstrap interval.
    difference: float
    low: float
    high: float
    verdict: Literal["better", "worse", "inconclusive"]


class CompareResponse(BaseModel):
    baseline: str
    treatment: str
    split: str
    questions: int
    comparisons: list[MetricComparison]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    ollama: str
    configs: list[str]
    default_config: str
    models: list[str]
