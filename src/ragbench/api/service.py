"""The work behind each endpoint, kept apart from HTTP so it can be tested with fakes."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ragbench.chunking.models import Chunk
from ragbench.dataset.models import Question, read_questions
from ragbench.experiments.config import RetrievalConfig
from ragbench.experiments.runner import QuestionResult, RunRecord, run_retrieval, summarize
from ragbench.experiments.sweep import SweepSpec, expand_sweep, paired_difference
from ragbench.experiments.tracker import capture_environment, read_lock
from ragbench.generation.ollama_client import OllamaClient, OllamaError
from ragbench.generation.prompts import PROMPT_VERSION, Passage
from ragbench.generation.rag_pipeline import GenerationSettings, generate_answer
from ragbench.performance.latency import timed
from ragbench.retrieval.base import MetadataFilter, Retriever, ScoredChunk

from .schemas import (
    Citation,
    CompareRequest,
    CompareResponse,
    EvaluationRequest,
    EvaluationResult,
    ExperimentSummary,
    MetricComparison,
    QueryLatency,
    QueryRequest,
    QueryResponse,
    RetrievedPassage,
)

NO_PASSAGES_ANSWER = "No documentation passages matched this question, so it cannot be answered."


class ApiError(Exception):
    """An error with an HTTP status and a stable code that clients can act on."""

    def __init__(self, status: int, code: str, message: str, details=None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details


@dataclass
class RagService:
    configs: dict[str, RetrievalConfig]
    default_config: str
    models: list[str]
    # Builds (and may cache) the retriever for a configuration.
    retriever_for: Callable[[RetrievalConfig], Retriever]
    client: OllamaClient
    settings: GenerationSettings
    data_dir: Path
    reports_dir: Path
    corpus_hash: str
    dataset_hash: str
    # A client with a short timeout, so a health check never hangs on a stuck server.
    probe_client: OllamaClient | None = None
    load_questions: Callable[[str], list[Question]] | None = None
    # The source and version values present in a configuration's chunks, to validate filters.
    facets_for: Callable[[RetrievalConfig], dict[str, set[str]]] | None = None
    _experiments: dict[str, Path] = field(default_factory=dict, init=False)

    @classmethod
    def from_project(
        cls, root: Path = Path("."), ollama_url: str = "http://localhost:11434"
    ) -> "RagService":
        from ragbench.experiments.resources import Resources

        data_dir = root / "data"
        generation = yaml.safe_load((root / "configs/generation.yaml").read_text("utf-8"))
        sweep = yaml.safe_load((root / "configs/retrieval_sweep.yaml").read_text("utf-8"))
        configs = {c.name: c for c in expand_sweep(SweepSpec.model_validate(sweep))}
        resources = Resources(
            data_dir, root / "configs/chunking.yaml", root / "configs/embeddings.yaml"
        )
        dataset_dir = data_dir / "datasets" / "v1"
        return cls(
            configs=configs,
            default_config=generation["retrieval_configs"][0],
            models=list(generation["models"]),
            retriever_for=resources.retriever,
            client=OllamaClient(ollama_url),
            probe_client=OllamaClient(ollama_url, timeout_s=3.0),
            settings=GenerationSettings(**generation["settings"]),
            data_dir=data_dir,
            reports_dir=root / "reports",
            corpus_hash=resources.corpus_hash,
            dataset_hash=read_lock(dataset_dir / "dataset.lock.json")["dataset_hash"],
            load_questions=lambda split: read_questions(dataset_dir / f"{split}.jsonl"),
            facets_for=lambda config: chunk_facets(resources.chunks(config.chunking)),
        )

    # Health

    def ollama_status(self) -> str:
        try:
            return f"ok {(self.probe_client or self.client).version()}"
        except OllamaError:
            return "unreachable"

    # Querying

    def config(self, name: str | None) -> RetrievalConfig:
        name = name or self.default_config
        if name not in self.configs:
            raise ApiError(
                404,
                "unknown_config",
                f"No retrieval configuration named {name!r}.",
                {"available": sorted(self.configs)},
            )
        return self.configs[name]

    def retriever(self, config: RetrievalConfig) -> Retriever:
        try:
            return self.retriever_for(config)
        except FileNotFoundError as exc:
            raise ApiError(503, "config_unavailable", str(exc)) from exc
        except (ImportError, OSError) as exc:
            # Dense retrieval and reranking need PyTorch (the `ml` extra), which may be missing
            # or blocked on this machine.
            raise ApiError(
                503,
                "config_unavailable",
                f"{config.name} needs model inference, which is not available here: {exc}",
            ) from exc

    def query(self, request: QueryRequest, query_id: str) -> QueryResponse:
        config = self.config(request.config)
        model = request.model or self.models[0]
        if model not in self.models:
            raise ApiError(
                404,
                "unknown_model",
                f"Model {model!r} is not configured.",
                {"available": self.models},
            )
        retriever = self.retriever(config)
        restriction = self._restriction(config, request)
        ranked, retrieval_ms = timed(
            lambda: retriever.retrieve(request.question, request.top_n, restriction)
        )
        passages = [Passage(r.chunk.chunk_id, r.chunk.title, r.chunk.text) for r in ranked]
        if passages:
            try:
                answer = generate_answer(
                    self.client,
                    model,
                    query_id,
                    request.question,
                    passages,
                    "retrieved",
                    self.settings,
                )
            except OllamaError as exc:
                raise ApiError(502, "generation_failed", str(exc)) from exc
            text, abstained, parsed = answer.answer, answer.abstained, answer.parsed
            cited, generation_ms = answer.citations, answer.total_ms
        else:
            # Nothing to ground an answer in, so abstain without asking the model.
            text, abstained, parsed = NO_PASSAGES_ANSWER, True, True
            cited, generation_ms = [], 0.0

        by_id = {r.chunk.chunk_id: r for r in ranked}
        return QueryResponse(
            query_id=query_id,
            question=request.question,
            answer=text,
            abstained=abstained,
            citations=[_citation(by_id[chunk_id]) for chunk_id in cited],
            parsed=parsed,
            config=config.name,
            config_id=config.config_id,
            model=model,
            prompt_version=PROMPT_VERSION,
            latency=QueryLatency(
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=generation_ms,
                total_ms=round(retrieval_ms + generation_ms, 1),
            ),
            passages=[
                RetrievedPassage(
                    **_citation(r).model_dump(), rank=rank, score=r.score, text=r.chunk.text
                )
                for rank, r in enumerate(ranked, start=1)
            ]
            if request.include_passages
            else None,
        )

    def _restriction(self, config: RetrievalConfig, request: QueryRequest) -> MetadataFilter | None:
        """The metadata filter for a query. Unknown values are an error rather than a filter
        that silently matches nothing."""
        if not (request.sources or request.versions):
            return None
        if self.facets_for is not None:
            facets = self.facets_for(config)
            for name, given in (("sources", request.sources), ("versions", request.versions)):
                unknown = sorted(set(given or []) - facets[name])
                if unknown:
                    raise ApiError(
                        422,
                        "unknown_filter_value",
                        f"Unknown {name}: {', '.join(map(repr, unknown))}.",
                        {"available": sorted(facets[name])},
                    )
        return MetadataFilter(
            frozenset(request.sources) if request.sources else None,
            frozenset(request.versions) if request.versions else None,
        )

    # Evaluation

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        config = self.config(request.config)
        retriever = self.retriever(config)
        if self.load_questions is None:
            raise ApiError(503, "no_dataset", "No dataset is configured.")
        questions = self.load_questions(request.split)
        results = run_retrieval(config, questions, retriever)
        record = summarize(
            config,
            results,
            questions,
            request.split,
            self.dataset_hash,
            self.corpus_hash,
            capture_environment(),
        )
        return EvaluationResult(
            config=config.name,
            config_id=record.config_id,
            split=record.split,
            dataset_hash=record.dataset_hash,
            corpus_hash=record.corpus_hash,
            questions_scored=record.questions_scored,
            questions_skipped=record.questions_skipped,
            latency_p50_ms=record.latency_p50_ms,
            latency_p95_ms=record.latency_p95_ms,
            metrics=record.metrics,
        )

    # Stored experiments and reports

    def experiment_dirs(self) -> dict[str, Path]:
        """Saved retrieval runs, by ID `<split>.<run name>`. Unfinished runs have no run.json."""
        root = self.data_dir / "runs" / "retrieval"
        return {
            f"{path.parent.parent.name}.{path.parent.name}": path.parent
            for path in sorted(root.glob("*/*/run.json"))
        }

    def experiments(self, split: str | None = None) -> list[ExperimentSummary]:
        summaries = []
        for experiment_id, directory in self.experiment_dirs().items():
            record = _read_record(directory)
            if split is None or record.split == split:
                summaries.append(_summary(experiment_id, record))
        return summaries

    def experiment(self, experiment_id: str) -> tuple[RunRecord, list[QuestionResult]]:
        directory = self.experiment_dirs().get(experiment_id)
        if directory is None:
            raise ApiError(404, "unknown_experiment", f"No experiment {experiment_id!r}.")
        with (directory / "questions.jsonl").open(encoding="utf-8") as lines:
            results = [QuestionResult.model_validate(json.loads(line)) for line in lines]
        return _read_record(directory), results

    def compare(self, request: CompareRequest) -> CompareResponse:
        base_record, base_results = self.experiment(request.baseline)
        treat_record, treat_results = self.experiment(request.treatment)
        if base_record.split != treat_record.split:
            raise ApiError(
                422, "not_comparable", "Experiments on different splits cannot be paired."
            )
        unknown = [m for m in request.metrics if m not in base_record.metrics]
        if unknown:
            raise ApiError(
                422,
                "unknown_metric",
                f"Unknown metrics: {', '.join(unknown)}.",
                {"available": sorted(base_record.metrics)},
            )
        comparisons = []
        for metric in request.metrics:
            try:
                difference = paired_difference(base_results, treat_results, metric)
            except ValueError as exc:
                raise ApiError(422, "not_comparable", str(exc)) from exc
            comparisons.append(
                MetricComparison(
                    metric=metric,
                    baseline=base_record.metrics[metric],
                    treatment=treat_record.metrics[metric],
                    difference=round(difference.estimate, 4),
                    low=round(difference.low, 4),
                    high=round(difference.high, 4),
                    verdict=difference.verdict,
                )
            )
        return CompareResponse(
            baseline=request.baseline,
            treatment=request.treatment,
            split=base_record.split,
            questions=len(base_results),
            comparisons=comparisons,
        )

    def reports(self) -> list[str]:
        return sorted(path.stem for path in self.reports_dir.glob("*.md"))

    def report(self, report_id: str) -> str:
        # Only names from the listing, so a request can never reach outside the reports folder.
        if report_id not in self.reports():
            raise ApiError(404, "unknown_report", f"No report {report_id!r}.")
        return (self.reports_dir / f"{report_id}.md").read_text(encoding="utf-8")


def chunk_facets(chunks: list[Chunk]) -> dict[str, set[str]]:
    return {
        "sources": {chunk.source for chunk in chunks},
        "versions": {chunk.version for chunk in chunks},
    }


def _citation(result: ScoredChunk) -> Citation:
    chunk = result.chunk
    return Citation(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        source=chunk.source,
        source_url=chunk.source_url,
        version=chunk.version,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
    )


def _read_record(directory: Path) -> RunRecord:
    return RunRecord.model_validate_json((directory / "run.json").read_text(encoding="utf-8"))


def _summary(experiment_id: str, record: RunRecord) -> ExperimentSummary:
    return ExperimentSummary(
        experiment_id=experiment_id,
        name=record.config.name,
        split=record.split,
        config_id=record.config_id,
        questions_scored=record.questions_scored,
        latency_p50_ms=record.latency_p50_ms,
        metrics=record.metrics,
        git_commit=record.environment.git_commit,
    )
