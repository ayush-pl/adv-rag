import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from ragbench.api.app import EXECUTION_HEADER, ApiSettings, create_app
from ragbench.api.service import NO_PASSAGES_ANSWER, RagService, chunk_facets
from ragbench.chunking.models import Chunk
from ragbench.dataset.models import Provenance, Question
from ragbench.evaluation.gold import EvidenceGroup, GoldSpan
from ragbench.experiments.config import ChunkingChoice, RetrievalConfig
from ragbench.experiments.runner import QuestionResult, save_run, summarize
from ragbench.experiments.tracker import Environment
from ragbench.generation.ollama_client import OllamaClient
from ragbench.generation.rag_pipeline import GenerationSettings
from ragbench.retrieval.base import Retriever, ScoredChunk

CHUNKING = ChunkingChoice(strategy="recursive", chunk_size=512, overlap=50)
CONFIGS = {
    "bm25": RetrievalConfig(name="bm25", chunking=CHUNKING, method="bm25"),
    "dense": RetrievalConfig(name="dense", chunking=CHUNKING, method="dense", embedder="m"),
    "missing": RetrievalConfig(name="missing", chunking=CHUNKING, method="bm25", top_k=5),
}
ENVIRONMENT = Environment(
    git_commit="abc", git_dirty=False, python="3.12", platform="t", torch=None, cuda_device=None
)


def chunk(document_id: str, start: int, end: int, source="kubernetes", version="1.36") -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}::{start}-{end}",
        document_id=document_id,
        char_start=start,
        char_end=end,
        text=f"text of {document_id}",
        token_count=3,
        section="",
        source=source,
        source_url=f"https://example.org/{document_id}",
        title=f"Title {document_id}",
        version=version,
    )


PASSAGES = [chunk("gold", 0, 50), chunk("other", 0, 50, source="docker"), chunk("third", 0, 50)]


class FakeRetriever(Retriever):
    def __init__(self) -> None:
        self.filters = []

    def retrieve(self, query, top_k, metadata_filter=None):
        self.filters.append(metadata_filter)
        kept = [c for c in PASSAGES if metadata_filter is None or metadata_filter.matches(c)]
        return [ScoredChunk(c, 1.0 / (i + 1)) for i, c in enumerate(kept[:top_k])]


def question(question_id: str, answerable: bool = True) -> Question:
    span = GoldSpan(document_id="gold", char_start=0, char_end=40, quote="q")
    return Question(
        question_id=question_id,
        question=f"Question {question_id}?",
        question_type="single_hop" if answerable else "unanswerable",
        expected_answer="A",
        evidence=[EvidenceGroup(spans=[span])] if answerable else [],
        answerable=answerable,
        human_verified=True,
        provenance=Provenance(method="manual"),
    )


def ollama(answer: dict | None = None, fail: bool = False) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("refused")
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0-test"})
        body = answer or {"answer": "Use a Deployment.", "citations": [1, 9], "abstained": False}
        return httpx.Response(
            200,
            json={
                "model": "m",
                "response": json.dumps(body),
                "prompt_eval_count": 100,
                "eval_count": 10,
                "total_duration": 5_000_000,
            },
        )

    return OllamaClient(transport=httpx.MockTransport(handler))


def retriever_for(retriever: Retriever):
    def build(config: RetrievalConfig) -> Retriever:
        if config.name == "missing":
            raise FileNotFoundError("No chunks for missing. Run build_chunks.py.")
        if config.method == "dense":
            raise ImportError("DLL load failed while importing torch")
        return retriever

    return build


def save_experiment(data_dir, name: str, recalls: list[float]) -> None:
    config = CONFIGS["bm25"].model_copy(update={"name": name})
    results = [
        QuestionResult(
            question_id=f"q{i}",
            question_type="single_hop",
            chunk_ids=[],
            scores=[],
            latency_ms=1.0,
            metrics={"recall_at_5": r, "mrr": r, "ndcg_at_10": r},
        )
        for i, r in enumerate(recalls)
    ]
    record = summarize(
        config,
        results,
        [question(f"q{i}") for i in range(len(recalls))],
        "development",
        "sha256-d",
        "sha256-c",
        ENVIRONMENT,
    )
    save_run(data_dir / "runs" / "retrieval" / "development" / config.run_name, record, results)


@pytest.fixture
def project(tmp_path):
    save_experiment(tmp_path / "data", "baseline", [0.0] * 20 + [1.0] * 10)
    save_experiment(tmp_path / "data", "improved", [1.0] * 30)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "retrieval_sweep_development.md").write_text("# Sweep\n", "utf-8")
    return tmp_path


def make_service(project, client=None, retriever=None) -> RagService:
    return RagService(
        configs=CONFIGS,
        default_config="bm25",
        models=["m", "other-model"],
        retriever_for=retriever_for(retriever or FakeRetriever()),
        client=client or ollama(),
        settings=GenerationSettings(),
        data_dir=project / "data",
        reports_dir=project / "reports",
        corpus_hash="sha256-c",
        dataset_hash="sha256-d",
        load_questions=lambda split: [question("q1"), question("q2"), question("u", False)],
        facets_for=lambda config: chunk_facets(PASSAGES),
    )


def api(project, settings=None, **service_options) -> TestClient:
    return TestClient(create_app(make_service(project, **service_options), settings))


def wait_for(client: TestClient, job_id: str, headers=None) -> dict:
    for _ in range(200):
        job = client.get(f"/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("the job did not finish")


class TestHealth:
    def test_reports_configs_and_ollama(self, project):
        response = api(project).get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["ollama"] == "ok 0.0-test"
        assert body["default_config"] == "bm25"
        assert response.headers[EXECUTION_HEADER].startswith("exec-")

    def test_degraded_when_ollama_is_down(self, project):
        body = api(project, client=ollama(fail=True)).get("/health").json()
        assert (body["status"], body["ollama"]) == ("degraded", "unreachable")


class TestQuery:
    def test_answers_with_resolved_citations_and_drops_invalid_ones(self, project):
        response = api(project).post("/query", json={"question": "How do I run an app?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Use a Deployment."
        assert body["query_id"] == response.headers[EXECUTION_HEADER]
        # Citation 9 points past the 3 passages, so it is dropped.
        assert [c["chunk_id"] for c in body["citations"]] == ["gold::0-50"]
        assert body["citations"][0]["source_url"] == "https://example.org/gold"
        assert [p["rank"] for p in body["passages"]] == [1, 2, 3]
        assert body["config"] == "bm25" and body["model"] == "m"
        assert body["latency"]["generation_ms"] == 5.0

    def test_abstention_is_returned(self, project):
        client = ollama({"answer": "Not in the docs.", "citations": [], "abstained": True})
        body = (
            api(project, client=client)
            .post(
                "/query",
                json={"question": "What is the capital of France?", "include_passages": False},
            )
            .json()
        )
        assert body["abstained"] is True
        assert body["citations"] == [] and body["passages"] is None

    def test_metadata_filter_reaches_the_retriever(self, project):
        retriever = FakeRetriever()
        body = (
            api(project, retriever=retriever)
            .post("/query", json={"question": "How do I run an app?", "sources": ["docker"]})
            .json()
        )
        assert retriever.filters[-1].sources == frozenset({"docker"})
        assert [p["source"] for p in body["passages"]] == ["docker"]

    def test_unknown_filter_values_are_rejected_with_the_valid_ones(self, project):
        response = api(project).post("/query", json={"question": "Why?", "sources": ["postgresql"]})
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "unknown_filter_value"
        assert error["details"]["available"] == ["docker", "kubernetes"]

    def test_no_passages_means_abstaining_without_the_model(self, project):
        class EmptyRetriever(FakeRetriever):
            def retrieve(self, query, top_k, metadata_filter=None):
                return []

        # Ollama is down, so the test fails if the model is asked anything.
        body = (
            api(project, client=ollama(fail=True), retriever=EmptyRetriever())
            .post("/query", json={"question": "Why?"})
            .json()
        )
        assert body["abstained"] is True
        assert body["answer"] == NO_PASSAGES_ANSWER
        assert body["passages"] == [] and body["latency"]["generation_ms"] == 0

    @pytest.mark.parametrize(
        "payload",
        [{"question": ""}, {"question": "x" * 2001}, {"question": "ok?", "top_n": 0}, {}],
    )
    def test_invalid_input_gets_a_structured_error(self, project, payload):
        response = api(project).post("/query", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "invalid_request"
        assert body["error"]["details"]
        assert body["execution_id"] == response.headers[EXECUTION_HEADER]

    def test_unknown_config_and_model(self, project):
        client = api(project)
        config = client.post("/query", json={"question": "Why?", "config": "nope"})
        assert config.status_code == 404
        assert config.json()["error"]["code"] == "unknown_config"
        assert "bm25" in config.json()["error"]["details"]["available"]
        model = client.post("/query", json={"question": "Why?", "model": "gpt"})
        assert model.json()["error"]["code"] == "unknown_model"

    @pytest.mark.parametrize("config", ["dense", "missing"])
    def test_unavailable_config_is_a_503(self, project, config):
        response = api(project).post("/query", json={"question": "Why?", "config": config})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "config_unavailable"

    def test_ollama_failure_is_a_502(self, project):
        response = api(project, client=ollama(fail=True)).post("/query", json={"question": "Why?"})
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "generation_failed"

    def test_slow_query_times_out(self, project):
        class SlowRetriever(FakeRetriever):
            def retrieve(self, query, top_k, metadata_filter=None):
                time.sleep(0.5)
                return super().retrieve(query, top_k, metadata_filter)

        settings = ApiSettings(query_timeout_s=0.05)
        response = api(project, settings, retriever=SlowRetriever()).post(
            "/query", json={"question": "Why?"}
        )
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "timeout"


class TestAuth:
    def test_key_is_required_except_for_health(self, project):
        client = api(project, ApiSettings(api_key="secret"))
        assert client.get("/health").status_code == 200
        denied = client.get("/experiments")
        assert denied.status_code == 401
        assert denied.json()["error"]["code"] == "unauthorized"
        assert client.get("/experiments", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/experiments", headers={"X-API-Key": "secret"}).status_code == 200


class TestEvaluate:
    def test_runs_in_the_background_and_reports_metrics(self, project):
        client = api(project)
        response = client.post("/evaluate", json={"config": "bm25"})
        assert response.status_code == 202
        job = wait_for(client, response.json()["job_id"])
        assert job["status"] == "succeeded"
        result = job["result"]
        assert (result["questions_scored"], result["questions_skipped"]) == (2, 1)
        # The fake ranks the gold passage first for every question.
        assert result["metrics"]["recall_at_5"] == 1.0
        assert result["dataset_hash"] == "sha256-d"
        assert [j["job_id"] for j in client.get("/jobs").json()] == [job["job_id"]]

    def test_the_test_split_is_not_available(self, project):
        response = api(project).post("/evaluate", json={"config": "bm25", "split": "test"})
        assert response.status_code == 422

    def test_unknown_config_is_rejected_before_queueing(self, project):
        response = api(project).post("/evaluate", json={"config": "nope"})
        assert response.status_code == 404

    def test_a_failing_job_records_its_error(self, project):
        client = api(project)
        job = wait_for(client, client.post("/evaluate", json={"config": "dense"}).json()["job_id"])
        assert job["status"] == "failed"
        assert job["error"]["code"] == "config_unavailable"

    def test_unknown_job(self, project):
        response = api(project).get("/jobs/job-nope")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "unknown_job"


class TestExperiments:
    def test_lists_and_filters_stored_runs(self, project):
        client = api(project)
        runs = client.get("/experiments").json()
        assert sorted(r["name"] for r in runs) == ["baseline", "improved"]
        assert runs[0]["experiment_id"].startswith("development.")
        assert client.get("/experiments", params={"split": "test"}).json() == []

    def test_detail_with_and_without_questions(self, project):
        client = api(project)
        experiment_id = client.get("/experiments").json()[0]["experiment_id"]
        brief = client.get(f"/experiments/{experiment_id}").json()
        assert brief["record"]["split"] == "development" and "questions" not in brief
        full = client.get(f"/experiments/{experiment_id}", params={"questions": True}).json()
        assert len(full["questions"]) == 30
        assert client.get("/experiments/development.nope").status_code == 404

    def test_compare_gives_a_paired_verdict(self, project):
        client = api(project)
        ids = {r["name"]: r["experiment_id"] for r in client.get("/experiments").json()}
        response = client.post(
            "/experiments/compare",
            json={"baseline": ids["baseline"], "treatment": ids["improved"], "metrics": ["mrr"]},
        )
        assert response.status_code == 200
        [comparison] = response.json()["comparisons"]
        assert comparison["difference"] == pytest.approx(2 / 3, abs=1e-3)
        assert comparison["verdict"] == "better"
        assert response.json()["questions"] == 30

    def test_compare_rejects_unknown_metrics(self, project):
        client = api(project)
        ids = [r["experiment_id"] for r in client.get("/experiments").json()]
        response = client.post(
            "/experiments/compare",
            json={"baseline": ids[0], "treatment": ids[1], "metrics": ["accuracy"]},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "unknown_metric"


class TestReports:
    def test_lists_and_serves_markdown(self, project):
        client = api(project)
        assert client.get("/reports").json() == ["retrieval_sweep_development"]
        response = client.get("/reports/retrieval_sweep_development")
        assert response.text == "# Sweep\n"
        assert response.headers["content-type"].startswith("text/markdown")

    @pytest.mark.parametrize("name", ["nope", "..%2Fpyproject", "%2E%2E"])
    def test_only_listed_reports_are_served(self, project, name):
        response = api(project).get(f"/reports/{name}")
        assert response.status_code == 404


def test_unknown_route_uses_the_error_format(project):
    response = api(project).get("/nowhere")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
