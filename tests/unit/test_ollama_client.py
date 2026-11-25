import json

import httpx
import pytest

from ragbench.generation.ollama_client import OllamaClient, OllamaError


def client_returning(handler) -> OllamaClient:
    return OllamaClient(transport=httpx.MockTransport(handler))


def test_generate_sends_reproducibility_options_and_parses_metadata():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "response": "hello",
                "prompt_eval_count": 5,
                "eval_count": 2,
                "load_duration": 1_000_000,
                "prompt_eval_duration": 2_000_000,
                "eval_duration": 3_000_000,
                "total_duration": 6_000_000,
            },
        )

    result = client_returning(handler).generate("test-model", "hi", seed=7)

    assert seen["options"] == {"temperature": 0.0, "seed": 7}
    assert seen["stream"] is False
    assert result.text == "hello"
    assert (result.prompt_tokens, result.completion_tokens) == (5, 2)
    assert result.load_ms == 1.0
    assert result.total_ms == 6.0


def test_generate_passes_json_schema_as_format():
    seen = {}
    schema = {"type": "object", "properties": {"abstained": {"type": "boolean"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "m", "response": "{}"})

    client_returning(handler).generate("m", "hi", json_schema=schema)

    assert seen["format"] == schema


def test_model_digests_maps_name_to_digest():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "m:1", "digest": "abc"}]})

    assert client_returning(handler).model_digests() == {"m:1": "abc"}


def test_http_error_raises_ollama_error_with_details():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    with pytest.raises(OllamaError, match="404.*model not found"):
        client_returning(handler).generate("missing", "hi")


def test_connection_failure_raises_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="Could not reach Ollama"):
        client_returning(handler).version()
