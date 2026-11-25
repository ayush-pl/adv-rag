"""Minimal client for the Ollama HTTP API."""

from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 300.0


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


@dataclass(frozen=True)
class Generation:
    """One completion, with the metadata needed for run records and latency reports."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    load_ms: float
    prompt_eval_ms: float
    generation_ms: float
    total_ms: float


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def version(self) -> str:
        return self._request("GET", "/api/version")["version"]

    def model_digests(self) -> dict[str, str]:
        """Map each installed model name to its digest, for recording in run records."""
        models = self._request("GET", "/api/tags")["models"]
        return {model["name"]: model["digest"] for model in models}

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Generation:
        options: dict[str, Any] = {"temperature": temperature, "seed": seed}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system is not None:
            payload["system"] = system
        if json_schema is not None:
            payload["format"] = json_schema

        data = self._request("POST", "/api/generate", json=payload)
        return Generation(
            text=data["response"],
            model=data["model"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            load_ms=_ns_to_ms(data.get("load_duration", 0)),
            prompt_eval_ms=_ns_to_ms(data.get("prompt_eval_duration", 0)),
            generation_ms=_ns_to_ms(data.get("eval_duration", 0)),
            total_ms=_ns_to_ms(data.get("total_duration", 0)),
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self._http.base_url}: {exc}") from exc
        if response.is_error:
            raise OllamaError(f"Ollama returned {response.status_code}: {response.text}")
        return response.json()


def _ns_to_ms(nanoseconds: int) -> float:
    return nanoseconds / 1_000_000
