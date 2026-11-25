"""Checks against real hardware and services. Run with: pytest -m "gpu or ollama"."""

import pytest

from ragbench.generation.ollama_client import OllamaClient

GENERATOR_MODEL = "llama3.1:8b"


@pytest.mark.gpu
def test_tensor_runs_on_gpu():
    torch = pytest.importorskip("torch")

    assert torch.cuda.is_available()
    product = torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda")
    assert product.device.type == "cuda"
    assert product[0, 0].item() == 64.0


@pytest.mark.ollama
def test_ollama_generates_with_configured_model():
    client = OllamaClient()

    assert GENERATOR_MODEL in client.model_digests()
    result = client.generate(GENERATOR_MODEL, "Reply with the word ready.", max_tokens=10)
    assert result.text.strip()
    assert result.completion_tokens > 0
