"""Report whether this machine has what the benchmark needs. Exits non-zero on failure."""

import platform
import sys

from ragbench.generation.ollama_client import OllamaClient, OllamaError

GENERATOR_MODEL = "llama3.1:8b"


def check_gpu() -> bool:
    try:
        import torch
    except ImportError:
        print("[FAIL] torch is not installed. Run: uv sync --extra ml")
        return False

    print(f"[info] torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available")
        return False

    # Blackwell GPUs can report as available on incompatible builds, so run a real operation.
    try:
        product = torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda")
    except RuntimeError as exc:
        print(f"[FAIL] GPU tensor operation failed: {exc}")
        return False
    if product[0, 0].item() != 64.0:
        print("[FAIL] GPU tensor operation returned a wrong result")
        return False

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[ OK ] GPU tensor operation on {name} ({vram_gb:.1f} GB)")
    return True


def check_ollama() -> bool:
    client = OllamaClient()
    try:
        print(f"[info] Ollama {client.version()}")
        digests = client.model_digests()
        if GENERATOR_MODEL not in digests:
            print(f"[FAIL] {GENERATOR_MODEL} is not installed. Found: {sorted(digests)}")
            return False
        result = client.generate(GENERATOR_MODEL, "Reply with the word ready.", max_tokens=10)
    except OllamaError as exc:
        print(f"[FAIL] {exc}")
        return False

    print(
        f"[ OK ] {GENERATOR_MODEL} replied {result.text.strip()!r} "
        f"({result.completion_tokens} tokens, {result.total_ms:.0f} ms)"
    )
    return True


def main() -> int:
    print(f"[info] Python {platform.python_version()} on {platform.platform()}")
    checks = [check_gpu(), check_ollama()]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
