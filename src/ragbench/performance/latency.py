"""Latency measurement helpers."""

import time
from collections.abc import Callable


def timed[T](operation: Callable[[], T]) -> tuple[T, float]:
    """Run an operation and return its result and duration in milliseconds."""
    started = time.perf_counter()
    result = operation()
    return result, (time.perf_counter() - started) * 1000
