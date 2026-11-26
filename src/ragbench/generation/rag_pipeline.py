"""Grounded answer generation from given passages, with citations and abstention."""

import json
from typing import Literal

from pydantic import BaseModel

from ragbench.generation.ollama_client import OllamaClient
from ragbench.generation.prompts import ANSWER_SCHEMA, PROMPT_VERSION, SYSTEM, Passage, build_prompt


class GenerationSettings(BaseModel):
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 512
    context_tokens: int = 8192


class GeneratedAnswer(BaseModel):
    question_id: str
    model: str
    mode: Literal["retrieved", "oracle"]
    prompt_version: str
    context_chunk_ids: list[str]
    answer: str
    # Chunk IDs of the passages the model cited. Numbers outside the context are dropped.
    citations: list[str]
    invalid_citations: list[int]
    abstained: bool
    # False when the output was not the requested JSON, in which case the raw text is kept.
    parsed: bool
    raw_output: str
    prompt_tokens: int
    completion_tokens: int
    # True when the prompt came within the output budget of the context window, so it may
    # have been cut, and the answer should not be trusted.
    possibly_truncated: bool
    load_ms: float
    generation_ms: float
    total_ms: float
    # Set when the model failed to produce output at all, such as Ollama aborting a
    # degenerate, repeating generation. Such an answer counts as a failure of the system.
    error: str = ""


# Ollama errors that describe the model's own output, not a broken connection or server.
_MODEL_FAILURES = ("token repeat limit",)


def is_model_failure(message: str) -> bool:
    return any(marker in message for marker in _MODEL_FAILURES)


def failed_answer(
    question_id: str,
    model: str,
    passages: list[Passage],
    mode: Literal["retrieved", "oracle"],
    error: str,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        question_id=question_id,
        model=model,
        mode=mode,
        prompt_version=PROMPT_VERSION,
        context_chunk_ids=[passage.chunk_id for passage in passages],
        answer="",
        citations=[],
        invalid_citations=[],
        abstained=False,
        parsed=False,
        raw_output="",
        prompt_tokens=0,
        completion_tokens=0,
        possibly_truncated=False,
        load_ms=0.0,
        generation_ms=0.0,
        total_ms=0.0,
        error=error,
    )


def generate_answer(
    client: OllamaClient,
    model: str,
    question_id: str,
    question: str,
    passages: list[Passage],
    mode: Literal["retrieved", "oracle"],
    settings: GenerationSettings,
) -> GeneratedAnswer:
    generation = client.generate(
        model,
        build_prompt(question, passages),
        system=SYSTEM,
        temperature=settings.temperature,
        seed=settings.seed,
        max_tokens=settings.max_tokens,
        context_tokens=settings.context_tokens,
        json_schema=ANSWER_SCHEMA,
    )
    answer, numbers, abstained, parsed = _parse(generation.text)
    valid = [n for n in dict.fromkeys(numbers) if 1 <= n <= len(passages)]
    return GeneratedAnswer(
        question_id=question_id,
        model=model,
        mode=mode,
        prompt_version=PROMPT_VERSION,
        context_chunk_ids=[passage.chunk_id for passage in passages],
        answer=answer,
        citations=[passages[n - 1].chunk_id for n in valid],
        invalid_citations=[n for n in numbers if n not in valid],
        abstained=abstained,
        parsed=parsed,
        raw_output=generation.text,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        possibly_truncated=(
            generation.prompt_tokens >= settings.context_tokens - settings.max_tokens
        ),
        load_ms=round(generation.load_ms, 1),
        generation_ms=round(generation.generation_ms, 1),
        total_ms=round(generation.total_ms, 1),
    )


def _parse(text: str) -> tuple[str, list[int], bool, bool]:
    try:
        data = json.loads(text)
        answer = str(data["answer"]).strip()
        numbers = [int(n) for n in data.get("citations", [])]
        return answer, numbers, bool(data["abstained"]), True
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return text.strip(), [], False, False
