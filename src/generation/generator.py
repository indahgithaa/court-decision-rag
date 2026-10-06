"""Provider-neutral interface and timing for answer generation."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Protocol


class TextGenerator(Protocol):
    """Minimal contract implemented by a provider-specific model adapter."""

    @property
    def model_name(self) -> str:
        """Return a reproducible provider/model identifier."""

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return decoding parameters recorded with every answer."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate one answer from the frozen prompt."""


def balanced_generation_order(
    records: list[Mapping[str, Any]], *, seed: int = 42
) -> list[Mapping[str, Any]]:
    """Randomize query order and alternate which paired strategy runs first."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        query_id = str(record.get("query_id", "")).strip()
        strategy = str(record.get("strategy", "")).strip()
        if not query_id or not strategy:
            raise ValueError("Every record must have non-empty query_id and strategy")
        grouped[query_id].append(record)

    expected_strategies: set[str] | None = None
    for query_id, pair in grouped.items():
        strategies = [str(record["strategy"]) for record in pair]
        if len(strategies) != 2 or len(set(strategies)) != 2:
            raise ValueError(f"Query {query_id} must have exactly two distinct strategies")
        if expected_strategies is None:
            expected_strategies = set(strategies)
        elif set(strategies) != expected_strategies:
            raise ValueError("All query pairs must contain the same two strategies")

    query_ids = sorted(grouped)
    random.Random(seed).shuffle(query_ids)
    ordered: list[Mapping[str, Any]] = []
    for index, query_id in enumerate(query_ids):
        pair = sorted(grouped[query_id], key=lambda record: str(record["strategy"]))
        if index % 2:
            pair.reverse()
        ordered.extend(pair)
    return ordered


def generate_answer(
    input_record: Mapping[str, Any],
    generator: TextGenerator,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Generate and time one answer while preserving its experiment metadata."""
    system_prompt = str(input_record.get("system_prompt", "")).strip()
    user_prompt = str(input_record.get("user_prompt", "")).strip()
    if not system_prompt or not user_prompt:
        raise ValueError("input record must contain non-empty system_prompt and user_prompt")

    started = clock()
    answer = generator.generate(system_prompt=system_prompt, user_prompt=user_prompt).strip()
    elapsed_ms = (clock() - started) * 1_000
    if not answer:
        raise ValueError("generator returned an empty answer")

    result = dict(input_record)
    result.update(
        {
            "answer": answer,
            "generation_model": generator.model_name,
            "generation_parameters": dict(generator.parameters),
            "generation_latency_ms": elapsed_ms,
        }
    )
    return result

