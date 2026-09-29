"""Dependency-free summaries for per-query latency measurements."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def summarize_latency(values_ms: Sequence[float]) -> dict[str, float | int]:
    """Summarize non-negative latency observations in milliseconds."""
    values = sorted(float(value) for value in values_ms)
    if not values:
        raise ValueError("latency values must not be empty")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latency values must be finite and non-negative")
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile from an ordered sequence."""
    if not sorted_values:
        raise ValueError("values must not be empty")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )

