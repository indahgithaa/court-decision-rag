"""Sample-size utilities for paired, document-clustered retrieval studies."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence


def matched_binary_sample_size(
    effect: float,
    discordance_rate: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Approximate required paired queries for a two-sided McNemar comparison.

    ``effect`` is the target absolute difference in paired hit probabilities.
    ``discordance_rate`` is the expected fraction of pairs where only one
    strategy succeeds. The approximation is used for planning, not inference.
    """
    effect = abs(float(effect))
    discordance_rate = float(discordance_rate)
    if not 0 < effect <= discordance_rate <= 1:
        raise ValueError("require 0 < effect <= discordance_rate <= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not 0 < power < 1:
        raise ValueError("power must be between zero and one")

    normal = statistics.NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    variance_under_alternative = discordance_rate - effect**2
    numerator = (
        z_alpha * math.sqrt(discordance_rate)
        + z_power * math.sqrt(variance_under_alternative)
    ) ** 2
    return math.ceil(numerator / effect**2)


def cluster_adjusted_plan(
    independent_queries: int,
    *,
    questions_per_document: int,
    intracluster_correlation: float,
) -> dict[str, float | int]:
    """Inflate a query target with the standard equal-cluster design effect."""
    if independent_queries <= 0:
        raise ValueError("independent_queries must be positive")
    if questions_per_document <= 0:
        raise ValueError("questions_per_document must be positive")
    if not 0 <= intracluster_correlation < 1:
        raise ValueError("intracluster_correlation must be in [0, 1)")

    design_effect = 1 + (questions_per_document - 1) * intracluster_correlation
    adjusted_queries = math.ceil(independent_queries * design_effect)
    documents = math.ceil(adjusted_queries / questions_per_document)
    return {
        "design_effect": design_effect,
        "documents": documents,
        "queries": documents * questions_per_document,
    }


def estimate_equal_cluster_icc(
    values_by_cluster: Mapping[str, Sequence[float]],
) -> float:
    """Estimate ICC(1,1) for equal-size clusters, clipped to [0, 1]."""
    clusters = [tuple(float(value) for value in values) for values in values_by_cluster.values()]
    if len(clusters) < 2 or any(not values for values in clusters):
        raise ValueError("at least two non-empty clusters are required")
    sizes = {len(values) for values in clusters}
    if len(sizes) != 1:
        raise ValueError("clusters must have equal sizes")
    size = sizes.pop()
    if size < 2:
        raise ValueError("clusters must contain at least two observations")

    cluster_means = [statistics.fmean(values) for values in clusters]
    grand_mean = statistics.fmean(value for values in clusters for value in values)
    between = size * sum((mean - grand_mean) ** 2 for mean in cluster_means)
    between /= len(clusters) - 1
    within = sum(
        sum((value - mean) ** 2 for value in values)
        for values, mean in zip(clusters, cluster_means)
    )
    within /= len(clusters) * (size - 1)
    denominator = between + (size - 1) * within
    if denominator == 0:
        return 0.0
    return min(1.0, max(0.0, (between - within) / denominator))
