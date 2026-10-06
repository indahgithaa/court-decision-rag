import pytest

from src.evaluation.sample_size import (
    cluster_adjusted_plan,
    estimate_equal_cluster_icc,
    paired_mean_sample_size,
)


def test_paired_mean_sample_size_increases_for_smaller_effect() -> None:
    large_effect = paired_mean_sample_size(0.10, 0.50)
    small_effect = paired_mean_sample_size(0.05, 0.50)

    assert large_effect == 197
    assert small_effect == 785


def test_cluster_plan_rounds_to_complete_documents() -> None:
    plan = cluster_adjusted_plan(
        197,
        questions_per_document=4,
        intracluster_correlation=0.2,
    )

    assert plan["design_effect"] == pytest.approx(1.6)
    assert plan["documents"] == 79
    assert plan["queries"] == 316


def test_equal_cluster_icc_detects_clustered_values() -> None:
    icc = estimate_equal_cluster_icc(
        {"d1": [1, 1, 1, 1], "d2": [-1, -1, -1, -1], "d3": [0, 0, 0, 0]}
    )

    assert icc == pytest.approx(1.0)


def test_sample_size_rejects_non_positive_standard_deviation() -> None:
    with pytest.raises(ValueError, match="difference_standard_deviation"):
        paired_mean_sample_size(0.1, 0)
