import pytest

from src.evaluation.sample_size import (
    cluster_adjusted_plan,
    estimate_equal_cluster_icc,
    matched_binary_sample_size,
)


def test_matched_binary_sample_size_increases_for_smaller_effect() -> None:
    large_effect = matched_binary_sample_size(0.10, 0.325)
    small_effect = matched_binary_sample_size(0.05, 0.325)

    assert large_effect == 253
    assert small_effect == 1018


def test_cluster_plan_rounds_to_complete_documents() -> None:
    plan = cluster_adjusted_plan(
        253,
        questions_per_document=4,
        intracluster_correlation=0.2,
    )

    assert plan["design_effect"] == pytest.approx(1.6)
    assert plan["documents"] == 102
    assert plan["queries"] == 408


def test_equal_cluster_icc_detects_clustered_values() -> None:
    icc = estimate_equal_cluster_icc(
        {"d1": [1, 1, 1, 1], "d2": [-1, -1, -1, -1], "d3": [0, 0, 0, 0]}
    )

    assert icc == pytest.approx(1.0)


def test_sample_size_rejects_effect_larger_than_discordance() -> None:
    with pytest.raises(ValueError, match="effect"):
        matched_binary_sample_size(0.4, 0.3)
