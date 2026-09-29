import pytest

from src.evaluation.latency import summarize_latency
from src.evaluation.retrieval_report import evaluate_paired_runs, render_markdown


def _record(query_id: str, strategy: str, ranking: list[str], latency: float) -> dict:
    return {
        "query_id": query_id,
        "target_section_label": "fakta",
        "strategy": strategy,
        "latency_ms": latency,
        "ranking": [
            {"rank": rank, "chunk_id": chunk_id}
            for rank, chunk_id in enumerate(ranking, start=1)
        ],
    }


def test_paired_report_compares_structure_aware_with_fixed_size() -> None:
    qrels = [
        {"query_id": "q1", "strategy": "fixed_size", "chunk_id": "f1", "relevance_grade": "2"},
        {"query_id": "q1", "strategy": "structure_aware", "chunk_id": "s1", "relevance_grade": "2"},
        {"query_id": "q2", "strategy": "fixed_size", "chunk_id": "f2", "relevance_grade": "2"},
        {"query_id": "q2", "strategy": "structure_aware", "chunk_id": "s2", "relevance_grade": "2"},
    ]
    runs = {
        "fixed_size": [
            _record("q1", "fixed_size", ["wrong", "f1"], 10.0),
            _record("q2", "fixed_size", ["wrong"], 20.0),
        ],
        "structure_aware": [
            _record("q1", "structure_aware", ["s1"], 12.0),
            _record("q2", "structure_aware", ["s2"], 18.0),
        ],
    }

    result = evaluate_paired_runs(
        qrels,
        runs,
        ks=(1, 5),
        bootstrap_samples=100,
        seed=7,
    )

    assert result["strategies"]["fixed_size"]["overall"]["aggregate"]["hit@1"] == 0
    assert result["strategies"]["structure_aware"]["overall"]["aggregate"]["hit@1"] == 1
    assert result["paired_difference"]["hit@1"]["mean_difference"] == 1
    assert "SAC - fixed" in render_markdown(result)


def test_latency_summary_uses_interpolated_p95() -> None:
    result = summarize_latency([10.0, 20.0, 30.0])

    assert result["mean_ms"] == pytest.approx(20.0)
    assert result["median_ms"] == pytest.approx(20.0)
    assert result["p95_ms"] == pytest.approx(29.0)
