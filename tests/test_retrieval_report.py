import pytest

from src.evaluation.latency import summarize_latency
from src.evaluation.retrieval_report import (
    evaluate_paired_runs,
    paired_cluster_bootstrap_interval,
    render_markdown,
)


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
        {"query_id": "q1", "document_id": "d1", "strategy": "fixed_size", "chunk_id": "f1", "relevance_grade": "2"},
        {"query_id": "q1", "document_id": "d1", "strategy": "structure_aware", "chunk_id": "s1", "relevance_grade": "2"},
        {"query_id": "q2", "document_id": "d2", "strategy": "fixed_size", "chunk_id": "f2", "relevance_grade": "2"},
        {"query_id": "q2", "document_id": "d2", "strategy": "structure_aware", "chunk_id": "s2", "relevance_grade": "2"},
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
    assert result["bootstrap_unit"] == "document"
    assert result["document_count"] == 2
    markdown = render_markdown(result)
    assert "SAC - fixed" in markdown
    assert "## Candidate coverage" in markdown
    assert "| 5 | 0.5000 | 1.0000 | +0.5000 |" in markdown


def test_latency_summary_uses_interpolated_p95() -> None:
    result = summarize_latency([10.0, 20.0, 30.0])

    assert result["mean_ms"] == pytest.approx(20.0)
    assert result["median_ms"] == pytest.approx(20.0)
    assert result["p95_ms"] == pytest.approx(29.0)


def test_report_marks_gold_document_oracle_scope() -> None:
    qrels = [
        {"query_id": "q1", "document_id": "d1", "strategy": "fixed_size", "chunk_id": "f1", "relevance_grade": "2"},
        {"query_id": "q1", "document_id": "d1", "strategy": "structure_aware", "chunk_id": "s1", "relevance_grade": "2"},
    ]
    fixed = _record("q1", "fixed_size", ["f1"], 1.0)
    structure = _record("q1", "structure_aware", ["s1"], 1.0)
    fixed["retrieval_scope"] = "gold_document_oracle"
    structure["retrieval_scope"] = "gold_document_oracle"

    result = evaluate_paired_runs(
        qrels,
        {"fixed_size": [fixed], "structure_aware": [structure]},
        ks=(1,),
        bootstrap_samples=10,
    )

    markdown = render_markdown(result)
    assert result["retrieval_scope"] == "gold_document_oracle"
    assert "gold-document oracle" in markdown
    assert "bukan performa retrieval end-to-end" in markdown


def test_report_marks_final_holdout_without_pilot_warning() -> None:
    qrels = [
        {"query_id": "q1", "document_id": "d1", "strategy": "fixed_size", "chunk_id": "f1", "relevance_grade": "2"},
        {"query_id": "q1", "document_id": "d1", "strategy": "structure_aware", "chunk_id": "s1", "relevance_grade": "2"},
    ]
    result = evaluate_paired_runs(
        qrels,
        {
            "fixed_size": [_record("q1", "fixed_size", ["f1"], 1.0)],
            "structure_aware": [_record("q1", "structure_aware", ["s1"], 1.0)],
        },
        ks=(1,),
        bootstrap_samples=10,
    )
    result["evaluation_stage"] = "final_holdout"

    markdown = render_markdown(result)

    assert "tidak digunakan untuk pemilihan desain" in markdown
    assert "Hasil pilot" not in markdown


def test_cluster_bootstrap_keeps_document_questions_together() -> None:
    low, high = paired_cluster_bootstrap_interval(
        {"d1": [1.0, 1.0], "d2": [-1.0, -1.0]},
        samples=2_000,
        seed=7,
    )

    assert low == pytest.approx(-1.0)
    assert high == pytest.approx(1.0)
