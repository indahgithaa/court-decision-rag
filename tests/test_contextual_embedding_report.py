import pytest

from src.evaluation.contextual_embedding_report import (
    evaluate_contextual_embedding_runs,
    render_contextual_embedding_markdown,
)


def _record(method: str, query_id: str, ranking: list[tuple[str, str]]) -> dict:
    return {
        "query_id": query_id,
        "embedding_method": method,
        "retrieval_scope": "full_corpus",
        "latency_ms": 1.0,
        "ranking": [
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "retrieved_document_id": document_id,
            }
            for rank, (chunk_id, document_id) in enumerate(ranking, start=1)
        ],
    }


def test_report_isolates_contextual_embedding_effect() -> None:
    questions = [
        {"query_id": "q1", "document_id": "d1", "target_section_label": "fakta"},
        {"query_id": "q2", "document_id": "d2", "target_section_label": "hukum"},
    ]
    qrels = [
        {
            "query_id": "q1",
            "chunk_id": "c1",
            "evidence_id": "q1:e1",
            "relevance_grade": "2",
        },
        {
            "query_id": "q2",
            "chunk_id": "c2",
            "evidence_id": "q2:e1",
            "relevance_grade": "2",
        },
    ]
    runs = {
        "independent": [
            _record("independent", "q1", [("x", "d2"), ("c1", "d1")]),
            _record("independent", "q2", [("x", "d1")]),
        ],
        "contextual": [
            _record("contextual", "q1", [("c1", "d1")]),
            _record("contextual", "q2", [("c2", "d2")]),
        ],
    }

    result = evaluate_contextual_embedding_runs(
        qrels, questions, runs, ks=(1, 5), bootstrap_samples=100, seed=7
    )

    assert result["paired_difference"]["recall@1"]["mean_difference"] == 1.0
    assert result["paired_difference"]["drm@1"]["mean_difference"] == -1.0
    markdown = render_contextual_embedding_markdown(result)
    assert "SAC independen" in markdown
    assert "Context. Recall" in markdown
    assert "Diagnosis provenance dokumen" in markdown
    assert "`drm@5`" in markdown


def test_report_rejects_wrong_method_label() -> None:
    questions = [
        {"query_id": "q1", "document_id": "d1", "target_section_label": "fakta"}
    ]
    qrels = [
        {
            "query_id": "q1",
            "chunk_id": "c1",
            "evidence_id": "q1:e1",
            "relevance_grade": "2",
        }
    ]
    wrong = _record("contextual", "q1", [("c1", "d1")])

    with pytest.raises(ValueError, match="Expected embedding_method"):
        evaluate_contextual_embedding_runs(
            qrels,
            questions,
            {"independent": [wrong], "contextual": [wrong]},
            ks=(1,),
            bootstrap_samples=10,
        )


def test_report_accepts_historical_independent_run_without_method_field() -> None:
    questions = [
        {"query_id": "q1", "document_id": "d1", "target_section_label": "fakta"}
    ]
    qrels = [
        {
            "query_id": "q1",
            "chunk_id": "c1",
            "evidence_id": "q1:e1",
            "relevance_grade": "2",
        }
    ]
    independent = _record("independent", "q1", [("c1", "d1")])
    independent.pop("embedding_method")

    result = evaluate_contextual_embedding_runs(
        qrels,
        questions,
        {
            "independent": [independent],
            "contextual": [_record("contextual", "q1", [("c1", "d1")])],
        },
        ks=(1,),
        bootstrap_samples=10,
    )

    assert result["paired_difference"]["recall@1"]["mean_difference"] == 0.0


def test_report_rejects_contextual_run_without_method_field() -> None:
    questions = [
        {"query_id": "q1", "document_id": "d1", "target_section_label": "fakta"}
    ]
    qrels = [
        {
            "query_id": "q1",
            "chunk_id": "c1",
            "evidence_id": "q1:e1",
            "relevance_grade": "2",
        }
    ]
    contextual = _record("contextual", "q1", [("c1", "d1")])
    contextual.pop("embedding_method")

    with pytest.raises(ValueError, match="Contextual runs must contain"):
        evaluate_contextual_embedding_runs(
            qrels,
            questions,
            {
                "independent": [
                    _record("independent", "q1", [("c1", "d1")])
                ],
                "contextual": [contextual],
            },
            ks=(1,),
            bootstrap_samples=10,
        )
