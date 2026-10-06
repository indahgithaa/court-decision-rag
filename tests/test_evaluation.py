"""Tests for structure-review and retrieval evaluation metrics."""

import pytest

from src.evaluation import (
    build_qrel_candidates,
    evaluate_retrieval,
    score_structure_review,
    validate_questions,
)


def test_evaluate_retrieval_scores_ranked_results_and_missing_runs() -> None:
    qrels = {
        "q1": {"a": 2, "b": 1},
        "q2": {"c": 1},
    }
    runs = {"q1": ["x", "b", "a"]}

    score = evaluate_retrieval(qrels, runs, ks=(1, 3))

    assert score["query_count"] == 2
    assert score["per_query"]["q1"]["recall@1"] == 0
    assert score["per_query"]["q1"]["recall@3"] == 1
    assert score["per_query"]["q1"]["mrr@3"] == 0.5
    assert 0 < score["per_query"]["q1"]["ndcg@3"] < 1
    assert score["per_query"]["q2"]["recall@3"] == 0
    assert score["aggregate"]["recall@3"] == 0.5


def test_evaluate_retrieval_deduplicates_ranked_chunks() -> None:
    score = evaluate_retrieval(
        {"q1": {"relevant": 1}},
        {"q1": ["noise", "noise", "relevant"]},
        ks=(2,),
    )

    assert score["per_query"]["q1"]["mrr@2"] == 0.5


def test_evaluate_retrieval_rejects_query_without_positive_qrel() -> None:
    with pytest.raises(ValueError, match="no relevant chunks"):
        evaluate_retrieval({"q1": {"a": 0}}, {"q1": ["a"]})


def test_structure_review_scores_completed_sheet() -> None:
    labels = [
        "kepala_putusan",
        "identitas_terdakwa",
        "riwayat_penahanan",
        "fakta",
        "pertimbangan_hukum",
        "amar_putusan",
        "penutup",
    ]
    rows = [
        {
            "document_id": "doc-1",
            "section_label": label,
            "detected": "yes",
            "gold_present": "yes",
            "review_status": "near" if label == "identitas_terdakwa" else "exact",
        }
        for label in labels
    ]

    score = score_structure_review(rows)

    assert score["complete"] is True
    assert score["acceptance"]["passed"] is True
    assert score["per_label"]["identitas_terdakwa"]["exact_recall"] == 0
    assert score["per_label"]["identitas_terdakwa"]["tolerant_recall"] == 1


def test_structure_review_defers_acceptance_until_complete() -> None:
    rows = [
        {
            "document_id": "doc-1",
            "section_label": "amar_putusan",
            "detected": "yes",
            "gold_present": "yes",
            "review_status": "exact",
        },
        {
            "document_id": "doc-2",
            "section_label": "amar_putusan",
            "detected": "yes",
            "gold_present": "",
            "review_status": "",
        },
    ]

    score = score_structure_review(rows)

    assert score["completion"] == 0.5
    assert score["acceptance"] is None


def test_structure_review_requires_gold_presence_for_reviewed_rows() -> None:
    with pytest.raises(ValueError, match="gold_present"):
        score_structure_review(
            [
                {
                    "document_id": "doc-1",
                    "section_label": "amar_putusan",
                    "detected": "yes",
                    "gold_present": "",
                    "review_status": "exact",
                }
            ]
        )


def test_structure_review_rejects_status_inconsistent_with_detection() -> None:
    with pytest.raises(ValueError, match="requires detected=yes"):
        score_structure_review(
            [
                {
                    "document_id": "doc-1",
                    "section_label": "amar_putusan",
                    "detected": "no",
                    "gold_present": "yes",
                    "review_status": "exact",
                }
            ]
        )


def test_validate_questions_checks_approved_evidence_span() -> None:
    rows = [
        {
            "query_id": "q1",
            "document_id": "doc-1",
            "target_section_label": "amar_putusan",
            "question": "Apa amar putusan?",
            "reference_answer": "Pidana penjara.",
            "evidence_start_position": "6",
            "evidence_end_position": "19",
            "difficulty": "easy",
            "review_status": "approved",
        }
    ]

    result = validate_questions(rows, {"doc-1": "awal. Pidana penjara."})

    assert result["valid"] is True
    assert result["ready"] is True
    assert result["approved_count"] == 1


def test_validate_questions_allows_unreviewed_template_rows() -> None:
    rows = [
        {
            "query_id": "q1",
            "document_id": "doc-1",
            "target_section_label": "fakta",
            "review_status": "",
        }
    ]

    result = validate_questions(rows, {"doc-1": "Teks dokumen."})

    assert result["valid"] is True
    assert result["ready"] is False
    assert result["status_counts"] == {"unreviewed": 1}


def test_build_qrel_candidates_grades_full_and_partial_evidence() -> None:
    questions = [
        {
            "query_id": "q1",
            "document_id": "doc-1",
            "target_section_label": "fakta",
            "review_status": "approved",
            "evidence_start_position": "10",
            "evidence_end_position": "20",
        }
    ]
    chunks = [
        {
            "chunk_id": "fixed-1",
            "document_id": "doc-1",
            "strategy": "fixed_size",
            "start_position": 0,
            "end_position": 30,
        },
        {
            "chunk_id": "sac-1",
            "document_id": "doc-1",
            "strategy": "structure_aware",
            "start_position": 5,
            "end_position": 15,
        },
        {
            "chunk_id": "irrelevant",
            "document_id": "doc-1",
            "strategy": "structure_aware",
            "start_position": 20,
            "end_position": 30,
        },
    ]

    candidates = build_qrel_candidates(questions, chunks)

    assert [candidate["auto_grade"] for candidate in candidates] == [2, 1]
    assert candidates[0]["relevance_grade"] == 2
    assert candidates[1]["relevance_grade"] == ""
    assert candidates[1]["evidence_coverage"] == 0.5
