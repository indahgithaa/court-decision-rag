import pytest

from src.evaluation.document_retrieval_metrics import evaluate_document_retrieval


def test_document_retrieval_reports_drm_hit_and_mrr() -> None:
    result = evaluate_document_retrieval(
        {"q1": "d1", "q2": "d2"},
        {
            "q1": ["wrong", "d1", "wrong"],
            "q2": ["d2", "d2", "wrong"],
        },
        ks=(1, 3),
    )

    q1 = result["per_query"]["q1"]
    assert q1["drm@1"] == 1.0
    assert q1["drm@3"] == pytest.approx(2 / 3)
    assert q1["document_hit@3"] == 1.0
    assert q1["document_mrr@3"] == 0.5
    assert result["aggregate"]["document_hit@1"] == 0.5


def test_document_retrieval_treats_missing_ranking_as_mismatch() -> None:
    result = evaluate_document_retrieval({"q": "d"}, {}, ks=(5,))

    assert result["aggregate"]["drm@5"] == 1.0
    assert result["aggregate"]["document_hit@5"] == 0.0


def test_document_retrieval_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        evaluate_document_retrieval({}, {}, ks=(1,))
    with pytest.raises(ValueError):
        evaluate_document_retrieval({"q": "d"}, {}, ks=(0,))
