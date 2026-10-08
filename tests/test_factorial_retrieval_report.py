from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluation.factorial_retrieval_report import (
    evaluate_factorial_retrieval,
    render_factorial_retrieval_markdown,
)
from src.utils.config import load_config


QUESTIONS = [
    {
        "query_id": "q1",
        "document_id": "d1",
        "target_section_label": "fakta",
        "review_status": "approved",
    },
    {
        "query_id": "q2",
        "document_id": "d2",
        "target_section_label": "hukum",
        "review_status": "approved",
    },
]


def _qrels(strategy: str, prefix: str) -> list[dict[str, str]]:
    return [
        {
            "query_id": "q1",
            "document_id": "d1",
            "target_section_label": "fakta",
            "evidence_id": "q1:e1",
            "strategy": strategy,
            "chunk_id": f"{prefix}1",
            "evidence_start_position": "10",
            "evidence_end_position": "20",
            "relevance_grade": "2",
        },
        {
            "query_id": "q2",
            "document_id": "d2",
            "target_section_label": "hukum",
            "evidence_id": "q2:e1",
            "strategy": strategy,
            "chunk_id": f"{prefix}2",
            "evidence_start_position": "30",
            "evidence_end_position": "40",
            "relevance_grade": "2",
        },
    ]


def _record(
    query_id: str,
    *,
    strategy: str,
    embedding: str,
    chunks: list[tuple[str, str]],
) -> dict:
    document_id = "d1" if query_id == "q1" else "d2"
    section = "fakta" if query_id == "q1" else "hukum"
    return {
        "query_id": query_id,
        "document_id": document_id,
        "target_section_label": section,
        "strategy": strategy,
        "embedding_method": embedding,
        "model_name": "test-model",
        "retrieval_scope": "full_corpus",
        "latency_ms": 2.0,
        "ranking": [
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "retrieved_document_id": retrieved_document,
            }
            for rank, (chunk_id, retrieved_document) in enumerate(chunks, start=1)
        ],
    }


def _runs() -> dict[str, list[dict]]:
    return {
        "fixed_independent": [
            _record(
                "q1",
                strategy="fixed_size",
                embedding="independent",
                chunks=[("fx1", "other"), ("f1", "d1")],
            ),
            _record(
                "q2",
                strategy="fixed_size",
                embedding="independent",
                chunks=[("fx2", "other"), ("fx3", "other")],
            ),
        ],
        "fixed_contextual": [
            _record(
                "q1",
                strategy="fixed_size",
                embedding="contextual",
                chunks=[("f1", "d1"), ("fx1", "other")],
            ),
            _record(
                "q2",
                strategy="fixed_size",
                embedding="contextual",
                chunks=[("fx2", "other"), ("fx3", "other")],
            ),
        ],
        "sac_independent": [
            _record(
                "q1",
                strategy="structure_aware",
                embedding="independent",
                chunks=[("sx1", "other"), ("s1", "d1")],
            ),
            _record(
                "q2",
                strategy="structure_aware",
                embedding="independent",
                chunks=[("s2", "d2"), ("sx2", "other")],
            ),
        ],
        "sac_contextual": [
            _record(
                "q1",
                strategy="structure_aware",
                embedding="contextual",
                chunks=[("s1", "d1"), ("sx1", "other")],
            ),
            _record(
                "q2",
                strategy="structure_aware",
                embedding="contextual",
                chunks=[("s2", "d2"), ("sx2", "other")],
            ),
        ],
    }


def test_factorial_report_computes_planned_contrasts_and_wtl() -> None:
    result = evaluate_factorial_retrieval(
        _qrels("fixed_size", "f"),
        _qrels("structure_aware", "s"),
        QUESTIONS,
        _runs(),
        ks=(1, 2),
        bootstrap_samples=100,
        seed=7,
    )

    assert result["design"] == "2x2_chunking_x_embedding"
    assert result["evidence_units_per_query"] == {
        "min": 1,
        "max": 1,
        "mean": 1.0,
    }
    assert result["positive_relevance_grades"] == {
        "fixed": [2.0],
        "sac": [2.0],
    }
    assert result["positive_chunk_qrel_counts"] == {"fixed": 2, "sac": 2}
    assert result["arms"]["sac_contextual"]["overall"]["aggregate"][
        "recall@1"
    ] == 1.0
    contrasts = result["planned_contrasts"]
    assert contrasts["contextual_effect_within_fixed"]["metrics"]["recall@1"][
        "mean_difference"
    ] == 0.5
    assert contrasts["combined_vs_conventional_baseline"]["metrics"]["recall@1"][
        "mean_difference"
    ] == 1.0
    assert contrasts["chunking_embedding_interaction"]["metrics"]["recall@1"][
        "mean_difference"
    ] == 0.0
    outcome = contrasts["contextual_effect_within_fixed"]["metrics"]["recall@1"][
        "win_tie_loss"
    ]
    assert outcome == {
        "wins": 1,
        "ties": 1,
        "losses": 0,
        "win_rate": 0.5,
        "tie_rate": 0.5,
        "loss_rate": 0.0,
    }

    markdown = render_factorial_retrieval_markdown(result)
    assert "Empat kondisi eksperimen" in markdown
    assert "chunking_embedding_interaction" in markdown
    assert "Fixed + contextual" in markdown
    assert "Hasil per bagian" in markdown
    assert "secara numerik sama dengan evidence Hit@K" in markdown


def test_factorial_report_rejects_nonshared_evidence_units() -> None:
    sac_qrels = deepcopy(_qrels("structure_aware", "s"))
    sac_qrels[1]["evidence_id"] = "q2:different"

    with pytest.raises(ValueError, match="identical evidence units"):
        evaluate_factorial_retrieval(
            _qrels("fixed_size", "f"),
            sac_qrels,
            QUESTIONS,
            _runs(),
            ks=(1, 2),
            bootstrap_samples=10,
        )


def test_factorial_report_allows_different_occurrences_for_shared_evidence() -> None:
    fixed_qrels = _qrels("fixed_size", "f")
    fixed_qrels.append(
        {
            **fixed_qrels[0],
            "chunk_id": "f1-equivalent",
            "evidence_start_position": "50",
            "evidence_end_position": "60",
        }
    )
    sac_qrels = _qrels("structure_aware", "s")
    sac_qrels.extend(
        [
            {
                **sac_qrels[0],
                "chunk_id": "s1-equivalent-a",
                "evidence_start_position": "70",
                "evidence_end_position": "80",
            },
            {
                **sac_qrels[0],
                "chunk_id": "s1-equivalent-b",
                "evidence_start_position": "90",
                "evidence_end_position": "100",
            },
        ]
    )

    result = evaluate_factorial_retrieval(
        fixed_qrels,
        sac_qrels,
        QUESTIONS,
        _runs(),
        ks=(1, 2),
        bootstrap_samples=10,
    )

    assert result["recall_unit"] == "shared_evidence"
    assert result["query_count"] == 2


def test_factorial_report_rejects_unlabelled_contextual_run() -> None:
    runs = _runs()
    runs["fixed_contextual"][0].pop("embedding_method")

    with pytest.raises(ValueError, match="must declare embedding_method"):
        evaluate_factorial_retrieval(
            _qrels("fixed_size", "f"),
            _qrels("structure_aware", "s"),
            QUESTIONS,
            runs,
            ks=(1, 2),
            bootstrap_samples=10,
        )


def test_fixed_contextual_config_matches_factorial_control() -> None:
    project_root = Path(__file__).resolve().parents[1]
    independent = load_config(project_root / "configs/indolaw_fixed_independent.yaml")
    contextual = load_config(project_root / "configs/indolaw_fixed_contextual.yaml")

    assert independent["chunking"] == contextual["chunking"]
    assert independent["embedding"]["model_name"] == contextual["embedding"][
        "model_name"
    ]
    assert independent["embedding"]["method"] == "independent"
    assert contextual["embedding"]["method"] == "contextual"
    assert contextual["embedding"]["context_window_tokens"] == 512
    assert contextual["retrieval"]["top_k"] == 50
