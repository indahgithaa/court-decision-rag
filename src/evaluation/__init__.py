"""Evaluation metrics for structure detection and retrieval experiments."""

from src.evaluation.ground_truth import (
    build_qrel_candidates,
    join_clean_pages,
    validate_questions,
)
from src.evaluation.retrieval_metrics import (
    evaluate_retrieval,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.evaluation.structure_metrics import (
    render_structure_score,
    score_structure_review,
)
from src.evaluation.sac_compliance import (
    render_sac_compliance,
    validate_sac_compliance,
)

__all__ = [
    "build_qrel_candidates",
    "evaluate_retrieval",
    "hit_rate_at_k",
    "join_clean_pages",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "render_structure_score",
    "render_sac_compliance",
    "score_structure_review",
    "validate_questions",
    "validate_sac_compliance",
]
from .retrieval_report import evaluate_paired_runs, render_markdown

__all__ = ["evaluate_paired_runs", "render_markdown"]
