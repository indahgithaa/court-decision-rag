"""Evaluation metrics for structure detection and retrieval experiments."""

from src.evaluation.error_analysis import (
    analyze_retrieval_errors,
    render_error_markdown,
)
from src.evaluation.document_retrieval_metrics import evaluate_document_retrieval
from src.evaluation.contextual_embedding_report import (
    evaluate_contextual_embedding_runs,
    render_contextual_embedding_markdown,
)
from src.evaluation.ground_truth import (
    build_qrel_candidates,
    join_clean_pages,
    validate_questions,
)
from src.evaluation.retrieval_metrics import (
    evidence_recall_at_k,
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.evaluation.retrieval_report import (
    evaluate_paired_runs,
    paired_cluster_bootstrap_interval,
    render_markdown,
)
from src.evaluation.sample_size import (
    cluster_adjusted_plan,
    estimate_equal_cluster_icc,
    paired_mean_sample_size,
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
    "analyze_retrieval_errors",
    "build_qrel_candidates",
    "evidence_recall_at_k",
    "evaluate_document_retrieval",
    "evaluate_contextual_embedding_runs",
    "evaluate_retrieval",
    "evaluate_paired_runs",
    "paired_cluster_bootstrap_interval",
    "paired_mean_sample_size",
    "cluster_adjusted_plan",
    "estimate_equal_cluster_icc",
    "join_clean_pages",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "render_error_markdown",
    "render_contextual_embedding_markdown",
    "render_markdown",
    "render_structure_score",
    "render_sac_compliance",
    "score_structure_review",
    "validate_questions",
    "validate_sac_compliance",
]
