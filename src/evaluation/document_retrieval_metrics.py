"""Document-level retrieval diagnostics for legal corpora."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def evaluate_document_retrieval(
    expected_documents: Mapping[str, str],
    retrieved_documents: Mapping[str, Sequence[str]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Evaluate source-document fidelity, including Reuter et al.'s DRM.

    DRM@K is the fraction of the first K retrieved chunks that originate from
    the wrong source document.  Lower values are better.  Missing run entries
    are treated as entirely mismatched.
    """
    cutoffs = tuple(sorted(set(int(k) for k in ks)))
    if not expected_documents:
        raise ValueError("expected_documents must not be empty")
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("ks must contain positive integers")

    per_query: dict[str, dict[str, float]] = {}
    for query_id, expected in sorted(expected_documents.items()):
        ranking = list(retrieved_documents.get(query_id, ()))
        result: dict[str, float] = {}
        for k in cutoffs:
            top = ranking[:k]
            correct = sum(document_id == expected for document_id in top)
            # A short or missing ranking contributes mismatches for absent slots.
            result[f"drm@{k}"] = 1.0 - (correct / k)
            result[f"document_hit@{k}"] = float(correct > 0)
            first = next(
                (
                    rank
                    for rank, document_id in enumerate(top, start=1)
                    if document_id == expected
                ),
                None,
            )
            result[f"document_mrr@{k}"] = 1.0 / first if first is not None else 0.0
        per_query[query_id] = result

    metric_names = next(iter(per_query.values()))
    aggregate = {
        metric: statistics.fmean(row[metric] for row in per_query.values())
        for metric in metric_names
    }
    return {
        "query_count": len(per_query),
        "aggregate": aggregate,
        "per_query": per_query,
    }
