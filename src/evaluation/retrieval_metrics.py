"""Dependency-free Recall, MRR, and nDCG metrics for retrieval runs."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


Qrels = Mapping[str, Mapping[str, float]]
Runs = Mapping[str, Sequence[str]]
EvidenceQrels = Mapping[str, Mapping[str, Sequence[str]]]


def evaluate_retrieval(
    qrels: Qrels,
    runs: Runs,
    *,
    evidence_qrels: EvidenceQrels,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Evaluate ranked chunk IDs for every query in ``qrels``.

    Relevance values may be binary or graded. Recall is calculated over shared
    evidence units rather than strategy-specific chunk IDs; MRR treats any
    positive chunk grade as relevant and nDCG preserves the supplied chunk
    grades. Queries absent from ``runs`` receive zero for every metric.
    """
    cutoffs = tuple(sorted(set(ks)))
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("ks must contain positive integers")
    if not qrels:
        raise ValueError("qrels must contain at least one query")
    if set(evidence_qrels) != set(qrels):
        raise ValueError("evidence_qrels must contain the same query IDs as qrels")

    per_query: dict[str, dict[str, float]] = {}
    for query_id in sorted(qrels):
        relevance = qrels[query_id]
        if not any(value > 0 for value in relevance.values()):
            raise ValueError(f"Query {query_id!r} has no relevant chunks")
        evidence = evidence_qrels[query_id]
        if not evidence or any(not chunk_ids for chunk_ids in evidence.values()):
            raise ValueError(
                f"Query {query_id!r} must map every evidence unit to chunks"
            )
        ranking = _deduplicate(runs.get(query_id, ()))
        result: dict[str, float] = {}
        for k in cutoffs:
            result[f"recall@{k}"] = evidence_recall_at_k(evidence, ranking, k)
            result[f"mrr@{k}"] = reciprocal_rank_at_k(relevance, ranking, k)
            result[f"ndcg@{k}"] = ndcg_at_k(relevance, ranking, k)
        per_query[query_id] = result

    metric_names = next(iter(per_query.values()))
    aggregate = {
        metric: statistics.fmean(result[metric] for result in per_query.values())
        for metric in metric_names
    }
    return {
        "query_count": len(per_query),
        "recall_unit": "evidence",
        "aggregate": aggregate,
        "per_query": per_query,
    }


def evidence_recall_at_k(
    evidence: Mapping[str, Sequence[str]], ranking: Sequence[str], k: int
) -> float:
    """Return the fraction of shared evidence units covered in the first ``k``.

    One evidence unit may be contained by several overlapping chunks. It is
    counted once when any supporting chunk is retrieved, so changing chunk size
    or overlap cannot change the Recall denominator.
    """
    _validate_k(k)
    if not evidence:
        raise ValueError("evidence must contain at least one unit")
    retrieved = set(ranking[:k])
    covered = sum(bool(retrieved.intersection(chunk_ids)) for chunk_ids in evidence.values())
    return covered / len(evidence)


def recall_at_k(
    relevance: Mapping[str, float], ranking: Sequence[str], k: int
) -> float:
    """Return the fraction of relevant chunks retrieved in the first ``k``."""
    _validate_k(k)
    relevant = {chunk_id for chunk_id, value in relevance.items() if value > 0}
    if not relevant:
        raise ValueError("relevance must contain at least one positive value")
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def reciprocal_rank_at_k(
    relevance: Mapping[str, float], ranking: Sequence[str], k: int
) -> float:
    """Return reciprocal rank of the first relevant chunk, capped at ``k``."""
    _validate_k(k)
    for rank, chunk_id in enumerate(ranking[:k], start=1):
        if relevance.get(chunk_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    relevance: Mapping[str, float], ranking: Sequence[str], k: int
) -> float:
    """Return normalized discounted cumulative gain at ``k``."""
    _validate_k(k)
    gains = [relevance.get(chunk_id, 0) for chunk_id in ranking[:k]]
    ideal = sorted((value for value in relevance.values() if value > 0), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def _dcg(relevances: Sequence[float]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def _deduplicate(ranking: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(ranking))


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
