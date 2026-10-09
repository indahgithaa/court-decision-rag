"""Method-independent qrels, retrieval metrics, and paired inference."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


DEFAULT_KS = (1, 3, 5, 10, 20, 50)


def map_evidence_to_chunks(
    questions: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, list[str]]], list[dict[str, Any]]]:
    """Map source spans to canonical chunks once for all methods."""
    by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_document[str(chunk["document_id"])].append(chunk)
    qrels: dict[str, dict[str, float]] = {}
    evidence_qrels: dict[str, dict[str, list[str]]] = {}
    rows: list[dict[str, Any]] = []
    for question in questions:
        query_id = str(question["query_id"])
        document_id = str(question["document_id"])
        start = int(question["evidence_start_position"])
        end = int(question["evidence_end_position"])
        evidence_id = str(question.get("evidence_id") or f"{query_id}:{start}:{end}")
        if end <= start:
            raise ValueError(f"Invalid evidence span for {query_id}: {start}:{end}")
        grades: dict[str, float] = {}
        candidates: list[str] = []
        for chunk in by_document.get(document_id, ()):
            chunk_start = int(chunk["start_position"])
            chunk_end = int(chunk["end_position"])
            overlap = max(0, min(end, chunk_end) - max(start, chunk_start))
            if not overlap:
                continue
            contains = chunk_start <= start and end <= chunk_end
            grade = 2.0 if contains else 1.0
            chunk_id = str(chunk["chunk_id"])
            grades[chunk_id] = grade
            candidates.append(chunk_id)
            rows.append(
                {
                    "query_id": query_id,
                    "evidence_id": evidence_id,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "relevance_grade": int(grade),
                    "evidence_coverage": overlap / (end - start),
                    "mapping_rule": "complete_span_grade_2_partial_overlap_grade_1",
                }
            )
        if not candidates:
            raise ValueError(f"No canonical chunk overlaps evidence for {query_id}")
        qrels[query_id] = grades
        evidence_qrels[query_id] = {evidence_id: candidates}
    return qrels, evidence_qrels, rows


def evaluate_method(
    questions: Sequence[Mapping[str, Any]],
    ranking_records: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, float]],
    evidence_qrels: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Compute evidence Recall, first-hit MRR, graded nDCG, and DRM."""
    question_by_id = {str(row["query_id"]): row for row in questions}
    run_by_id = {str(row["query_id"]): row for row in ranking_records}
    if set(question_by_id) != set(qrels) or set(question_by_id) != set(evidence_qrels):
        raise ValueError("questions and qrels must have identical query IDs")
    per_query: dict[str, dict[str, Any]] = {}
    for query_id in sorted(question_by_id):
        question = question_by_id[query_id]
        run = run_by_id.get(query_id)
        ranking = list(run.get("ranking", ())) if run else []
        ranked_ids = [str(item["chunk_id"]) for item in ranking]
        ranked_docs = [str(item["document_id"]) for item in ranking]
        relevance = qrels[query_id]
        evidence = evidence_qrels[query_id]
        valid_documents = _valid_documents(question)
        metrics: dict[str, float] = {}
        for k in sorted(set(int(value) for value in ks)):
            top_ids = ranked_ids[:k]
            top_docs = ranked_docs[:k]
            retrieved = set(top_ids)
            covered = sum(
                bool(retrieved.intersection(chunk_ids))
                for chunk_ids in evidence.values()
            )
            metrics[f"recall@{k}"] = covered / len(evidence)
            metrics[f"mrr@{k}"] = _reciprocal_rank(relevance, top_ids)
            metrics[f"ndcg@{k}"] = _ndcg(relevance, top_ids, k)
            metrics[f"drm@{k}"] = (
                sum(document not in valid_documents for document in top_docs) / len(top_docs)
                if top_docs
                else 1.0
            )
        per_query[query_id] = {
            "query_id": query_id,
            "document_id": str(question["document_id"]),
            "target_section_label": str(question["target_section_label"]),
            **metrics,
        }
    metric_names = [
        key for key in next(iter(per_query.values())) if "@" in key
    ]
    aggregate = {
        metric: statistics.fmean(float(row[metric]) for row in per_query.values())
        for metric in metric_names
    }
    by_section: dict[str, dict[str, Any]] = {}
    labels = sorted({str(row["target_section_label"]) for row in per_query.values()})
    for label in labels:
        selected = [
            row for row in per_query.values() if row["target_section_label"] == label
        ]
        by_section[label] = {
            "query_count": len(selected),
            **{
                metric: statistics.fmean(float(row[metric]) for row in selected)
                for metric in metric_names
            },
        }
    macro_by_section = {
        metric: statistics.fmean(float(row[metric]) for row in by_section.values())
        for metric in metric_names
    }
    return {
        "query_count": len(per_query),
        "metric_definitions": {
            "recall": "fraction of annotated evidence units covered within top-k",
            "mrr": "reciprocal rank of first positive canonical chunk within top-k",
            "ndcg": "graded nDCG with complete-span grade 2 and partial-overlap grade 1",
            "drm": "fraction of returned top-k chunks outside valid source documents; lower is better",
        },
        "aggregate_micro": aggregate,
        "aggregate_macro_sections": macro_by_section,
        "by_section": by_section,
        "per_query": per_query,
    }


def paired_cluster_comparisons(
    method_results: Mapping[str, Mapping[str, Any]],
    *,
    pairs: Sequence[tuple[str, str]] = (("B1", "B2"), ("B2", "M1"), ("B1", "M1")),
    metrics: Sequence[str] = ("recall@5", "mrr@5", "ndcg@5", "drm@5"),
    seed: int = 42,
    bootstrap_samples: int = 2_000,
    permutation_samples: int = 10_000,
) -> list[dict[str, Any]]:
    """Paired document-cluster bootstrap CIs and sign-flip randomization."""
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for baseline, candidate in pairs:
        base_rows = method_results[baseline]["per_query"]
        cand_rows = method_results[candidate]["per_query"]
        if set(base_rows) != set(cand_rows):
            raise ValueError(f"Paired query mismatch for {baseline}/{candidate}")
        for metric in metrics:
            observations = [
                (
                    query_id,
                    str(base_rows[query_id]["document_id"]),
                    float(base_rows[query_id][metric]),
                    float(cand_rows[query_id][metric]),
                )
                for query_id in sorted(base_rows)
            ]
            baseline_mean = statistics.fmean(row[2] for row in observations)
            candidate_mean = statistics.fmean(row[3] for row in observations)
            difference = candidate_mean - baseline_mean
            clusters: dict[str, list[float]] = defaultdict(list)
            for _, document_id, base_value, candidate_value in observations:
                clusters[document_id].append(candidate_value - base_value)
            document_ids = sorted(clusters)
            bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
            for index in range(bootstrap_samples):
                sampled = rng.choice(document_ids, size=len(document_ids), replace=True)
                values = [value for document_id in sampled for value in clusters[str(document_id)]]
                bootstrap[index] = float(np.mean(values))
            cluster_sums = np.asarray(
                [sum(clusters[document_id]) for document_id in document_ids], dtype=np.float64
            )
            total_questions = sum(len(values) for values in clusters.values())
            extreme = 0
            for _ in range(permutation_samples):
                signs = rng.choice((-1.0, 1.0), size=len(document_ids))
                permuted = float(np.sum(cluster_sums * signs) / total_questions)
                extreme += abs(permuted) >= abs(difference)
            relative = difference / abs(baseline_mean) if baseline_mean else None
            output.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "query_count": len(observations),
                    "document_count": len(document_ids),
                    "baseline_mean": baseline_mean,
                    "candidate_mean": candidate_mean,
                    "absolute_difference": difference,
                    "relative_difference": relative,
                    "ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "paired_cluster_sign_flip_p": (extreme + 1) / (permutation_samples + 1),
                    "bootstrap_samples": bootstrap_samples,
                    "permutation_samples": permutation_samples,
                    "seed": seed,
                }
            )
    return output


def per_section_comparisons(
    method_results: Mapping[str, Mapping[str, Any]],
    *,
    pairs: Sequence[tuple[str, str]] = (("B1", "B2"), ("B2", "M1"), ("B1", "M1")),
    metric: str = "recall@5",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for baseline, candidate in pairs:
        base_rows = method_results[baseline]["per_query"]
        cand_rows = method_results[candidate]["per_query"]
        labels = sorted({str(row["target_section_label"]) for row in base_rows.values()})
        for label in labels:
            query_ids = [
                query_id
                for query_id, row in base_rows.items()
                if str(row["target_section_label"]) == label
            ]
            differences = [
                float(cand_rows[query_id][metric]) - float(base_rows[query_id][metric])
                for query_id in query_ids
            ]
            output.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "section_label": label,
                    "metric": metric,
                    "query_count": len(query_ids),
                    "baseline_mean": statistics.fmean(
                        float(base_rows[query_id][metric]) for query_id in query_ids
                    ),
                    "candidate_mean": statistics.fmean(
                        float(cand_rows[query_id][metric]) for query_id in query_ids
                    ),
                    "absolute_difference": statistics.fmean(differences),
                }
            )
    return output


def classify_failures(
    questions: Sequence[Mapping[str, Any]],
    ranking_records: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, float]],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    question_by_id = {str(row["query_id"]): row for row in questions}
    run_by_id = {str(row["query_id"]): row for row in ranking_records}
    details: list[dict[str, Any]] = []
    for query_id, question in sorted(question_by_id.items()):
        ranking = list(run_by_id[query_id]["ranking"])
        relevant = {chunk_id for chunk_id, grade in qrels[query_id].items() if grade > 0}
        first_relevant = next(
            (int(item["rank"]) for item in ranking if str(item["chunk_id"]) in relevant),
            None,
        )
        if first_relevant is not None and first_relevant <= k:
            category = "relevant_evidence_in_top_k"
        else:
            top = ranking[:k]
            valid_docs = _valid_documents(question)
            correct_document = [
                item for item in top if str(item["document_id"]) in valid_docs
            ]
            target_section = str(question["target_section_label"])
            correct_section = [
                item
                for item in correct_document
                if str(chunks_by_id[str(item["chunk_id"])].get("section_label"))
                == target_section
            ]
            if not correct_document:
                category = "wrong_source_document"
            elif not correct_section:
                category = "correct_document_wrong_section"
            else:
                category = "correct_document_and_section_wrong_passage"
            if first_relevant is not None and first_relevant > k:
                category += "+evidence_beyond_top_k"
        details.append(
            {
                "query_id": query_id,
                "document_id": str(question["document_id"]),
                "target_section_label": str(question["target_section_label"]),
                "category": category,
                "first_relevant_rank": first_relevant,
            }
        )
    return {
        "k": k,
        "query_count": len(details),
        "category_counts": dict(sorted(Counter(row["category"] for row in details).items())),
        "details": details,
    }


def _valid_documents(question: Mapping[str, Any]) -> set[str]:
    values = question.get("relevant_document_ids")
    if values:
        if isinstance(values, str):
            return {value.strip() for value in values.split("|") if value.strip()}
        return {str(value) for value in values}
    return {str(question["document_id"])}


def _reciprocal_rank(relevance: Mapping[str, float], ranking: Sequence[str]) -> float:
    for rank, chunk_id in enumerate(ranking, start=1):
        if float(relevance.get(chunk_id, 0.0)) > 0:
            return 1.0 / rank
    return 0.0


def _ndcg(relevance: Mapping[str, float], ranking: Sequence[str], k: int) -> float:
    gains = [float(relevance.get(chunk_id, 0.0)) for chunk_id in ranking[:k]]
    ideal = sorted((float(value) for value in relevance.values() if value > 0), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def _dcg(values: Sequence[float]) -> float:
    return sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(values, 1))
