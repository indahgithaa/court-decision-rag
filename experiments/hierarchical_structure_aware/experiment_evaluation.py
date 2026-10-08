"""Ground-truth mapping, retrieval metrics, uncertainty, and error diagnosis."""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.evaluation.ground_truth import build_qrel_candidates
from src.evaluation.retrieval_metrics import evaluate_retrieval


def build_positive_qrels(
    questions: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    sections_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Create positive qrels using the frozen evidence-span protocol.

    The alternate-statute rule is reproduced from the validated Indo-Law qrel
    builder: another occurrence of the same cited provision inside the gold
    reasoning section is equivalent evidence for the same evidence unit.
    """
    candidates = build_qrel_candidates(questions, chunks)
    positives = [dict(row) for row in candidates if int(row["auto_grade"]) == 2]
    existing = {(str(row["query_id"]), str(row["chunk_id"])) for row in positives}
    chunks_by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[str(chunk["document_id"])].append(chunk)

    for question in questions:
        if str(question["target_section_label"]) != "pertimbangan_hukum":
            continue
        key_match = re.search(
            r"\bpasal\s+\d+(?:\s+ayat\s+\d+)?(?:\s+huruf\s+[a-z])?",
            str(question["reference_answer"]),
            flags=re.IGNORECASE,
        )
        if not key_match:
            continue
        document_id = str(question["document_id"])
        section = next(
            row
            for row in sections_by_document[document_id]
            if str(row["section_label"]) == "pertimbangan_hukum"
        )
        section_text = str(section["section_text"])
        section_start = int(section["start_position"])
        for occurrence in re.finditer(
            re.escape(key_match.group(0)), section_text, flags=re.IGNORECASE
        ):
            evidence_start = section_start + occurrence.start()
            evidence_end = section_start + occurrence.end()
            for chunk in chunks_by_document[document_id]:
                chunk_start = int(chunk["start_position"])
                chunk_end = int(chunk["end_position"])
                pair = (str(question["query_id"]), str(chunk["chunk_id"]))
                if pair in existing or not (
                    chunk_start <= evidence_start and evidence_end <= chunk_end
                ):
                    continue
                existing.add(pair)
                positives.append(
                    {
                        "query_id": str(question["query_id"]),
                        "document_id": document_id,
                        "target_section_label": str(
                            question["target_section_label"]
                        ),
                        "evidence_id": (
                            f"{question['query_id']}:"
                            f"{question['evidence_start_position']}:"
                            f"{question['evidence_end_position']}"
                        ),
                        "strategy": str(chunk["strategy"]),
                        "chunk_id": str(chunk["chunk_id"]),
                        "chunk_start_position": chunk_start,
                        "chunk_end_position": chunk_end,
                        "evidence_start_position": evidence_start,
                        "evidence_end_position": evidence_end,
                        "evidence_coverage": 1.0,
                        "auto_grade": 2,
                        "relevance_grade": 2,
                        "reviewer_notes": (
                            "Equivalent statute occurrence in the annotated "
                            "reasoning section."
                        ),
                    }
                )

    expected = {str(question["query_id"]) for question in questions}
    covered = {str(row["query_id"]) for row in positives}
    if covered != expected:
        missing = sorted(expected - covered)
        raise RuntimeError(f"Chunks lack complete evidence for {missing[:10]}")
    return sorted(
        positives,
        key=lambda row: (
            str(row["query_id"]),
            int(row["chunk_start_position"]),
            str(row["chunk_id"]),
        ),
    )


def evaluate_method(
    qrel_rows: Sequence[Mapping[str, Any]],
    run_records: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    sections_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Evaluate child rankings and separately diagnose delivered context."""
    qrels, evidence_qrels = _parse_qrels(qrel_rows)
    records = {str(record["query_id"]): record for record in run_records}
    question_by_id = {str(row["query_id"]): row for row in questions}
    chunk_by_id = {str(row["chunk_id"]): row for row in chunks}
    expected = set(question_by_id)
    if set(records) != expected or set(qrels) != expected:
        raise ValueError("Question, run, and qrel query IDs must match")
    rankings = {
        query_id: [
            str(item["chunk_id"])
            for item in sorted(
                records[query_id]["ranking"], key=lambda row: int(row["rank"])
            )
        ]
        for query_id in sorted(expected)
    }
    overall = evaluate_retrieval(
        qrels, rankings, evidence_qrels=evidence_qrels, ks=ks
    )
    labels = {
        query_id: str(question_by_id[query_id]["target_section_label"])
        for query_id in expected
    }
    by_section: dict[str, Any] = {}
    for label in sorted(set(labels.values())):
        query_ids = sorted(qid for qid, value in labels.items() if value == label)
        by_section[label] = evaluate_retrieval(
            {qid: qrels[qid] for qid in query_ids},
            {qid: rankings[qid] for qid in query_ids},
            evidence_qrels={qid: evidence_qrels[qid] for qid in query_ids},
            ks=ks,
        )

    evidence_spans: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    for row in qrel_rows:
        evidence_spans[str(row["query_id"])].add(
            (
                str(row["document_id"]),
                int(row["evidence_start_position"]),
                int(row["evidence_end_position"]),
            )
        )

    relevant_ranks: dict[str, int | None] = {}
    context_hits: dict[str, float] = {}
    context_words: list[int] = []
    context_units: list[int] = []
    error_types: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for query_id in sorted(expected):
        record = records[query_id]
        ranking = rankings[query_id]
        relevant = {cid for cid, grade in qrels[query_id].items() if grade > 0}
        relevant_rank = next(
            (rank for rank, cid in enumerate(ranking, start=1) if cid in relevant),
            None,
        )
        relevant_ranks[query_id] = relevant_rank
        contexts = list(record.get("returned_context", ()))
        context_words.append(int(record.get("returned_context_words", 0)))
        context_units.append(len(contexts))
        hit = any(
            str(context["document_id"]) == document_id
            and int(context["start_position"]) <= evidence_start
            and evidence_end <= int(context["end_position"])
            for context in contexts
            for document_id, evidence_start, evidence_end in evidence_spans[query_id]
        )
        context_hits[query_id] = float(hit)

        if relevant_rank is not None and relevant_rank <= 5:
            error_types["success_top5"] += 1
            continue
        question = question_by_id[query_id]
        gold_document = str(question["document_id"])
        target_label = str(question["target_section_label"])
        top5_chunks = [chunk_by_id[cid] for cid in ranking[:5]]
        if not any(str(chunk["document_id"]) == gold_document for chunk in top5_chunks):
            category = "wrong_document_top5"
        elif not any(
            str(chunk["document_id"]) == gold_document
            and _overlaps_target_section(
                chunk, target_label, sections_by_document[gold_document]
            )
            for chunk in top5_chunks
        ):
            category = "wrong_section_top5"
        elif relevant_rank is not None:
            category = "evidence_rank_6_to_50"
        else:
            category = "evidence_not_in_top50"
        error_types[category] += 1
        failures.append(
            {
                "query_id": query_id,
                "document_id": gold_document,
                "target_section_label": target_label,
                "category": category,
                "relevant_rank": relevant_rank,
                "top5_document_ids": [
                    str(chunk["document_id"]) for chunk in top5_chunks
                ],
                "top5_chunk_ids": [str(chunk["chunk_id"]) for chunk in top5_chunks],
                "context_contains_evidence": hit,
            }
        )

    context_by_section: dict[str, dict[str, float | int]] = {}
    for label in sorted(set(labels.values())):
        values = [
            context_hits[qid] for qid, value in labels.items() if value == label
        ]
        context_by_section[label] = {
            "query_count": len(values),
            "evidence_hit_rate": statistics.fmean(values),
        }

    return {
        "overall": overall,
        "by_section": by_section,
        "context_delivery": {
            "evidence_hit_rate": statistics.fmean(context_hits.values()),
            "mean_returned_words": statistics.fmean(context_words),
            "median_returned_words": statistics.median(context_words),
            "mean_context_units": statistics.fmean(context_units),
            "by_section": context_by_section,
            "per_query": context_hits,
        },
        "relevant_rank": relevant_ranks,
        "error_analysis": {
            "counts": dict(sorted(error_types.items())),
            "failures": failures,
        },
    }


def paired_cluster_bootstrap(
    left_per_query: Mapping[str, Mapping[str, float]],
    right_per_query: Mapping[str, Mapping[str, float]],
    query_documents: Mapping[str, str],
    *,
    metrics: Sequence[str],
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Estimate paired mean differences by resampling whole documents."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    query_ids = sorted(left_per_query)
    if set(query_ids) != set(right_per_query) or set(query_ids) != set(query_documents):
        raise ValueError("paired inputs must have identical query IDs")
    by_document: dict[str, list[str]] = defaultdict(list)
    for query_id in query_ids:
        by_document[str(query_documents[query_id])].append(query_id)
    documents = sorted(by_document)
    generator = random.Random(seed)
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        observed = statistics.fmean(
            float(left_per_query[qid][metric])
            - float(right_per_query[qid][metric])
            for qid in query_ids
        )
        means: list[float] = []
        for _ in range(samples):
            sampled_documents = [
                documents[generator.randrange(len(documents))]
                for _ in documents
            ]
            differences = [
                float(left_per_query[qid][metric])
                - float(right_per_query[qid][metric])
                for document_id in sampled_documents
                for qid in by_document[document_id]
            ]
            means.append(statistics.fmean(differences))
        means.sort()
        result[metric] = {
            "mean_difference": observed,
            "ci95_low": _percentile(means, 0.025),
            "ci95_high": _percentile(means, 0.975),
        }
    return result


def summarize_chunk_lengths(
    chunks: Sequence[Mapping[str, Any]], token_lengths: Sequence[int], *, limit: int
) -> dict[str, Any]:
    if len(chunks) != len(token_lengths) or not chunks:
        raise ValueError("chunks and token lengths must be non-empty and aligned")
    word_lengths = [len(re.findall(r"\S+", str(chunk["text"]))) for chunk in chunks]
    ordered_tokens = sorted(int(value) for value in token_lengths)
    ordered_words = sorted(word_lengths)
    return {
        "chunk_count": len(chunks),
        "word_length": _distribution(ordered_words),
        "encoder_token_length_with_prefix": _distribution(ordered_tokens),
        "encoder_limit": limit,
        "over_encoder_limit": sum(value > limit for value in token_lengths),
    }


def _parse_qrels(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, tuple[str, ...]]],
]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    evidence: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        query_id = str(row["query_id"])
        chunk_id = str(row["chunk_id"])
        grade = float(row["relevance_grade"])
        qrels[query_id][chunk_id] = grade
        if grade > 0:
            evidence[query_id][str(row["evidence_id"])].add(chunk_id)
    return dict(qrels), {
        query_id: {
            evidence_id: tuple(sorted(chunk_ids))
            for evidence_id, chunk_ids in by_evidence.items()
        }
        for query_id, by_evidence in evidence.items()
    }


def _overlaps_target_section(
    chunk: Mapping[str, Any],
    target_label: str,
    sections: Sequence[Mapping[str, Any]],
) -> bool:
    chunk_start = int(chunk["start_position"])
    chunk_end = int(chunk["end_position"])
    return any(
        str(section["section_label"]) == target_label
        and max(chunk_start, int(section["start_position"]))
        < min(chunk_end, int(section["end_position"]))
        for section in sections
    )


def _distribution(sorted_values: Sequence[int]) -> dict[str, float | int]:
    return {
        "min": int(sorted_values[0]),
        "mean": statistics.fmean(sorted_values),
        "median": statistics.median(sorted_values),
        "p95": _percentile(sorted_values, 0.95),
        "max": int(sorted_values[-1]),
    }


def _percentile(sorted_values: Sequence[float | int], probability: float) -> float:
    if not sorted_values:
        raise ValueError("values must not be empty")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower]) + (
        float(sorted_values[upper]) - float(sorted_values[lower])
    ) * fraction

