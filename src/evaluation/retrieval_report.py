"""Paired evaluation report for fixed-size and structure-aware retrieval."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .latency import summarize_latency
from .retrieval_metrics import evaluate_retrieval


STRATEGIES = ("fixed_size", "structure_aware")


def evaluate_paired_runs(
    qrel_rows: Sequence[Mapping[str, Any]],
    run_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate two strategy runs and paired SAC-minus-baseline differences.

    Confidence intervals resample documents, not individual questions. The
    four questions attached to one judgment share language and evidence, so a
    question-level bootstrap can be overconfident.
    """
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    qrels = {strategy: _build_qrels(qrel_rows, strategy) for strategy in STRATEGIES}
    parsed_runs = {
        strategy: _parse_run_records(run_records.get(strategy, ()), strategy)
        for strategy in STRATEGIES
    }
    expected_queries = set(qrels["fixed_size"])
    if set(qrels["structure_aware"]) != expected_queries:
        raise ValueError("Strategies must contain qrels for the same query IDs")
    query_documents = _query_documents(qrel_rows, expected_queries)

    labels = parsed_runs["fixed_size"]["labels"]
    if set(labels) != expected_queries:
        raise ValueError("Fixed-size run query IDs do not match qrels")
    if parsed_runs["structure_aware"]["labels"] != labels:
        raise ValueError("Runs must contain identical query IDs and section labels")
    scopes = {parsed_runs[strategy]["scope"] for strategy in STRATEGIES}
    if len(scopes) != 1:
        raise ValueError("Runs must use the same retrieval scope")
    retrieval_scope = scopes.pop()

    strategies: dict[str, Any] = {}
    for strategy in STRATEGIES:
        rankings = parsed_runs[strategy]["rankings"]
        overall = evaluate_retrieval(qrels[strategy], rankings, ks=ks)
        by_section: dict[str, Any] = {}
        for label in sorted(set(labels.values())):
            query_ids = {query_id for query_id, value in labels.items() if value == label}
            by_section[label] = evaluate_retrieval(
                {query_id: qrels[strategy][query_id] for query_id in query_ids},
                {query_id: rankings[query_id] for query_id in query_ids},
                ks=ks,
            )
        strategies[strategy] = {
            "overall": overall,
            "by_section": by_section,
            "latency": summarize_latency(parsed_runs[strategy]["latencies"]),
        }

    fixed_per_query = strategies["fixed_size"]["overall"]["per_query"]
    structure_per_query = strategies["structure_aware"]["overall"]["per_query"]
    metric_names = list(strategies["fixed_size"]["overall"]["aggregate"])
    paired: dict[str, Any] = {}
    for metric in metric_names:
        differences = [
            structure_per_query[query_id][metric] - fixed_per_query[query_id][metric]
            for query_id in sorted(expected_queries)
        ]
        differences_by_document: dict[str, list[float]] = defaultdict(list)
        for query_id, difference in zip(sorted(expected_queries), differences):
            differences_by_document[query_documents[query_id]].append(difference)
        low, high = paired_cluster_bootstrap_interval(
            differences_by_document,
            samples=bootstrap_samples,
            seed=seed,
        )
        paired[metric] = {
            "mean_difference": statistics.fmean(differences),
            "ci95_low": low,
            "ci95_high": high,
        }

    return {
        "query_count": len(expected_queries),
        "document_count": len(set(query_documents.values())),
        "cutoffs": sorted(set(ks)),
        "comparison": "structure_aware_minus_fixed_size",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "document",
        "seed": seed,
        "retrieval_scope": retrieval_scope,
        "strategies": strategies,
        "paired_difference": paired,
    }


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile 95% CI for a paired mean difference."""
    values = [float(value) for value in differences]
    if not values:
        raise ValueError("differences must not be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return (_percentile(means, 0.025), _percentile(means, 0.975))


def paired_cluster_bootstrap_interval(
    differences_by_cluster: Mapping[str, Sequence[float]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile 95% CI after resampling whole clusters."""
    clusters = [
        tuple(float(value) for value in values)
        for values in differences_by_cluster.values()
    ]
    if not clusters or any(not values for values in clusters):
        raise ValueError("differences_by_cluster must contain non-empty clusters")
    if samples <= 0:
        raise ValueError("samples must be positive")
    generator = random.Random(seed)
    count = len(clusters)
    means: list[float] = []
    for _ in range(samples):
        sampled = [clusters[generator.randrange(count)] for _ in range(count)]
        means.append(
            statistics.fmean(value for cluster in sampled for value in cluster)
        )
    means.sort()
    return (_percentile(means, 0.025), _percentile(means, 0.975))


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the compact, thesis-oriented portion of an evaluation result."""
    strategies = result["strategies"]
    retrieval_scope = str(result.get("retrieval_scope", "full_corpus"))
    title_suffix = (
        " - gold-document oracle"
        if retrieval_scope == "gold_document_oracle"
        else " - corpus-wide"
    )
    dataset_label = str(result.get("dataset", "exploration_20"))
    evaluation_stage = str(result.get("evaluation_stage", "pilot"))
    lines = [
        f"# Evaluasi retrieval {dataset_label}{title_suffix}",
        "",
        f"Pertanyaan: {result['query_count']}",
        "",
        "## Metrik keseluruhan",
        "",
        "| Metrik | Fixed-size | Structure-aware | SAC - fixed | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    fixed = strategies["fixed_size"]["overall"]["aggregate"]
    structure = strategies["structure_aware"]["overall"]["aggregate"]
    paired = result["paired_difference"]
    for metric in fixed:
        delta = paired[metric]
        lines.append(
            f"| `{metric}` | {fixed[metric]:.4f} | {structure[metric]:.4f} | "
            f"{delta['mean_difference']:+.4f} | "
            f"[{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] |"
        )

    lines.extend(
        [
            "",
            "## Candidate coverage",
            "",
            "| Cutoff | Fixed-size Hit@k | Structure-aware Hit@k | SAC - fixed |",
            "|---:|---:|---:|---:|",
        ]
    )
    for k in result["cutoffs"]:
        metric = f"hit@{k}"
        lines.append(
            f"| {k} | {fixed[metric]:.4f} | {structure[metric]:.4f} | "
            f"{paired[metric]['mean_difference']:+.4f} |"
        )

    section_cutoff = 5 if 5 in result["cutoffs"] else max(result["cutoffs"])
    section_metric = f"hit@{section_cutoff}"
    lines.extend(
        [
            "",
            f"## Hit@{section_cutoff} per bagian",
            "",
            "| Bagian | N | Fixed-size | Structure-aware |",
            "|---|---:|---:|---:|",
        ]
    )
    fixed_sections = strategies["fixed_size"]["by_section"]
    structure_sections = strategies["structure_aware"]["by_section"]
    for label in fixed_sections:
        lines.append(
            f"| `{label}` | {fixed_sections[label]['query_count']} | "
            f"{fixed_sections[label]['aggregate'][section_metric]:.4f} | "
            f"{structure_sections[label]['aggregate'][section_metric]:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Latency retrieval",
            "",
            "| Strategi | Mean (ms) | Median (ms) | P95 (ms) |",
            "|---|---:|---:|---:|",
        ]
    )
    for strategy in STRATEGIES:
        latency = strategies[strategy]["latency"]
        lines.append(
            f"| `{strategy}` | {latency['mean_ms']:.2f} | "
            f"{latency['median_ms']:.2f} | {latency['p95_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Interval kepercayaan dihitung dengan paired cluster bootstrap pada unit "
            "dokumen; seluruh pertanyaan dari dokumen yang terpilih diresample bersama.",
            *(
                [
                    "Hasil gold-document oracle hanya mendiagnosis ranking chunk di dalam "
                    "dokumen dan bukan performa retrieval end-to-end."
                ]
                if retrieval_scope == "gold_document_oracle"
                else []
            ),
            *(
                [
                    "Hasil ini berasal dari holdout dokumen yang tidak digunakan untuk "
                    "pemilihan desain."
                ]
                if evaluation_stage == "final_holdout"
                else [
                    "Hasil pilot digunakan untuk diagnosis pipeline, bukan klaim final; "
                    "klaim utama memerlukan holdout dokumen terpisah."
                ]
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _build_qrels(
    rows: Sequence[Mapping[str, Any]], strategy: str
) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if str(row.get("strategy", "")) != strategy:
            continue
        grade_text = str(row.get("relevance_grade", "")).strip()
        if not grade_text:
            raise ValueError(f"Blank relevance grade for {row.get('query_id')}")
        grade = float(grade_text)
        if grade < 0:
            raise ValueError("Relevance grades must be non-negative")
        qrels[str(row["query_id"])][str(row["chunk_id"])] = grade
    if not qrels:
        raise ValueError(f"No qrels found for strategy {strategy}")
    for query_id, relevance in qrels.items():
        if not any(value > 0 for value in relevance.values()):
            raise ValueError(f"No positive qrel for {strategy}/{query_id}")
    return dict(qrels)


def _query_documents(
    rows: Sequence[Mapping[str, Any]], expected_queries: set[str]
) -> dict[str, str]:
    documents: dict[str, str] = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if query_id not in expected_queries:
            continue
        document_id = str(row.get("document_id", "")).strip()
        if not document_id:
            raise ValueError(f"Missing document_id for {query_id}")
        previous = documents.setdefault(query_id, document_id)
        if previous != document_id:
            raise ValueError(f"Mixed document_id values for {query_id}")
    if set(documents) != expected_queries:
        raise ValueError("Every query must map to exactly one document")
    return documents


def _parse_run_records(
    records: Sequence[Mapping[str, Any]], strategy: str
) -> dict[str, Any]:
    rankings: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    latencies: list[float] = []
    scopes: set[str] = set()
    for record in records:
        query_id = str(record["query_id"])
        if query_id in rankings:
            raise ValueError(f"Duplicate run query_id: {query_id}")
        record_strategy = str(record.get("strategy", ""))
        if record_strategy != strategy:
            raise ValueError(
                f"Run strategy mismatch for {query_id}: {record_strategy!r}"
            )
        ranking = sorted(record.get("ranking", ()), key=lambda item: int(item["rank"]))
        rankings[query_id] = [str(item["chunk_id"]) for item in ranking]
        labels[query_id] = str(record["target_section_label"])
        latencies.append(float(record["latency_ms"]))
        scopes.add(str(record.get("retrieval_scope", "full_corpus")))
    if not rankings:
        raise ValueError(f"Run is empty for strategy {strategy}")
    if len(scopes) != 1:
        raise ValueError(f"Mixed retrieval scopes in {strategy} run")
    return {
        "rankings": rankings,
        "labels": labels,
        "latencies": latencies,
        "scope": scopes.pop(),
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
