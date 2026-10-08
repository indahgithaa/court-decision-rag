"""Paired evaluation for independent and contextual embeddings of identical chunks."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .document_retrieval_metrics import evaluate_document_retrieval
from .latency import summarize_latency
from .retrieval_metrics import evaluate_retrieval
from .retrieval_report import paired_cluster_bootstrap_interval


METHODS = ("independent", "contextual")


def evaluate_contextual_embedding_runs(
    qrel_rows: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10, 50),
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare contextual versus independent vectors with shared SAC qrels."""
    qrels, evidence_qrels = _build_qrels(qrel_rows)
    documents = {str(row["query_id"]): str(row["document_id"]) for row in questions}
    sections = {
        str(row["query_id"]): str(row["target_section_label"]) for row in questions
    }
    if set(qrels) != set(documents) or set(documents) != set(sections):
        raise ValueError("Question and qrel query IDs must match exactly")

    evaluations: dict[str, Any] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        parsed[method] = _parse_run(runs.get(method, ()), method)
        if set(parsed[method]["rankings"]) != set(qrels):
            raise ValueError(f"Run/query mismatch for {method}")
        overall = evaluate_retrieval(
            qrels,
            parsed[method]["rankings"],
            evidence_qrels=evidence_qrels,
            ks=ks,
        )
        by_section: dict[str, Any] = {}
        for section in sorted(set(sections.values())):
            query_ids = {key for key, value in sections.items() if value == section}
            by_section[section] = evaluate_retrieval(
                {key: qrels[key] for key in query_ids},
                {key: parsed[method]["rankings"][key] for key in query_ids},
                evidence_qrels={key: evidence_qrels[key] for key in query_ids},
                ks=ks,
            )
        evaluations[method] = {
            "overall": overall,
            "by_section": by_section,
            "document_retrieval": evaluate_document_retrieval(
                documents, parsed[method]["document_rankings"], ks=ks
            ),
            "latency": summarize_latency(parsed[method]["latencies"]),
        }

    scopes = {parsed[method]["scope"] for method in METHODS}
    if len(scopes) != 1:
        raise ValueError("Both runs must use the same retrieval scope")
    paired = _paired_differences(
        evaluations,
        documents,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "metric_schema_version": "retrieval-v2-evidence-recall+drm",
        "comparison": "contextual_minus_independent",
        "query_count": len(documents),
        "document_count": len(set(documents.values())),
        "cutoffs": sorted(set(int(k) for k in ks)),
        "retrieval_scope": scopes.pop(),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "document",
        "seed": seed,
        "methods": evaluations,
        "paired_difference": paired,
    }


def render_contextual_embedding_markdown(result: Mapping[str, Any]) -> str:
    """Render a compact report focused on the isolated embedding effect."""
    methods = result["methods"]
    independent = methods["independent"]["overall"]["aggregate"]
    contextual = methods["contextual"]["overall"]["aggregate"]
    paired = result["paired_difference"]
    lines = [
        "# SAC dengan Contextualized Chunk Embeddings",
        "",
        (
            f"{result['document_count']} dokumen; {result['query_count']} pertanyaan; "
            f"scope `{result['retrieval_scope']}`. Batas chunk SAC identik pada "
            "kedua metode."
        ),
        "",
        "## Hasil keseluruhan",
        "",
        "| Metrik | SAC independen | SAC kontekstual | Delta | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in independent:
        delta = paired[metric]
        lines.append(
            f"| `{metric}` | {independent[metric]:.4f} | {contextual[metric]:.4f} | "
            f"{delta['mean_difference']:+.4f} | "
            f"[{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] |"
        )

    cutoff = 5 if 5 in result["cutoffs"] else max(result["cutoffs"])
    lines.extend(
        [
            "",
            f"## Per bagian pada @{cutoff}",
            "",
            "| Bagian | N | Indep. Recall | Context. Recall | Indep. nDCG | "
            "Context. nDCG |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    independent_sections = methods["independent"]["by_section"]
    contextual_sections = methods["contextual"]["by_section"]
    for section in independent_sections:
        left = independent_sections[section]
        right = contextual_sections[section]
        lines.append(
            f"| `{section}` | {left['query_count']} | "
            f"{left['aggregate'][f'recall@{cutoff}']:.4f} | "
            f"{right['aggregate'][f'recall@{cutoff}']:.4f} | "
            f"{left['aggregate'][f'ndcg@{cutoff}']:.4f} | "
            f"{right['aggregate'][f'ndcg@{cutoff}']:.4f} |"
        )
    independent_documents = methods["independent"]["document_retrieval"]["aggregate"]
    contextual_documents = methods["contextual"]["document_retrieval"]["aggregate"]
    lines.extend(
        [
            "",
            "## Diagnosis provenance dokumen",
            "",
            "| Metrik | SAC independen | SAC kontekstual | Delta | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in (f"drm@{cutoff}", f"document_hit@{cutoff}"):
        delta = paired[metric]
        lines.append(
            f"| `{metric}` | {independent_documents[metric]:.4f} | "
            f"{contextual_documents[metric]:.4f} | "
            f"{delta['mean_difference']:+.4f} | "
            f"[{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "Delta adalah kontekstual dikurangi independen. Confidence interval "
            "memakai paired cluster bootstrap pada unit dokumen.",
            "",
            "Karena corpus ini telah dipakai selama pengembangan, hasil ini "
            "bersifat komparatif eksploratif, bukan bukti generalisasi eksternal.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_qrels(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[str, ...]]]]:
    qrels: dict[str, dict[str, float]] = {}
    evidence: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        grade = float(row["relevance_grade"])
        if grade <= 0:
            continue
        query_id = str(row["query_id"])
        chunk_id = str(row["chunk_id"])
        evidence_id = str(row.get("evidence_id", "")).strip()
        if not evidence_id:
            raise ValueError("Every positive qrel must contain evidence_id")
        qrels.setdefault(query_id, {})[chunk_id] = grade
        evidence.setdefault(query_id, {}).setdefault(evidence_id, set()).add(chunk_id)
    if not qrels:
        raise ValueError("No positive qrels found")
    return qrels, {
        query_id: {
            evidence_id: tuple(sorted(chunk_ids))
            for evidence_id, chunk_ids in by_evidence.items()
        }
        for query_id, by_evidence in evidence.items()
    }


def _parse_run(
    records: Sequence[Mapping[str, Any]], expected_method: str
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"Run for {expected_method} is empty")
    rankings: dict[str, list[str]] = {}
    document_rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    scopes: set[str] = set()
    for record in records:
        # Historical independent SAC runs predate the explicit method field.
        raw_method = record.get("embedding_method")
        if raw_method is None and expected_method != "independent":
            raise ValueError("Contextual runs must contain embedding_method")
        method = str(raw_method or "independent")
        if method != expected_method:
            raise ValueError(
                f"Expected embedding_method={expected_method!r}, found {method!r}"
            )
        query_id = str(record["query_id"])
        if query_id in rankings:
            raise ValueError(
                f"Duplicate query_id {query_id!r} in {expected_method} run"
            )
        ranking = sorted(record["ranking"], key=lambda item: int(item["rank"]))
        rankings[query_id] = [str(item["chunk_id"]) for item in ranking]
        document_rankings[query_id] = [
            str(item["retrieved_document_id"]) for item in ranking
        ]
        latencies.append(float(record["latency_ms"]))
        scopes.add(str(record.get("retrieval_scope", "full_corpus")))
    if len(scopes) != 1:
        raise ValueError(f"Mixed retrieval scopes in {expected_method} run")
    return {
        "rankings": rankings,
        "document_rankings": document_rankings,
        "latencies": latencies,
        "scope": scopes.pop(),
    }


def _paired_differences(
    evaluations: Mapping[str, Any],
    documents: Mapping[str, str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left = evaluations["independent"]
    right = evaluations["contextual"]
    metric_sources = (
        (left["overall"], right["overall"]),
        (left["document_retrieval"], right["document_retrieval"]),
    )
    result: dict[str, Any] = {}
    for left_source, right_source in metric_sources:
        for metric in left_source["aggregate"]:
            by_document: dict[str, list[float]] = defaultdict(list)
            flat: list[float] = []
            for query_id, document_id in documents.items():
                difference = (
                    right_source["per_query"][query_id][metric]
                    - left_source["per_query"][query_id][metric]
                )
                by_document[document_id].append(difference)
                flat.append(difference)
            low, high = paired_cluster_bootstrap_interval(
                by_document, samples=samples, seed=seed
            )
            result[metric] = {
                "mean_difference": statistics.fmean(flat),
                "ci95_low": low,
                "ci95_high": high,
            }
    return result
