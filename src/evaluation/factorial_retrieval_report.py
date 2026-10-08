"""Factorial evaluation of chunking and contextualized chunk embeddings."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .document_retrieval_metrics import evaluate_document_retrieval
from .latency import summarize_latency
from .retrieval_metrics import evaluate_retrieval


ARM_SPECS: dict[str, dict[str, str]] = {
    "fixed_independent": {
        "chunking": "fixed_size",
        "embedding": "independent",
    },
    "fixed_contextual": {
        "chunking": "fixed_size",
        "embedding": "contextual",
    },
    "sac_independent": {
        "chunking": "structure_aware",
        "embedding": "independent",
    },
    "sac_contextual": {
        "chunking": "structure_aware",
        "embedding": "contextual",
    },
}

# Every contrast is expressed as a linear combination of per-query arm scores.
# The first four are simple effects, followed by the combined endpoint,
# factorial marginal effects, and the interaction (difference in differences).
PLANNED_CONTRASTS: dict[str, dict[str, float]] = {
    "contextual_effect_within_fixed": {
        "fixed_contextual": 1.0,
        "fixed_independent": -1.0,
    },
    "contextual_effect_within_sac": {
        "sac_contextual": 1.0,
        "sac_independent": -1.0,
    },
    "sac_effect_with_independent": {
        "sac_independent": 1.0,
        "fixed_independent": -1.0,
    },
    "sac_effect_with_contextual": {
        "sac_contextual": 1.0,
        "fixed_contextual": -1.0,
    },
    "combined_vs_conventional_baseline": {
        "sac_contextual": 1.0,
        "fixed_independent": -1.0,
    },
    "contextual_marginal_effect": {
        "fixed_contextual": 0.5,
        "sac_contextual": 0.5,
        "fixed_independent": -0.5,
        "sac_independent": -0.5,
    },
    "sac_marginal_effect": {
        "sac_independent": 0.5,
        "sac_contextual": 0.5,
        "fixed_independent": -0.5,
        "fixed_contextual": -0.5,
    },
    "chunking_embedding_interaction": {
        "sac_contextual": 1.0,
        "sac_independent": -1.0,
        "fixed_contextual": -1.0,
        "fixed_independent": 1.0,
    },
}


def evaluate_factorial_retrieval(
    fixed_qrel_rows: Sequence[Mapping[str, Any]],
    sac_qrel_rows: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10, 50),
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate the four arms of a chunking x embedding 2x2 experiment.

    Qrels are strategy-specific because fixed-size and SAC produce different
    chunk IDs. Their positive evidence units must nevertheless match exactly.
    Contextual and independent arms of one strategy reuse the same qrels.
    """
    cutoffs = tuple(sorted(set(int(k) for k in ks)))
    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("ks must contain positive integers")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    question_metadata = _parse_questions(questions)
    qrels: dict[str, dict[str, dict[str, float]]] = {}
    evidence_qrels: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    evidence_signatures: dict[
        str, dict[str, tuple[tuple[int, int], ...]]
    ] = {}
    for factor, rows, strategy in (
        ("fixed", fixed_qrel_rows, "fixed_size"),
        ("sac", sac_qrel_rows, "structure_aware"),
    ):
        (
            qrels[factor],
            evidence_qrels[factor],
            evidence_signatures[factor],
        ) = _parse_qrels(rows, strategy, question_metadata)
    _validate_shared_evidence(evidence_signatures["fixed"], evidence_signatures["sac"])

    expected_queries = set(question_metadata)
    parsed_runs: dict[str, dict[str, Any]] = {}
    for arm, specification in ARM_SPECS.items():
        if arm not in runs:
            raise ValueError(f"Missing run for arm {arm!r}")
        parsed_runs[arm] = _parse_run(
            runs[arm],
            arm=arm,
            expected_strategy=specification["chunking"],
            expected_embedding=specification["embedding"],
            questions=question_metadata,
            required_depth=max(cutoffs),
        )
        if set(parsed_runs[arm]["rankings"]) != expected_queries:
            raise ValueError(f"Run/query mismatch for arm {arm!r}")

    scopes = {parsed_runs[arm]["scope"] for arm in ARM_SPECS}
    if len(scopes) != 1:
        raise ValueError("All four runs must use the same retrieval scope")
    models = {parsed_runs[arm]["model_name"] for arm in ARM_SPECS}
    if len(models) != 1:
        raise ValueError("All four runs must use the same embedding model")

    arms: dict[str, Any] = {}
    section_labels = sorted(
        {metadata["section"] for metadata in question_metadata.values()}
    )
    expected_documents = {
        query_id: metadata["document_id"]
        for query_id, metadata in question_metadata.items()
    }
    for arm, specification in ARM_SPECS.items():
        qrel_family = "fixed" if specification["chunking"] == "fixed_size" else "sac"
        parsed = parsed_runs[arm]
        overall = evaluate_retrieval(
            qrels[qrel_family],
            parsed["rankings"],
            evidence_qrels=evidence_qrels[qrel_family],
            ks=cutoffs,
        )
        document_retrieval = evaluate_document_retrieval(
            expected_documents,
            parsed["document_rankings"],
            ks=cutoffs,
        )
        by_section: dict[str, Any] = {}
        for section in section_labels:
            query_ids = {
                query_id
                for query_id, metadata in question_metadata.items()
                if metadata["section"] == section
            }
            by_section[section] = {
                "retrieval": evaluate_retrieval(
                    {query_id: qrels[qrel_family][query_id] for query_id in query_ids},
                    {query_id: parsed["rankings"][query_id] for query_id in query_ids},
                    evidence_qrels={
                        query_id: evidence_qrels[qrel_family][query_id]
                        for query_id in query_ids
                    },
                    ks=cutoffs,
                ),
                "document_retrieval": evaluate_document_retrieval(
                    {
                        query_id: expected_documents[query_id]
                        for query_id in query_ids
                    },
                    {
                        query_id: parsed["document_rankings"][query_id]
                        for query_id in query_ids
                    },
                    ks=cutoffs,
                ),
            }
        arms[arm] = {
            "factors": dict(specification),
            "overall": overall,
            "document_retrieval": document_retrieval,
            "by_section": by_section,
            "latency": summarize_latency(parsed["latencies"]),
        }

    query_documents = {
        query_id: metadata["document_id"]
        for query_id, metadata in question_metadata.items()
    }
    contrasts = _evaluate_planned_contrasts(
        arms,
        query_documents,
        samples=bootstrap_samples,
        seed=seed,
    )
    evidence_counts = [
        len(by_evidence) for by_evidence in evidence_signatures["fixed"].values()
    ]
    positive_grades = {
        family: sorted(
            {
                float(grade)
                for by_chunk in qrels[family].values()
                for grade in by_chunk.values()
                if grade > 0
            }
        )
        for family in ("fixed", "sac")
    }
    positive_chunk_qrel_counts = {
        family: sum(
            grade > 0
            for by_chunk in qrels[family].values()
            for grade in by_chunk.values()
        )
        for family in ("fixed", "sac")
    }
    return {
        "metric_schema_version": "retrieval-v3-factorial-contextual",
        "design": "2x2_chunking_x_embedding",
        "factors": {
            "chunking": ["fixed_size", "structure_aware"],
            "embedding": ["independent", "contextual"],
        },
        "query_count": len(question_metadata),
        "document_count": len(set(query_documents.values())),
        "cutoffs": list(cutoffs),
        "retrieval_scope": scopes.pop(),
        "model_name": models.pop(),
        "recall_unit": "shared_evidence",
        "ndcg_unit": "strategy_specific_graded_chunks",
        "evidence_units_per_query": {
            "min": min(evidence_counts),
            "max": max(evidence_counts),
            "mean": statistics.fmean(evidence_counts),
        },
        "positive_relevance_grades": positive_grades,
        "positive_chunk_qrel_counts": positive_chunk_qrel_counts,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "document",
        "bootstrap_method": "paired_percentile_cluster_bootstrap",
        "seed": seed,
        "arms": arms,
        "planned_contrasts": contrasts,
    }


def render_factorial_retrieval_markdown(result: Mapping[str, Any]) -> str:
    """Render a thesis-oriented Markdown report of the factorial evaluation."""
    cutoff = 5 if 5 in result["cutoffs"] else max(result["cutoffs"])
    arms = result["arms"]
    labels = {
        "fixed_independent": "Fixed + independent",
        "fixed_contextual": "Fixed + contextual",
        "sac_independent": "SAC + independent",
        "sac_contextual": "SAC + contextual",
    }
    lines = [
        "# Evaluasi faktorial chunking × contextualized embeddings",
        "",
        (
            f"{result['query_count']} pertanyaan dari {result['document_count']} dokumen; "
            f"retrieval `{result['retrieval_scope']}`; encoder "
            f"`{result['model_name']}`."
        ),
        "",
        "## Empat kondisi eksperimen",
        "",
        f"| Kondisi | Recall@{cutoff} | MRR@{cutoff} | nDCG@{cutoff} | "
        f"DRM@{cutoff} ↓ | Document hit@{cutoff} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_SPECS:
        retrieval = arms[arm]["overall"]["aggregate"]
        documents = arms[arm]["document_retrieval"]["aggregate"]
        lines.append(
            f"| {labels[arm]} | {retrieval[f'recall@{cutoff}']:.4f} | "
            f"{retrieval[f'mrr@{cutoff}']:.4f} | "
            f"{retrieval[f'ndcg@{cutoff}']:.4f} | "
            f"{documents[f'drm@{cutoff}']:.4f} | "
            f"{documents[f'document_hit@{cutoff}']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Retrieval pada seluruh cutoff",
            "",
            "| Metrik | Fixed + indep. | Fixed + context. | SAC + indep. | "
            "SAC + context. |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for k in result["cutoffs"]:
        for metric_name in ("recall", "mrr", "ndcg"):
            metric = f"{metric_name}@{k}"
            values = [arms[arm]["overall"]["aggregate"][metric] for arm in ARM_SPECS]
            lines.append(
                f"| `{metric}` | {values[0]:.4f} | {values[1]:.4f} | "
                f"{values[2]:.4f} | {values[3]:.4f} |"
            )

    lines.extend(
        [
            "",
            f"## Kontras terencana pada @{cutoff}",
            "",
            "| Kontras | Metrik | Delta | 95% CI | W/T/L |",
            "|---|---|---:|---:|---:|",
        ]
    )
    selected_metrics = (
        f"recall@{cutoff}",
        f"mrr@{cutoff}",
        f"ndcg@{cutoff}",
        f"drm@{cutoff}",
        f"document_hit@{cutoff}",
    )
    for contrast_name, contrast in result["planned_contrasts"].items():
        for metric in selected_metrics:
            score = contrast["metrics"][metric]
            outcome = score["win_tie_loss"]
            lines.append(
                f"| `{contrast_name}` | `{metric}` | "
                f"{score['mean_difference']:+.4f} | "
                f"[{score['ci95_low']:+.4f}, {score['ci95_high']:+.4f}] | "
                f"{outcome['wins']}/{outcome['ties']}/{outcome['losses']} |"
            )

    lines.extend(
        [
            "",
            f"## Hasil per bagian pada @{cutoff}",
            "",
            "| Bagian | Kondisi | N | Recall | MRR | nDCG | DRM ↓ |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    sections = arms["fixed_independent"]["by_section"]
    for section in sections:
        for arm in ARM_SPECS:
            retrieval = arms[arm]["by_section"][section]["retrieval"]
            documents = arms[arm]["by_section"][section]["document_retrieval"]
            lines.append(
                f"| `{section}` | {labels[arm]} | {retrieval['query_count']} | "
                f"{retrieval['aggregate'][f'recall@{cutoff}']:.4f} | "
                f"{retrieval['aggregate'][f'mrr@{cutoff}']:.4f} | "
                f"{retrieval['aggregate'][f'ndcg@{cutoff}']:.4f} | "
                f"{documents['aggregate'][f'drm@{cutoff}']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Diagnosis provenance dokumen pada seluruh cutoff",
            "",
            "| Metrik | Fixed + indep. | Fixed + context. | SAC + indep. | "
            "SAC + context. |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for k in result["cutoffs"]:
        for metric_name in ("drm", "document_hit", "document_mrr"):
            metric = f"{metric_name}@{k}"
            values = [
                arms[arm]["document_retrieval"]["aggregate"][metric]
                for arm in ARM_SPECS
            ]
            lines.append(
                f"| `{metric}` | {values[0]:.4f} | {values[1]:.4f} | "
                f"{values[2]:.4f} | {values[3]:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Latency retrieval",
            "",
            "Latency berikut hanya diagnostik. Run independent dan contextual "
            "dihasilkan pada eksekusi berbeda, sehingga angka lintas metode "
            "embedding tidak merupakan perbandingan terkontrol dan tidak boleh "
            "dipakai untuk menyimpulkan metode mana yang lebih cepat.",
            "",
            "| Kondisi | Mean (ms) | Median (ms) | P95 (ms) |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm in ARM_SPECS:
        latency = arms[arm]["latency"]
        lines.append(
            f"| {labels[arm]} | {latency['mean_ms']:.2f} | "
            f"{latency['median_ms']:.2f} | {latency['p95_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Delta mengikuti formula kontras pada JSON. Interval kepercayaan memakai "
            "paired percentile cluster bootstrap dengan dokumen sebagai unit resampling. "
            "W/T/L telah disesuaikan dengan arah metrik, sehingga DRM yang lebih rendah "
            "dihitung sebagai kemenangan.",
            "",
            "Recall memakai evidence unit bersama antara Fixed dan SAC. nDCG memakai "
            "graded chunk qrels spesifik strategi; karena overlap dapat menghasilkan "
            "jumlah chunk relevan berbeda, nDCG harus dibaca bersama Recall dan MRR.",
        ]
    )
    evidence_summary = result.get("evidence_units_per_query", {})
    if evidence_summary.get("min") == evidence_summary.get("max") == 1:
        lines.extend(
            [
                "",
                "Setiap query memiliki tepat satu evidence unit, sehingga evidence "
                "Recall@K pada eksperimen ini secara numerik sama dengan evidence "
                "Hit@K.",
            ]
        )
    grade_summary = result.get("positive_relevance_grades", {})
    if grade_summary and all(
        len(grades) == 1 for grades in grade_summary.values()
    ):
        counts = result.get("positive_chunk_qrel_counts", {})
        lines.extend(
            [
                "",
                "Seluruh qrel positif memakai grade yang sama. Karena jumlah chunk "
                f"positif Fixed ({counts.get('fixed', 'n/a')}) dan SAC "
                f"({counts.get('sac', 'n/a')}) berbeda, nDCG lintas chunker dan "
                "interaksinya bersifat deskriptif, bukan bukti faktorial primer.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _parse_questions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    if not rows:
        raise ValueError("Question file is empty")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        document_id = str(row.get("document_id", "")).strip()
        section = str(row.get("target_section_label", "")).strip()
        if not query_id or not document_id or not section:
            raise ValueError("Every question requires query_id, document_id, and section")
        if query_id in result:
            raise ValueError(f"Duplicate question query_id {query_id!r}")
        review_status = str(row.get("review_status", "")).strip().lower()
        if review_status and review_status != "approved":
            raise ValueError(f"Question {query_id!r} is not approved")
        result[query_id] = {"document_id": document_id, "section": section}
    return result


def _parse_qrels(
    rows: Sequence[Mapping[str, Any]],
    expected_strategy: str,
    questions: Mapping[str, Mapping[str, str]],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, tuple[str, ...]]],
    dict[str, dict[str, tuple[tuple[int, int], ...]]],
]:
    if not rows:
        raise ValueError(f"Qrels are empty for strategy {expected_strategy!r}")
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    evidence: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    signatures: dict[str, dict[str, set[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        strategy = str(row.get("strategy", "")).strip()
        if strategy != expected_strategy:
            raise ValueError(
                f"Expected qrel strategy {expected_strategy!r}, found {strategy!r}"
            )
        query_id = str(row.get("query_id", "")).strip()
        if query_id not in questions:
            raise ValueError(f"Unknown qrel query_id {query_id!r}")
        expected = questions[query_id]
        if str(row.get("document_id", "")).strip() != expected["document_id"]:
            raise ValueError(f"Qrel document mismatch for {query_id!r}")
        if str(row.get("target_section_label", "")).strip() != expected["section"]:
            raise ValueError(f"Qrel section mismatch for {query_id!r}")
        chunk_id = str(row.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError(f"Missing qrel chunk_id for {query_id!r}")
        grade_text = str(row.get("relevance_grade", "")).strip()
        if not grade_text:
            raise ValueError(f"Blank relevance grade for {query_id!r}")
        grade = float(grade_text)
        if grade < 0:
            raise ValueError("Relevance grades must be non-negative")
        previous_grade = qrels[query_id].get(chunk_id)
        qrels[query_id][chunk_id] = (
            grade if previous_grade is None else max(previous_grade, grade)
        )
        if grade <= 0:
            continue
        evidence_id = str(row.get("evidence_id", "")).strip()
        if not evidence_id:
            raise ValueError(f"Missing evidence_id for {query_id!r}")
        try:
            span = (
                int(str(row["evidence_start_position"])),
                int(str(row["evidence_end_position"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid evidence span for {query_id!r}") from error
        if span[0] < 0 or span[1] <= span[0]:
            raise ValueError(f"Invalid evidence span for {query_id!r}: {span}")
        # One semantic evidence unit may have several equivalent occurrences
        # in a judgment (for example, a statute repeated in the reasoning).
        # Preserve and compare the full occurrence set across chunking methods.
        signatures[query_id][evidence_id].add(span)
        evidence[query_id][evidence_id].add(chunk_id)

    expected_queries = set(questions)
    if set(qrels) != expected_queries or set(evidence) != expected_queries:
        raise ValueError(
            f"Qrel/query mismatch for strategy {expected_strategy!r}"
        )
    for query_id in expected_queries:
        if not any(grade > 0 for grade in qrels[query_id].values()):
            raise ValueError(f"No positive qrel for {expected_strategy}/{query_id}")
    return (
        {query_id: dict(values) for query_id, values in qrels.items()},
        {
            query_id: {
                evidence_id: tuple(sorted(chunk_ids))
                for evidence_id, chunk_ids in by_evidence.items()
            }
            for query_id, by_evidence in evidence.items()
        },
        {
            query_id: {
                evidence_id: tuple(sorted(spans))
                for evidence_id, spans in values.items()
            }
            for query_id, values in signatures.items()
        },
    )


def _validate_shared_evidence(
    fixed: Mapping[str, Mapping[str, tuple[tuple[int, int], ...]]],
    sac: Mapping[str, Mapping[str, tuple[tuple[int, int], ...]]],
) -> None:
    if set(fixed) != set(sac):
        raise ValueError("Fixed and SAC qrels must contain identical query IDs")
    for query_id in fixed:
        # The evidence ID is the shared semantic unit and fixes the Recall
        # denominator. Strategy-specific qrel generation may attach additional
        # equivalent occurrences with different character spans when the same
        # statute/fact is repeated in a decision, so occurrence sets need not
        # be byte-identical across chunking methods.
        if set(fixed[query_id]) != set(sac[query_id]):
            raise ValueError(
                "Fixed and SAC must use identical evidence units for "
                f"query {query_id!r}"
            )


def _parse_run(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    expected_strategy: str,
    expected_embedding: str,
    questions: Mapping[str, Mapping[str, str]],
    required_depth: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError(f"Run for arm {arm!r} is empty")
    rankings: dict[str, list[str]] = {}
    document_rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    scopes: set[str] = set()
    models: set[str] = set()
    for record in records:
        query_id = str(record.get("query_id", "")).strip()
        if query_id not in questions:
            raise ValueError(f"Unknown run query_id {query_id!r} in {arm!r}")
        if query_id in rankings:
            raise ValueError(f"Duplicate run query_id {query_id!r} in {arm!r}")
        if str(record.get("strategy", "")).strip() != expected_strategy:
            raise ValueError(f"Run strategy mismatch for {arm!r}/{query_id}")
        raw_embedding = record.get("embedding_method")
        if raw_embedding is None:
            if expected_embedding != "independent":
                raise ValueError(
                    f"Contextual run {arm!r} must declare embedding_method"
                )
            actual_embedding = "independent"
        else:
            actual_embedding = str(raw_embedding).strip()
        if actual_embedding != expected_embedding:
            raise ValueError(f"Run embedding method mismatch for {arm!r}/{query_id}")
        metadata = questions[query_id]
        if str(record.get("document_id", "")).strip() != metadata["document_id"]:
            raise ValueError(f"Run document mismatch for {arm!r}/{query_id}")
        if (
            str(record.get("target_section_label", "")).strip()
            != metadata["section"]
        ):
            raise ValueError(f"Run section mismatch for {arm!r}/{query_id}")

        raw_ranking = list(record.get("ranking", ()))
        ranking = sorted(raw_ranking, key=lambda item: int(item["rank"]))
        ranks = [int(item["rank"]) for item in ranking]
        if ranks != list(range(1, len(ranking) + 1)):
            raise ValueError(f"Ranking must be contiguous for {arm!r}/{query_id}")
        if len(ranking) < required_depth:
            raise ValueError(
                f"Run {arm!r}/{query_id} has depth {len(ranking)}, "
                f"but @{required_depth} was requested"
            )
        chunk_ids = [str(item.get("chunk_id", "")).strip() for item in ranking]
        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError(f"Blank retrieved chunk ID for {arm!r}/{query_id}")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(f"Duplicate retrieved chunk ID for {arm!r}/{query_id}")
        retrieved_documents = [
            str(item.get("retrieved_document_id", "")).strip() for item in ranking
        ]
        if any(not document_id for document_id in retrieved_documents):
            raise ValueError(f"Blank retrieved document ID for {arm!r}/{query_id}")

        rankings[query_id] = chunk_ids
        document_rankings[query_id] = retrieved_documents
        latencies.append(float(record["latency_ms"]))
        scopes.add(str(record.get("retrieval_scope", "full_corpus")))
        model_name = str(record.get("model_name", "")).strip()
        if not model_name:
            raise ValueError(f"Missing model_name for {arm!r}/{query_id}")
        models.add(model_name)
    if len(scopes) != 1:
        raise ValueError(f"Mixed retrieval scopes in arm {arm!r}")
    if len(models) != 1:
        raise ValueError(f"Mixed embedding models in arm {arm!r}")
    return {
        "rankings": rankings,
        "document_rankings": document_rankings,
        "latencies": latencies,
        "scope": scopes.pop(),
        "model_name": models.pop(),
    }


def _evaluate_planned_contrasts(
    arms: Mapping[str, Any],
    query_documents: Mapping[str, str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    query_ids = sorted(query_documents)
    document_ids = sorted(set(query_documents.values()))
    document_index = {document_id: index for index, document_id in enumerate(document_ids)}
    cluster_for_query = np.asarray(
        [document_index[query_documents[query_id]] for query_id in query_ids],
        dtype=np.int64,
    )
    cluster_sizes = np.bincount(cluster_for_query, minlength=len(document_ids)).astype(
        np.float64
    )
    generator = np.random.default_rng(seed)
    bootstrap_weights = generator.multinomial(
        len(document_ids),
        np.full(len(document_ids), 1.0 / len(document_ids)),
        size=samples,
    ).astype(np.float64)
    bootstrap_denominators = bootstrap_weights @ cluster_sizes

    retrieval_metrics = list(
        arms["fixed_independent"]["overall"]["aggregate"]
    )
    document_metrics = list(
        arms["fixed_independent"]["document_retrieval"]["aggregate"]
    )
    metric_names = retrieval_metrics + document_metrics
    per_arm: dict[str, np.ndarray] = {}
    for arm in ARM_SPECS:
        rows: list[list[float]] = []
        for query_id in query_ids:
            retrieval = arms[arm]["overall"]["per_query"][query_id]
            documents = arms[arm]["document_retrieval"]["per_query"][query_id]
            rows.append(
                [float(retrieval[metric]) for metric in retrieval_metrics]
                + [float(documents[metric]) for metric in document_metrics]
            )
        per_arm[arm] = np.asarray(rows, dtype=np.float64)

    result: dict[str, Any] = {}
    tolerance = 1e-12
    for contrast_name, coefficients in PLANNED_CONTRASTS.items():
        per_query = sum(
            coefficient * per_arm[arm]
            for arm, coefficient in coefficients.items()
        )
        cluster_sums = np.zeros(
            (len(document_ids), len(metric_names)), dtype=np.float64
        )
        np.add.at(cluster_sums, cluster_for_query, per_query)
        bootstrap_means = (
            bootstrap_weights @ cluster_sums
        ) / bootstrap_denominators[:, None]
        bounds = np.quantile(
            bootstrap_means,
            (0.025, 0.975),
            axis=0,
            method="linear",
        )
        metrics: dict[str, Any] = {}
        for index, metric in enumerate(metric_names):
            values = per_query[:, index]
            directed = -values if metric.startswith("drm@") else values
            wins = int(np.sum(directed > tolerance))
            losses = int(np.sum(directed < -tolerance))
            ties = len(values) - wins - losses
            metrics[metric] = {
                "mean_difference": statistics.fmean(values.tolist()),
                "ci95_low": float(bounds[0, index]),
                "ci95_high": float(bounds[1, index]),
                "higher_is_better": not metric.startswith("drm@"),
                "win_tie_loss": {
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": wins / len(values),
                    "tie_rate": ties / len(values),
                    "loss_rate": losses / len(values),
                },
            }
        result[contrast_name] = {
            "formula": " + ".join(
                f"{coefficient:+g}*{arm}" for arm, coefficient in coefficients.items()
            ).lstrip("+"),
            "weights": dict(coefficients),
            "metrics": metrics,
        }
    return result
