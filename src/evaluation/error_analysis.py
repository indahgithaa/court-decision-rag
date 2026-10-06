"""Error taxonomy for paired chunk-retrieval experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


STRATEGIES = ("fixed_size", "structure_aware")


def analyze_retrieval_errors(
    questions: Sequence[Mapping[str, Any]],
    qrel_rows: Sequence[Mapping[str, Any]],
    run_records: Mapping[str, Sequence[Mapping[str, Any]]],
    chunks: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Classify misses as document-level or within-document chunk failures."""
    if k <= 0:
        raise ValueError("k must be positive")
    question_by_id = {str(row["query_id"]): row for row in questions}
    if len(question_by_id) != len(questions):
        raise ValueError("Question IDs must be unique")
    positives = _positive_qrels(qrel_rows)
    runs = {
        strategy: _index_runs(run_records.get(strategy, ()), strategy)
        for strategy in STRATEGIES
    }

    details: list[dict[str, Any]] = []
    for query_id in sorted(question_by_id):
        question = question_by_id[query_id]
        target_document = str(question["document_id"])
        for strategy in STRATEGIES:
            relevant = positives.get(strategy, {}).get(query_id, set())
            if not relevant:
                raise ValueError(f"No positive qrels for {strategy}/{query_id}")
            try:
                run = runs[strategy][query_id]
            except KeyError as error:
                raise ValueError(f"Missing run record for {strategy}/{query_id}") from error
            ranking = run["ranking"]
            relevant_rank = _first_rank(
                ranking, lambda item: str(item["chunk_id"]) in relevant
            )
            document_rank = _first_rank(
                ranking,
                lambda item: str(item["retrieved_document_id"]) == target_document,
            )
            top_k = ranking[:k]
            if relevant_rank is not None and relevant_rank <= k:
                status = "relevant_found"
            elif any(
                str(item["retrieved_document_id"]) == target_document for item in top_k
            ):
                status = "document_found_chunk_miss"
            else:
                status = "document_miss"
            top = top_k[0] if top_k else None
            top_chunk = chunks.get(str(top["chunk_id"])) if top else None
            details.append(
                {
                    "query_id": query_id,
                    "strategy": strategy,
                    "target_section_label": str(question["target_section_label"]),
                    "target_document_id": target_document,
                    "question": str(question["question"]),
                    "status_at_k": status,
                    "k": k,
                    "first_relevant_rank": relevant_rank,
                    "first_target_document_rank": document_rank,
                    "top_chunk_id": str(top["chunk_id"]) if top else None,
                    "top_document_id": (
                        str(top["retrieved_document_id"]) if top else None
                    ),
                    "top_score": float(top["score"]) if top else None,
                    "top_section_label": (
                        str(top_chunk.get("section_label", "")) if top_chunk else None
                    ),
                    "top_text_preview": (
                        _preview(str(top_chunk.get("text", ""))) if top_chunk else None
                    ),
                }
            )

    detail_index = {
        (row["query_id"], row["strategy"]): row
        for row in details
    }
    paired: list[dict[str, Any]] = []
    for query_id in sorted(question_by_id):
        fixed = detail_index[(query_id, "fixed_size")]
        structure = detail_index[(query_id, "structure_aware")]
        fixed_found = fixed["status_at_k"] == "relevant_found"
        structure_found = structure["status_at_k"] == "relevant_found"
        if fixed_found and structure_found:
            outcome = "both_found"
        elif structure_found:
            outcome = "structure_only"
        elif fixed_found:
            outcome = "fixed_only"
        else:
            outcome = "neither_found"
        paired.append(
            {
                "query_id": query_id,
                "target_section_label": fixed["target_section_label"],
                "question": fixed["question"],
                "outcome_at_k": outcome,
                "fixed_status": fixed["status_at_k"],
                "structure_status": structure["status_at_k"],
                "fixed_relevant_rank": fixed["first_relevant_rank"],
                "structure_relevant_rank": structure["first_relevant_rank"],
            }
        )

    return {
        "query_count": len(question_by_id),
        "k": k,
        "strategy_summary": {
            strategy: _strategy_summary(details, strategy) for strategy in STRATEGIES
        },
        "paired_summary": dict(sorted(Counter(row["outcome_at_k"] for row in paired).items())),
        "paired_by_section": _paired_by_section(paired),
        "details": details,
        "paired": paired,
    }


def render_error_markdown(result: Mapping[str, Any]) -> str:
    """Render an error report with aggregate taxonomy and one-sided wins."""
    k = result["k"]
    lines = [
        "# Analisis error retrieval exploration_20",
        "",
        f"Pertanyaan: {result['query_count']}  ",
        f"Cutoff diagnosis: top-{k}",
        "",
        "## Jenis hasil per strategi",
        "",
        "| Strategi | Chunk relevan ditemukan | Dokumen benar, chunk salah | Dokumen tidak ditemukan |",
        "|---|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        counts = result["strategy_summary"][strategy]["status_counts"]
        lines.append(
            f"| `{strategy}` | {counts.get('relevant_found', 0)} | "
            f"{counts.get('document_found_chunk_miss', 0)} | "
            f"{counts.get('document_miss', 0)} |"
        )

    paired = result["paired_summary"]
    lines.extend(
        [
            "",
            "## Perbandingan berpasangan",
            "",
            "| Outcome | Jumlah |",
            "|---|---:|",
            f"| Keduanya berhasil | {paired.get('both_found', 0)} |",
            f"| Hanya structure-aware | {paired.get('structure_only', 0)} |",
            f"| Hanya fixed-size | {paired.get('fixed_only', 0)} |",
            f"| Keduanya gagal | {paired.get('neither_found', 0)} |",
            "",
            "## Perbandingan per bagian",
            "",
            "| Bagian | Both | SAC only | Fixed only | Neither |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, counts in result["paired_by_section"].items():
        lines.append(
            f"| `{label}` | {counts.get('both_found', 0)} | "
            f"{counts.get('structure_only', 0)} | {counts.get('fixed_only', 0)} | "
            f"{counts.get('neither_found', 0)} |"
        )

    lines.extend(["", "## Kemenangan satu strategi", ""])
    one_sided = [
        row
        for row in result["paired"]
        if row["outcome_at_k"] in {"structure_only", "fixed_only"}
    ]
    if not one_sided:
        lines.append("Tidak ada kemenangan satu strategi.")
    for row in one_sided:
        winner = "SAC" if row["outcome_at_k"] == "structure_only" else "fixed-size"
        lines.extend(
            [
                f"### {row['query_id']} - {winner}",
                "",
                f"- Bagian: `{row['target_section_label']}`",
                f"- Pertanyaan: {row['question']}",
                f"- Fixed-size: `{row['fixed_status']}`; relevant rank: "
                f"`{row['fixed_relevant_rank']}`",
                f"- Structure-aware: `{row['structure_status']}`; relevant rank: "
                f"`{row['structure_relevant_rank']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _positive_qrels(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        strategy = str(row["strategy"])
        if strategy not in STRATEGIES:
            continue
        grade = float(str(row["relevance_grade"]))
        if grade > 0:
            result[strategy][str(row["query_id"])].add(str(row["chunk_id"]))
    return {strategy: dict(queries) for strategy, queries in result.items()}


def _index_runs(
    records: Sequence[Mapping[str, Any]], strategy: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = str(record["query_id"])
        if str(record.get("strategy", "")) != strategy:
            raise ValueError(f"Run strategy mismatch for {query_id}")
        if query_id in result:
            raise ValueError(f"Duplicate run query_id: {query_id}")
        ranking = sorted(record.get("ranking", ()), key=lambda item: int(item["rank"]))
        result[query_id] = {"ranking": ranking}
    return result


def _first_rank(
    ranking: Sequence[Mapping[str, Any]],
    predicate: Any,
) -> int | None:
    for rank, item in enumerate(ranking, start=1):
        if predicate(item):
            return rank
    return None


def _strategy_summary(details: Sequence[Mapping[str, Any]], strategy: str) -> dict[str, Any]:
    selected = [row for row in details if row["strategy"] == strategy]
    by_section: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        by_section[str(row["target_section_label"])][str(row["status_at_k"])] += 1
    return {
        "status_counts": dict(sorted(Counter(row["status_at_k"] for row in selected).items())),
        "by_section": {
            label: dict(sorted(counts.items())) for label, counts in sorted(by_section.items())
        },
    }


def _paired_by_section(paired: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in paired:
        result[str(row["target_section_label"])][str(row["outcome_at_k"])] += 1
    return {
        label: dict(sorted(counts.items())) for label, counts in sorted(result.items())
    }


def _preview(text: str, limit: int = 280) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."
