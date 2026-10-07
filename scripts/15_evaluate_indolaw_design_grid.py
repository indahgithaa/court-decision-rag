"""Evaluate development runs and freeze one design per chunking family."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.latency import summarize_latency
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.utils.io import read_jsonl


CONFIGS = (
    "fixed_w150_o30",
    "fixed_w300_o60",
    "fixed_w500_o100",
    "sac_w150_o30_s0",
    "sac_w300_o60_s0",
    "sac_w500_o100_s0",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Indo-Law design grid.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/indolaw_200/development_questions.csv"),
    )
    parser.add_argument(
        "--qrels-root", type=Path, default=Path("data/evaluation/indolaw_200")
    )
    parser.add_argument(
        "--chunks-root",
        type=Path,
        default=Path("data/chunks/indolaw_200/development"),
    )
    parser.add_argument(
        "--runs-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/results/indolaw_200_development_design_grid.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("experiments/results/indolaw_200_development_design_grid.md"),
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("experiments/indolaw_200_selected_design_v2.json"),
    )
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="Write the diagnostic report without writing a frozen selection.",
    )
    args = parser.parse_args(argv)

    outputs = [args.output_json, args.output_md]
    if not args.no_freeze:
        outputs.append(args.selection_output)
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")

    with args.questions.open("r", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    labels = {row["query_id"]: row["target_section_label"] for row in questions}
    evaluations: dict[str, Any] = {}
    shared_evidence_units: dict[str, set[str]] | None = None
    for name in args.configs:
        qrels, evidence_qrels = _read_qrels(
            args.qrels_root / f"development_{name}_qrels.csv"
        )
        current_evidence_units = {
            query_id: set(units) for query_id, units in evidence_qrels.items()
        }
        if shared_evidence_units is None:
            shared_evidence_units = current_evidence_units
        elif current_evidence_units != shared_evidence_units:
            raise ValueError(
                f"Evidence units differ for {name}; all designs must use the same units"
            )
        records = list(
            read_jsonl(args.runs_root / f"indolaw_200_development_{name}_top50.jsonl")
        )
        rankings = {
            str(record["query_id"]): [
                str(item["chunk_id"])
                for item in sorted(record["ranking"], key=lambda item: int(item["rank"]))
            ]
            for record in records
        }
        overall = evaluate_retrieval(
            qrels,
            rankings,
            evidence_qrels=evidence_qrels,
            ks=(1, 3, 5, 10, 50),
        )
        by_section: dict[str, Any] = {}
        for label in sorted(set(labels.values())):
            query_ids = {query_id for query_id, value in labels.items() if value == label}
            by_section[label] = evaluate_retrieval(
                {query_id: qrels[query_id] for query_id in query_ids},
                {query_id: rankings[query_id] for query_id in query_ids},
                evidence_qrels={
                    query_id: evidence_qrels[query_id] for query_id in query_ids
                },
                ks=(1, 3, 5, 10, 50),
            )
        chunk_count = sum(1 for _ in read_jsonl(args.chunks_root / name / "chunks.jsonl"))
        evaluations[name] = {
            "family": "fixed_size" if name.startswith("fixed") else "structure_aware",
            "chunk_count": chunk_count,
            "overall": overall,
            "by_section": by_section,
            "latency": summarize_latency([float(row["latency_ms"]) for row in records]),
        }

    selected = {
        family: _select(evaluations, family)
        for family in ("fixed_size", "structure_aware")
    }
    result = {
        "dataset": "indolaw_200",
        "split": "development",
        "document_count": 40,
        "query_count": len(questions),
        "selection_rule": "maximize nDCG@5, then MRR@5, then Recall@5, then minimize chunk count",
        "sentence_overlap_note": (
            "s0 and s2 were byte-identical because normalized XML lacks sentence punctuation; "
            "only s0 was indexed"
        ),
        "selection_status": (
            "diagnostic_candidate" if args.no_freeze else "frozen"
        ),
        "selected": selected,
        "evaluations": evaluations,
    }
    markdown = render_markdown(result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown, encoding="utf-8")
    frozen = {
        "status": "frozen",
        "metric_schema_version": "retrieval-v2-evidence-recall",
        "recall_unit": "evidence",
        "dataset": result["dataset"],
        "selected_on": "development_only",
        "development_documents": 40,
        "holdout_documents": 160,
        "reported_metrics": ["recall@k", "mrr@k", "ndcg@k"],
        "selection_rule": result["selection_rule"],
        "fixed_size": selected["fixed_size"],
        "structure_aware": selected["structure_aware"],
        "embedding_model": "intfloat/multilingual-e5-small",
        "reranking": "none (dense-only, frozen from prior pilot)",
    }
    if not args.no_freeze:
        args.selection_output.write_text(
            json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
        )
    print(markdown)


def _read_qrels(
    path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[str, ...]]]]:
    qrels: dict[str, dict[str, float]] = {}
    evidence: dict[str, dict[str, set[str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            grade = float(row["relevance_grade"])
            if grade > 0:
                query_id = row["query_id"]
                chunk_id = row["chunk_id"]
                evidence_id = row.get("evidence_id", "").strip()
                if not evidence_id:
                    raise ValueError(
                        f"Missing evidence_id for {query_id}; regenerate {path}"
                    )
                qrels.setdefault(query_id, {})[chunk_id] = grade
                evidence.setdefault(query_id, {}).setdefault(evidence_id, set()).add(
                    chunk_id
                )
    frozen_evidence = {
        query_id: {
            evidence_id: tuple(sorted(chunk_ids))
            for evidence_id, chunk_ids in evidence_by_id.items()
        }
        for query_id, evidence_by_id in evidence.items()
    }
    return qrels, frozen_evidence


def _select(evaluations: dict[str, Any], family: str) -> str:
    candidates = [name for name, result in evaluations.items() if result["family"] == family]

    def criterion(name: str) -> tuple[float, float, float, int]:
        result = evaluations[name]
        aggregate = result["overall"]["aggregate"]
        return (
            aggregate["ndcg@5"],
            aggregate["mrr@5"],
            aggregate["recall@5"],
            -int(result["chunk_count"]),
        )

    return max(candidates, key=criterion)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Seleksi desain chunking Indo-Law development",
        "",
        f"Dokumen: {result['document_count']}; pertanyaan: {result['query_count']}.",
        "",
        "| Konfigurasi | Chunk | Recall@5 | MRR@5 | nDCG@5 | Recall@10 | Recall@50 | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, evaluation in result["evaluations"].items():
        aggregate = evaluation["overall"]["aggregate"]
        lines.append(
            f"| `{name}` | {evaluation['chunk_count']} | {aggregate['recall@5']:.4f} | "
            f"{aggregate['mrr@5']:.4f} | {aggregate['ndcg@5']:.4f} | "
            f"{aggregate['recall@10']:.4f} | {aggregate['recall@50']:.4f} | "
            f"{evaluation['latency']['p95_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "## Desain dibekukan"
                if result.get("selection_status") == "frozen"
                else "## Kandidat berdasarkan aturan development"
            ),
            "",
            f"- Fixed-size: `{result['selected']['fixed_size']}`",
            f"- Structure-aware: `{result['selected']['structure_aware']}`",
            "",
            f"Aturan: {result['selection_rule']}.",
            "",
            "Overlap dua kalimat tidak menjadi kondisi efektif karena teks XML ternormalisasi "
            "tidak memiliki tanda baca kalimat; hasil s0 dan s2 identik.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
