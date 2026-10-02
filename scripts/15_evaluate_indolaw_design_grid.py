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
        default=Path("experiments/indolaw_200_selected_design.json"),
    )
    args = parser.parse_args(argv)

    with args.questions.open("r", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    labels = {row["query_id"]: row["target_section_label"] for row in questions}
    evaluations: dict[str, Any] = {}
    for name in CONFIGS:
        qrels = _read_qrels(args.qrels_root / f"development_{name}_qrels.csv")
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
        overall = evaluate_retrieval(qrels, rankings, ks=(1, 3, 5, 10, 50))
        by_section: dict[str, Any] = {}
        for label in sorted(set(labels.values())):
            query_ids = {query_id for query_id, value in labels.items() if value == label}
            by_section[label] = evaluate_retrieval(
                {query_id: qrels[query_id] for query_id in query_ids},
                {query_id: rankings[query_id] for query_id in query_ids},
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
        "selection_rule": "maximize Hit@5, then MRR@5, then nDCG@5, then minimize chunk count",
        "sentence_overlap_note": (
            "s0 and s2 were byte-identical because normalized XML lacks sentence punctuation; "
            "only s0 was indexed"
        ),
        "selected": selected,
        "evaluations": evaluations,
    }
    markdown = render_markdown(result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown, encoding="utf-8")
    frozen = {
        "dataset": result["dataset"],
        "selected_on": "development_only",
        "development_documents": 40,
        "holdout_documents": 160,
        "primary_metric": "hit@5",
        "selection_rule": result["selection_rule"],
        "fixed_size": selected["fixed_size"],
        "structure_aware": selected["structure_aware"],
        "embedding_model": "intfloat/multilingual-e5-small",
        "reranking": "none (dense-only, frozen from prior pilot)",
    }
    args.selection_output.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(markdown)


def _read_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            grade = float(row["relevance_grade"])
            if grade > 0:
                qrels.setdefault(row["query_id"], {})[row["chunk_id"]] = grade
    return qrels


def _select(evaluations: dict[str, Any], family: str) -> str:
    candidates = [name for name, result in evaluations.items() if result["family"] == family]

    def criterion(name: str) -> tuple[float, float, float, int]:
        result = evaluations[name]
        aggregate = result["overall"]["aggregate"]
        return (
            aggregate["hit@5"],
            aggregate["mrr@5"],
            aggregate["ndcg@5"],
            -int(result["chunk_count"]),
        )

    return max(candidates, key=criterion)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Seleksi desain chunking Indo-Law development",
        "",
        f"Dokumen: {result['document_count']}; pertanyaan: {result['query_count']}.",
        "",
        "| Konfigurasi | Chunk | Hit@1 | Hit@5 | MRR@5 | nDCG@5 | Hit@10 | Hit@50 | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, evaluation in result["evaluations"].items():
        aggregate = evaluation["overall"]["aggregate"]
        lines.append(
            f"| `{name}` | {evaluation['chunk_count']} | {aggregate['hit@1']:.4f} | "
            f"{aggregate['hit@5']:.4f} | {aggregate['mrr@5']:.4f} | "
            f"{aggregate['ndcg@5']:.4f} | {aggregate['hit@10']:.4f} | "
            f"{aggregate['hit@50']:.4f} | {evaluation['latency']['p95_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Desain dibekukan",
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
