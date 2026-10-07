"""Evaluate Summary-Augmented Structure-Aware Chunking and its ablations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.document_retrieval_metrics import evaluate_document_retrieval
from src.evaluation.latency import summarize_latency
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.evaluation.retrieval_report import paired_cluster_bootstrap_interval
from src.utils.io import read_jsonl


DEFAULT_CONFIGS = (
    "fixed_w150_o30",
    "fixed_w300_o60",
    "fixed_w500_o100",
    "sac_w150_o30_s0",
    "sac_w300_o60_s0",
    "sac_w500_o100_s0",
    "summary_fixed_w150_o30",
    "summary_fixed_w300_o60",
    "summary_fixed_w500_o100",
    "hybrid_w150_o30_s0",
    "hybrid_w300_o60_s0",
    "hybrid_w500_o100_s0",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="indolaw_200")
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument(
        "--questions-root", type=Path, default=Path("data/evaluation/indolaw_200")
    )
    parser.add_argument(
        "--chunks-root", type=Path, default=Path("data/chunks/indolaw_200")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-stage",
        choices=("development", "corrected_exploratory_holdout", "final_holdout"),
        default="development",
    )
    args = parser.parse_args(argv)
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("Refusing to overwrite a hybrid evaluation report")

    question_path = args.questions_root / f"{args.split}_questions.csv"
    with question_path.open("r", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    query_documents = {row["query_id"]: row["document_id"] for row in questions}
    labels = {row["query_id"]: row["target_section_label"] for row in questions}

    evaluations: dict[str, Any] = {}
    shared_evidence: dict[str, set[str]] | None = None
    for design in args.configs:
        qrels, evidence_qrels = _read_qrels(
            args.questions_root / f"{args.split}_{design}_qrels.csv"
        )
        current_evidence = {
            query_id: set(values) for query_id, values in evidence_qrels.items()
        }
        if shared_evidence is None:
            shared_evidence = current_evidence
        elif current_evidence != shared_evidence:
            raise ValueError(f"Evidence units differ for {design}")

        run_path = args.runs_root / (
            f"{args.dataset_id}_{args.split}_{design}_top50.jsonl"
        )
        records = list(read_jsonl(run_path))
        rankings = {
            str(record["query_id"]): [
                str(item["chunk_id"])
                for item in sorted(record["ranking"], key=lambda item: int(item["rank"]))
            ]
            for record in records
        }
        document_rankings = {
            str(record["query_id"]): [
                str(item["retrieved_document_id"])
                for item in sorted(record["ranking"], key=lambda item: int(item["rank"]))
            ]
            for record in records
        }
        if set(rankings) != set(query_documents):
            raise ValueError(f"Run/query mismatch for {design}")
        retrieval = evaluate_retrieval(
            qrels,
            rankings,
            evidence_qrels=evidence_qrels,
            ks=(1, 3, 5, 10, 50),
        )
        by_section: dict[str, Any] = {}
        for label in sorted(set(labels.values())):
            ids = {query_id for query_id, value in labels.items() if value == label}
            by_section[label] = evaluate_retrieval(
                {query_id: qrels[query_id] for query_id in ids},
                {query_id: rankings[query_id] for query_id in ids},
                evidence_qrels={query_id: evidence_qrels[query_id] for query_id in ids},
                ks=(1, 3, 5, 10, 50),
            )
        document = evaluate_document_retrieval(
            query_documents,
            document_rankings,
            ks=(1, 3, 5, 10, 50),
        )
        chunk_path = args.chunks_root / args.split / design / "chunks.jsonl"
        evaluations[design] = {
            "family": _family(design),
            "chunk_count": sum(1 for _ in read_jsonl(chunk_path)),
            "retrieval": retrieval,
            "by_section": by_section,
            "document_retrieval": document,
            "latency": summarize_latency([float(row["latency_ms"]) for row in records]),
        }

    families = sorted({result["family"] for result in evaluations.values()})
    selected = {family: _select(evaluations, family) for family in families}
    hybrid_design = selected.get("structure_summary_augmented")
    if hybrid_design is None:
        raise ValueError("At least one hybrid design is required")
    comparisons = {
        design: _paired_comparison(
            evaluations[hybrid_design],
            evaluations[design],
            query_documents,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for family, design in selected.items()
        if family != "structure_summary_augmented"
    }
    result = {
        "dataset": args.dataset_id,
        "split": args.split,
        "evaluation_stage": args.evaluation_stage,
        "metric_schema_version": "retrieval-v2-evidence-recall+drm",
        "query_count": len(questions),
        "document_count": len(set(query_documents.values())),
        "selection_rule": "maximize nDCG@5, then MRR@5, Recall@5, then minimize chunks",
        "selected": selected,
        "hybrid_reference": hybrid_design,
        "evaluations": evaluations,
        "paired_hybrid_minus_baseline": comparisons,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "document",
        "seed": args.seed,
    }
    markdown = render_markdown(result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


def _family(design: str) -> str:
    if design.startswith("summary_fixed_"):
        return "summary_augmented_fixed"
    if design.startswith("hybrid_"):
        return "structure_summary_augmented"
    if design.startswith("fixed_"):
        return "fixed_size"
    if design.startswith("sac_"):
        return "structure_aware"
    raise ValueError(f"Unknown design family: {design}")


def _select(evaluations: dict[str, Any], family: str) -> str:
    candidates = [name for name, value in evaluations.items() if value["family"] == family]

    def key(name: str) -> tuple[float, float, float, int]:
        value = evaluations[name]
        aggregate = value["retrieval"]["aggregate"]
        return (
            aggregate["ndcg@5"],
            aggregate["mrr@5"],
            aggregate["recall@5"],
            -int(value["chunk_count"]),
        )

    return max(candidates, key=key)


def _paired_comparison(
    hybrid: dict[str, Any],
    baseline: dict[str, Any],
    query_documents: dict[str, str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    hybrid_rows = {
        **hybrid["retrieval"]["per_query"],
    }
    baseline_rows = baseline["retrieval"]["per_query"]
    metric_names = list(hybrid["retrieval"]["aggregate"])
    results: dict[str, Any] = {}
    for metric in metric_names:
        differences: dict[str, list[float]] = defaultdict(list)
        flat: list[float] = []
        for query_id in sorted(query_documents):
            difference = hybrid_rows[query_id][metric] - baseline_rows[query_id][metric]
            differences[query_documents[query_id]].append(difference)
            flat.append(difference)
        low, high = paired_cluster_bootstrap_interval(
            differences,
            samples=samples,
            seed=seed,
        )
        results[metric] = {
            "mean_difference": statistics.fmean(flat),
            "ci95_low": low,
            "ci95_high": high,
        }
    for metric in ("drm@1", "drm@5", "document_hit@1", "document_hit@5"):
        differences = defaultdict(list)
        flat = []
        for query_id in sorted(query_documents):
            difference = (
                hybrid["document_retrieval"]["per_query"][query_id][metric]
                - baseline["document_retrieval"]["per_query"][query_id][metric]
            )
            differences[query_documents[query_id]].append(difference)
            flat.append(difference)
        low, high = paired_cluster_bootstrap_interval(
            differences,
            samples=samples,
            seed=seed,
        )
        results[metric] = {
            "mean_difference": statistics.fmean(flat),
            "ci95_low": low,
            "ci95_high": high,
        }
    return results


def _read_qrels(
    path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[str, ...]]]]:
    qrels: dict[str, dict[str, float]] = {}
    evidence: dict[str, dict[str, set[str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            grade = float(row["relevance_grade"])
            if grade <= 0:
                continue
            query_id = row["query_id"]
            chunk_id = row["chunk_id"]
            evidence_id = row.get("evidence_id", "").strip()
            if not evidence_id:
                raise ValueError(f"Missing evidence_id in {path}")
            qrels.setdefault(query_id, {})[chunk_id] = grade
            evidence.setdefault(query_id, {}).setdefault(evidence_id, set()).add(chunk_id)
    return qrels, {
        query_id: {
            evidence_id: tuple(sorted(chunk_ids))
            for evidence_id, chunk_ids in by_id.items()
        }
        for query_id, by_id in evidence.items()
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Evaluasi hybrid chunking — {result['split']}",
        "",
        f"Dokumen: {result['document_count']}; pertanyaan: {result['query_count']}.",
        "",
        "| Desain | Family | Chunk | Recall@5 | MRR@5 | nDCG@5 | DRM@5 ↓ | DocHit@5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["evaluations"].items():
        retrieval = value["retrieval"]["aggregate"]
        document = value["document_retrieval"]["aggregate"]
        lines.append(
            f"| `{name}` | `{value['family']}` | {value['chunk_count']} | "
            f"{retrieval['recall@5']:.4f} | {retrieval['mrr@5']:.4f} | "
            f"{retrieval['ndcg@5']:.4f} | {document['drm@5']:.4f} | "
            f"{document['document_hit@5']:.4f} |"
        )
    lines.extend(["", "## Terpilih per family", ""])
    for family, design in result["selected"].items():
        lines.append(f"- `{family}`: `{design}`")
    lines.extend(
        [
            "",
            "## Hybrid dibanding baseline terpilih",
            "",
            "Selisih adalah hybrid dikurangi baseline; untuk DRM, nilai negatif lebih baik.",
            "",
            "| Baseline | ΔRecall@5 | 95% CI | ΔnDCG@5 | 95% CI | ΔDRM@5 | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline, metrics in result["paired_hybrid_minus_baseline"].items():
        recall = metrics["recall@5"]
        ndcg = metrics["ndcg@5"]
        drm = metrics["drm@5"]
        lines.append(
            f"| `{baseline}` | {recall['mean_difference']:+.4f} | "
            f"[{recall['ci95_low']:+.4f}, {recall['ci95_high']:+.4f}] | "
            f"{ndcg['mean_difference']:+.4f} | "
            f"[{ndcg['ci95_low']:+.4f}, {ndcg['ci95_high']:+.4f}] | "
            f"{drm['mean_difference']:+.4f} | "
            f"[{drm['ci95_low']:+.4f}, {drm['ci95_high']:+.4f}] |"
        )
    reference = result["hybrid_reference"]
    lines.extend(["", f"## Per bagian — `{reference}`", ""])
    lines.extend(
        [
            "| Bagian | N | Recall@5 | MRR@5 | nDCG@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, metrics in result["evaluations"][reference]["by_section"].items():
        aggregate = metrics["aggregate"]
        lines.append(
            f"| `{label}` | {metrics['query_count']} | {aggregate['recall@5']:.4f} | "
            f"{aggregate['mrr@5']:.4f} | {aggregate['ndcg@5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Confidence interval memakai paired cluster bootstrap pada unit dokumen.",
            "",
        ]
    )
    if result["evaluation_stage"] == "corrected_exploratory_holdout":
        lines.extend(
            [
                "Holdout ini pernah dibuka pada eksperimen sebelumnya; hasil bukan "
                "konfirmasi blind.",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
