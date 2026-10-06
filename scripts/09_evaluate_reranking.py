"""Run a paired dense/BM25 reranking sensitivity experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.retrieval_report import evaluate_paired_runs
from src.retrieval import rerank_records
from src.utils.io import read_jsonl, write_jsonl


STRATEGIES = ("fixed_size", "structure_aware")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 reranking weights on paired top-N dense runs."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/exploration_20_questions.csv"),
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("data/evaluation/exploration_20_qrel_candidates.csv"),
    )
    parser.add_argument(
        "--fixed-run",
        type=Path,
        default=Path(
            "experiments/results/exploration_20_fixed_size_top50_run.jsonl"
        ),
    )
    parser.add_argument(
        "--structure-run",
        type=Path,
        default=Path(
            "experiments/results/exploration_20_structure_aware_top50_run.jsonl"
        ),
    )
    parser.add_argument(
        "--fixed-chunks",
        type=Path,
        default=Path("data/chunks/exploration_20/fixed_size/chunks.jsonl"),
    )
    parser.add_argument(
        "--structure-chunks",
        type=Path,
        default=Path("data/chunks/exploration_20/structure_aware/chunks.jsonl"),
    )
    parser.add_argument(
        "--dense-weights",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
    )
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("experiments/results/reranking_runs"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "experiments/results/exploration_20_reranking_sensitivity.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/results/exploration_20_reranking_sensitivity.md"
        ),
    )
    args = parser.parse_args(argv)

    weights = _validate_weights(args.dense_weights)
    questions = _read_csv(args.questions)
    qrels = _read_csv(args.qrels)
    source_runs = {
        "fixed_size": list(read_jsonl(args.fixed_run)),
        "structure_aware": list(read_jsonl(args.structure_run)),
    }
    chunks = {
        "fixed_size": list(read_jsonl(args.fixed_chunks)),
        "structure_aware": list(read_jsonl(args.structure_chunks)),
    }
    evaluations: dict[str, Any] = {}
    run_files: dict[str, dict[str, str]] = {}
    for weight in weights:
        key = _weight_key(weight)
        paired_runs = {
            strategy: rerank_records(
                questions,
                source_runs[strategy],
                chunks[strategy],
                dense_weight=weight,
                rrf_constant=args.rrf_constant,
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
            )
            for strategy in STRATEGIES
        }
        run_files[key] = {}
        for strategy in STRATEGIES:
            path = args.runs_dir / f"{strategy}_dense_weight_{key}.jsonl"
            write_jsonl(path, paired_runs[strategy])
            run_files[key][strategy] = str(path)
        evaluations[key] = evaluate_paired_runs(
            qrels,
            paired_runs,
            ks=(1, 3, 5, 10, 50),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )

    selected_key = _select_weight(evaluations)
    candidate_depth = min(
        len(record["ranking"])
        for strategy in STRATEGIES
        for record in source_runs[strategy]
    )
    result = {
        "method": "weighted_rrf_dense_bm25",
        "candidate_depth": candidate_depth,
        "dense_weights": weights,
        "rrf_constant": args.rrf_constant,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "selection_rule": (
            "highest mean nDCG@5 across both strategies, then mean MRR@5, "
            "mean Recall@5, then larger dense weight"
        ),
        "selected_dense_weight": float(selected_key),
        "evaluations": evaluations,
        "run_files": run_files,
    }
    markdown = render_sensitivity_markdown(result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"Wrote JSON report to {args.output_json}")
    print(f"Wrote Markdown report to {args.output_md}")


def render_sensitivity_markdown(result: Mapping[str, Any]) -> str:
    """Render the compact comparison used to choose a pilot reranking weight."""
    lines = [
        "# Sensitivitas reranking dense + BM25",
        "",
        "Eksperimen pilot ini mererank kandidat dense top-"
        f"{result['candidate_depth']} dengan weighted reciprocal-rank fusion. ",
        "Bobot dan parameter identik digunakan untuk fixed-size dan SAC.",
        "",
        "| Dense | BM25 | Fixed Recall@5 | SAC Recall@5 | Fixed MRR@5 | SAC MRR@5 | Fixed nDCG@5 | SAC nDCG@5 | SAC - fixed nDCG | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    evaluations = result["evaluations"]
    for key, evaluation in evaluations.items():
        dense_weight = float(key)
        fixed = evaluation["strategies"]["fixed_size"]["overall"]["aggregate"]
        structure = evaluation["strategies"]["structure_aware"]["overall"][
            "aggregate"
        ]
        delta = evaluation["paired_difference"]["ndcg@5"]
        lines.append(
            f"| {dense_weight:.2f} | {1 - dense_weight:.2f} | "
            f"{fixed['recall@5']:.4f} | {structure['recall@5']:.4f} | "
            f"{fixed['mrr@5']:.4f} | {structure['mrr@5']:.4f} | "
            f"{fixed['ndcg@5']:.4f} | {structure['ndcg@5']:.4f} | "
            f"{delta['mean_difference']:+.4f} | "
            f"[{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] |"
        )

    selected_key = _weight_key(float(result["selected_dense_weight"]))
    selected = evaluations[selected_key]
    fixed_sections = selected["strategies"]["fixed_size"]["by_section"]
    structure_sections = selected["strategies"]["structure_aware"]["by_section"]
    lines.extend(
        [
            "",
            "## Operating point pilot",
            "",
            f"Dense weight terpilih: **{float(selected_key):.2f}**; BM25 weight: "
            f"**{1 - float(selected_key):.2f}**.",
            "",
            "Aturan seleksi: " + str(result["selection_rule"]) + ".",
            "",
            (
                "Keputusan pilot: pertahankan dense-only; BM25 tidak meningkatkan "
                "operating point bersama kedua strategi."
                if float(selected_key) == 1.0
                else "Keputusan pilot: bekukan bobot fusion ini untuk evaluasi holdout."
            ),
            "",
            "| Bagian | N | Fixed Recall@5 | SAC Recall@5 | Fixed MRR@5 | SAC MRR@5 | Fixed nDCG@5 | SAC nDCG@5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in fixed_sections:
        lines.append(
            f"| `{label}` | {fixed_sections[label]['query_count']} | "
            f"{fixed_sections[label]['aggregate']['recall@5']:.4f} | "
            f"{structure_sections[label]['aggregate']['recall@5']:.4f} | "
            f"{fixed_sections[label]['aggregate']['mrr@5']:.4f} | "
            f"{structure_sections[label]['aggregate']['mrr@5']:.4f} | "
            f"{fixed_sections[label]['aggregate']['ndcg@5']:.4f} | "
            f"{structure_sections[label]['aggregate']['ndcg@5']:.4f} |"
        )
    fixed_overall = selected["strategies"]["fixed_size"]["overall"]["aggregate"]
    structure_overall = selected["strategies"]["structure_aware"]["overall"][
        "aggregate"
    ]
    lines.extend(
        [
            "",
            "## Batas interpretasi",
            "",
            f"Candidate Recall@50: fixed-size {fixed_overall['recall@50']:.4f}; "
            f"SAC {structure_overall['recall@50']:.4f}.",
            "Bobot dipilih pada exploration_20 dan hanya boleh diperlakukan sebagai "
            "konfigurasi development. Klaim utama tetap memerlukan evaluasi holdout.",
            "",
        ]
    )
    return chr(10).join(lines)


def _select_weight(evaluations: Mapping[str, Mapping[str, Any]]) -> str:
    def criterion(key: str) -> tuple[float, float, float, float]:
        evaluation = evaluations[key]
        aggregates = [
            evaluation["strategies"][strategy]["overall"]["aggregate"]
            for strategy in STRATEGIES
        ]
        mean_ndcg = sum(values["ndcg@5"] for values in aggregates) / len(aggregates)
        mean_mrr = sum(values["mrr@5"] for values in aggregates) / len(aggregates)
        mean_recall = sum(values["recall@5"] for values in aggregates) / len(
            aggregates
        )
        return (mean_ndcg, mean_mrr, mean_recall, float(key))

    return max(evaluations, key=criterion)


def _validate_weights(values: Sequence[float]) -> list[float]:
    weights = [float(value) for value in values]
    if not weights or any(not 0 <= value <= 1 for value in weights):
        raise ValueError("dense weights must be between zero and one")
    if len(set(weights)) != len(weights):
        raise ValueError("dense weights must be unique")
    return weights


def _weight_key(weight: float) -> str:
    return f"{weight:.2f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    main()
