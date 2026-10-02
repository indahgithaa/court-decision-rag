"""Plan a powered document-level holdout from paired pilot Hit@k outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.sample_size import (
    cluster_adjusted_plan,
    estimate_equal_cluster_icc,
    matched_binary_sample_size,
)
from src.utils.io import read_jsonl


STRATEGIES = ("fixed_size", "structure_aware")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plan the independent retrieval holdout.")
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("data/evaluation/exploration_20_qrel_candidates.csv"),
    )
    parser.add_argument(
        "--fixed-run",
        type=Path,
        default=Path("experiments/results/exploration_20_fixed_size_run.jsonl"),
    )
    parser.add_argument(
        "--structure-run",
        type=Path,
        default=Path("experiments/results/exploration_20_structure_aware_run.jsonl"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--questions-per-document", type=int, default=4)
    parser.add_argument("--effects", type=float, nargs="+", default=(0.10, 0.075, 0.05))
    parser.add_argument("--icc-scenarios", type=float, nargs="+", default=(0.0, 0.1, 0.2, 0.3))
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("experiments/results/holdout_sample_size.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("experiments/results/holdout_sample_size.md"),
    )
    args = parser.parse_args(argv)
    if args.k <= 0:
        raise ValueError("k must be positive")

    with args.qrels.open("r", encoding="utf-8-sig", newline="") as file:
        qrel_rows = list(csv.DictReader(file))
    runs = {
        "fixed_size": list(read_jsonl(args.fixed_run)),
        "structure_aware": list(read_jsonl(args.structure_run)),
    }
    pilot = summarize_pilot(qrel_rows, runs, k=args.k)
    plans: list[dict[str, Any]] = []
    for effect in args.effects:
        independent = matched_binary_sample_size(effect, pilot["discordance_rate"])
        for icc in args.icc_scenarios:
            adjusted = cluster_adjusted_plan(
                independent,
                questions_per_document=args.questions_per_document,
                intracluster_correlation=icc,
            )
            plans.append(
                {
                    "minimum_detectable_effect": effect,
                    "icc_scenario": icc,
                    "independent_query_target": independent,
                    **adjusted,
                }
            )

    result = {
        "method": "normal approximation for paired binary outcomes with cluster design-effect sensitivity",
        "alpha": 0.05,
        "power": 0.80,
        "questions_per_document": args.questions_per_document,
        "pilot": pilot,
        "plans": plans,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(result, k=args.k), encoding="utf-8")
    print(args.markdown_output.read_text(encoding="utf-8"))


def summarize_pilot(
    qrel_rows: Sequence[dict[str, str]],
    runs: dict[str, Sequence[dict[str, Any]]],
    *,
    k: int,
) -> dict[str, Any]:
    relevant: dict[str, dict[str, set[str]]] = {
        strategy: defaultdict(set) for strategy in STRATEGIES
    }
    documents: dict[str, str] = {}
    for row in qrel_rows:
        query_id = row["query_id"]
        documents[query_id] = row["document_id"]
        if float(row["relevance_grade"]) > 0:
            relevant[row["strategy"]][query_id].add(row["chunk_id"])

    hits: dict[str, dict[str, int]] = {}
    for strategy in STRATEGIES:
        strategy_hits: dict[str, int] = {}
        for record in runs[strategy]:
            ranking = sorted(record["ranking"], key=lambda item: int(item["rank"]))
            retrieved = {item["chunk_id"] for item in ranking[:k]}
            query_id = str(record["query_id"])
            strategy_hits[query_id] = int(bool(retrieved & relevant[strategy][query_id]))
        hits[strategy] = strategy_hits

    query_ids = sorted(documents)
    if any(set(hits[strategy]) != set(query_ids) for strategy in STRATEGIES):
        raise ValueError("run and qrel query IDs do not match")
    differences_by_document: dict[str, list[float]] = defaultdict(list)
    structure_only = fixed_only = both = neither = 0
    for query_id in query_ids:
        fixed = hits["fixed_size"][query_id]
        structure = hits["structure_aware"][query_id]
        differences_by_document[documents[query_id]].append(float(structure - fixed))
        if fixed and structure:
            both += 1
        elif structure:
            structure_only += 1
        elif fixed:
            fixed_only += 1
        else:
            neither += 1

    discordant = structure_only + fixed_only
    return {
        "query_count": len(query_ids),
        "document_count": len(differences_by_document),
        "both_hit": both,
        "structure_only": structure_only,
        "fixed_only": fixed_only,
        "neither_hit": neither,
        "discordance_rate": discordant / len(query_ids),
        "observed_hit_difference": (structure_only - fixed_only) / len(query_ids),
        "estimated_icc": estimate_equal_cluster_icc(differences_by_document),
    }


def render_markdown(result: dict[str, Any], *, k: int) -> str:
    pilot = result["pilot"]
    lines = [
        "# Perencanaan ukuran holdout retrieval",
        "",
        f"Pilot: {pilot['query_count']} pertanyaan dalam {pilot['document_count']} dokumen; "
        f"discordance Hit@{k} = {pilot['discordance_rate']:.3f}; "
        f"ICC selisih berpasangan = {pilot['estimated_icc']:.3f}.",
        "",
        "| MDE absolut | Asumsi ICC | Design effect | Query independen | Dokumen | Query aktual |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for plan in result["plans"]:
        lines.append(
            f"| {plan['minimum_detectable_effect']:.3f} | {plan['icc_scenario']:.2f} | "
            f"{plan['design_effect']:.2f} | {plan['independent_query_target']} | "
            f"{plan['documents']} | {plan['queries']} |"
        )
    lines.extend(
        [
            "",
            "Perhitungan memakai pendekatan normal McNemar dua sisi (alpha 0,05; power 0,80) "
            "dan inflasi design effect untuk empat pertanyaan per dokumen. Angka ini adalah "
            "panduan perencanaan, bukan hasil inferensial.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
