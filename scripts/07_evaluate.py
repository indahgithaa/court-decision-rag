"""Evaluate fixed-size and structure-aware retrieval runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from src.evaluation.retrieval_report import evaluate_paired_runs, render_markdown
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired retrieval runs.")
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
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/results/exploration_20_retrieval_evaluation.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("experiments/results/exploration_20_retrieval_evaluation.md"),
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    with args.qrels.open("r", encoding="utf-8-sig", newline="") as file:
        qrel_rows = list(csv.DictReader(file))
    result = evaluate_paired_runs(
        qrel_rows,
        {
            "fixed_size": list(read_jsonl(args.fixed_run)),
            "structure_aware": list(read_jsonl(args.structure_run)),
        },
        ks=args.ks,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    markdown = render_markdown(result)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.output_md.write_text(markdown, encoding="utf-8", newline="\n")
    print(markdown, end="")
    print(f"Wrote JSON report to {args.output_json}")
    print(f"Wrote Markdown report to {args.output_md}")


if __name__ == "__main__":
    main()

