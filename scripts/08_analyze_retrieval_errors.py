"""Diagnose document-level and chunk-level retrieval failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from src.evaluation.error_analysis import analyze_retrieval_errors, render_error_markdown
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze paired retrieval errors.")
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
        default=Path("experiments/results/exploration_20_fixed_size_run.jsonl"),
    )
    parser.add_argument(
        "--structure-run",
        type=Path,
        default=Path("experiments/results/exploration_20_structure_aware_run.jsonl"),
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
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/results/exploration_20_retrieval_errors.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("experiments/results/exploration_20_retrieval_errors.md"),
    )
    args = parser.parse_args(argv)

    with args.questions.open("r", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    with args.qrels.open("r", encoding="utf-8-sig", newline="") as file:
        qrels = list(csv.DictReader(file))
    chunks = {
        str(chunk["chunk_id"]): chunk
        for path in (args.fixed_chunks, args.structure_chunks)
        for chunk in read_jsonl(path)
    }
    result = analyze_retrieval_errors(
        questions,
        qrels,
        {
            "fixed_size": list(read_jsonl(args.fixed_run)),
            "structure_aware": list(read_jsonl(args.structure_run)),
        },
        chunks,
        k=args.k,
    )
    markdown = render_error_markdown(result)
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
