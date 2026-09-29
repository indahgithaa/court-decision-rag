"""Create per-strategy qrel candidates from approved evidence spans."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.ground_truth import build_qrel_candidates
from src.utils.io import read_jsonl


FIELDNAMES = [
    "query_id",
    "document_id",
    "target_section_label",
    "strategy",
    "chunk_id",
    "chunk_start_position",
    "chunk_end_position",
    "evidence_start_position",
    "evidence_end_position",
    "evidence_coverage",
    "auto_grade",
    "relevance_grade",
    "reviewer_notes",
]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare qrel review candidates.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/exploration_20_questions.csv"),
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
        "--output",
        type=Path,
        default=Path("data/evaluation/exploration_20_qrel_candidates.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {args.output}; pass --overwrite")
    with args.questions.open("r", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    approved = [
        row
        for row in questions
        if str(row.get("review_status", "")).strip().lower() == "approved"
    ]
    if not approved:
        raise ValueError("No approved questions are available")

    chunks: list[dict[str, Any]] = []
    chunks.extend(read_jsonl(args.fixed_chunks))
    chunks.extend(read_jsonl(args.structure_chunks))
    candidates = build_qrel_candidates(approved, chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Wrote {len(candidates)} qrel candidates to {args.output}")


if __name__ == "__main__":
    main()
