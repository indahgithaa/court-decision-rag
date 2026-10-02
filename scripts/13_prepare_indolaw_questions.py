"""Create exact-span benchmark questions for each Indo-Law split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from src.evaluation.indolaw_questions import build_indolaw_questions
from src.evaluation.question_drafting import QUESTION_FIELDNAMES
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare Indo-Law benchmark questions.")
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed/indolaw_200")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/evaluation/indolaw_200")
    )
    args = parser.parse_args(argv)
    total_questions = 0
    total_failures = 0
    for split in ("development", "holdout"):
        sections = list(read_jsonl(args.processed_root / split / "sections.jsonl"))
        rows, failures = build_indolaw_questions(
            sections, query_prefix=f"indolaw-{split[:3]}"
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        output = args.output_root / f"{split}_questions.csv"
        with output.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=QUESTION_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        failure_path = args.output_root / f"{split}_question_failures.json"
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        total_questions += len(rows)
        total_failures += len(failures)
        print(
            f"{split}: {len(rows)} questions from {len(rows) // 4} documents; "
            f"{len(failures)} excluded documents"
        )
    print(f"total: {total_questions} questions; {total_failures} exclusions")


if __name__ == "__main__":
    main()
