"""Create exact-span benchmark questions for the complete Indo-Law corpus."""

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
    sections = list(read_jsonl(args.processed_root / "corpus" / "sections.jsonl"))
    rows, failures = build_indolaw_questions(sections, query_prefix="indolaw-cor")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "corpus_questions.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=QUESTION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    failure_path = args.output_root / "corpus_question_failures.json"
    failure_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"corpus: {len(rows)} questions from {len(rows) // 4} documents; "
        f"{len(failures)} excluded documents"
    )


if __name__ == "__main__":
    main()
