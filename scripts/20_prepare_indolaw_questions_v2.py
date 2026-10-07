"""Create enriched, human-reviewable Indo-Law question drafts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from src.evaluation.indolaw_questions import (
    INDOLAW_V2_FIELDNAMES,
    build_indolaw_questions_v2,
)
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare enriched Indo-Law question drafts for human review."
    )
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed/indolaw_200")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/evaluation/indolaw_200_v2")
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("development", "holdout"),
        default=("development",),
        help="Defaults to development; holdout must be requested explicitly.",
    )
    args = parser.parse_args(argv)

    for split in args.splits:
        output = args.output_root / f"{split}_questions_draft.csv"
        failure_path = args.output_root / f"{split}_question_failures.json"
        if output.exists() or failure_path.exists():
            raise FileExistsError(f"Refusing to overwrite v2 drafts for {split}")
        sections = list(
            read_jsonl(args.processed_root / split / "sections.jsonl")
        )
        rows, failures = build_indolaw_questions_v2(
            sections, query_prefix=f"indolaw2-{split[:3]}"
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=INDOLAW_V2_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        types = Counter(row["question_type"] for row in rows)
        anchors = Counter(row["query_anchor_scope"] for row in rows)
        print(
            f"{split}: {len(rows)} draft questions; types={dict(types)}; "
            f"anchors={dict(anchors)}; failures={len(failures)}"
        )


if __name__ == "__main__":
    main()
