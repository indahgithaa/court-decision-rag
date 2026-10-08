"""Evaluate SAC independent versus contextualized chunk embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.contextual_embedding_report import (
    evaluate_contextual_embedding_runs,
    render_contextual_embedding_markdown,
)
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/indolaw_200/corpus_questions.csv"),
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path(
            "data/evaluation/indolaw_200/corpus_sac_w150_o30_s0_qrels.csv"
        ),
    )
    parser.add_argument("--independent-run", type=Path, required=True)
    parser.add_argument("--contextual-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("Refusing to overwrite contextual embedding report")

    questions = _read_csv(args.questions)
    qrels = _read_csv(args.qrels)
    result = evaluate_contextual_embedding_runs(
        qrels,
        questions,
        {
            "independent": list(read_jsonl(args.independent_run)),
            "contextual": list(read_jsonl(args.contextual_run)),
        },
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result["questions_path"] = args.questions.as_posix()
    result["qrels_path"] = args.qrels.as_posix()
    result["independent_run_path"] = args.independent_run.as_posix()
    result["contextual_run_path"] = args.contextual_run.as_posix()
    markdown = render_contextual_embedding_markdown(result)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    main()
