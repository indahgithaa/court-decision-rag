"""Evaluate the frozen fixed-size and SAC designs on the locked holdout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from src.evaluation.retrieval_report import evaluate_paired_runs, render_markdown
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Indo-Law holdout designs.")
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("experiments/indolaw_200_selected_design.json"),
    )
    parser.add_argument(
        "--evaluation-root", type=Path, default=Path("data/evaluation/indolaw_200")
    )
    parser.add_argument(
        "--runs-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/results/indolaw_200_holdout_evaluation.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("experiments/results/indolaw_200_holdout_evaluation.md"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    fixed_name = str(selection["fixed_size"])
    structure_name = str(selection["structure_aware"])
    qrel_rows = _read_csv(
        args.evaluation_root / f"holdout_{fixed_name}_qrels.csv"
    ) + _read_csv(args.evaluation_root / f"holdout_{structure_name}_qrels.csv")
    runs = {
        "fixed_size": list(
            read_jsonl(args.runs_root / f"indolaw_200_holdout_{fixed_name}_top50.jsonl")
        ),
        "structure_aware": list(
            read_jsonl(
                args.runs_root / f"indolaw_200_holdout_{structure_name}_top50.jsonl"
            )
        ),
    }
    result = evaluate_paired_runs(
        qrel_rows,
        runs,
        ks=(1, 3, 5, 10, 50),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result["dataset"] = "Indo-Law 200 holdout"
    result["evaluation_stage"] = "final_holdout"
    result["frozen_designs"] = {
        "fixed_size": fixed_name,
        "structure_aware": structure_name,
    }
    result["scope_note"] = (
        "Normalized XML text with corpus-provided section boundaries; this is an "
        "oracle-structure robustness study, not a PDF extraction evaluation."
    )
    markdown = render_markdown(result)
    markdown += (
        "\n## Desain dan batas interpretasi\n\n"
        f"Fixed-size: `{fixed_name}`; structure-aware: `{structure_name}`. "
        "Keduanya dipilih hanya pada development 40 dokumen.\n\n"
        "Corpus menggunakan teks XML ternormalisasi dan batas section dari Indo-Law. "
        "Hasil ini menguji oracle-structure chunking dan tidak mencakup error ekstraksi PDF "
        "atau deteksi section otomatis.\n"
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    main()
