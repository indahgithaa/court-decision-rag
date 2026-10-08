"""Evaluate the Fixed/SAC x independent/contextual 2x2 retrieval design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.factorial_retrieval_report import (
    evaluate_factorial_retrieval,
    render_factorial_retrieval_markdown,
)
from src.evaluation import factorial_retrieval_report as report_module
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/indolaw_200/corpus_questions.csv"),
    )
    parser.add_argument(
        "--fixed-qrels",
        type=Path,
        default=Path(
            "data/evaluation/indolaw_200/corpus_fixed_w150_o30_qrels.csv"
        ),
    )
    parser.add_argument(
        "--sac-qrels",
        type=Path,
        default=Path(
            "data/evaluation/indolaw_200/corpus_sac_w150_o30_s0_qrels.csv"
        ),
    )
    parser.add_argument(
        "--fixed-independent-run",
        type=Path,
        default=Path(
            "experiments/results/indolaw_200_corpus_fixed_w150_o30_top50.jsonl"
        ),
    )
    parser.add_argument("--fixed-contextual-run", type=Path, required=True)
    parser.add_argument(
        "--sac-independent-run",
        type=Path,
        default=Path(
            "experiments/results/indolaw_200_corpus_sac_w150_o30_s0_top50.jsonl"
        ),
    )
    parser.add_argument("--sac-contextual-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("Refusing to overwrite factorial evaluation output")

    run_paths = {
        "fixed_independent": args.fixed_independent_run,
        "fixed_contextual": args.fixed_contextual_run,
        "sac_independent": args.sac_independent_run,
        "sac_contextual": args.sac_contextual_run,
    }
    result = evaluate_factorial_retrieval(
        _read_csv(args.fixed_qrels),
        _read_csv(args.sac_qrels),
        _read_csv(args.questions),
        {arm: list(read_jsonl(path)) for arm, path in run_paths.items()},
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result["inputs"] = {
        "questions": args.questions.as_posix(),
        "fixed_qrels": args.fixed_qrels.as_posix(),
        "sac_qrels": args.sac_qrels.as_posix(),
        "runs": {arm: path.as_posix() for arm, path in run_paths.items()},
    }
    input_paths = {
        "questions": args.questions,
        "fixed_qrels": args.fixed_qrels,
        "sac_qrels": args.sac_qrels,
        **{f"run_{arm}": path for arm, path in run_paths.items()},
    }
    result["evaluation_stage"] = "single_corpus_exploratory"
    result["provenance"] = {
        "input_sha256": {
            name: _sha256(path) for name, path in input_paths.items()
        },
        "implementation_sha256": {
            "runner": _sha256(Path(__file__).resolve()),
            "evaluator": _sha256(Path(report_module.__file__).resolve()),
        },
    }
    markdown = render_factorial_retrieval_markdown(result)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(markdown, encoding="utf-8")
    # Windows consoles commonly use a legacy code page that cannot encode
    # symbols used in the Markdown report.  Keep the UTF-8 artifact intact,
    # while making the optional console preview best-effort.
    console_encoding = sys.stdout.encoding or "utf-8"
    print(
        markdown.encode(console_encoding, errors="replace").decode(
            console_encoding
        )
    )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
