"""Prepare paired, word-budget-matched inputs for answer generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.generation.context import assemble_ranked_context, index_chunks
from src.generation.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from src.utils.io import read_jsonl, write_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare matched fixed-size and SAC answer-generation inputs."
    )
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("experiments/indolaw_200_selected_design.json"),
    )
    parser.add_argument(
        "--evaluation-root", type=Path, default=Path("data/evaluation/indolaw_200")
    )
    parser.add_argument(
        "--chunks-root", type=Path, default=Path("data/chunks/indolaw_200")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--word-budget", type=int, default=1_200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    designs = {
        "fixed_size": str(selection["fixed_size"]),
        "structure_aware": str(selection["structure_aware"]),
    }
    questions = _read_approved_questions(
        args.evaluation_root / f"{args.split}_questions.csv"
    )
    question_ids = set(questions)
    records: list[dict[str, Any]] = []

    for strategy, design in designs.items():
        run_path = args.runs_root / f"indolaw_200_{args.split}_{design}_top50.jsonl"
        runs = _index_runs(read_jsonl(run_path))
        if set(runs) != question_ids:
            missing = sorted(question_ids - set(runs))
            unexpected = sorted(set(runs) - question_ids)
            raise ValueError(
                f"Run/query mismatch for {strategy}; missing={missing}, "
                f"unexpected={unexpected}"
            )
        chunks = index_chunks(
            read_jsonl(args.chunks_root / args.split / design / "chunks.jsonl")
        )
        for query_id, question in questions.items():
            run = runs[query_id]
            assembled = assemble_ranked_context(
                run["ranking"], chunks, word_budget=args.word_budget
            )
            evidence_start = int(question["evidence_start_position"])
            evidence_end = int(question["evidence_end_position"])
            primary_evidence_in_context = any(
                source["document_id"] == question["document_id"]
                and source["included_start_position"] <= evidence_start
                and evidence_end <= source["included_end_position"]
                for source in assembled["selected_chunks"]
            )
            records.append(
                {
                    "query_id": query_id,
                    "document_id": question["document_id"],
                    "strategy": strategy,
                    "design": design,
                    "question": question["question"],
                    "reference_answer": question["reference_answer"],
                    "target_section_label": question["target_section_label"],
                    "evidence_start_position": evidence_start,
                    "evidence_end_position": evidence_end,
                    "primary_evidence_in_context": primary_evidence_in_context,
                    "prompt_version": PROMPT_VERSION,
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": build_user_prompt(
                        question=question["question"], context=assembled["context"]
                    ),
                    "retrieval_latency_ms": float(run["latency_ms"]),
                    **assembled,
                }
            )

    records.sort(key=lambda item: (item["query_id"], item["strategy"]))
    output = args.output or (
        args.runs_root / f"indolaw_200_{args.split}_answer_inputs.jsonl"
    )
    count = write_jsonl(output, records)
    print(
        f"Wrote {count} records ({len(questions)} paired questions) to {output}; "
        f"context={args.word_budget} words per record; prompt={PROMPT_VERSION}"
    )


def _read_approved_questions(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    approved: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("review_status", "").strip().lower() != "approved":
            continue
        query_id = row.get("query_id", "").strip()
        if not query_id:
            raise ValueError("Approved question has an empty query_id")
        if query_id in approved:
            raise ValueError(f"Duplicate approved query_id: {query_id}")
        approved[query_id] = row
    if not approved:
        raise ValueError(f"No approved questions found in {path}")
    return approved


def _index_runs(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id", "")).strip()
        if not query_id:
            raise ValueError("Every run must have a non-empty query_id")
        if query_id in indexed:
            raise ValueError(f"Duplicate run query_id: {query_id}")
        indexed[query_id] = record
    return indexed


if __name__ == "__main__":
    main()
