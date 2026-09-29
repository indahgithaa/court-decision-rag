"""Create evidence-grounded draft questions and a human review packet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping, Sequence

from src.evaluation.ground_truth import join_clean_pages, validate_questions
from src.evaluation.question_drafting import QUESTION_FIELDNAMES, draft_questions
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Draft pilot questions from exact document evidence spans."
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("data/evaluation/exploration_20_questions.csv"),
    )
    parser.add_argument(
        "--sections",
        type=Path,
        default=Path("data/processed/exploration_20/sections.jsonl"),
    )
    parser.add_argument(
        "--pages",
        type=Path,
        default=Path("data/processed/exploration_20/pages.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/exploration_20_questions_draft.csv"),
    )
    parser.add_argument(
        "--review-packet",
        type=Path,
        default=Path("experiments/results/exploration_20_question_review.md"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    for target in (args.output, args.review_packet):
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to replace {target}; pass --overwrite")

    with args.template.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    sections = list(read_jsonl(args.sections))
    pages = list(read_jsonl(args.pages))
    drafts = draft_questions(rows, sections)
    documents = join_clean_pages(pages)
    validation = validate_questions(drafts, documents)
    if not validation["valid"]:
        errors = "\n".join(str(error) for error in validation["errors"])
        raise ValueError(f"Generated drafts failed validation:\n{errors}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=QUESTION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(drafts)

    packet = render_review_packet(drafts, documents)
    args.review_packet.parent.mkdir(parents=True, exist_ok=True)
    args.review_packet.write_text(packet, encoding="utf-8", newline="\n")
    print(f"Wrote {len(drafts)} drafts to {args.output}")
    print(f"Wrote review packet to {args.review_packet}")


def render_review_packet(
    rows: Sequence[Mapping[str, str]],
    documents: Mapping[str, str],
    *,
    context_chars: int = 220,
) -> str:
    """Render evidence plus nearby context for efficient human approval."""
    lines = [
        "# Review pertanyaan retrieval pilot",
        "",
        "Semua entri berikut adalah draft heuristik. Reviewer harus memeriksa "
        "pertanyaan, jawaban, dan kecukupan rentang bukti sebelum memberi status "
        "`approved`.",
        "",
    ]
    for row in rows:
        document = documents[row["document_id"]]
        start = int(row["evidence_start_position"])
        end = int(row["evidence_end_position"])
        context_start = max(0, start - context_chars)
        context_end = min(len(document), end + context_chars)
        before = _single_line(document[context_start:start])
        evidence = _single_line(document[start:end])
        after = _single_line(document[end:context_end])
        lines.extend(
            [
                f"## {row['query_id']} · {row['target_section_label']}",
                "",
                f"- Dokumen: `{row['document_id']}`",
                f"- Pertanyaan: {row['question']}",
                f"- Jawaban rujukan: {row['reference_answer']}",
                f"- Offset: `{start}:{end}`",
                f"- Difficulty: `{row['difficulty']}`",
                "",
                f"> …{before} **[{evidence}]** {after}…",
                "",
                "Keputusan reviewer: ☐ approved ☐ revise ☐ rejected",
                "",
            ]
        )
    return "\n".join(lines)


def _single_line(text: str) -> str:
    return " ".join(text.split()).replace("[", "\\[").replace("]", "\\]")


if __name__ == "__main__":
    main()
