"""Build the frozen fixed-size and structure-aware design grid."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.chunking import FixedSizeChunker, StructureAwareChunker
from src.evaluation.ground_truth import build_qrel_candidates
from src.utils.io import read_jsonl, write_jsonl


FIELDNAMES = [
    "query_id",
    "document_id",
    "target_section_label",
    "evidence_id",
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
    parser = argparse.ArgumentParser(description="Build the Indo-Law chunking grid.")
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed/indolaw_200")
    )
    parser.add_argument(
        "--questions-root", type=Path, default=Path("data/evaluation/indolaw_200")
    )
    parser.add_argument(
        "--chunks-root", type=Path, default=Path("data/chunks/indolaw_200")
    )
    parser.add_argument("--splits", nargs="+", default=("development", "holdout"))
    parser.add_argument(
        "--designs",
        nargs="+",
        help="Build only named designs; experimental designs require explicit selection.",
    )
    args = parser.parse_args(argv)

    available_designs = dict(design_grid(include_experimental=True))
    selected_names = args.designs or [
        name for name in available_designs if not name.endswith("_tail")
    ]
    unknown = sorted(set(selected_names) - set(available_designs))
    if unknown:
        raise ValueError(f"Unknown design(s): {', '.join(unknown)}")

    for split in args.splits:
        pages = list(read_jsonl(args.processed_root / split / "pages.jsonl"))
        sections = list(read_jsonl(args.processed_root / split / "sections.jsonl"))
        with (args.questions_root / f"{split}_questions.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            questions = list(csv.DictReader(file))
        text_by_document = {str(page["document_id"]): str(page["clean_text"]) for page in pages}
        sections_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for section in sections:
            sections_by_document[str(section["document_id"])].append(section)
        for values in sections_by_document.values():
            values.sort(key=lambda row: int(row["start_position"]))

        for name in selected_names:
            chunker = available_designs[name]
            directory = args.chunks_root / split / name
            qrels_path = args.questions_root / f"{split}_{name}_qrels.csv"
            if (directory / "chunks.jsonl").exists() or qrels_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite artifacts for {split}/{name}"
                )
            chunks: list[dict[str, Any]] = []
            for document_id, text in sorted(text_by_document.items()):
                rows = chunker.chunk(
                    document_id,
                    text,
                    sections=sections_by_document.get(document_id),
                )
                chunks.extend(row.to_dict() for row in rows)
            write_jsonl(directory / "chunks.jsonl", chunks)
            candidates = build_qrel_candidates(questions, chunks)
            complete = [row for row in candidates if int(row["auto_grade"]) == 2]
            complete.extend(
                _alternate_statute_qrels(
                    questions,
                    chunks,
                    sections_by_document,
                    existing={(row["query_id"], row["chunk_id"]) for row in complete},
                )
            )
            covered = {str(row["query_id"]) for row in complete}
            expected = {str(row["query_id"]) for row in questions}
            if covered != expected:
                missing = sorted(expected - covered)
                raise RuntimeError(
                    f"{split}/{name} lacks complete evidence chunks for {missing[:5]}"
                )
            with qrels_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(complete)
            print(
                f"{split}/{name}: {len(chunks)} chunks; {len(complete)} positive qrels"
            )


def design_grid(
    *, include_experimental: bool = False
) -> list[tuple[str, FixedSizeChunker | StructureAwareChunker]]:
    """Return legacy candidates and optional post-hoc development ablations."""
    designs: list[tuple[str, FixedSizeChunker | StructureAwareChunker]] = []
    for size in (150, 300, 500):
        overlap = size // 5
        designs.append(
            (f"fixed_w{size}_o{overlap}", FixedSizeChunker(max_words=size, overlap_words=overlap))
        )
        for sentences in (0, 2):
            designs.append(
                (
                    f"sac_w{size}_o{overlap}_s{sentences}",
                    StructureAwareChunker(
                        max_words=size,
                        overlap_words=overlap,
                        overlap_sentences=sentences,
                    ),
                )
            )
    if include_experimental:
        designs.extend(
            [
                (
                    "sac_w150_o30_s0_tail",
                    StructureAwareChunker(
                        max_words=150,
                        overlap_words=30,
                        overlap_sentences=0,
                        backfill_short_tail=True,
                    ),
                ),
                (
                    "sac_w150_o30_s0_head",
                    StructureAwareChunker(
                        max_words=150,
                        overlap_words=30,
                        overlap_sentences=0,
                        embedding_context="section",
                    ),
                ),
                (
                    "sac_w150_o30_s0_ctx",
                    StructureAwareChunker(
                        max_words=150,
                        overlap_words=30,
                        overlap_sentences=0,
                        embedding_context="section_document",
                    ),
                ),
                (
                    "sac_w150_o30_s0_adaptive",
                    StructureAwareChunker(
                        max_words=150,
                        overlap_words=30,
                        overlap_sentences=0,
                        embedding_context="section_reasoning_document",
                    ),
                ),
            ]
        )
    return designs


def _alternate_statute_qrels(
    questions: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    sections_by_document: dict[str, list[dict[str, Any]]],
    *,
    existing: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    chunks_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[str(chunk["document_id"])].append(chunk)
    additions: list[dict[str, Any]] = []
    for question in questions:
        if question["target_section_label"] != "pertimbangan_hukum":
            continue
        key_match = re.search(
            r"\bpasal\s+\d+(?:\s+ayat\s+\d+)?(?:\s+huruf\s+[a-z])?",
            question["reference_answer"],
            flags=re.IGNORECASE,
        )
        if not key_match:
            continue
        document_id = question["document_id"]
        section = next(
            row
            for row in sections_by_document[document_id]
            if row["section_label"] == "pertimbangan_hukum"
        )
        section_text = str(section["section_text"])
        section_start = int(section["start_position"])
        for occurrence in re.finditer(
            re.escape(key_match.group(0)), section_text, flags=re.IGNORECASE
        ):
            evidence_start = section_start + occurrence.start()
            evidence_end = section_start + occurrence.end()
            for chunk in chunks_by_document[document_id]:
                chunk_start = int(chunk["start_position"])
                chunk_end = int(chunk["end_position"])
                pair = (question["query_id"], str(chunk["chunk_id"]))
                if pair in existing or not (
                    chunk_start <= evidence_start and evidence_end <= chunk_end
                ):
                    continue
                existing.add(pair)
                additions.append(
                    {
                        "query_id": question["query_id"],
                        "document_id": document_id,
                        "target_section_label": question["target_section_label"],
                        "evidence_id": (
                            f"{question['query_id']}:"
                            f"{question['evidence_start_position']}:"
                            f"{question['evidence_end_position']}"
                        ),
                        "strategy": str(chunk["strategy"]),
                        "chunk_id": str(chunk["chunk_id"]),
                        "chunk_start_position": chunk_start,
                        "chunk_end_position": chunk_end,
                        "evidence_start_position": evidence_start,
                        "evidence_end_position": evidence_end,
                        "evidence_coverage": 1.0,
                        "auto_grade": 2,
                        "relevance_grade": 2,
                        "reviewer_notes": "Equivalent statute occurrence in the annotated reasoning section.",
                    }
                )
    return additions


if __name__ == "__main__":
    main()
