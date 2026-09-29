"""Retrieve ranked chunks for an approved question set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from src.embedding import SentenceTransformerEmbedder
from src.retrieval import DenseRetriever, DenseVectorStore
from src.utils.config import load_config
from src.utils.io import write_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run dense retrieval.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/evaluation/exploration_20_questions.csv"),
    )
    parser.add_argument("--index", type=Path, help="Override retrieval.index_path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, help="Override retrieval.top_k")
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    embedding = config.get("embedding", {})
    retrieval = config.get("retrieval", {})
    index_path = args.index or _project_path(args.config, str(retrieval["index_path"]))
    with args.questions.open("r", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    _require_approved(questions)

    store = DenseVectorStore.load(index_path)
    embedder = SentenceTransformerEmbedder(
        store.model_name,
        batch_size=int(embedding.get("batch_size", 16)),
        device=args.device or embedding.get("device"),
        query_prefix=str(embedding.get("query_prefix", "query: ")),
        document_prefix=str(embedding.get("document_prefix", "passage: ")),
        normalize_embeddings=bool(embedding.get("normalize_embeddings", True)),
    )
    retriever = DenseRetriever(embedder, store)
    k = args.k or int(retrieval.get("top_k", 10))
    rankings = retriever.retrieve_many([row["question"] for row in questions], k=k)
    records = []
    for question, results in zip(questions, rankings, strict=True):
        records.append(
            {
                "query_id": question["query_id"],
                "document_id": question["document_id"],
                "target_section_label": question["target_section_label"],
                "ranking": [
                    {
                        "rank": rank,
                        "chunk_id": result.chunk_id,
                        "score": result.score,
                        "retrieved_document_id": str(result.chunk["document_id"]),
                    }
                    for rank, result in enumerate(results, start=1)
                ],
            }
        )
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} retrieval rankings (top-{k}) to {args.output}")


def _require_approved(questions: Sequence[dict[str, str]]) -> None:
    if not questions:
        raise ValueError("Question file is empty")
    unapproved = [
        row.get("query_id", "<unknown>")
        for row in questions
        if row.get("review_status", "").strip().lower() != "approved"
    ]
    if unapproved:
        preview = ", ".join(unapproved[:5])
        suffix = "..." if len(unapproved) > 5 else ""
        raise ValueError(
            f"Refusing retrieval: {len(unapproved)} question(s) are not approved "
            f"({preview}{suffix})"
        )


def _project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parent.parent / path


if __name__ == "__main__":
    main()

