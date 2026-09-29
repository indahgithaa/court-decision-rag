"""Build one exact dense index from a configured chunk strategy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.embedding import SentenceTransformerEmbedder
from src.retrieval import DenseVectorStore
from src.utils.config import load_config
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a dense retrieval index.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, help="Override chunking.output_path")
    parser.add_argument("--output", type=Path, help="Override retrieval.index_path")
    parser.add_argument("--model", help="Override embedding.model_name")
    parser.add_argument("--device", help="SentenceTransformers device, e.g. cpu or cuda")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    chunking = config.get("chunking", {})
    embedding = config.get("embedding", {})
    retrieval = config.get("retrieval", {})
    chunks_path = args.chunks or _project_path(args.config, str(chunking["output_path"]))
    output = args.output or _project_path(args.config, str(retrieval["index_path"]))
    chunks = list(read_jsonl(chunks_path))
    model_name = args.model or str(
        embedding.get("model_name", "intfloat/multilingual-e5-small")
    )
    embedder = SentenceTransformerEmbedder(
        model_name,
        batch_size=int(embedding.get("batch_size", 16)),
        device=args.device or embedding.get("device"),
        query_prefix=str(embedding.get("query_prefix", "query: ")),
        document_prefix=str(embedding.get("document_prefix", "passage: ")),
        normalize_embeddings=bool(embedding.get("normalize_embeddings", True)),
    )
    matrix = embedder.embed_documents([str(chunk["text"]) for chunk in chunks])
    store = DenseVectorStore(chunks, matrix, model_name=model_name)
    store.save(output)
    print(
        f"Built {len(chunks)}-chunk index ({store.dimension} dimensions) at {output}"
    )


def _project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parent.parent / path


if __name__ == "__main__":
    main()

