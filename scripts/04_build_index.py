"""Build one exact dense index from a configured chunk strategy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from src.embedding import (
    ContextualSentenceTransformerEmbedder,
    SentenceTransformerEmbedder,
)
from src.retrieval import DenseVectorStore
from src.utils.config import load_config
from src.utils.io import read_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a dense retrieval index.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, help="Override chunking.output_path")
    parser.add_argument("--output", type=Path, help="Override retrieval.index_path")
    parser.add_argument("--model", help="Override embedding.model_name")
    parser.add_argument(
        "--embedding-method",
        choices=("independent", "contextual"),
        help="Override embedding.method",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        help="Source pages JSONL required by contextual embedding",
    )
    parser.add_argument(
        "--device", help="SentenceTransformers device, e.g. cpu or cuda"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    chunking = config.get("chunking", {})
    embedding = config.get("embedding", {})
    retrieval = config.get("retrieval", {})
    chunks_path = args.chunks or _project_path(
        args.config, str(chunking["output_path"])
    )
    output = args.output or _project_path(args.config, str(retrieval["index_path"]))
    chunks = list(read_jsonl(chunks_path))
    model_name = args.model or str(
        embedding.get("model_name", "intfloat/multilingual-e5-small")
    )
    method = args.embedding_method or str(embedding.get("method", "independent"))
    common_options = {
        "batch_size": int(embedding.get("batch_size", 16)),
        "device": args.device or embedding.get("device"),
        "query_prefix": str(embedding.get("query_prefix", "query: ")),
        "document_prefix": str(embedding.get("document_prefix", "passage: ")),
        "normalize_embeddings": bool(embedding.get("normalize_embeddings", True)),
        "trust_remote_code": bool(embedding.get("trust_remote_code", False)),
        "max_sequence_length": _optional_int(embedding.get("max_sequence_length")),
    }
    documents_path: Path | None = None
    if method == "independent":
        embedder = SentenceTransformerEmbedder(model_name, **common_options)
        matrix = embedder.embed_documents(
            [str(chunk.get("embedding_text") or chunk["text"]) for chunk in chunks]
        )
        embedding_text_field = "embedding_text_or_text"
    elif method == "contextual":
        configured_documents = embedding.get("documents_path") or config.get(
            "paths", {}
        ).get("processed_pages")
        if args.documents is not None:
            documents_path = args.documents
        elif configured_documents:
            documents_path = _project_path(args.config, str(configured_documents))
        else:
            raise ValueError(
                "Contextual embedding requires --documents or embedding.documents_path"
            )
        documents = _read_documents(documents_path)
        contextual_embedder = ContextualSentenceTransformerEmbedder(
            model_name,
            context_window_tokens=_optional_int(
                embedding.get("context_window_tokens")
            ),
            context_window_overlap_tokens=int(
                embedding.get("context_window_overlap_tokens", 512)
            ),
            **common_options,
        )
        matrix = contextual_embedder.embed_chunks(chunks, documents)
        embedding_text_field = "source_text_span_pooling"
    else:
        raise ValueError("embedding.method must be independent or contextual")
    store = DenseVectorStore(chunks, matrix, model_name=model_name)
    store.save(
        output,
        provenance={
            "config_path": args.config.as_posix(),
            "chunks_path": chunks_path.as_posix(),
            "chunks_sha256": _sha256(chunks_path),
            "documents_path": documents_path.as_posix() if documents_path else None,
            "documents_sha256": (
                _sha256(documents_path) if documents_path is not None else None
            ),
            "embedding_method": method,
            "embedding_text_field": embedding_text_field,
            "query_prefix": common_options["query_prefix"],
            "document_prefix": common_options["document_prefix"],
            "normalize_embeddings": common_options["normalize_embeddings"],
            "max_sequence_length": common_options["max_sequence_length"],
            "context_window_tokens": embedding.get("context_window_tokens"),
            "context_window_overlap_tokens": embedding.get(
                "context_window_overlap_tokens"
            ),
        },
    )
    print(
        f"Built {len(chunks)}-chunk index ({store.dimension} dimensions) at {output}"
    )


def _project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parent.parent / path


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _read_documents(path: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    for row in read_jsonl(path):
        document_id = str(row.get("document_id", ""))
        text = str(row.get("clean_text") or row.get("raw_text") or "")
        if not document_id or not text:
            raise ValueError(f"Invalid source document record in {path}")
        if document_id in documents:
            raise ValueError(f"Duplicate document_id {document_id!r} in {path}")
        documents[document_id] = text
    if not documents:
        raise ValueError(f"No source documents found in {path}")
    return documents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

