"""Small exact cosine-similarity store for reproducible pilot experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.utils.io import read_jsonl, write_jsonl


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and its cosine score."""

    chunk: Mapping[str, Any]
    score: float

    @property
    def chunk_id(self) -> str:
        return str(self.chunk["chunk_id"])

    def to_dict(self) -> dict[str, Any]:
        return {**dict(self.chunk), "score": self.score}


class DenseVectorStore:
    """Exact dense search; sufficient and deterministic for the pilot corpus."""

    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        embeddings: NDArray[np.floating[Any]],
        *,
        model_name: str,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must contain at least one record")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] == 0:
            raise ValueError(
                "embeddings must be a non-empty 2D matrix with one row per chunk"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings contain non-finite values")
        chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("every chunk must have a chunk_id")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_id values must be unique")
        self.chunks = [dict(chunk) for chunk in chunks]
        self.embeddings = _normalize_rows(matrix)
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def search(
        self,
        query_embedding: NDArray[np.floating[Any]],
        *,
        k: int = 10,
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise ValueError(
                f"query embedding must have shape ({self.dimension},), got {query.shape}"
            )
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("query embedding must be finite and non-zero")
        scores = self.embeddings @ (query / norm)
        limit = min(k, len(self.chunks))
        indices = np.argsort(-scores, kind="stable")[:limit]
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(scores[int(index)]))
            for index in indices
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings, allow_pickle=False)
        write_jsonl(directory / "chunks.jsonl", self.chunks)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_name": self.model_name,
            "chunk_count": len(self.chunks),
            "dimension": self.dimension,
            "normalized": True,
            "strategies": sorted(
                {str(chunk.get("strategy", "")) for chunk in self.chunks}
            ),
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, directory: Path) -> "DenseVectorStore":
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported index schema: {manifest.get('schema_version')}")
        chunks = list(read_jsonl(directory / "chunks.jsonl"))
        embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
        store = cls(chunks, embeddings, model_name=str(manifest["model_name"]))
        if len(chunks) != int(manifest["chunk_count"]):
            raise ValueError("Index chunk count does not match manifest")
        if store.dimension != int(manifest["dimension"]):
            raise ValueError("Index dimension does not match manifest")
        return store


def _normalize_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embeddings contain a zero vector")
    return np.asarray(matrix / norms, dtype=np.float32)

