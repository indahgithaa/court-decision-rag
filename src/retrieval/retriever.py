"""Dense retriever that keeps query encoding separate from passage encoding."""

from __future__ import annotations

from collections.abc import Sequence

from src.embedding import Embedder

from .vector_store import DenseVectorStore, SearchResult


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: DenseVectorStore) -> None:
        if embedder.model_name != store.model_name:
            raise ValueError(
                f"Embedder model {embedder.model_name!r} does not match index "
                f"model {store.model_name!r}"
            )
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, *, k: int = 10) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be blank")
        embedding = self.embedder.embed_queries([query])[0]
        return self.store.search(embedding, k=k)

    def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        k: int = 10,
    ) -> list[list[SearchResult]]:
        if not queries or any(not query.strip() for query in queries):
            raise ValueError("queries must contain non-blank strings")
        embeddings = self.embedder.embed_queries(queries)
        return [self.store.search(embedding, k=k) for embedding in embeddings]

