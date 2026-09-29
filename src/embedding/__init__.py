"""Embedding interfaces reserved for a later milestone."""

from .embedder import Embedder

__all__ = ["Embedder"]

from .embedder import Embedder, FloatMatrix, SentenceTransformerEmbedder

__all__ = ["Embedder", "FloatMatrix", "SentenceTransformerEmbedder"]
