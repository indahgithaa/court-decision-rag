"""Retrieval interfaces reserved for a later milestone."""

from .retriever import DenseRetriever
from .vector_store import DenseVectorStore, SearchResult

__all__ = ["DenseRetriever", "DenseVectorStore", "SearchResult"]
