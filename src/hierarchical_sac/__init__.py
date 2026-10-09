"""Controlled B1/B2/M1 retrieval experiment for hierarchical SAC."""

from .chunking import RecursiveCharacterChunker, assign_chunk_context
from .metrics import evaluate_method, paired_cluster_comparisons
from .representations import build_embedding_representations

__all__ = [
    "RecursiveCharacterChunker",
    "assign_chunk_context",
    "build_embedding_representations",
    "evaluate_method",
    "paired_cluster_comparisons",
]
