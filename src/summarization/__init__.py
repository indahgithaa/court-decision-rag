"""Document-level summarization used by summary-augmented chunking."""

from .document_fingerprint import (
    QwenDocumentFingerprintGenerator,
    build_hierarchical_extractive_contexts,
    build_representative_document,
    build_structured_extractive_fingerprint,
    truncate_at_word_boundary,
)

__all__ = [
    "QwenDocumentFingerprintGenerator",
    "build_hierarchical_extractive_contexts",
    "build_representative_document",
    "build_structured_extractive_fingerprint",
    "truncate_at_word_boundary",
]
