"""Embedding contracts and a multilingual E5 adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float32]


class Embedder(Protocol):
    """Asymmetric embedding interface used by indexing and retrieval."""

    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed corpus passages in input order."""

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        """Embed search queries in input order."""


class SentenceTransformerEmbedder:
    """Lazy SentenceTransformers adapter for multilingual E5 models.

    E5 is trained asymmetrically and requires ``query:`` and ``passage:``
    prefixes even for non-English retrieval. Keeping those transformations in
    the adapter prevents index/query mismatches elsewhere in the pipeline.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        batch_size: int = 16,
        device: str | None = None,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        normalize_embeddings: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.normalize_embeddings = normalize_embeddings
        self._model: object | None = None

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts, prefix=self.document_prefix)

    def embed_queries(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts, prefix=self.query_prefix)

    def _encode(self, texts: Sequence[str], *, prefix: str) -> FloatMatrix:
        if not texts:
            raise ValueError("texts must contain at least one item")
        model = self._load_model()
        values = model.encode(  # type: ignore[attr-defined]
            [prefix + str(text) for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(f"Embedding model returned unexpected shape: {matrix.shape}")
        return matrix

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Dense retrieval requires the optional dependencies. Install "
                "the project with `python -m pip install -e .[retrieval]` in a "
                "Python version supported by PyTorch (Python 3.12 recommended)."
            ) from error
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

