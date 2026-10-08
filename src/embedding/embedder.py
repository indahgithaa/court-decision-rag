"""Embedding contracts and a multilingual E5 adapter."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

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
        trust_remote_code: bool = False,
        max_sequence_length: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code
        self.max_sequence_length = max_sequence_length
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
            raise ValueError(
                f"Embedding model returned unexpected shape: {matrix.shape}"
            )
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
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        if self.max_sequence_length is not None:
            if self.max_sequence_length <= 2:
                raise ValueError("max_sequence_length must be greater than two")
            self._model.max_seq_length = self.max_sequence_length
        return self._model


class ContextualSentenceTransformerEmbedder(SentenceTransformerEmbedder):
    """Create contextual chunk vectors by pooling document token embeddings.

    Chunk boundaries and source text remain unchanged. The full document is
    tokenized first, contextual windows are encoded, and each chunk vector is
    pooled only from tokens overlapping its source character offsets. This is
    late-chunking-style contextualization, with overlapping macro windows for
    documents longer than the encoder context window.
    """

    def __init__(
        self,
        model_name: str,
        *,
        context_window_tokens: int | None = None,
        context_window_overlap_tokens: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        if context_window_tokens is not None and context_window_tokens <= 2:
            raise ValueError("context_window_tokens must be greater than two")
        if context_window_overlap_tokens < 0:
            raise ValueError("context_window_overlap_tokens must be non-negative")
        self.context_window_tokens = context_window_tokens
        self.context_window_overlap_tokens = context_window_overlap_tokens

    def embed_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        documents: Mapping[str, str],
    ) -> FloatMatrix:
        """Embed chunks in input order using their complete document context."""
        if not chunks:
            raise ValueError("chunks must contain at least one item")
        grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            document_id = str(chunk.get("document_id", ""))
            if not document_id:
                raise ValueError("every chunk must have a document_id")
            if document_id not in documents:
                raise ValueError(f"Missing source document {document_id!r}")
            grouped[document_id].append((index, chunk))

        vectors: list[FloatMatrix | None] = [None] * len(chunks)
        for document_id, indexed_chunks in grouped.items():
            document_vectors = self.embed_document_chunks(
                str(documents[document_id]),
                [chunk for _, chunk in indexed_chunks],
            )
            for (global_index, _), vector in zip(indexed_chunks, document_vectors):
                vectors[global_index] = vector

        if any(vector is None for vector in vectors):
            raise RuntimeError(
                "Contextual embedding did not produce every chunk vector"
            )
        return np.asarray(vectors, dtype=np.float32)

    def embed_document_chunks(
        self,
        document_text: str,
        chunks: Sequence[Mapping[str, Any]],
    ) -> FloatMatrix:
        """Embed one document's chunks, preserving the supplied chunk order."""
        return self.embed_document_chunk_groups(
            document_text, {"chunks": chunks}
        )["chunks"]

    def embed_document_chunk_groups(
        self,
        document_text: str,
        chunk_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> dict[str, FloatMatrix]:
        """Embed several boundary strategies with independent window assignment.

        Each strategy computes its own contextual windows and chunk-to-window
        assignments. Identical assigned windows are encoded once across groups,
        which saves compute without allowing one strategy's boundaries to alter
        another strategy's contextual representation.
        """
        matrices = self._embed_document_chunk_groups(document_text, chunk_groups)
        if self.normalize_embeddings:
            return {name: _normalize_rows(matrix) for name, matrix in matrices.items()}
        return matrices

    def _embed_document_chunks(
        self,
        document_text: str,
        chunks: Sequence[Mapping[str, Any]],
    ) -> FloatMatrix:
        """Backward-compatible raw single-group embedding helper."""
        return self._embed_document_chunk_groups(
            document_text, {"chunks": chunks}
        )["chunks"]

    def _embed_document_chunk_groups(
        self,
        document_text: str,
        chunk_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> dict[str, FloatMatrix]:
        if not document_text:
            raise ValueError("source document text must not be empty")
        if not chunk_groups or any(not chunks for chunks in chunk_groups.values()):
            raise ValueError("chunk_groups must contain non-empty chunk sequences")
        model = self._load_model()
        transformer = model[0]  # type: ignore[index]
        tokenizer = transformer.tokenizer
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(
                "Contextual chunk embeddings require a fast tokenizer with offset "
                "mapping"
            )

        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        try:
            if isinstance(tokenizer_limit, int):
                tokenizer.model_max_length = max(tokenizer_limit, len(document_text))
            tokenized = tokenizer(
                document_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
                truncation=False,
            )
        finally:
            if isinstance(tokenizer_limit, int):
                tokenizer.model_max_length = tokenizer_limit
        token_ids = list(tokenized["input_ids"])
        offsets = [tuple(value) for value in tokenized["offset_mapping"]]
        if not token_ids:
            raise ValueError("source document produced no tokens")

        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        prefix_tokens = (
            len(
                tokenizer(
                    self.document_prefix,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    truncation=False,
                )["input_ids"]
            )
            if self.document_prefix
            else 0
        )
        configured_limit = self.context_window_tokens or int(
            model.max_seq_length  # type: ignore[attr-defined]
        )
        available_tokens = configured_limit - special_tokens - prefix_tokens
        # Retokenizing a character-sliced window can create a few extra
        # boundary subwords compared with slicing the full-document token IDs.
        # Reserve a small margin instead of silently truncating source tokens.
        boundary_margin = min(8, max(0, available_tokens // 16))
        capacity = available_tokens - boundary_margin
        if capacity <= 0:
            raise ValueError("context window leaves no room for document tokens")

        group_states: dict[str, dict[str, Any]] = {}
        all_active_windows: set[tuple[int, int]] = set()
        for group_name, chunks in chunk_groups.items():
            character_spans: list[tuple[int, int]] = []
            spans: list[tuple[int, int]] = []
            for chunk in chunks:
                start = int(chunk.get("start_position", -1))
                end = int(chunk.get("end_position", -1))
                text = str(chunk.get("text", ""))
                if start < 0 or end <= start or end > len(document_text):
                    raise ValueError(f"Invalid chunk offsets {start}:{end}")
                if document_text[start:end] != text:
                    raise ValueError(
                        "Chunk text does not match its source offsets; contextual "
                        f"pooling is unsafe at {start}:{end}"
                    )
                character_spans.append((start, end))
                spans.append(_character_span_to_token_span(offsets, start, end))
            windows = _context_windows(
                len(token_ids),
                capacity=capacity,
                overlap=self.context_window_overlap_tokens,
                required_spans=spans,
            )
            assignments = _assign_spans_to_windows(spans, windows)
            chunks_by_window: dict[tuple[int, int], list[int]] = defaultdict(list)
            for chunk_index, window_index in enumerate(assignments):
                chunks_by_window[windows[window_index]].append(chunk_index)
            all_active_windows.update(chunks_by_window)
            group_states[group_name] = {
                "character_spans": character_spans,
                "chunks_by_window": chunks_by_window,
                "output_vectors": [None] * len(chunks),
            }

        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Contextual embeddings require PyTorch") from error

        model_device = getattr(model, "device", self.device or "cpu")
        active_windows = sorted(all_active_windows)
        with torch.no_grad():
            for batch_start in range(0, len(active_windows), self.batch_size):
                window_batch = active_windows[
                    batch_start : batch_start + self.batch_size
                ]
                character_windows = [
                    (offsets[window_start][0], offsets[window_end - 1][1])
                    for window_start, window_end in window_batch
                ]
                prepared = tokenizer(
                    [
                        self.document_prefix
                        + document_text[character_start:character_end]
                        for character_start, character_end in character_windows
                    ],
                    add_special_tokens=True,
                    padding=True,
                    truncation=False,
                    return_attention_mask=True,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                )
                if int(prepared["input_ids"].shape[1]) > configured_limit:
                    raise RuntimeError(
                        "Retokenized contextual window exceeds the configured "
                        "sequence length; increase the boundary margin"
                    )
                batch_offsets = prepared.pop("offset_mapping").tolist()
                features = {
                    key: value.to(model_device)
                    for key, value in prepared.items()
                }
                encoded = transformer(features)
                for batch_index, window in enumerate(window_batch):
                    prepared_offsets = [
                        tuple(value) for value in batch_offsets[batch_index]
                    ]
                    window_character_start, _ = character_windows[batch_index]
                    token_embeddings = encoded["token_embeddings"][batch_index]
                    for state in group_states.values():
                        chunks_by_window = state["chunks_by_window"]
                        for chunk_index in chunks_by_window.get(window, ()):
                            character_start, character_end = state[
                                "character_spans"
                            ][chunk_index]
                            local_character_start = (
                                len(self.document_prefix)
                                + character_start
                                - window_character_start
                            )
                            local_character_end = (
                                len(self.document_prefix)
                                + character_end
                                - window_character_start
                            )
                            local_start, local_end = _character_span_to_token_span(
                                prepared_offsets,
                                local_character_start,
                                local_character_end,
                            )
                            pooled = token_embeddings[local_start:local_end].mean(
                                dim=0
                            )
                            pooled = _apply_post_pooling_modules(model, pooled)
                            state["output_vectors"][chunk_index] = np.asarray(
                                pooled.detach().cpu().numpy(), dtype=np.float32
                            )

        matrices: dict[str, FloatMatrix] = {}
        for group_name, state in group_states.items():
            output_vectors = state["output_vectors"]
            if any(vector is None for vector in output_vectors):
                raise RuntimeError(
                    f"A contextual chunk span was not pooled for {group_name!r}"
                )
            matrices[group_name] = np.asarray(output_vectors, dtype=np.float32)
        return matrices


def _character_span_to_token_span(
    offsets: Sequence[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    if not indices:
        raise ValueError(f"Character span {start}:{end} contains no tokens")
    return indices[0], indices[-1] + 1


def _context_windows(
    token_count: int,
    *,
    capacity: int,
    overlap: int,
    required_spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    if token_count <= 0 or capacity <= 0:
        raise ValueError("token_count and capacity must be positive")
    if overlap < 0 or overlap >= capacity:
        raise ValueError("overlap must satisfy 0 <= overlap < capacity")
    if any(end - start > capacity for start, end in required_spans):
        raise ValueError("A chunk is longer than the contextual token capacity")
    if token_count <= capacity:
        return [(0, token_count)]

    step = capacity - overlap
    starts = list(range(0, token_count - capacity + 1, step))
    final_start = token_count - capacity
    if starts[-1] != final_start:
        starts.append(final_start)
    windows = [(start, start + capacity) for start in starts]

    for span_start, span_end in required_spans:
        if any(start <= span_start and span_end <= end for start, end in windows):
            continue
        lower_bound = max(0, span_end - capacity)
        upper_bound = min(span_start, token_count - capacity)
        centered_start = (span_start + span_end - capacity) // 2
        ideal_start = max(lower_bound, min(centered_start, upper_bound))
        windows.append((ideal_start, ideal_start + capacity))
    return sorted(set(windows))


def _assign_spans_to_windows(
    spans: Sequence[tuple[int, int]], windows: Sequence[tuple[int, int]]
) -> list[int]:
    assignments: list[int] = []
    for span_start, span_end in spans:
        candidates = [
            (index, start, end)
            for index, (start, end) in enumerate(windows)
            if start <= span_start and span_end <= end
        ]
        if not candidates:
            raise ValueError(
                f"No contextual window covers token span {span_start}:{span_end}"
            )
        best = max(
            candidates,
            key=lambda value: (
                min(span_start - value[1], value[2] - span_end),
                (span_start - value[1]) + (value[2] - span_end),
                -value[0],
            ),
        )
        assignments.append(best[0])
    return assignments


def _normalize_rows(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embeddings contain a zero vector")
    return np.asarray(matrix / norms, dtype=np.float32)


def _apply_post_pooling_modules(model: Any, pooled: Any) -> Any:
    """Apply projection/normalization modules that follow ST's pooling module."""
    modules = list(getattr(model, "_modules", {}).values())
    pooling_index = next(
        (
            index
            for index, module in enumerate(modules)
            if module.__class__.__name__ == "Pooling"
        ),
        None,
    )
    if pooling_index is None:
        return pooled
    features = {"sentence_embedding": pooled.unsqueeze(0)}
    for module in modules[pooling_index + 1 :]:
        features = module(features)
    return features["sentence_embedding"][0]

