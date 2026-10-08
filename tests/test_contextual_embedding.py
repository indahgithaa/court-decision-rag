import numpy as np
import pytest
import torch

from src.embedding.embedder import (
    ContextualSentenceTransformerEmbedder,
    _assign_spans_to_windows,
    _character_span_to_token_span,
    _context_windows,
)


class _FakeTokenizer:
    is_fast = True

    def __call__(self, text: str | list[str], **kwargs: object) -> dict[str, object]:
        if isinstance(text, list):
            encoded = [
                self._encode(value, bool(kwargs.get("add_special_tokens")))
                for value in text
            ]
            width = max(len(ids) for ids, _ in encoded)
            padded_ids = [ids + ([0] * (width - len(ids))) for ids, _ in encoded]
            padded_offsets = [
                offsets + ([(0, 0)] * (width - len(offsets)))
                for ids, offsets in encoded
            ]
            masks = [
                ([1] * len(ids)) + ([0] * (width - len(ids)))
                for ids, _ in encoded
            ]
            return {
                "input_ids": torch.tensor(padded_ids),
                "attention_mask": torch.tensor(masks),
                "offset_mapping": torch.tensor(padded_offsets),
            }
        input_ids, offsets = self._encode(
            text, bool(kwargs.get("add_special_tokens"))
        )
        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
                "offset_mapping": torch.tensor([offsets]),
            }
        return {"input_ids": input_ids, "offset_mapping": offsets}

    @staticmethod
    def _encode(
        text: str, add_special_tokens: bool
    ) -> tuple[list[int], list[tuple[int, int]]]:
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token_id, word in enumerate(text.split(), start=1):
            start = text.index(word, cursor)
            end = start + len(word)
            input_ids.append(token_id)
            offsets.append((start, end))
            cursor = end
        if add_special_tokens:
            input_ids = [101, *input_ids, 102]
            offsets = [(0, 0), *offsets, (0, 0)]
        return input_ids, offsets

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return 2

class _FakeTransformer:
    tokenizer = _FakeTokenizer()

    def __call__(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ids = features["input_ids"].to(torch.float32)
        # Two non-zero dimensions are enough to exercise pooling and normalization.
        return {"token_embeddings": torch.stack((ids, ids + 1.0), dim=-1)}


class _FakeModel:
    max_seq_length = 8
    device = "cpu"

    def __getitem__(self, index: int) -> _FakeTransformer:
        assert index == 0
        return _FakeTransformer()


def test_character_span_maps_to_overlapping_tokens() -> None:
    offsets = [(0, 4), (5, 9), (10, 15)]

    assert _character_span_to_token_span(offsets, 5, 15) == (1, 3)


def test_context_windows_cover_chunks_that_cross_regular_boundaries() -> None:
    spans = [(0, 3), (7, 12), (18, 20)]

    windows = _context_windows(
        20,
        capacity=10,
        overlap=2,
        required_spans=spans,
    )

    assert all(
        any(
            window_start <= start and end <= window_end
            for window_start, window_end in windows
        )
        for start, end in spans
    )


def test_span_assignment_prefers_balanced_surrounding_context() -> None:
    windows = [(0, 10), (4, 14)]

    assert _assign_spans_to_windows([(6, 8)], windows) == [0]
    assert _assign_spans_to_windows([(9, 11)], windows) == [1]


def test_context_windows_reject_chunk_larger_than_capacity() -> None:
    with pytest.raises(ValueError, match="longer"):
        _context_windows(20, capacity=5, overlap=1, required_spans=[(2, 8)])


def test_contextual_embedder_preserves_input_order_and_normalizes() -> None:
    embedder = ContextualSentenceTransformerEmbedder(
        "fake",
        context_window_tokens=8,
        context_window_overlap_tokens=2,
        document_prefix="",
    )
    embedder._model = _FakeModel()
    document = "aa bb cc dd ee ff gg hh ii jj"
    chunks = [
        {
            "chunk_id": "late",
            "document_id": "d1",
            "start_position": 24,
            "end_position": 29,
            "text": "ii jj",
        },
        {
            "chunk_id": "early",
            "document_id": "d1",
            "start_position": 0,
            "end_position": 5,
            "text": "aa bb",
        },
    ]

    vectors = embedder.embed_chunks(chunks, {"d1": document})

    assert vectors.shape == (2, 2)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])
    assert not np.allclose(vectors[0], vectors[1])


def test_contextual_embedder_rejects_offset_text_mismatch() -> None:
    embedder = ContextualSentenceTransformerEmbedder(
        "fake", context_window_tokens=8, document_prefix=""
    )
    embedder._model = _FakeModel()

    with pytest.raises(ValueError, match="does not match"):
        embedder.embed_chunks(
            [
                {
                    "chunk_id": "bad",
                    "document_id": "d1",
                    "start_position": 0,
                    "end_position": 2,
                    "text": "wrong",
                }
            ],
            {"d1": "aa bb"},
        )
