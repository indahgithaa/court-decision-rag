"""Deterministic ONNX fallback for multilingual E5 embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


class _TokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> list[int]:
        del truncation
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return str(
            self._tokenizer.decode(
                list(ids), skip_special_tokens=skip_special_tokens
            )
        )


def load_e5_tokenizer(tokenizer_path: Path) -> _TokenizerAdapter:
    """Load the fast E5 tokenizer without importing Transformers or PyTorch."""
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("E5 token budgeting requires tokenizers") from error
    return _TokenizerAdapter(Tokenizer.from_file(str(tokenizer_path)))


class OnnxE5Encoder:
    """Small SentenceTransformer-compatible wrapper around the official E5 ONNX export."""

    def __init__(
        self,
        *,
        model_path: Path,
        tokenizer_path: Path,
        max_seq_length: int = 512,
    ) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError("ONNX fallback requires onnxruntime and tokenizers") from error

        raw_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        pad_id = raw_tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("E5 tokenizer has no <pad> token")
        raw_tokenizer.enable_padding(
            direction="right", pad_id=pad_id, pad_type_id=0, pad_token="<pad>"
        )
        self._raw_tokenizer = raw_tokenizer
        self.tokenizer = _TokenizerAdapter(raw_tokenizer)
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.max_seq_length = max_seq_length
        self.runtime_version = ort.__version__

    def eval(self) -> "OnnxE5Encoder":
        return self

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        del show_progress_bar
        if not convert_to_numpy:
            raise ValueError("ONNX E5 wrapper only returns NumPy arrays")
        self._raw_tokenizer.enable_truncation(max_length=int(self.max_seq_length))
        results: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            encodings = self._raw_tokenizer.encode_batch(list(texts[start : start + batch_size]))
            input_ids = np.asarray([row.ids for row in encodings], dtype=np.int64)
            attention_mask = np.asarray(
                [row.attention_mask for row in encodings], dtype=np.int64
            )
            token_type_ids = np.asarray(
                [row.type_ids for row in encodings], dtype=np.int64
            )
            hidden = self._session.run(
                ["last_hidden_state"],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )[0]
            mask = attention_mask[..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            if normalize_embeddings:
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                pooled = pooled / np.maximum(norms, np.finfo(np.float32).eps)
            results.append(pooled.astype(np.float32))
        if not results:
            return np.empty((0, 384), dtype=np.float32)
        return np.concatenate(results, axis=0)
