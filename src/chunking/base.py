"""Shared chunk representation and interface."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with provenance back to the cleaned document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    strategy: str
    text: str
    start_position: int
    end_position: int
    section_label: str | None = None
    section_heading: str | None = None
    embedding_text: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        if self.embedding_text is None:
            payload.pop("embedding_text")
        return payload


class BaseChunker(ABC):
    """Consistent interface implemented by every experimental chunker."""

    strategy: str

    def __init__(self, *, max_words: int = 300, overlap_words: int = 50) -> None:
        if max_words <= 0:
            raise ValueError("max_words must be greater than zero")
        if overlap_words < 0 or overlap_words >= max_words:
            raise ValueError("overlap_words must satisfy 0 <= overlap_words < max_words")
        self.max_words = max_words
        self.overlap_words = overlap_words

    @abstractmethod
    def chunk(
        self,
        document_id: str,
        text: str,
        *,
        sections: Sequence[Mapping[str, object]] | None = None,
    ) -> list[Chunk]:
        """Create chunks from one cleaned document."""

    def _windows(self, text: str, *, offset: int = 0) -> list[tuple[str, int, int]]:
        """Return word-window text and global character offsets."""
        words = list(re.finditer(r"\S+", text))
        if not words:
            return []
        step = self.max_words - self.overlap_words
        windows: list[tuple[str, int, int]] = []
        for word_start in range(0, len(words), step):
            word_end = min(word_start + self.max_words, len(words))
            char_start = words[word_start].start()
            char_end = words[word_end - 1].end()
            windows.append((text[char_start:char_end], offset + char_start, offset + char_end))
            if word_end == len(words):
                break
        return windows

    def _make_id(self, document_id: str, index: int, start: int, end: int) -> str:
        payload = f"{self.strategy}|{document_id}|{index}|{start}|{end}"
        digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
        return f"{document_id}-{self.strategy}-{digest}"

