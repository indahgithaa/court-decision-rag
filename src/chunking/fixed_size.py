"""Fixed-size word-window baseline."""

from __future__ import annotations

from typing import Mapping, Sequence

from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Split the whole document without considering structural boundaries."""

    strategy = "fixed_size"

    def chunk(
        self,
        document_id: str,
        text: str,
        *,
        sections: Sequence[Mapping[str, object]] | None = None,
    ) -> list[Chunk]:
        del sections  # The baseline intentionally ignores document structure.
        chunks: list[Chunk] = []
        for index, (chunk_text, start, end) in enumerate(self._windows(text)):
            chunks.append(
                Chunk(
                    chunk_id=self._make_id(document_id, index, start, end),
                    document_id=document_id,
                    chunk_index=index,
                    strategy=self.strategy,
                    text=chunk_text,
                    start_position=start,
                    end_position=end,
                )
            )
        return chunks

