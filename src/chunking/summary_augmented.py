"""Document-summary augmentation for arbitrary source-preserving chunkers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .base import BaseChunker, Chunk


class SummaryAugmentedChunker(BaseChunker):
    """Prepend a document fingerprint to chunks only for embedding.

    This implements the augmentation stage of Reuter et al. (2025): one concise
    summary is shared by every chunk from the same source document.  The wrapped
    chunker's source text and character offsets remain unchanged, so evaluation
    and citations continue to address verbatim evidence.
    """

    strategy = "summary_augmented"

    def __init__(
        self,
        base_chunker: BaseChunker,
        summaries: Mapping[str, str],
        *,
        strategy: str = "summary_augmented",
        max_summary_chars: int = 170,
    ) -> None:
        super().__init__(
            max_words=base_chunker.max_words,
            overlap_words=base_chunker.overlap_words,
        )
        if not strategy.strip():
            raise ValueError("strategy must not be blank")
        if max_summary_chars <= 0:
            raise ValueError("max_summary_chars must be positive")
        self.base_chunker = base_chunker
        self.strategy = strategy
        self.max_summary_chars = max_summary_chars
        self.summaries = {
            str(document_id): self._normalize_summary(summary)
            for document_id, summary in summaries.items()
        }

    def chunk(
        self,
        document_id: str,
        text: str,
        *,
        sections: Sequence[Mapping[str, object]] | None = None,
    ) -> list[Chunk]:
        summary = self.summaries.get(document_id)
        if summary is None:
            raise KeyError(f"Missing document summary for {document_id}")
        if len(summary) > self.max_summary_chars:
            raise ValueError(
                f"Summary for {document_id} has {len(summary)} characters; "
                f"maximum is {self.max_summary_chars}"
            )

        base_chunks = self.base_chunker.chunk(
            document_id,
            text,
            sections=sections,
        )
        chunks: list[Chunk] = []
        for index, base in enumerate(base_chunks):
            embedding_source = base.embedding_text or base.text
            chunks.append(
                Chunk(
                    chunk_id=self._make_id(
                        document_id,
                        index,
                        base.start_position,
                        base.end_position,
                    ),
                    document_id=document_id,
                    chunk_index=index,
                    strategy=self.strategy,
                    text=base.text,
                    start_position=base.start_position,
                    end_position=base.end_position,
                    section_label=base.section_label,
                    section_heading=base.section_heading,
                    embedding_text=(
                        f"ringkasan dokumen: {summary}\n\n{embedding_source}"
                    ),
                )
            )
        return chunks

    @staticmethod
    def _normalize_summary(value: object) -> str:
        summary = " ".join(str(value).split())
        if not summary:
            raise ValueError("document summaries must not be blank")
        return summary


class HierarchicalSummaryAugmentedChunker(BaseChunker):
    """Augment structure-aware chunks with document and section contexts."""

    strategy = "hierarchical_summary_augmented"

    def __init__(
        self,
        base_chunker: BaseChunker,
        document_contexts: Mapping[str, str],
        section_contexts: Mapping[str, Mapping[str, str]],
        *,
        strategy: str = "hierarchical_summary_augmented",
        max_context_chars: int = 240,
    ) -> None:
        super().__init__(
            max_words=base_chunker.max_words,
            overlap_words=base_chunker.overlap_words,
        )
        if not strategy.strip():
            raise ValueError("strategy must not be blank")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        self.base_chunker = base_chunker
        self.strategy = strategy
        self.max_context_chars = max_context_chars
        self.document_contexts = {
            str(document_id): SummaryAugmentedChunker._normalize_summary(context)
            for document_id, context in document_contexts.items()
        }
        self.section_contexts = {
            str(document_id): {
                str(label): SummaryAugmentedChunker._normalize_summary(context)
                for label, context in contexts.items()
            }
            for document_id, contexts in section_contexts.items()
        }

    def chunk(
        self,
        document_id: str,
        text: str,
        *,
        sections: Sequence[Mapping[str, object]] | None = None,
    ) -> list[Chunk]:
        document_context = self.document_contexts.get(document_id)
        contexts = self.section_contexts.get(document_id)
        if document_context is None or contexts is None:
            raise KeyError(f"Missing hierarchical contexts for {document_id}")
        base_chunks = self.base_chunker.chunk(
            document_id,
            text,
            sections=sections,
        )
        chunks: list[Chunk] = []
        for index, base in enumerate(base_chunks):
            label = base.section_label or "unknown"
            section_context = contexts.get(label)
            if section_context is None:
                raise KeyError(
                    f"Missing section context for {document_id}/{label}"
                )
            context_chars = len(document_context) + len(section_context)
            if context_chars > self.max_context_chars:
                raise ValueError(
                    f"Contexts for {document_id}/{label} have {context_chars} "
                    f"characters; maximum is {self.max_context_chars}"
                )
            embedding_source = base.embedding_text or base.text
            chunks.append(
                Chunk(
                    chunk_id=self._make_id(
                        document_id,
                        index,
                        base.start_position,
                        base.end_position,
                    ),
                    document_id=document_id,
                    chunk_index=index,
                    strategy=self.strategy,
                    text=base.text,
                    start_position=base.start_position,
                    end_position=base.end_position,
                    section_label=base.section_label,
                    section_heading=base.section_heading,
                    embedding_text=(
                        f"konteks dokumen: {document_context}\n"
                        f"konteks bagian: {section_context}\n\n"
                        f"{embedding_source}"
                    ),
                )
            )
        return chunks
