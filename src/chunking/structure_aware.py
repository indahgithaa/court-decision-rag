"""Chunking that respects detected section boundaries."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from .base import BaseChunker, Chunk


class StructureAwareChunker(BaseChunker):
    """Split sections with SAC-H+ sentence-aware overlap."""

    strategy = "structure_aware"

    def __init__(
        self,
        *,
        max_words: int = 300,
        overlap_words: int = 50,
        overlap_sentences: int = 2,
        backfill_short_tail: bool = False,
        embedding_context: str = "none",
    ) -> None:
        super().__init__(max_words=max_words, overlap_words=overlap_words)
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be greater than or equal to zero")
        self.overlap_sentences = overlap_sentences
        self.backfill_short_tail = backfill_short_tail
        if embedding_context not in {
            "none",
            "section",
            "section_document",
            "section_reasoning_document",
        }:
            raise ValueError(
                "embedding_context must be none, section, section_document, "
                "or section_reasoning_document"
            )
        self.embedding_context = embedding_context

    def chunk(
        self,
        document_id: str,
        text: str,
        *,
        sections: Sequence[Mapping[str, object]] | None = None,
    ) -> list[Chunk]:
        usable_sections = list(sections or [])
        if not usable_sections and text:
            usable_sections = [
                {
                    "document_id": document_id,
                    "section_label": "unknown",
                    "section_heading": None,
                    "start_position": 0,
                    "end_position": len(text),
                    "section_text": text,
                }
            ]

        document_context = self._document_context(usable_sections)
        chunks: list[Chunk] = []
        for section in usable_sections:
            section_text = str(section.get("section_text", ""))
            section_start = int(section.get("start_position", 0))
            for chunk_text, start, end in self._sentence_windows(
                section_text, offset=section_start
            ):
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=self._make_id(document_id, index, start, end),
                        document_id=document_id,
                        chunk_index=index,
                        strategy=self.strategy,
                        text=chunk_text,
                        start_position=start,
                        end_position=end,
                        section_label=str(section.get("section_label", "unknown")),
                        section_heading=(
                            str(section["section_heading"])
                            if section.get("section_heading") is not None
                            else None
                        ),
                        embedding_text=self._embedding_text(
                            chunk_text,
                            section_label=str(
                                section.get("section_label", "unknown")
                            ),
                            document_context=document_context,
                        ),
                    )
                )
        return chunks

    def _embedding_text(
        self,
        text: str,
        *,
        section_label: str,
        document_context: str | None,
    ) -> str | None:
        if self.embedding_context == "none":
            return None
        parts = [f"bagian dokumen: {section_label.replace('_', ' ')}"]
        include_document = self.embedding_context == "section_document" or (
            self.embedding_context == "section_reasoning_document"
            and section_label in {"pertimbangan_hukum", "fakta_hukum"}
        )
        if include_document and document_context:
            parts.insert(0, f"perkara terdakwa: {document_context}")
        return ". ".join(parts) + ".\n" + text

    @staticmethod
    def _document_context(
        sections: Sequence[Mapping[str, object]],
    ) -> str | None:
        identity = next(
            (
                str(section.get("section_text", ""))
                for section in sections
                if str(section.get("section_label", "")) == "identitas_terdakwa"
            ),
            "",
        )
        match = re.search(
            r"\bnama lengkap\s+(.{1,160}?)(?=\s+(?:tempat lahir|umur|tanggal lahir|"
            r"jenis kelamin|kebangsaan)\b)",
            identity,
            flags=re.IGNORECASE,
        )
        return " ".join(match.group(1).split()) if match else None

    def _sentence_windows(self, text: str, *, offset: int) -> list[tuple[str, int, int]]:
        """Pack complete sentences and carry the previous two into the next chunk.

        A sentence longer than the word budget falls back to the shared word
        window implementation, preserving the configured word overlap.
        """
        sentences = self._sentence_spans(text)
        if not sentences:
            return []

        counts = [len(re.findall(r"\S+", text[start:end])) for start, end in sentences]
        windows: list[tuple[str, int, int]] = []
        sentence_start = 0
        while sentence_start < len(sentences):
            if counts[sentence_start] > self.max_words:
                start, end = sentences[sentence_start]
                windows.extend(self._windows(text[start:end], offset=offset + start))
                sentence_start += 1
                continue

            sentence_end = sentence_start
            words = 0
            while sentence_end < len(sentences):
                candidate_words = counts[sentence_end]
                if sentence_end > sentence_start and words + candidate_words > self.max_words:
                    break
                words += candidate_words
                sentence_end += 1

            char_start = sentences[sentence_start][0]
            char_end = sentences[sentence_end - 1][1]
            windows.append(
                (text[char_start:char_end], offset + char_start, offset + char_end)
            )
            if sentence_end == len(sentences):
                break

            next_start = max(sentence_start + 1, sentence_end - self.overlap_sentences)
            # Retain as much sentence overlap as fits while guaranteeing that
            # the next window also contains unseen content.
            while (
                next_start < sentence_end
                and sum(counts[next_start : sentence_end + 1]) > self.max_words
            ):
                next_start += 1
            sentence_start = next_start

        windows = self._ensure_context_overlap(text, windows, offset=offset)
        return self._backfill_tail(text, windows, offset=offset)

    def _backfill_tail(
        self,
        text: str,
        windows: list[tuple[str, int, int]],
        *,
        offset: int,
    ) -> list[tuple[str, int, int]]:
        """Anchor an undersized final window to the end of its section.

        Section-local windowing can leave a very short orphan chunk at a
        boundary. Such a chunk often contains the concluding legal basis but
        too little surrounding reasoning for dense retrieval. Backfilling adds
        preceding words from the same section only; source offsets and the
        maximum word budget remain exact.
        """
        if not self.backfill_short_tail or len(windows) < 2:
            return windows

        last_text, _, last_end = windows[-1]
        if len(re.findall(r"\S+", last_text)) > max(1, self.max_words // 2):
            return windows

        relative_end = last_end - offset
        words = [match for match in re.finditer(r"\S+", text) if match.end() <= relative_end]
        if len(words) <= self.max_words:
            return windows
        relative_start = words[-self.max_words].start()
        backfilled = (
            text[relative_start:relative_end],
            offset + relative_start,
            last_end,
        )
        return [*windows[:-1], backfilled]

    def _ensure_context_overlap(
        self,
        text: str,
        windows: list[tuple[str, int, int]],
        *,
        offset: int,
    ) -> list[tuple[str, int, int]]:
        """Bridge resets between oversized-sentence fallbacks.

        SAC-H+ normally overlaps complete sentences. A sentence that already
        exceeds the word budget cannot be carried intact, so the word overlap
        is used across that boundary. If expanding the next window would exceed
        the budget, a bridge window is inserted before it.
        """
        if len(windows) < 2 or self.overlap_words == 0:
            return windows

        result = [windows[0]]
        for current in windows[1:]:
            previous = result[-1]
            if current[1] < previous[2]:
                result.append(current)
                continue

            previous_start = previous[1] - offset
            previous_end = previous[2] - offset
            current_start = current[1] - offset
            current_end = current[2] - offset
            previous_words = list(re.finditer(r"\S+", text[previous_start:previous_end]))
            current_words = list(re.finditer(r"\S+", text[current_start:current_end]))
            context_count = min(self.overlap_words, len(previous_words))
            context_start = (
                previous_start + previous_words[-context_count].start()
                if context_count
                else current_start
            )

            if context_count + len(current_words) <= self.max_words:
                result.append(
                    (
                        text[context_start:current_end],
                        offset + context_start,
                        offset + current_end,
                    )
                )
                continue

            new_word_budget = self.max_words - context_count
            if new_word_budget <= 0:
                result.append(current)
                continue
            bridge_end = current_start + current_words[new_word_budget - 1].end()
            result.append(
                (
                    text[context_start:bridge_end],
                    offset + context_start,
                    offset + bridge_end,
                )
            )
            result.append(current)
        return result

    @staticmethod
    def _sentence_spans(text: str) -> list[tuple[int, int]]:
        boundaries = list(re.finditer(r"[.!?;]+(?=\s|$)", text))
        spans: list[tuple[int, int]] = []
        cursor = 0
        for boundary in boundaries:
            start_match = re.search(r"\S", text[cursor : boundary.end()])
            if start_match:
                spans.append((cursor + start_match.start(), boundary.end()))
            cursor = boundary.end()
        tail_match = re.search(r"\S", text[cursor:])
        if tail_match:
            spans.append((cursor + tail_match.start(), len(text.rstrip())))
        return spans

