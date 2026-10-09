"""Canonical recursive-character chunks shared by B1, B2, and M1."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_SEPARATORS = ("\n\n", "\n", "!", "?", ".", ":", ";", ",", " ", "")


@dataclass(frozen=True)
class CanonicalChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    original_text: str
    start_position: int
    end_position: int
    page_numbers: tuple[int, ...]
    section_label: str
    section_heading: str | None
    section_overlap_chars: int
    section_assignment_status: str
    mixed_section_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["page_numbers"] = list(self.page_numbers)
        row["mixed_section_labels"] = list(self.mixed_section_labels)
        return row


class RecursiveCharacterChunker:
    """Deterministic recursive-character splitter with source-exact offsets.

    At every window the splitter selects the highest-priority separator that
    yields a reasonably filled chunk. If no such separator exists, it falls
    through the hierarchy and finally uses the hard character boundary. This
    is deliberately dependency-free and versioned as ``rcts-offset-v1``.
    """

    implementation_version = "rcts-offset-v1"

    def __init__(
        self,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        preferred_fill_ratio: float = 0.5,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < size")
        if not separators or separators[-1] != "":
            raise ValueError("separators must end with the empty-string fallback")
        if not 0 < preferred_fill_ratio <= 1:
            raise ValueError("preferred_fill_ratio must be in (0, 1]")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)
        self.preferred_fill_ratio = preferred_fill_ratio

    def split(self, document_id: str, text: str) -> list[dict[str, Any]]:
        if not document_id.strip():
            raise ValueError("document_id must not be blank")
        if not text:
            return []
        spans: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            hard_end = min(cursor + self.chunk_size, len(text))
            end = hard_end if hard_end == len(text) else self._boundary(text, cursor, hard_end)
            raw_start, raw_end = cursor, end
            while raw_start < raw_end and text[raw_start].isspace():
                raw_start += 1
            while raw_end > raw_start and text[raw_end - 1].isspace():
                raw_end -= 1
            if raw_end > raw_start:
                spans.append((raw_start, raw_end))
            if end >= len(text):
                break
            next_cursor = end - self.chunk_overlap
            if next_cursor <= cursor:
                next_cursor = cursor + 1
            cursor = next_cursor

        chunks: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(spans):
            original = text[start:end]
            digest = hashlib.sha256(
                f"{document_id}|{index}|{start}|{end}|{original}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                {
                    "chunk_id": f"{document_id}-rcts-{index:05d}-{digest}",
                    "document_id": document_id,
                    "chunk_index": index,
                    "original_text": original,
                    "start_position": start,
                    "end_position": end,
                }
            )
        return chunks

    def _boundary(self, text: str, start: int, hard_end: int) -> int:
        minimum = start + int(self.chunk_size * self.preferred_fill_ratio)
        window = text[start:hard_end]
        for separator in self.separators:
            if not separator:
                return hard_end
            position = window.rfind(separator)
            if position < 0:
                continue
            candidate = start + position + len(separator)
            if candidate >= minimum:
                return candidate
        return hard_end


def assign_chunk_context(
    chunks: Sequence[Mapping[str, Any]],
    *,
    sections: Sequence[Mapping[str, Any]],
    page_spans: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Assign each base chunk to a section by maximum character overlap."""
    enriched: list[dict[str, Any]] = []
    ordered_sections = sorted(sections, key=lambda row: int(row["start_position"]))
    for chunk in chunks:
        start = int(chunk["start_position"])
        end = int(chunk["end_position"])
        overlaps: list[tuple[int, str, str | None]] = []
        for section in ordered_sections:
            overlap = _overlap(
                start,
                end,
                int(section["start_position"]),
                int(section["end_position"]),
            )
            if overlap:
                overlaps.append(
                    (
                        overlap,
                        str(section.get("section_label") or "unknown"),
                        str(section["section_heading"])
                        if section.get("section_heading") is not None
                        else None,
                    )
                )
        if not overlaps:
            label, heading, best_overlap, status = "unknown", None, 0, "unassigned"
            mixed: tuple[str, ...] = ()
        else:
            overlaps.sort(key=lambda item: item[0], reverse=True)
            best_overlap, label, heading = overlaps[0]
            tied = [item for item in overlaps if item[0] == best_overlap]
            mixed = tuple(item[1] for item in overlaps)
            if len(tied) > 1:
                status = "mixed_tie_first_section"
            elif len(overlaps) > 1:
                status = "mixed_maximum_overlap"
            else:
                status = "single_section"
        pages = tuple(
            int(page["page_number"])
            for page in page_spans
            if _overlap(start, end, int(page["start_position"]), int(page["end_position"]))
        )
        enriched.append(
            CanonicalChunk(
                chunk_id=str(chunk["chunk_id"]),
                document_id=str(chunk["document_id"]),
                chunk_index=int(chunk["chunk_index"]),
                original_text=str(chunk["original_text"]),
                start_position=start,
                end_position=end,
                page_numbers=pages,
                section_label=label,
                section_heading=heading,
                section_overlap_chars=best_overlap,
                section_assignment_status=status,
                mixed_section_labels=mixed,
            ).to_dict()
        )
    return enriched


def build_page_spans(pages: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, int]]]:
    """Reconstruct the pipeline's ``\n\n`` page join and its character spans."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for page in pages:
        grouped.setdefault(str(page["document_id"]), []).append(page)
    output: dict[str, list[dict[str, int]]] = {}
    for document_id, document_pages in grouped.items():
        cursor = 0
        spans: list[dict[str, int]] = []
        ordered = sorted(document_pages, key=lambda row: int(row["page_number"]))
        for index, page in enumerate(ordered):
            text = str(page.get("clean_text", ""))
            spans.append(
                {
                    "page_number": int(page["page_number"]),
                    "start_position": cursor,
                    "end_position": cursor + len(text),
                }
            )
            cursor += len(text)
            if index + 1 < len(ordered):
                cursor += 2
        output[document_id] = spans
    return output


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))
