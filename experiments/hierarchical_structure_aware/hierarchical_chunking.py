"""Two-level chunking and parent-context reconstruction for the experiment.

The two hierarchical arms intentionally share all hierarchy parameters and
child splitting logic.  Their only design difference is the parent boundary:
B3 creates linear parents over the complete document, while M1 creates parents
inside source sections and never crosses a section boundary.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any


def word_windows(
    text: str,
    *,
    max_words: int,
    overlap_words: int,
    offset: int = 0,
) -> list[tuple[str, int, int]]:
    """Return exact source slices using deterministic whitespace-token windows."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap < max_words")
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    step = max_words - overlap_words
    windows: list[tuple[str, int, int]] = []
    for word_start in range(0, len(words), step):
        word_end = min(word_start + max_words, len(words))
        char_start = words[word_start].start()
        char_end = words[word_end - 1].end()
        windows.append(
            (text[char_start:char_end], offset + char_start, offset + char_end)
        )
        if word_end == len(words):
            break
    return windows


class HierarchicalChunker:
    """Create retrievable children linked to bounded source-text parents."""

    def __init__(
        self,
        *,
        structure_aware: bool,
        parent_words: int = 500,
        parent_overlap_words: int = 100,
        child_words: int = 150,
        child_overlap_words: int = 30,
    ) -> None:
        if parent_words <= child_words:
            raise ValueError("parent_words must be greater than child_words")
        if parent_overlap_words < child_overlap_words:
            raise ValueError(
                "parent overlap must be at least child overlap to preserve boundaries"
            )
        self.structure_aware = structure_aware
        self.parent_words = parent_words
        self.parent_overlap_words = parent_overlap_words
        self.child_words = child_words
        self.child_overlap_words = child_overlap_words
        self.strategy = (
            "hierarchical_structure_aware"
            if structure_aware
            else "hierarchical"
        )

    def chunk(
        self,
        document_id: str,
        text: str,
        *,
        sections: Sequence[Mapping[str, object]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ``(parents, children)`` with exact global source offsets."""
        parent_sources = self._parent_sources(document_id, text, sections)
        parents: list[dict[str, Any]] = []
        children: list[dict[str, Any]] = []
        for source in parent_sources:
            source_text = str(source["text"])
            source_start = int(source["start_position"])
            for parent_text, parent_start, parent_end in word_windows(
                source_text,
                max_words=self.parent_words,
                overlap_words=self.parent_overlap_words,
                offset=source_start,
            ):
                parent_index = len(parents)
                parent_id = self._make_id(
                    "parent", document_id, parent_index, parent_start, parent_end
                )
                parent = {
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "parent_index": parent_index,
                    "strategy": self.strategy,
                    "text": parent_text,
                    "start_position": parent_start,
                    "end_position": parent_end,
                    "section_label": source.get("section_label"),
                    "section_heading": source.get("section_heading"),
                }
                parents.append(parent)
                for child_text, child_start, child_end in word_windows(
                    parent_text,
                    max_words=self.child_words,
                    overlap_words=self.child_overlap_words,
                    offset=parent_start,
                ):
                    child_index = len(children)
                    child_id = self._make_id(
                        "child",
                        document_id,
                        child_index,
                        child_start,
                        child_end,
                        parent_id=parent_id,
                    )
                    children.append(
                        {
                            "chunk_id": child_id,
                            "document_id": document_id,
                            "chunk_index": child_index,
                            "strategy": self.strategy,
                            "text": child_text,
                            "start_position": child_start,
                            "end_position": child_end,
                            "parent_id": parent_id,
                            "parent_start_position": parent_start,
                            "parent_end_position": parent_end,
                            "section_label": source.get("section_label"),
                            "section_heading": source.get("section_heading"),
                        }
                    )
        return parents, children

    def _parent_sources(
        self,
        document_id: str,
        text: str,
        sections: Sequence[Mapping[str, object]] | None,
    ) -> list[dict[str, object]]:
        if not self.structure_aware:
            return [
                {
                    "text": text,
                    "start_position": 0,
                    "section_label": None,
                    "section_heading": None,
                }
            ]
        usable = list(sections or [])
        if not usable and text:
            usable = [
                {
                    "document_id": document_id,
                    "section_text": text,
                    "start_position": 0,
                    "end_position": len(text),
                    "section_label": "unknown",
                    "section_heading": None,
                }
            ]
        sources: list[dict[str, object]] = []
        for section in sorted(usable, key=lambda row: int(row["start_position"])):
            section_text = str(section.get("section_text", ""))
            start = int(section.get("start_position", 0))
            end = int(section.get("end_position", start + len(section_text)))
            if end - start != len(section_text) or text[start:end] != section_text:
                raise ValueError(
                    f"Section provenance mismatch for {document_id} at {start}:{end}"
                )
            sources.append(
                {
                    "text": section_text,
                    "start_position": start,
                    "section_label": str(
                        section.get("section_label", "unknown")
                    ),
                    "section_heading": section.get("section_heading"),
                }
            )
        return sources

    def _make_id(
        self,
        level: str,
        document_id: str,
        index: int,
        start: int,
        end: int,
        *,
        parent_id: str = "",
    ) -> str:
        payload = (
            f"{self.strategy}|{level}|{document_id}|{index}|{start}|{end}|{parent_id}"
        )
        digest = hashlib.sha1(
            payload.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:12]
        return f"{document_id}-{self.strategy}-{level}-{digest}"


def aggregate_parent_ranking(
    child_ranking: Sequence[Mapping[str, Any]],
    parents_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate parents and score each with its best retrieved child."""
    best: dict[str, dict[str, Any]] = {}
    for child in child_ranking:
        parent_id = str(child.get("parent_id", ""))
        if not parent_id:
            raise ValueError("hierarchical child is missing parent_id")
        if parent_id not in parents_by_id:
            raise ValueError(f"Unknown parent_id {parent_id!r}")
        score = float(child["score"])
        previous = best.get(parent_id)
        if previous is None or score > float(previous["score"]):
            parent = dict(parents_by_id[parent_id])
            parent["score"] = score
            parent["best_child_id"] = str(child["chunk_id"])
            parent["best_child_rank"] = int(child["rank"])
            best[parent_id] = parent
    ordered = sorted(
        best.values(),
        key=lambda row: (
            -float(row["score"]),
            int(row["best_child_rank"]),
            str(row["parent_id"]),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    return ordered


def select_context_budget(
    ranking: Sequence[Mapping[str, Any]], *, budget_words: int
) -> tuple[list[dict[str, Any]], int]:
    """Select ranked, unique context units without exceeding a word budget."""
    if budget_words <= 0:
        raise ValueError("budget_words must be positive")
    selected: list[dict[str, Any]] = []
    used = 0
    seen: set[str] = set()
    for item in ranking:
        unit_id = str(item.get("parent_id") or item.get("chunk_id") or "")
        if not unit_id or unit_id in seen:
            continue
        words = len(re.findall(r"\S+", str(item["text"])))
        if selected and used + words > budget_words:
            continue
        if not selected and words > budget_words:
            # Return at least the highest-ranked unit; record its actual size.
            selected.append(dict(item))
            return selected, words
        selected.append(dict(item))
        seen.add(unit_id)
        used += words
        if used == budget_words:
            break
    return selected, used

