"""Deterministic context assembly for paired answer-generation experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def index_chunks(chunks: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index chunks by ID and reject missing or duplicate identifiers."""
    indexed: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("Every chunk must have a non-empty chunk_id")
        if chunk_id in indexed:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        indexed[chunk_id] = dict(chunk)
    return indexed


def assemble_ranked_context(
    ranking: Sequence[Mapping[str, Any]],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    *,
    word_budget: int,
) -> dict[str, Any]:
    """Assemble ranked chunks up to an exact whitespace-word budget.

    The final selected chunk is truncated when needed. Source headers are not
    counted toward the budget, so both experimental arms receive the same
    amount of retrieved document text.
    """
    if word_budget <= 0:
        raise ValueError("word_budget must be positive")
    if not ranking:
        raise ValueError("ranking must not be empty")

    ordered = sorted(ranking, key=lambda item: int(item["rank"]))
    ranks = [int(item["rank"]) for item in ordered]
    if ranks != list(range(1, len(ordered) + 1)):
        raise ValueError("ranking ranks must be unique and contiguous from 1")

    seen_chunk_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    context_parts: list[str] = []
    remaining = word_budget

    for ranked_item in ordered:
        chunk_id = str(ranked_item.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("Every ranking item must have a non-empty chunk_id")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate ranked chunk_id: {chunk_id}")
        seen_chunk_ids.add(chunk_id)

        try:
            chunk = chunks_by_id[chunk_id]
        except KeyError as error:
            raise ValueError(f"Ranked chunk not found: {chunk_id}") from error

        text = str(chunk.get("text", ""))
        word_matches = list(re.finditer(r"\S+", text))
        if not word_matches:
            continue
        included_matches = word_matches[:remaining]
        included_text = " ".join(match.group() for match in included_matches)
        document_id = str(chunk.get("document_id", "")).strip()
        retrieved_document_id = str(
            ranked_item.get("retrieved_document_id", document_id)
        ).strip()
        if retrieved_document_id and retrieved_document_id != document_id:
            raise ValueError(
                f"Document mismatch for {chunk_id}: ranking={retrieved_document_id}, "
                f"chunk={document_id}"
            )

        source_number = len(selected) + 1
        section_label = chunk.get("section_label")
        section_heading = chunk.get("section_heading")
        chunk_start = int(chunk["start_position"])
        chunk_end = int(chunk["end_position"])
        included_end = (
            chunk_end
            if len(included_matches) == len(word_matches)
            else chunk_start + included_matches[-1].end()
        )
        context_parts.append(
            f"[Chunk {source_number}]\n{included_text}"
        )
        selected.append(
            {
                "source_number": source_number,
                "rank": int(ranked_item["rank"]),
                "chunk_id": chunk_id,
                "document_id": document_id,
                "score": float(ranked_item["score"]),
                "section_label": section_label,
                "section_heading": section_heading,
                "start_position": chunk_start,
                "end_position": chunk_end,
                "included_start_position": chunk_start,
                "included_end_position": included_end,
                "included_word_count": len(included_matches),
                "original_word_count": len(word_matches),
                "truncated": len(included_matches) < len(word_matches),
            }
        )
        remaining -= len(included_matches)
        if remaining == 0:
            break

    actual_word_count = word_budget - remaining
    if actual_word_count != word_budget:
        raise ValueError(
            f"Ranking supplies only {actual_word_count} words; required {word_budget}"
        )
    return {
        "context": "\n\n".join(context_parts),
        "context_word_budget": word_budget,
        "context_word_count": actual_word_count,
        "selected_chunks": selected,
    }
