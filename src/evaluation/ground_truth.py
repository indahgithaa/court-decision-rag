"""Validation and chunk-candidate generation for retrieval ground truth."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_REVIEW_STATUSES = {"draft", "approved", "rejected"}


def join_clean_pages(pages: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Rebuild the exact document text used by preprocessing and chunking."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for page in pages:
        grouped[str(page["document_id"])].append(page)
    return {
        document_id: "\n\n".join(
            str(page.get("clean_text", ""))
            for page in sorted(document_pages, key=lambda item: int(item["page_number"]))
        )
        for document_id, document_pages in grouped.items()
    }


def validate_questions(
    rows: Sequence[Mapping[str, Any]],
    documents: Mapping[str, str],
) -> dict[str, Any]:
    """Validate populated question rows and summarize annotation progress."""
    errors: list[str] = []
    query_ids: Counter[str] = Counter(str(row.get("query_id", "")).strip() for row in rows)
    duplicate_ids = sorted(query_id for query_id, count in query_ids.items() if query_id and count > 1)
    if duplicate_ids:
        errors.append(f"Duplicate query_id: {', '.join(duplicate_ids)}")

    status_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for number, row in enumerate(rows, start=2):
        query_id = str(row.get("query_id", "")).strip() or f"CSV row {number}"
        document_id = str(row.get("document_id", "")).strip()
        status = str(row.get("review_status", "")).strip().lower()
        label = str(row.get("target_section_label", "")).strip()
        status_counts[status or "unreviewed"] += 1
        label_counts[label] += 1

        if document_id not in documents:
            errors.append(f"{query_id}: unknown document_id {document_id!r}")
            continue
        if status and status not in VALID_REVIEW_STATUSES:
            errors.append(f"{query_id}: invalid review_status {status!r}")
            continue
        if status in {"draft", "approved"}:
            _validate_populated_row(query_id, row, documents[document_id], errors)

    approved = status_counts["approved"]
    return {
        "row_count": len(rows),
        "approved_count": approved,
        "completion": approved / len(rows) if rows else 0.0,
        "status_counts": dict(sorted(status_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "errors": errors,
        "valid": not errors,
        "ready": bool(rows) and approved == len(rows) and not errors,
    }


def build_qrel_candidates(
    questions: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map approved evidence spans to overlapping chunks.

    ``auto_grade`` is 2 only when a chunk contains the complete evidence span.
    Partial overlaps receive grade 1 as a review candidate; a human must decide
    whether the chunk still contains enough context to answer the question.
    """
    chunks_by_document: dict[str, list[Mapping[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_document.setdefault(str(chunk["document_id"]), []).append(chunk)

    candidates: list[dict[str, Any]] = []
    for question in questions:
        if str(question.get("review_status", "")).strip().lower() != "approved":
            continue
        query_id = str(question["query_id"])
        document_id = str(question["document_id"])
        evidence_start = int(str(question["evidence_start_position"]))
        evidence_end = int(str(question["evidence_end_position"]))
        evidence_id = str(question.get("evidence_id", "")).strip() or (
            f"{query_id}:{evidence_start}:{evidence_end}"
        )
        evidence_length = evidence_end - evidence_start
        found = False
        for chunk in chunks_by_document.get(document_id, []):
            chunk_start = int(chunk["start_position"])
            chunk_end = int(chunk["end_position"])
            overlap = max(0, min(evidence_end, chunk_end) - max(evidence_start, chunk_start))
            if not overlap:
                continue
            found = True
            contains_evidence = chunk_start <= evidence_start and evidence_end <= chunk_end
            candidates.append(
                {
                    "query_id": query_id,
                    "document_id": document_id,
                    "target_section_label": str(question.get("target_section_label", "")),
                    "evidence_id": evidence_id,
                    "strategy": str(chunk.get("strategy", "")),
                    "chunk_id": str(chunk["chunk_id"]),
                    "chunk_start_position": chunk_start,
                    "chunk_end_position": chunk_end,
                    "evidence_start_position": evidence_start,
                    "evidence_end_position": evidence_end,
                    "evidence_coverage": overlap / evidence_length,
                    "auto_grade": 2 if contains_evidence else 1,
                    "relevance_grade": 2 if contains_evidence else "",
                    "reviewer_notes": "",
                }
            )
        if not found:
            raise ValueError(f"No chunk overlaps evidence for {query_id}")
    return candidates


def _validate_populated_row(
    query_id: str,
    row: Mapping[str, Any],
    document_text: str,
    errors: list[str],
) -> None:
    for field in ("question", "reference_answer"):
        if not str(row.get(field, "")).strip():
            errors.append(f"{query_id}: {field} is required")

    difficulty = str(row.get("difficulty", "")).strip().lower()
    if difficulty not in VALID_DIFFICULTIES:
        errors.append(f"{query_id}: difficulty must be easy, medium, or hard")

    try:
        start = int(str(row.get("evidence_start_position", "")))
        end = int(str(row.get("evidence_end_position", "")))
    except ValueError:
        errors.append(f"{query_id}: evidence offsets must be integers")
        return
    if not 0 <= start < end <= len(document_text):
        errors.append(
            f"{query_id}: evidence span {start}:{end} is outside document length "
            f"{len(document_text)}"
        )
    elif not document_text[start:end].strip():
        errors.append(f"{query_id}: evidence span contains no text")
