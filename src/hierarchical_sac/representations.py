"""Embedding inputs for the three controlled retrieval configurations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


METHODS = ("B1", "B2", "M1")


def build_embedding_representations(
    chunks: Sequence[Mapping[str, Any]],
    *,
    document_summaries: Mapping[str, str],
    section_summaries: Mapping[tuple[str, str], str],
) -> dict[str, list[dict[str, Any]]]:
    """Build aligned method records without changing canonical chunk IDs."""
    output = {method: [] for method in METHODS}
    for chunk in chunks:
        document_id = str(chunk["document_id"])
        section_label = str(chunk.get("section_label") or "unknown")
        try:
            document_summary = _normalize(document_summaries[document_id])
        except KeyError as error:
            raise KeyError(f"Missing document summary for {document_id}") from error
        section_summary = section_summaries.get((document_id, section_label))
        if section_summary is None:
            section_summary = f"Bagian dokumen tidak teridentifikasi ({section_label})."
            section_status = "documented_unknown_section_fallback"
        else:
            section_summary = _normalize(section_summary)
            section_status = "summary"
        original = str(chunk["original_text"])
        texts = {
            "B1": original,
            "B2": (
                f"Ringkasan Dokumen:\n{document_summary}\n\n"
                f"Teks Asli:\n{original}"
            ),
            "M1": (
                f"Ringkasan Dokumen:\n{document_summary}\n\n"
                f"Ringkasan Bagian:\n{section_summary}\n\n"
                f"Teks Asli:\n{original}"
            ),
        }
        for method in METHODS:
            row = dict(chunk)
            row.update(
                {
                    "method": method,
                    "embedding_text": texts[method],
                    "document_summary": document_summary if method != "B1" else None,
                    "section_summary": section_summary if method == "M1" else None,
                    "section_summary_status": section_status if method == "M1" else None,
                }
            )
            output[method].append(row)
    assert_aligned(output)
    return output


def assert_aligned(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Require identical IDs, offsets, and original text across all methods."""
    if set(records) != set(METHODS):
        raise ValueError(f"Expected exactly {METHODS}; got {sorted(records)}")
    reference = [
        (
            str(row["chunk_id"]),
            str(row["document_id"]),
            int(row["start_position"]),
            int(row["end_position"]),
            str(row["original_text"]),
        )
        for row in records["B1"]
    ]
    for method in METHODS[1:]:
        candidate = [
            (
                str(row["chunk_id"]),
                str(row["document_id"]),
                int(row["start_position"]),
                int(row["end_position"]),
                str(row["original_text"]),
            )
            for row in records[method]
        ]
        if candidate != reference:
            raise ValueError(f"Canonical chunk alignment failed for {method}")


def apply_e5_token_budget(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_tokens: int = 512,
    passage_prefix: str = "passage: ",
) -> list[dict[str, Any]]:
    """Fit summaries into E5 while never truncating the original chunk."""
    if max_tokens <= 2:
        raise ValueError("max_tokens must be greater than two")
    output: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        original = str(row["original_text"])
        original_only_tokens = _token_count(tokenizer, passage_prefix + original)
        if original_only_tokens > max_tokens:
            raise ValueError(
                f"Original evidence exceeds E5 budget for {row['chunk_id']}: "
                f"{original_only_tokens}>{max_tokens}"
            )
        initial_text = str(row["embedding_text"])
        initial_tokens = _token_count(tokenizer, passage_prefix + initial_text)
        if initial_tokens <= max_tokens:
            final_text = initial_text
            truncated = False
        else:
            final_text = _fit_prefixed_representation(
                row,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                passage_prefix=passage_prefix,
            )
            truncated = True
        final_tokens = _token_count(tokenizer, passage_prefix + final_text)
        if final_tokens > max_tokens or original not in final_text:
            raise RuntimeError(f"Unsafe token-budget result for {row['chunk_id']}")
        row.update(
            {
                "embedding_text": final_text,
                "embedding_tokens_before_budget": initial_tokens,
                "embedding_tokens": final_tokens,
                "original_only_tokens": original_only_tokens,
                "summary_budget_applied": truncated,
                "source_text_preserved": True,
            }
        )
        output.append(row)
    return output


def _fit_prefixed_representation(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    max_tokens: int,
    passage_prefix: str,
) -> str:
    method = str(row["method"])
    original = str(row["original_text"])
    doc = str(row.get("document_summary") or "")
    sec = str(row.get("section_summary") or "")
    if method == "B1":
        return original

    def render(doc_value: str, sec_value: str) -> str:
        if method == "B2":
            return f"Ringkasan Dokumen:\n{doc_value}\n\nTeks Asli:\n{original}"
        return (
            f"Ringkasan Dokumen:\n{doc_value}\n\n"
            f"Ringkasan Bagian:\n{sec_value}\n\nTeks Asli:\n{original}"
        )

    # Remove summary tokens from the right in round-robin order. Evidence is
    # immutable and therefore survives even in the tightest budget.
    doc_ids = tokenizer.encode(doc, add_special_tokens=False)
    sec_ids = tokenizer.encode(sec, add_special_tokens=False)
    while True:
        doc_value = tokenizer.decode(doc_ids, skip_special_tokens=True).strip()
        sec_value = tokenizer.decode(sec_ids, skip_special_tokens=True).strip()
        candidate = render(doc_value, sec_value)
        if _token_count(tokenizer, passage_prefix + candidate) <= max_tokens:
            return candidate
        if sec_ids:
            sec_ids.pop()
        elif doc_ids:
            doc_ids.pop()
        else:
            # Labels alone can consume budget. Preserve only original evidence.
            return original


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def _normalize(value: object) -> str:
    normalized = " ".join(str(value).split())
    if not normalized:
        raise ValueError("summaries must not be blank")
    return normalized
