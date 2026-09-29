"""Tests for the common chunker contract and boundary behavior."""

import pytest

from src.chunking import FixedSizeChunker, StructureAwareChunker


def test_fixed_size_chunker_uses_configured_overlap() -> None:
    text = "satu dua tiga empat lima enam tujuh delapan sembilan sepuluh"
    chunks = FixedSizeChunker(max_words=4, overlap_words=1).chunk("doc", text)

    assert [chunk.text.split() for chunk in chunks] == [
        ["satu", "dua", "tiga", "empat"],
        ["empat", "lima", "enam", "tujuh"],
        ["tujuh", "delapan", "sembilan", "sepuluh"],
    ]
    assert all(text[chunk.start_position : chunk.end_position] == chunk.text for chunk in chunks)


def test_structure_aware_chunks_do_not_cross_section_boundaries() -> None:
    text = "A satu dua tiga\nB empat lima enam"
    second_start = text.index("B")
    sections = [
        {
            "section_label": "fakta",
            "section_heading": "A",
            "start_position": 0,
            "end_position": second_start,
            "section_text": text[:second_start],
        },
        {
            "section_label": "amar_putusan",
            "section_heading": "B",
            "start_position": second_start,
            "end_position": len(text),
            "section_text": text[second_start:],
        },
    ]

    chunks = StructureAwareChunker(max_words=3, overlap_words=0).chunk(
        "doc", text, sections=sections
    )

    assert {chunk.section_label for chunk in chunks} == {"fakta", "amar_putusan"}
    assert all(not ("A" in chunk.text and "B" in chunk.text) for chunk in chunks)
    assert all(text[chunk.start_position : chunk.end_position] == chunk.text for chunk in chunks)


def test_structure_aware_falls_back_without_sections() -> None:
    chunks = StructureAwareChunker(max_words=10, overlap_words=0).chunk("doc", "teks utuh")

    assert len(chunks) == 1
    assert chunks[0].section_label == "unknown"


def test_structure_aware_uses_two_sentence_overlap() -> None:
    text = "Satu dua tiga. Empat lima enam. Tujuh delapan sembilan. Sepuluh sebelas."
    sections = [
        {
            "section_label": "fakta",
            "section_heading": None,
            "start_position": 0,
            "end_position": len(text),
            "section_text": text,
        }
    ]

    chunks = StructureAwareChunker(
        max_words=9,
        overlap_words=1,
        overlap_sentences=2,
    ).chunk("doc", text, sections=sections)

    assert [chunk.text for chunk in chunks] == [
        "Satu dua tiga. Empat lima enam. Tujuh delapan sembilan.",
        "Empat lima enam. Tujuh delapan sembilan. Sepuluh sebelas.",
    ]


def test_structure_aware_rejects_negative_sentence_overlap() -> None:
    with pytest.raises(ValueError):
        StructureAwareChunker(overlap_sentences=-1)


def test_structure_aware_bridges_consecutive_oversized_sentences() -> None:
    text = "satu dua tiga empat lima. enam tujuh delapan sembilan sepuluh."
    sections = [
        {
            "section_label": "fakta",
            "section_heading": None,
            "start_position": 0,
            "end_position": len(text),
            "section_text": text,
        }
    ]

    chunks = StructureAwareChunker(
        max_words=4,
        overlap_words=1,
        overlap_sentences=2,
    ).chunk("doc", text, sections=sections)

    assert all(len(chunk.text.split()) <= 4 for chunk in chunks)
    assert all(
        current.start_position < previous.end_position
        for previous, current in zip(chunks, chunks[1:])
    )
    assert all(text[chunk.start_position : chunk.end_position] == chunk.text for chunk in chunks)


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(max_words=5, overlap_words=5)

