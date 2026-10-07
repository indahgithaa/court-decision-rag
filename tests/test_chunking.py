"""Tests for the common chunker contract and boundary behavior."""

import pytest

from src.chunking import (
    FixedSizeChunker,
    HierarchicalSummaryAugmentedChunker,
    StructureAwareChunker,
    SummaryAugmentedChunker,
)


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


def test_structure_aware_rejects_unknown_boundary_mode() -> None:
    with pytest.raises(ValueError, match="boundary_mode"):
        StructureAwareChunker(boundary_mode="query_aware")


def test_structure_aware_rejects_invalid_boundary_fill_ratio() -> None:
    with pytest.raises(ValueError, match="boundary_min_fill_ratio"):
        StructureAwareChunker(boundary_min_fill_ratio=0)


def test_pure_sac_rhetorical_boundaries_preserve_exact_source_text() -> None:
    text = (
        "Menimbang bahwa unsur setiap orang telah terpenuhi "
        "Terhadap unsur tanpa hak majelis menilai alat bukti yang diajukan "
        "Oleh karena itu semua unsur tindak pidana telah terpenuhi"
    )
    sections = [
        {
            "section_label": "pertimbangan_hukum",
            "section_heading": "PERTIMBANGAN HUKUM",
            "start_position": 0,
            "end_position": len(text),
            "section_text": text,
        }
    ]

    chunks = StructureAwareChunker(
        max_words=12,
        overlap_words=2,
        overlap_sentences=0,
        backfill_short_tail=True,
        boundary_mode="legal_rhetorical",
        embedding_context="none",
    ).chunk("doc", text, sections=sections)

    assert len(chunks) >= 3
    assert all(len(chunk.text.split()) <= 12 for chunk in chunks)
    assert all(chunk.embedding_text is None for chunk in chunks)
    assert all(
        text[chunk.start_position : chunk.end_position] == chunk.text
        for chunk in chunks
    )
    assert any("Terhadap unsur" in chunk.text for chunk in chunks)
    assert any("Oleh karena itu" in chunk.text for chunk in chunks)


def test_pure_sac_rhetorical_mode_remains_section_local() -> None:
    first = "Menimbang bahwa dasar dakwaan pertama telah terpenuhi"
    second = "Mengadili Menjatuhkan pidana penjara selama empat tahun"
    text = first + "\n" + second
    second_start = text.index(second)
    sections = [
        {
            "section_label": "pertimbangan_hukum",
            "start_position": 0,
            "end_position": len(first),
            "section_text": first,
        },
        {
            "section_label": "amar_putusan",
            "start_position": second_start,
            "end_position": len(text),
            "section_text": second,
        },
    ]

    chunks = StructureAwareChunker(
        max_words=20,
        overlap_words=4,
        overlap_sentences=0,
        boundary_mode="legal_rhetorical",
    ).chunk("doc", text, sections=sections)

    assert {chunk.section_label for chunk in chunks} == {
        "pertimbangan_hukum",
        "amar_putusan",
    }
    assert all(not ("Menimbang" in chunk.text and "Mengadili" in chunk.text) for chunk in chunks)


def test_pure_sac_aligns_full_windows_without_chunk_explosion() -> None:
    words = [f"w{index}" for index in range(7)]
    words += ["Menimbang", "bahwa"]
    words += [f"x{index}" for index in range(18)]
    text = " ".join(words)
    sections = [
        {
            "section_label": "pertimbangan_hukum",
            "start_position": 0,
            "end_position": len(text),
            "section_text": text,
        }
    ]

    chunks = StructureAwareChunker(
        max_words=10,
        overlap_words=2,
        overlap_sentences=0,
        boundary_mode="legal_rhetorical",
    ).chunk("doc", text, sections=sections)

    assert chunks[0].text.split() == words[:7]
    assert chunks[1].text.split()[:4] == words[5:9]
    assert len(chunks) <= 4
    assert all(len(chunk.text.split()) <= 10 for chunk in chunks)


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


def test_structure_aware_can_backfill_short_section_tail() -> None:
    text = " ".join(f"w{index}" for index in range(13))
    sections = [
        {
            "section_label": "pertimbangan_hukum",
            "section_heading": "PERTIMBANGAN HUKUM",
            "start_position": 0,
            "end_position": len(text),
            "section_text": text,
        }
    ]

    chunks = StructureAwareChunker(
        max_words=6,
        overlap_words=1,
        overlap_sentences=0,
        backfill_short_tail=True,
    ).chunk("doc", text, sections=sections)

    assert [len(chunk.text.split()) for chunk in chunks] == [6, 6, 6]
    assert chunks[-1].text.split() == ["w7", "w8", "w9", "w10", "w11", "w12"]
    assert chunks[-1].end_position == len(text)
    assert all(text[chunk.start_position : chunk.end_position] == chunk.text for chunk in chunks)


def test_structure_aware_contextualizes_embedding_without_changing_source_text() -> None:
    identity = "nama lengkap Siti Aminah tempat lahir Malang umur 30 tahun"
    reasoning = "menimbang bahwa pasal tersebut telah terpenuhi"
    text = identity + "\n" + reasoning
    reasoning_start = text.index(reasoning)
    sections = [
        {
            "section_label": "identitas_terdakwa",
            "start_position": 0,
            "end_position": len(identity),
            "section_text": identity,
        },
        {
            "section_label": "pertimbangan_hukum",
            "start_position": reasoning_start,
            "end_position": len(text),
            "section_text": reasoning,
        },
    ]

    chunks = StructureAwareChunker(
        max_words=20,
        overlap_words=0,
        embedding_context="section_document",
    ).chunk("doc", text, sections=sections)
    reasoning_chunk = next(
        chunk for chunk in chunks if chunk.section_label == "pertimbangan_hukum"
    )

    assert reasoning_chunk.text == reasoning
    assert reasoning_chunk.embedding_text == (
        "perkara terdakwa: Siti Aminah. bagian dokumen: pertimbangan hukum.\n"
        + reasoning
    )
    assert text[reasoning_chunk.start_position : reasoning_chunk.end_position] == reasoning


def test_structure_aware_adaptive_context_limits_identity_to_reasoning() -> None:
    identity = "nama lengkap Siti Aminah tempat lahir Malang umur 30 tahun"
    charge = "terdakwa didakwa melakukan tindak pidana"
    reasoning = "menimbang bahwa pasal tersebut telah terpenuhi"
    text = "\n".join((identity, charge, reasoning))
    charge_start = text.index(charge)
    reasoning_start = text.index(reasoning)
    sections = [
        {
            "section_label": "identitas_terdakwa",
            "start_position": 0,
            "end_position": len(identity),
            "section_text": identity,
        },
        {
            "section_label": "riwayat_dakwaan",
            "start_position": charge_start,
            "end_position": charge_start + len(charge),
            "section_text": charge,
        },
        {
            "section_label": "pertimbangan_hukum",
            "start_position": reasoning_start,
            "end_position": len(text),
            "section_text": reasoning,
        },
    ]
    chunks = StructureAwareChunker(
        max_words=20,
        overlap_words=0,
        embedding_context="section_reasoning_document",
    ).chunk("doc", text, sections=sections)
    by_label = {chunk.section_label: chunk for chunk in chunks}

    assert "perkara terdakwa: Siti Aminah" not in (
        by_label["riwayat_dakwaan"].embedding_text or ""
    )
    assert "perkara terdakwa: Siti Aminah" in (
        by_label["pertimbangan_hukum"].embedding_text or ""
    )


def test_summary_augmented_structure_chunking_preserves_source_evidence() -> None:
    text = "identitas singkat\nmenimbang bahwa unsur pasal terpenuhi"
    reasoning_start = text.index("menimbang")
    sections = [
        {
            "section_label": "identitas_terdakwa",
            "start_position": 0,
            "end_position": reasoning_start - 1,
            "section_text": text[: reasoning_start - 1],
        },
        {
            "section_label": "pertimbangan_hukum",
            "start_position": reasoning_start,
            "end_position": len(text),
            "section_text": text[reasoning_start:],
        },
    ]
    chunker = SummaryAugmentedChunker(
        StructureAwareChunker(
            max_words=20,
            overlap_words=0,
            overlap_sentences=0,
        ),
        {"doc": "Putusan perkara narkotika terdakwa Siti Aminah."},
        strategy="structure_summary_augmented",
    )

    chunks = chunker.chunk("doc", text, sections=sections)
    reasoning = next(
        chunk for chunk in chunks if chunk.section_label == "pertimbangan_hukum"
    )

    assert reasoning.strategy == "structure_summary_augmented"
    assert reasoning.text == text[reasoning.start_position : reasoning.end_position]
    assert reasoning.embedding_text == (
        "ringkasan dokumen: Putusan perkara narkotika terdakwa Siti Aminah.\n\n"
        + reasoning.text
    )
    assert reasoning.chunk_id != chunks[0].chunk_id


def test_summary_augmented_structure_chunking_can_retain_role_tag() -> None:
    text = "menimbang bahwa unsur pasal terpenuhi"
    sections = [
        {
            "section_label": "pertimbangan_hukum",
            "start_position": 0,
            "end_position": len(text),
            "section_text": text,
        }
    ]
    chunker = SummaryAugmentedChunker(
        StructureAwareChunker(
            max_words=20,
            overlap_words=0,
            overlap_sentences=0,
            embedding_context="section",
        ),
        {"doc": "perkara narkotika terdakwa Siti Aminah"},
        strategy="structure_summary_role_augmented",
    )

    [chunk] = chunker.chunk("doc", text, sections=sections)

    assert chunk.text == text
    assert chunk.embedding_text == (
        "ringkasan dokumen: perkara narkotika terdakwa Siti Aminah\n\n"
        "bagian dokumen: pertimbangan hukum.\n"
        + text
    )


def test_summary_augmented_chunking_requires_bounded_summary() -> None:
    chunker = SummaryAugmentedChunker(
        FixedSizeChunker(max_words=10, overlap_words=0),
        {"doc": "ringkasan terlalu panjang"},
        max_summary_chars=10,
    )

    with pytest.raises(ValueError, match="maximum is 10"):
        chunker.chunk("doc", "teks sumber")


def test_hierarchical_summary_augmentation_uses_matching_section_only() -> None:
    text = "dakwaan terdakwa\nmenimbang pasal"
    reasoning_start = text.index("menimbang")
    sections = [
        {
            "section_label": "riwayat_dakwaan",
            "start_position": 0,
            "end_position": reasoning_start - 1,
            "section_text": text[: reasoning_start - 1],
        },
        {
            "section_label": "pertimbangan_hukum",
            "start_position": reasoning_start,
            "end_position": len(text),
            "section_text": text[reasoning_start:],
        },
    ]
    chunker = HierarchicalSummaryAugmentedChunker(
        StructureAwareChunker(max_words=20, overlap_words=0),
        {"doc": "perkara satu terdakwa siti"},
        {
            "doc": {
                "riwayat_dakwaan": "bagian dokumen riwayat dakwaan",
                "pertimbangan_hukum": (
                    "bagian dokumen pertimbangan hukum; pasal 112 ayat 1"
                ),
            }
        },
    )

    chunks = chunker.chunk("doc", text, sections=sections)

    charge, reasoning = chunks
    assert "pasal 112" not in (charge.embedding_text or "")
    assert "pasal 112 ayat 1" in (reasoning.embedding_text or "")
    assert all(chunk.text == text[chunk.start_position : chunk.end_position] for chunk in chunks)


def test_summary_augmented_chunking_rejects_missing_document_summary() -> None:
    chunker = SummaryAugmentedChunker(
        FixedSizeChunker(max_words=10, overlap_words=0),
        {"other": "ringkasan"},
    )

    with pytest.raises(KeyError, match="Missing document summary"):
        chunker.chunk("doc", "teks sumber")


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(max_words=5, overlap_words=5)

