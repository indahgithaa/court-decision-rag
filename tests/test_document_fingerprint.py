import pytest

from src.summarization import (
    build_hierarchical_extractive_contexts,
    build_representative_document,
    build_structured_extractive_fingerprint,
    truncate_at_word_boundary,
)


def test_representative_document_balances_legal_sections() -> None:
    sections = [
        {
            "section_label": "kepala_putusan",
            "section_text": "putusan nomor 1 pengadilan negeri malang",
        },
        {
            "section_label": "pertimbangan_hukum",
            "section_text": "awal " + "tengah " * 100 + "kesimpulan akhir",
        },
        {
            "section_label": "unknown",
            "section_text": "tidak dipilih",
        },
    ]

    result = build_representative_document(sections, per_section_chars=200)

    assert "[kepala putusan]" in result
    assert "[pertimbangan hukum]" in result
    assert "awal" in result
    assert "kesimpulan akhir" in result
    assert "tidak dipilih" not in result
    assert "[...]" in result


def test_representative_document_requires_supported_section() -> None:
    with pytest.raises(ValueError, match="No supported legal sections"):
        build_representative_document(
            [{"section_label": "unknown", "section_text": "teks"}]
        )


def test_truncate_at_word_boundary_preserves_complete_word() -> None:
    assert truncate_at_word_boundary("satu dua tiga", 9) == "satu dua"


def test_truncate_at_word_boundary_rejects_nonpositive_limit() -> None:
    with pytest.raises(ValueError):
        truncate_at_word_boundary("teks", 0)


def test_structured_extractive_fingerprint_is_grounded_and_bounded() -> None:
    sections = [
        {
            "section_label": "kepala_putusan",
            "section_text": (
                "putusan nomor 12 pid sus 2025 pn malang demi keadilan dalam "
                "perkara terdakwa sebagai berikut siti aminah"
            ),
        },
        {
            "section_label": "pertimbangan_hukum",
            "section_text": "narkotika berdasarkan pasal 112 ayat 1 telah terpenuhi",
        },
        {
            "section_label": "amar_putusan",
            "section_text": "menjatuhkan pidana penjara selama empat tahun kepada terdakwa",
        },
    ]

    summary, metadata = build_structured_extractive_fingerprint(sections)

    assert len(summary) <= 170
    assert "12 pid sus 2025 pn malang" in summary
    assert "siti aminah" in summary
    assert "pasal 112 ayat 1" in summary
    assert metadata["decoding"] == "deterministic_source_extraction"


def test_fingerprint_prefers_clean_name_when_identity_fields_are_reordered() -> None:
    sections = [
        {
            "section_label": "identitas_terdakwa",
            "section_text": (
                "nama lengkap sopian sopandi alias pian bin salman bandung "
                "20 tahun 23 oktober 1994 tempat lahir umur tanggal lahir"
            ),
        },
        {
            "section_label": "riwayat_tuntutan",
            "section_text": (
                "menyatakan terdakwa sopian sopandi alias pian bin salman "
                "bersalah melakukan tindak pidana narkotika"
            ),
        },
        {
            "section_label": "amar_putusan",
            "section_text": (
                "menjatuhkan pidana kepada terdakwa dengan pidana penjara "
                "selama empat tahun"
            ),
        },
    ]

    summary, _ = build_structured_extractive_fingerprint(sections)

    assert "terdakwa sopian sopandi alias pian bin salman" in summary
    assert "23 oktober" not in summary


def test_fingerprint_prioritizes_sentence_over_guilt_declaration() -> None:
    sections = [
        {
            "section_label": "amar_putusan",
            "section_text": (
                "menyatakan terdakwa budi terbukti secara sah dan meyakinkan "
                "bersalah menjatuhkan pidana kepada terdakwa dengan pidana "
                "penjara selama lima tahun"
            ),
        }
    ]

    summary, _ = build_structured_extractive_fingerprint(sections)

    assert "menjatuhkan pidana" in summary


def test_fingerprint_prefers_final_controlling_statute() -> None:
    sections = [
        {
            "section_label": "pertimbangan_hukum",
            "section_text": (
                "pasal 55 ayat 1 disebut beberapa kali pasal 55 ayat 1 "
                "memperhatikan pasal 127 ayat 1 huruf a jo pasal 55 ayat 1"
            ),
        }
    ]

    summary, _ = build_structured_extractive_fingerprint(sections)

    assert "pasal 127 ayat 1 huruf a jo pasal 55 ayat 1" in summary


def test_hierarchical_context_keeps_legal_content_in_its_section() -> None:
    sections = [
        {
            "section_label": "kepala_putusan",
            "section_text": "putusan nomor 1 pid sus demi keadilan",
        },
        {
            "section_label": "identitas_terdakwa",
            "section_text": "nama lengkap siti aminah tempat lahir malang",
        },
        {
            "section_label": "riwayat_dakwaan",
            "section_text": (
                "terdakwa tanpa hak atau melawan hukum memiliki narkotika "
                "golongan satu sebagaimana dakwaan"
            ),
        },
        {
            "section_label": "pertimbangan_hukum",
            "section_text": (
                "memperhatikan pasal 112 ayat 1 undang undang narkotika"
            ),
        },
        {
            "section_label": "amar_putusan",
            "section_text": (
                "menjatuhkan pidana kepada terdakwa dengan pidana penjara "
                "selama empat tahun"
            ),
        },
    ]

    document, contexts = build_hierarchical_extractive_contexts(sections)

    assert "siti aminah" in document
    assert "pasal 112 ayat 1" not in document
    assert "pasal 112 ayat 1" in contexts["pertimbangan_hukum"]
    assert "pasal 112 ayat 1" not in contexts["amar_putusan"]
    assert "menjatuhkan pidana" in contexts["amar_putusan"]
