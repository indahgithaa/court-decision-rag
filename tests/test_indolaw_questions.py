from src.evaluation.indolaw_questions import (
    build_indolaw_questions,
    build_indolaw_questions_v2,
)


def _section(label: str, text: str, start: int) -> dict:
    return {
        "document_id": "doc-1",
        "section_label": label,
        "section_text": text,
        "start_position": start,
        "end_position": start + len(text),
    }


def test_build_indolaw_questions_creates_exact_global_spans() -> None:
    sections = [
        _section(
            "identitas_terdakwa",
            "nama lengkap budi bin aman tempat lahir bandung umur tanggal lahir 30 tahun",
            100,
        ),
        _section(
            "riwayat_dakwaan",
            "bahwa terdakwa tanpa hak atau melawan hukum memiliki narkotika golongan i "
            "perbuatan terdakwa dilakukan dengan cara berikut",
            300,
        ),
        _section(
            "pertimbangan_hukum",
            "majelis mempertimbangkan pasal 112 ayat 1 undang undang nomor 35 tahun 2009 "
            "tentang narkotika",
            600,
        ),
        _section(
            "amar_putusan",
            "mengadili 1 menjatuhkan pidana penjara selama empat tahun 2 menetapkan barang bukti",
            900,
        ),
    ]

    rows, failures = build_indolaw_questions(sections, query_prefix="test")

    assert not failures
    assert len(rows) == 4
    assert rows[0]["reference_answer"] == "bandung"
    assert rows[0]["evidence_start_position"] == "140"
    assert rows[-1]["target_section_label"] == "amar_putusan"
    assert rows[-1]["review_status"] == "approved"


def test_name_falls_back_when_identity_layout_is_broken() -> None:
    sections = [
        _section("identitas_terdakwa", "1 nama lengkap 2 tempat lahir", 0),
        _section("riwayat_penahanan", "1 penyidik sejak tanggal 1 januari 2020 sampai dengan tanggal 2 januari 2020", 50),
        _section("riwayat_dakwaan", "bahwa ia terdakwa budi santoso bin aman pada hari senin tanpa hak memiliki narkotika golongan i", 150),
        _section("pertimbangan_hukum", "pasal 112 ayat 1 uu ri no 35 tahun 2009 tentang narkotika", 300),
        _section("amar_putusan", "mengadili menyatakan terdakwa budi santoso bin aman terbukti bersalah", 400),
    ]

    rows, failures = build_indolaw_questions(sections, query_prefix="fallback")

    assert not failures
    assert "budi santoso bin aman" in rows[0]["question"]


def test_v2_keeps_birthplace_and_detention_as_independent_drafts() -> None:
    sections = [
        _section(
            "identitas_terdakwa",
            "nama lengkap budi bin aman tempat lahir bandung umur tanggal lahir 30 tahun",
            100,
        ),
        _section(
            "riwayat_penahanan",
            "1 penyidik sejak tanggal 1 januari 2020 sampai dengan tanggal 2 januari 2020",
            200,
        ),
        _section(
            "riwayat_dakwaan",
            "bahwa terdakwa tanpa hak atau melawan hukum memiliki narkotika golongan i "
            "perbuatan terdakwa dilakukan dengan cara berikut",
            300,
        ),
        _section(
            "pertimbangan_hukum",
            "majelis mempertimbangkan pasal 112 ayat 1 undang undang nomor 35 tahun 2009 "
            "tentang narkotika",
            600,
        ),
        _section(
            "amar_putusan",
            "mengadili 1 menjatuhkan pidana penjara selama empat tahun 2 menetapkan barang bukti",
            900,
        ),
    ]

    rows, failures = build_indolaw_questions_v2(sections, query_prefix="v2")

    assert not failures
    assert {row["question_type"] for row in rows} == {
        "birthplace",
        "detention",
        "charge",
        "statute",
        "disposition",
    }
    assert len(rows) == 5
    assert all(row["review_status"] == "draft" for row in rows)
    assert all(row["answer_scope"] == "single_section" for row in rows)
    assert all(row["query_anchor_scope"] == "external_identity" for row in rows[1:])


def test_v2_keeps_other_families_when_detention_evidence_is_missing() -> None:
    sections = [
        _section(
            "identitas_terdakwa",
            "nama lengkap budi bin aman tempat lahir bandung umur tanggal lahir 30 tahun",
            100,
        ),
        _section("riwayat_penahanan", "tidak terdapat tanggal penahanan", 200),
        _section(
            "riwayat_dakwaan",
            "bahwa terdakwa tanpa hak atau melawan hukum memiliki narkotika golongan i "
            "perbuatan terdakwa dilakukan dengan cara berikut",
            300,
        ),
        _section(
            "pertimbangan_hukum",
            "pasal 112 ayat 1 undang undang nomor 35 tahun 2009 tentang narkotika",
            600,
        ),
        _section(
            "amar_putusan",
            "mengadili 1 menjatuhkan pidana penjara selama empat tahun 2 menetapkan barang bukti",
            900,
        ),
    ]

    rows, failures = build_indolaw_questions_v2(sections, query_prefix="v2")

    assert len(rows) == 4
    assert "detention" not in {row["question_type"] for row in rows}
    assert failures == [
        {
            "document_id": "doc-1",
            "question_type": "detention",
            "error": "missing evidence for riwayat_penahanan",
        }
    ]


def test_v2_recovers_detention_embedded_in_identity_with_quality_flag() -> None:
    sections = [
        _section(
            "identitas_terdakwa",
            "nama lengkap budi bin aman tempat lahir bandung umur tanggal lahir 30 tahun "
            "terdakwa ditahan dalam tahanan rutan oleh penyidik sejak tanggal 1 januari "
            "2020 sampai dengan tanggal 2 januari 2020",
            100,
        ),
        _section(
            "riwayat_penahanan",
            "perpanjangan ketua pengadilan negeri sampai tanggal 3 februari 2020",
            300,
        ),
        _section(
            "riwayat_dakwaan",
            "bahwa terdakwa tanpa hak atau melawan hukum memiliki narkotika golongan i "
            "perbuatan terdakwa dilakukan dengan cara berikut",
            400,
        ),
        _section(
            "pertimbangan_hukum",
            "pasal 112 ayat 1 undang undang nomor 35 tahun 2009 tentang narkotika",
            700,
        ),
        _section(
            "amar_putusan",
            "mengadili 1 menjatuhkan pidana penjara selama empat tahun 2 menetapkan barang bukti",
            900,
        ),
    ]

    rows, failures = build_indolaw_questions_v2(sections, query_prefix="v2")
    detention = next(row for row in rows if row["question_type"] == "detention")

    assert not failures
    assert detention["target_section_label"] == "riwayat_penahanan"
    assert detention["evidence_section_label"] == "identitas_terdakwa"
    assert detention["extraction_method"] == "detention_embedded_in_identity"
    assert "semantic_target_differs_from_corpus_section" in detention["quality_flags"]


def test_v2_rejects_birthplace_values_that_are_field_labels() -> None:
    sections = [
        _section(
            "identitas_terdakwa",
            "nama lengkap budi bin aman tempat lahir umur tanggal lahir jenis kelamin",
            100,
        ),
        _section(
            "riwayat_penahanan",
            "1 penyidik sejak tanggal 1 januari 2020 sampai dengan tanggal 2 januari 2020",
            200,
        ),
        _section(
            "riwayat_dakwaan",
            "bahwa terdakwa tanpa hak atau melawan hukum memiliki narkotika golongan i "
            "perbuatan terdakwa dilakukan dengan cara berikut",
            300,
        ),
        _section(
            "pertimbangan_hukum",
            "pasal 112 ayat 1 undang undang nomor 35 tahun 2009 tentang narkotika",
            600,
        ),
        _section(
            "amar_putusan",
            "mengadili 1 menjatuhkan pidana penjara selama empat tahun 2 menetapkan barang bukti",
            900,
        ),
    ]

    rows, failures = build_indolaw_questions_v2(sections, query_prefix="v2")

    assert "birthplace" not in {row["question_type"] for row in rows}
    assert any(
        failure["question_type"] == "birthplace"
        and "high-confidence" in failure["error"]
        for failure in failures
    )
