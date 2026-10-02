from src.evaluation.indolaw_questions import build_indolaw_questions


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
