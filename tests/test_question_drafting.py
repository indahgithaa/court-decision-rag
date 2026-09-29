from src.evaluation.ground_truth import validate_questions
from src.evaluation.question_drafting import draft_questions


def _section(document_id: str, label: str, text: str, start: int) -> dict[str, object]:
    return {
        "document_id": document_id,
        "section_label": label,
        "section_text": text,
        "start_position": start,
        "end_position": start + len(text),
    }


def test_draft_questions_are_grounded_and_remain_unapproved() -> None:
    identity = (
        "1. Nama lengkap : BUDI BIN AMIR; 2. Tempat lahir : Palu; "
        "3. Umur/Tanggal lahir : 30 tahun/1 Januari 1994;"
    )
    disposition = (
        "MENGADILI: 1. Menyatakan Terdakwa bersalah; 2. Menjatuhkan pidana "
        "kepada Terdakwa dengan pidana penjara selama 2 (dua) tahun;"
    )
    separator = "\n\n"
    document = identity + separator + disposition
    rows = [
        {
            "query_id": "pilot-01-1",
            "document_id": "doc-1",
            "target_section_label": "identitas_terdakwa",
            "question": "",
            "reference_answer": "",
            "evidence_start_position": "",
            "evidence_end_position": "",
            "difficulty": "",
            "review_status": "",
            "notes": "",
        },
        {
            "query_id": "pilot-01-4",
            "document_id": "doc-1",
            "target_section_label": "amar_putusan",
            "question": "",
            "reference_answer": "",
            "evidence_start_position": "",
            "evidence_end_position": "",
            "difficulty": "",
            "review_status": "",
            "notes": "",
        },
    ]
    sections = [
        _section("doc-1", "identitas_terdakwa", identity, 0),
        _section("doc-1", "amar_putusan", disposition, len(identity + separator)),
    ]

    drafts = draft_questions(rows, sections)

    assert [row["review_status"] for row in drafts] == ["draft", "draft"]
    assert drafts[0]["reference_answer"] == "Palu"
    for row in drafts:
        start = int(row["evidence_start_position"])
        end = int(row["evidence_end_position"])
        assert document[start:end].strip()
    result = validate_questions(drafts, {"doc-1": document})
    assert result["valid"] is True
    assert result["ready"] is False
    assert result["status_counts"] == {"draft": 2}


def test_detention_draft_does_not_confuse_arrest_with_detention() -> None:
    identity = (
        "1. Nama lengkap : BUDI BIN AMIR; 2. Tempat lahir : Palu; "
        "3. Umur/Tanggal lahir : 30 tahun/1 Januari 1994;"
    )
    detention = (
        "Terdakwa ditangkap oleh Penyidik sejak tanggal 1 Januari 2024 sampai "
        "dengan tanggal 2 Januari 2024; Terdakwa ditahan oleh: 1. Penyidik "
        "sejak tanggal 3 Januari 2024 sampai dengan tanggal 22 Januari 2024;"
    )
    rows = [{
        "query_id": "pilot-01-1",
        "document_id": "doc-1",
        "target_section_label": "riwayat_penahanan",
        "question": "",
        "reference_answer": "",
        "evidence_start_position": "",
        "evidence_end_position": "",
        "difficulty": "",
        "review_status": "",
        "notes": "",
    }]
    sections = [
        _section("doc-1", "identitas_terdakwa", identity, 0),
        _section("doc-1", "riwayat_penahanan", detention, len(identity) + 2),
    ]

    [draft] = draft_questions(rows, sections)

    assert "3 Januari 2024" in draft["reference_answer"]
    assert "1 Januari 2024" not in draft["reference_answer"]
