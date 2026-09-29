"""Evidence-grounded draft questions for the retrieval pilot.

The rules in this module deliberately create *drafts*, not gold annotations.
They make the repetitive first pass reproducible while leaving semantic review
and approval to a human annotator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


QUESTION_FIELDNAMES = [
    "query_id",
    "document_id",
    "target_section_label",
    "question",
    "reference_answer",
    "evidence_start_position",
    "evidence_end_position",
    "difficulty",
    "review_status",
    "notes",
]


@dataclass(frozen=True)
class QuestionDraft:
    """A proposed question tied to one exact span in the cleaned document."""

    question: str
    reference_answer: str
    evidence_start_position: int
    evidence_end_position: int
    difficulty: str
    notes: str


def draft_question(
    section: Mapping[str, Any],
    *,
    defendant_name: str,
) -> QuestionDraft:
    """Create one deterministic draft for a supported rhetorical section."""
    label = str(section["section_label"])
    handlers = {
        "identitas_terdakwa": _draft_identity,
        "riwayat_penahanan": _draft_detention,
        "fakta": _draft_facts,
        "pertimbangan_hukum": _draft_reasoning,
        "amar_putusan": _draft_disposition,
    }
    try:
        handler = handlers[label]
    except KeyError as error:
        raise ValueError(f"Unsupported question label: {label}") from error
    return handler(section, defendant_name)


def extract_defendant_name(identity_section: Mapping[str, Any]) -> str:
    """Extract the name value from the identity block without changing offsets."""
    text = str(identity_section["section_text"])
    match = re.search(
        r"Nama\s+lengkap\s*:\s*(?P<value>.*?)"
        r"(?=\s*;?\s*(?:2\.\s*)?Tempat\s+lahir\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError("Identity section has no recognizable 'Nama lengkap' field")
    return _clean(match.group("value"))


def draft_questions(
    rows: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Populate template rows with drafts while preserving stable query IDs."""
    sections_by_key = {
        (str(section["document_id"]), str(section["section_label"])): section
        for section in sections
    }
    names: dict[str, str] = {}
    output: list[dict[str, str]] = []
    for row in rows:
        document_id = str(row["document_id"])
        label = str(row["target_section_label"])
        identity = sections_by_key.get((document_id, "identitas_terdakwa"))
        section = sections_by_key.get((document_id, label))
        if identity is None or section is None:
            raise ValueError(f"Missing section for {document_id}: {label}")
        if document_id not in names:
            names[document_id] = extract_defendant_name(identity)
        draft = draft_question(section, defendant_name=names[document_id])
        populated = {key: str(value) for key, value in row.items()}
        populated.update(
            {
                "question": draft.question,
                "reference_answer": draft.reference_answer,
                "evidence_start_position": str(draft.evidence_start_position),
                "evidence_end_position": str(draft.evidence_end_position),
                "difficulty": draft.difficulty,
                "review_status": "draft",
                "notes": draft.notes,
            }
        )
        output.append(populated)
    return output


def _draft_identity(section: Mapping[str, Any], name: str) -> QuestionDraft:
    text = str(section["section_text"])
    match = re.search(
        r"Tempat\s+lahir\s*:\s*(?P<value>.*?)"
        r"(?=\s*;?\s*(?:3\.\s*)?Umur\s*/?\s*(?:Tanggal|Tgl\.?)\s*\.?\s*lahir\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError("Identity section has no recognizable birthplace field")
    return _from_match(
        section,
        match,
        question=f"Di mana {name} dilahirkan?",
        answer=_clean(match.group("value")),
        difficulty="easy",
        rule="identity.birthplace",
    )


def _draft_detention(section: Mapping[str, Any], name: str) -> QuestionDraft:
    text = str(section["section_text"])
    match = re.search(
        r"(?:\d+\.\s*)?Penyidik(?!\s+Perpanjangan)[^;]{0,100}?"
        r"sejak\s+tanggal\s+[^;]{1,80}?sampai\s+dengan\s+tanggal\s+[^;.]{1,60}[;.]?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"Terdakwa[^;]{0,220}?(?:ditahan|ditangkap)[^;]{0,220}[;.]",
            text,
            flags=re.IGNORECASE,
        )
    if not match:
        raise ValueError("Detention section has no recognizable detention period")
    return _from_match(
        section,
        match,
        question=(
            "Kapan masa penahanan pada tingkat penyidikan untuk perkara "
            f"{name} dimulai dan berakhir?"
        ),
        answer=_clean(match.group(0)).rstrip(";"),
        difficulty="easy",
        rule="detention.investigator_period",
    )


def _draft_facts(section: Mapping[str, Any], name: str) -> QuestionDraft:
    text = str(section["section_text"])
    search_area = text[: min(len(text), 8000)]
    match = re.search(
        r"(?:Tanpa\s+hak|tanpa\s+hak|percobaan\s+atau\s+permufakatan\s+jahat)"
        r"[^;]{25,1200}?(?=,?\s*(?:perbuatan(?:\s+mana)?|yang\s+dilakukan)\b|;)",
        search_area,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        match = re.search(
            r"Bahwa\s+(?:ia\s+)?Terdakwa[^;]{80,900}?(?=;)",
            search_area,
            flags=re.IGNORECASE | re.DOTALL,
        )
    if not match:
        raise ValueError("Facts section has no recognizable charge narrative")
    return _from_match(
        section,
        match,
        question=(
            "Perbuatan pidana apa yang didakwakan kepada "
            f"{name} dalam uraian fakta perkara?"
        ),
        answer=_clean(match.group(0)).strip(" ,\"“”�"),
        difficulty="medium",
        rule="facts.charged_conduct",
    )


def _draft_reasoning(section: Mapping[str, Any], name: str) -> QuestionDraft:
    text = str(section["section_text"])
    search_area = text[: min(len(text), 9000)]
    match = re.search(
        r"Menimbang,?\s+bahwa\s+Terdakwa\s+telah\s+didakwa[^;]{30,1200}?"
        r"(?:Pasal|pasal)[^;]{5,400}?;",
        search_area,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        matches = list(
            re.finditer(
                r"(?:Memperhatikan|Mengingat),?[^;]{0,700}?(?:Pasal|pasal)[^;]*;?",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        match = matches[-1] if matches else None
    if not match:
        raise ValueError("Reasoning section has no recognizable selected charge or statute")
    return _from_match(
        section,
        match,
        question=(
            "Dakwaan atau ketentuan pidana mana yang dipilih Majelis Hakim "
            f"ketika menilai perkara {name}?"
        ),
        answer=_clean(match.group(0)).rstrip(";"),
        difficulty="hard",
        rule="reasoning.selected_charge",
    )


def _draft_disposition(section: Mapping[str, Any], name: str) -> QuestionDraft:
    text = str(section["section_text"])
    match = re.search(
        r"(?:\d+\.\s*)?Menjatuhkan\s+pidana[^;]{20,900}?;",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError("Disposition section has no recognizable sentence")
    return _from_match(
        section,
        match,
        question=f"Pidana apa yang dijatuhkan kepada {name}?",
        answer=_clean(match.group(0)).rstrip(";"),
        difficulty="easy",
        rule="disposition.sentence",
    )


def _from_match(
    section: Mapping[str, Any],
    match: re.Match[str],
    *,
    question: str,
    answer: str,
    difficulty: str,
    rule: str,
) -> QuestionDraft:
    start, end = _trim_span(str(section["section_text"]), match.start(), match.end())
    section_start = int(section["start_position"])
    return QuestionDraft(
        question=question,
        reference_answer=answer,
        evidence_start_position=section_start + start,
        evidence_end_position=section_start + end,
        difficulty=difficulty,
        notes=(
            f"Auto-draft rule={rule}; verify wording, minimal evidence, and ambiguity "
            "before changing review_status to approved."
        ),
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ;")
