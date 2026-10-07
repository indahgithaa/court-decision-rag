"""Deterministic answer-evidence questions for normalized Indo-Law XML."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


INDOLAW_V2_FIELDNAMES = [
    "query_id",
    "document_id",
    "question_type",
    "target_section_label",
    "evidence_section_label",
    "answer_scope",
    "query_anchor_scope",
    "extraction_method",
    "quality_flags",
    "question",
    "reference_answer",
    "evidence_start_position",
    "evidence_end_position",
    "difficulty",
    "review_status",
    "notes",
]


def build_indolaw_questions(
    sections: Sequence[Mapping[str, Any]], *, query_prefix: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build four grounded questions per document and return failures separately."""
    by_document: dict[str, dict[str, Mapping[str, Any]]] = {}
    for section in sections:
        by_document.setdefault(str(section["document_id"]), {})[
            str(section["section_label"])
        ] = section
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for document_index, (document_id, document_sections) in enumerate(
        sorted(by_document.items()), start=1
    ):
        try:
            name = _defendant_name(document_sections)
            drafts = (
                _administrative(document_sections, name),
                _charge(document_sections["riwayat_dakwaan"], name),
                _statute(document_sections["pertimbangan_hukum"], name),
                _disposition(document_sections["amar_putusan"], name),
            )
        except (KeyError, ValueError) as error:
            failures.append({"document_id": document_id, "error": str(error)})
            continue
        for question_index, draft in enumerate(drafts, start=1):
            rows.append(
                {
                    "query_id": f"{query_prefix}-{document_index:03d}-{question_index}",
                    "document_id": document_id,
                    "target_section_label": draft[0],
                    "question": draft[1],
                    "reference_answer": draft[2],
                    "evidence_start_position": str(draft[3]),
                    "evidence_end_position": str(draft[4]),
                    "difficulty": draft[5],
                    "review_status": "approved",
                    "notes": "Deterministic normalized-XML benchmark rule; exact span verified.",
                }
            )
    return rows, failures


def build_indolaw_questions_v2(
    sections: Sequence[Mapping[str, Any]], *, query_prefix: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build independent draft question families without all-or-none fallback.

    Unlike the legacy four-question benchmark, birthplace and detention are
    separate question families. Automatic span verification does not constitute
    semantic approval, so every generated row remains ``draft`` for human review.
    """
    by_document: dict[str, dict[str, Mapping[str, Any]]] = {}
    for section in sections:
        by_document.setdefault(str(section["document_id"]), {})[
            str(section["section_label"])
        ] = section

    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for document_index, (document_id, document_sections) in enumerate(
        sorted(by_document.items()), start=1
    ):
        try:
            name = _defendant_name(document_sections)
        except ValueError as error:
            failures.append(
                {
                    "document_id": document_id,
                    "question_type": "all",
                    "error": str(error),
                }
            )
            continue

        builders = (
            (
                "birthplace",
                lambda: _birthplace_v2(
                    document_sections["identitas_terdakwa"], name
                ),
            ),
            (
                "detention",
                lambda: _detention_v2(document_sections, name),
            ),
            (
                "charge",
                lambda: (
                    _charge(document_sections["riwayat_dakwaan"], name),
                    "riwayat_dakwaan",
                    "charge_pattern",
                ),
            ),
            (
                "statute",
                lambda: (
                    _statute(document_sections["pertimbangan_hukum"], name),
                    "pertimbangan_hukum",
                    "statute_pattern",
                ),
            ),
            (
                "disposition",
                lambda: (
                    _disposition(document_sections["amar_putusan"], name),
                    "amar_putusan",
                    "disposition_pattern",
                ),
            ),
        )
        for question_type, builder in builders:
            try:
                draft, evidence_label, extraction_method = builder()
            except (KeyError, ValueError) as error:
                failures.append(
                    {
                        "document_id": document_id,
                        "question_type": question_type,
                        "error": str(error),
                    }
                )
                continue
            target_section = document_sections[evidence_label]
            name_in_target = _clean(name).casefold() in _clean(
                str(target_section["section_text"])
            ).casefold()
            quality_flags = []
            if not name_in_target:
                quality_flags.append("query_name_not_in_evidence_section")
            if evidence_label != draft[0]:
                quality_flags.append("semantic_target_differs_from_corpus_section")
            rows.append(
                {
                    "query_id": (
                        f"{query_prefix}-{document_index:03d}-{question_type}"
                    ),
                    "document_id": document_id,
                    "question_type": question_type,
                    "target_section_label": draft[0],
                    "evidence_section_label": evidence_label,
                    "answer_scope": "single_section",
                    "query_anchor_scope": (
                        "target_section" if name_in_target else "external_identity"
                    ),
                    "extraction_method": extraction_method,
                    "quality_flags": ";".join(quality_flags),
                    "question": draft[1],
                    "reference_answer": draft[2],
                    "evidence_start_position": str(draft[3]),
                    "evidence_end_position": str(draft[4]),
                    "difficulty": draft[5],
                    "review_status": "draft",
                    "notes": (
                        "Deterministic normalized-XML draft; exact span verified; "
                        "human semantic, extraction, and name review required."
                    ),
                }
            )
    return rows, failures


def _defendant_name(sections: Mapping[str, Mapping[str, Any]]) -> str:
    sources = (
        (
            "identitas_terdakwa",
            r"\b(?:\d+\s+)?nama(?: lengkap)?\s+(.+?)"
            r"(?=\s+(?:\d+\s+)?tempat lahir\b)",
        ),
        (
            "amar_putusan",
            r"\bmenyatakan terdakwa\s+(.+?)"
            r"(?=\s+(?:tersebut\s+)?(?:telah\s+)?terbukti\b)",
        ),
        (
            "riwayat_tuntutan",
            r"\bmenyatakan terdakwa\s+(.+?)(?=\s+(?:telah\s+)?bersalah\b)",
        ),
        (
            "riwayat_dakwaan",
            r"\bbahwa (?:ia )?terdakwa\s+(.+?)"
            r"(?=\s+(?:pada|bersama sama|telah|yang)\b)",
        ),
    )
    for label, pattern in sources:
        section = sections.get(label)
        if section is None:
            continue
        match = re.search(pattern, str(section["section_text"]), flags=re.IGNORECASE)
        if match:
            candidate = _clean(match.group(1)).strip("0123456789 ")
            if len(re.findall(r"[a-z]+", candidate, flags=re.IGNORECASE)) >= 2:
                return candidate
    raise ValueError("missing defendant name")


def _administrative(
    sections: Mapping[str, Mapping[str, Any]], name: str
) -> tuple[str, str, str, int, int, str]:
    try:
        return _birthplace(sections["identitas_terdakwa"], name)
    except ValueError:
        return _detention(sections["riwayat_penahanan"], name)


def _birthplace(section: Mapping[str, Any], name: str) -> tuple[str, str, str, int, int, str]:
    match = re.search(
        r"\b(?:\d+\s+)?tempat lahir\s+(.+?)"
        r"(?=\s+(?:\d+\s+)?(?:umur(?:\s+atau)?(?:\s+(?:tanggal|tgl))?|tanggal|tangal|tgl) lahir\b)",
        str(section["section_text"]),
        flags=re.IGNORECASE,
    )
    return _draft(
        section,
        match,
        "identitas_terdakwa",
        f"Di mana {name} dilahirkan?",
        "easy",
        answer_group=1,
    )


def _birthplace_v2(
    section: Mapping[str, Any], name: str
) -> tuple[tuple[str, str, str, int, int, str], str, str]:
    """Extract a conservative birthplace draft and reject field-label leakage."""
    text = str(section["section_text"])
    direct = re.search(
        r"\b(?:\d+\s+)?tempat lahir\s+(.+?)"
        r"(?=\s+(?:\d+\s+)?(?:umur|tanggal|tangal|tgl)\b)",
        text,
        flags=re.IGNORECASE,
    )
    if direct is not None and _valid_birthplace(direct.group(1)):
        return (
            _draft(
                section,
                direct,
                "identitas_terdakwa",
                f"Di mana {name} dilahirkan?",
                "easy",
                answer_group=1,
            ),
            "identitas_terdakwa",
            "birthplace_inline_label",
        )

    # Some Indo-Law XML records serialize all field labels before their values.
    # In those records a reliable recovery is possible only when the extracted
    # defendant name is followed immediately by a short place and an age.
    adjacent = re.search(
        rf"\b{re.escape(_clean(name))}\s+([a-z][a-z .'-]{{1,60}}?)"
        r"(?=\s+\d{1,3}\s+tahun\b)",
        _clean(text),
        flags=re.IGNORECASE,
    )
    if adjacent is not None and _valid_birthplace(adjacent.group(1)):
        # Normalization only collapses whitespace. Locate the same candidate in
        # the original section so the gold offsets still address source text.
        candidate = _clean(adjacent.group(1))
        source_match = re.search(
            rf"\b{re.escape(candidate)}(?=\s+\d{{1,3}}\s+tahun\b)",
            text,
            flags=re.IGNORECASE,
        )
        if source_match is not None:
            return (
                _draft(
                    section,
                    source_match,
                    "identitas_terdakwa",
                    f"Di mana {name} dilahirkan?",
                    "easy",
                ),
                "identitas_terdakwa",
                "birthplace_name_adjacent",
            )
    raise ValueError("missing high-confidence evidence for identitas_terdakwa")


def _valid_birthplace(value: str) -> bool:
    candidate = _clean(value).casefold()
    words = candidate.split()
    forbidden = {
        "atau",
        "jenis",
        "kebangsaan",
        "kelamin",
        "lahir",
        "tanggal",
        "tangal",
        "tempat",
        "tahun",
        "tgl",
        "tinggal",
        "umur",
    }
    return (
        1 <= len(words) <= 5
        and not any(word in forbidden for word in words)
        and re.fullmatch(r"[a-z][a-z .'-]*", candidate, flags=re.IGNORECASE)
        is not None
    )


def _detention(
    section: Mapping[str, Any],
    name: str,
    *,
    max_answer_words: int = 28,
) -> tuple[str, str, str, int, int, str]:
    text = str(section["section_text"])
    match = re.search(
        r"\b(?:\d+\s+)?penyidik(?!\s+perpanjangan)[\s\S]{0,120}?"
        r"sejak tanggal\s+[\s\S]{1,50}?(?:sampai dengan|s d)\s+tanggal\s+[\s\S]{1,50}?"
        r"(?=\s+\d+\s+|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.search(
            r"\bterdakwa[\s\S]{0,220}?\b(?:ditangkap|ditahan)\b[\s\S]{0,180}",
            text,
            flags=re.IGNORECASE,
        )
    return _draft(
        section,
        match,
        "riwayat_penahanan",
        f"Bagaimana riwayat awal penahanan {name}?",
        "easy",
        max_answer_words=max_answer_words,
    )


def _detention_v2(
    sections: Mapping[str, Mapping[str, Any]], name: str
) -> tuple[tuple[str, str, str, int, int, str], str, str]:
    """Find initial detention even when the XML tag boundary splits it."""
    detention = sections.get("riwayat_penahanan")
    if detention is not None:
        try:
            return (
                _detention(detention, name, max_answer_words=40),
                "riwayat_penahanan",
                "detention_section",
            )
        except ValueError:
            pass
    identity = sections.get("identitas_terdakwa")
    if identity is not None:
        return (
            _detention(identity, name, max_answer_words=40),
            "identitas_terdakwa",
            "detention_embedded_in_identity",
        )
    raise ValueError("missing evidence for riwayat_penahanan")


def _charge(section: Mapping[str, Any], name: str) -> tuple[str, str, str, int, int, str]:
    text = str(section["section_text"])
    match = re.search(
        r"(?:tanpa hak atau melawan hukum|tanpa hak)[\s\S]{20,700}?"
        r"(?=\s+perbuatan (?:terdakwa|tersebut) dilakukan|\s+sebagaimana diatur|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.search(r"bahwa (?:ia )?terdakwa[\s\S]{40,500}", text, re.IGNORECASE)
    return _draft(
        section,
        match,
        "riwayat_dakwaan",
        f"Perbuatan apa yang didakwakan kepada {name}?",
        "medium",
    )


def _statute(section: Mapping[str, Any], name: str) -> tuple[str, str, str, int, int, str]:
    text = str(section["section_text"])
    matches = list(
        re.finditer(
            r"\bpasal\s+\d+[\s\S]{0,260}?"
            r"(?:undang undang(?: republik indonesia)?|uu(?: ri)?)\s+"
            r"(?:nomor|no)\s+35\s+tahun\s+2009\s+(?:tentang|tetang)\s+narkotika",
            text,
            flags=re.IGNORECASE,
        )
    )
    match = _most_discussed_statute(matches)
    if match is None:
        fallback = list(
            re.finditer(
                r"\bpasal\s+\d+[\s\S]{0,220}?(?=\s+yang unsur|\s+unsur unsurnya|$)",
                text,
                flags=re.IGNORECASE,
            )
        )
        match = fallback[-1] if fallback else None
    return _draft(
        section,
        match,
        "pertimbangan_hukum",
        f"Ketentuan pidana apa yang dipertimbangkan dalam perkara {name}?",
        "hard",
    )


def _most_discussed_statute(matches: Sequence[re.Match[str]]) -> re.Match[str] | None:
    if not matches:
        return None
    keyed: list[tuple[re.Match[str], str]] = []
    for match in matches:
        key_match = re.search(
            r"\bpasal\s+\d+(?:\s+ayat\s+\d+)?(?:\s+huruf\s+[a-z])?",
            match.group(0),
            flags=re.IGNORECASE,
        )
        if key_match:
            keyed.append((match, _clean(key_match.group(0)).lower()))
    if not keyed:
        return matches[0]
    non_definition = [(match, key) for match, key in keyed if not key.startswith("pasal 1 ")]
    candidates = non_definition or keyed
    counts: dict[str, int] = {}
    for _, key in candidates:
        counts[key] = counts.get(key, 0) + 1
    winning_key = max(counts, key=lambda key: (counts[key], -next(i for i, (_, value) in enumerate(candidates) if value == key)))
    return next(match for match, key in candidates if key == winning_key)


def _disposition(section: Mapping[str, Any], name: str) -> tuple[str, str, str, int, int, str]:
    text = str(section["section_text"])
    patterns = (
        r"\bmenjatuhkan pidana[\s\S]{10,450}?(?=\s+\d+\s+(?:menetapkan|memerintahkan|membebankan)|$)",
        r"\bmembebaskan terdakwa[\s\S]{0,180}?(?=\s+\d+\s+(?:menetapkan|memerintahkan|membebankan)|$)",
        r"\bmenyatakan terdakwa[\s\S]{20,300}?(?=\s+\d+\s+(?:menjatuhkan|menetapkan|memerintahkan)|$)",
    )
    match = next((found for pattern in patterns if (found := re.search(pattern, text, re.IGNORECASE))), None)
    if match is None:
        match = re.search(r"\S(?:[\s\S]{19,349}|[\s\S]*$)", text, re.IGNORECASE)
    return _draft(
        section,
        match,
        "amar_putusan",
        f"Apa isi amar putusan dalam perkara {name}?",
        "easy",
    )


def _draft(
    section: Mapping[str, Any],
    match: re.Match[str] | None,
    label: str,
    question: str,
    difficulty: str,
    answer_group: int | None = None,
    max_answer_words: int = 28,
) -> tuple[str, str, str, int, int, str]:
    if match is None:
        raise ValueError(f"missing evidence for {label}")
    text = str(section["section_text"])
    start, end = match.span(answer_group) if answer_group is not None else match.span()
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    words = list(re.finditer(r"\S+", text[start:end]))
    if len(words) > max_answer_words:
        end = start + words[max_answer_words - 1].end()
    answer = _clean(text[start:end])
    if not answer:
        raise ValueError(f"empty evidence for {label}")
    offset = int(section["start_position"])
    return label, question, answer, offset + start, offset + end, difficulty


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
