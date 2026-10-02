"""Metadata parsing for the external Indo-Law court-decision corpus."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from typing import Any


REQUIRED_SECTIONS = (
    "identitas",
    "riwayat_penahanan",
    "fakta",
    "pertimbangan_hukum",
    "amar_putusan",
)


def parse_indolaw_xml(xml_text: str, *, source_path: str) -> dict[str, Any]:
    """Extract non-text metadata needed for auditable holdout selection."""
    root = ET.fromstring(xml_text)
    if root.tag != "putusan":
        raise ValueError(f"Unexpected root tag in {source_path}: {root.tag}")
    heading = _element_text(root.find("kepala_putusan"))
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", heading)]
    case_number = _case_number(heading)
    section_lengths = {
        tag: len(_element_text(root.find(tag))) for tag in REQUIRED_SECTIONS
    }
    return {
        "source_path": source_path,
        "source_id": root.attrib.get("id", PurePosixPath(source_path).stem),
        "source_url": root.attrib.get("url", ""),
        "classification": root.attrib.get("klasifikasi", ""),
        "subclassification": root.attrib.get("sub_klasifikasi", ""),
        "court": root.attrib.get("lembaga_peradilan", ""),
        "province": root.attrib.get("provinsi", ""),
        "case_number_normalized": case_number,
        "case_year": years[0] if years else None,
        "section_lengths": section_lengths,
    }


def is_holdout_candidate(
    metadata: dict[str, Any],
    *,
    excluded_courts: set[str],
) -> bool:
    """Return whether metadata meets the frozen external-holdout criteria."""
    return (
        metadata["classification"] == "pidana-khusus"
        and metadata["subclassification"] == "narkotika-dan-psikotropika"
        and metadata["court"] not in excluded_courts
        and bool(metadata["source_url"])
        and metadata["case_year"] is not None
        and all(length > 0 for length in metadata["section_lengths"].values())
    )


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _case_number(heading: str) -> str:
    match = re.search(
        r"\b(?:nomor|no)\s+(.{1,100}?)(?=\s+demi\s+keadilan\b)",
        heading,
        flags=re.IGNORECASE,
    )
    return " ".join(match.group(1).split()) if match else ""
