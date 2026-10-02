"""Convert the frozen Indo-Law sample into pipeline pages and gold sections."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

from src.utils.io import write_jsonl


SECTION_MAP = (
    ("kepala_putusan", "kepala_putusan"),
    ("identitas", "identitas_terdakwa"),
    ("riwayat_penahanan", "riwayat_penahanan"),
    ("riwayat_perkara", "riwayat_perkara"),
    ("riwayat_tuntutan", "riwayat_tuntutan"),
    ("riwayat_dakwaan", "riwayat_dakwaan"),
    ("fakta", "fakta"),
    ("fakta_hukum", "fakta_hukum"),
    ("pertimbangan_hukum", "pertimbangan_hukum"),
    ("amar_putusan", "amar_putusan"),
    ("penutup", "penutup"),
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the Indo-Law 200 corpus.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("experiments/indolaw_200_manifest.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/indolaw_200")
    )
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_split: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in manifest["documents"]:
        split = str(record["split"])
        outputs = by_split.setdefault(split, {"pages": [], "sections": []})
        page, sections = convert_xml(Path(record["local_file"]))
        outputs["pages"].append(page)
        outputs["sections"].extend(sections)

    for split, outputs in sorted(by_split.items()):
        directory = args.output_root / split
        page_count = write_jsonl(directory / "pages.jsonl", outputs["pages"])
        section_count = write_jsonl(directory / "sections.jsonl", outputs["sections"])
        print(
            f"{split}: wrote {page_count} documents and {section_count} sections "
            f"to {directory}"
        )


def convert_xml(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a single logical page and character-exact annotated sections."""
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    document_id = root.attrib.get("id", path.stem)
    blocks: list[tuple[str, str, str]] = []
    for tag, label in SECTION_MAP:
        text = _element_text(root.find(tag))
        if text:
            blocks.append((tag, label, text))
    if not blocks:
        raise ValueError(f"No supported sections in {path}")

    document_parts: list[str] = []
    sections: list[dict[str, Any]] = []
    cursor = 0
    for index, (tag, label, text) in enumerate(blocks):
        section_text = text + ("\n\n" if index + 1 < len(blocks) else "")
        start = cursor
        end = start + len(section_text)
        document_parts.append(section_text)
        sections.append(
            {
                "document_id": document_id,
                "section_label": label,
                "section_heading": tag,
                "start_position": start,
                "end_position": end,
                "section_text": section_text,
            }
        )
        cursor = end
    document_text = "".join(document_parts)
    if "".join(row["section_text"] for row in sections) != document_text:
        raise RuntimeError(f"Section offsets failed for {document_id}")
    page = {
        "document_id": document_id,
        "filename": path.name,
        "page_number": 1,
        "raw_text": document_text,
        "clean_text": document_text,
        "source_format": "indolaw_xml",
    }
    return page, sections


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


if __name__ == "__main__":
    main()
