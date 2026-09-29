"""Thin command-line orchestration for the initial research pipeline."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.chunking import FixedSizeChunker, StructureAwareChunker
from src.preprocessing import PDFExtractor, StructureDetector, TextCleaner
from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logger import configure_logging

LOGGER = logging.getLogger(__name__)


def extract_main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for PDF extraction."""
    parser = argparse.ArgumentParser(description="Extract page text from court-decision PDFs.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--input-dir", type=Path, help="Override paths.raw_pdf_dir")
    parser.add_argument("--output", type=Path, help="Override paths.extracted_pages")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    config = load_config(args.config)
    input_dir = args.input_dir or _project_path(args.config, _get(config, "paths", "raw_pdf_dir"))
    output = args.output or _project_path(args.config, _get(config, "paths", "extracted_pages"))
    input_dir.mkdir(parents=True, exist_ok=True)

    extractor = PDFExtractor(
        exclude_rotated_text=bool(config.get("extraction", {}).get("exclude_rotated_text", True))
    )
    pages = extractor.extract_directory(
        input_dir,
        recursive=bool(config.get("extraction", {}).get("recursive", True)),
    )
    count = write_jsonl(output, (page.to_dict() for page in pages))
    documents = len({page.document_id for page in pages})
    LOGGER.info("Extracted %d page(s) from %d PDF document(s) to %s", count, documents, output)


def preprocess_main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for cleaning and structure detection."""
    parser = argparse.ArgumentParser(description="Clean page text and detect document sections.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--input", type=Path, help="Override paths.extracted_pages")
    parser.add_argument("--pages-output", type=Path, help="Override paths.processed_pages")
    parser.add_argument("--sections-output", type=Path, help="Override paths.detected_sections")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    config = load_config(args.config)
    input_path = args.input or _project_path(args.config, _get(config, "paths", "extracted_pages"))
    pages_output = args.pages_output or _project_path(args.config, _get(config, "paths", "processed_pages"))
    sections_output = args.sections_output or _project_path(args.config, _get(config, "paths", "detected_sections"))

    cleaning = config.get("preprocessing", {})
    cleaner = TextCleaner(
        normalize_unicode=bool(cleaning.get("normalize_unicode", True)),
        dehyphenate_line_breaks=bool(cleaning.get("dehyphenate_line_breaks", True)),
        collapse_whitespace=bool(cleaning.get("collapse_whitespace", True)),
        repair_mojibake=bool(cleaning.get("repair_mojibake", True)),
        remove_court_boilerplate=bool(cleaning.get("remove_court_boilerplate", True)),
    )
    processed_pages: list[dict[str, Any]] = []
    for page in read_jsonl(input_path):
        processed = dict(page)
        processed["clean_text"] = cleaner.clean(str(page.get("raw_text", "")))
        processed_pages.append(processed)
    processed_pages.sort(key=lambda page: (str(page["document_id"]), int(page["page_number"])))
    page_count = write_jsonl(pages_output, processed_pages)

    structure_detection = config.get("structure_detection", {})
    detector = StructureDetector(
        fallback_label=str(structure_detection.get("fallback_label", "unknown")),
        conclusion_search_fraction=float(
            structure_detection.get("conclusion_search_fraction", 0.20)
        ),
    )
    sections: list[dict[str, object]] = []
    for document_id, document_pages in _group_pages(processed_pages).items():
        document_text = _join_pages(document_pages)
        detected = detector.detect(document_id, document_text)
        if "".join(section.section_text for section in detected) != document_text:
            raise RuntimeError(f"Section detector did not preserve all text for {document_id}")
        sections.extend(section.to_dict() for section in detected)
    section_count = write_jsonl(sections_output, sections)
    LOGGER.info(
        "Processed %d page(s); wrote %d section(s) to %s",
        page_count,
        section_count,
        sections_output,
    )


def chunk_main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for either configured chunking strategy."""
    parser = argparse.ArgumentParser(description="Chunk cleaned court-decision text.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Override paths.processed_pages")
    parser.add_argument("--sections", type=Path, help="Override paths.detected_sections")
    parser.add_argument("--output", type=Path, help="Override chunking.output_path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    config = load_config(args.config)
    input_path = args.input or _project_path(args.config, _get(config, "paths", "processed_pages"))
    sections_path = args.sections or _project_path(args.config, _get(config, "paths", "detected_sections"))
    chunking = config.get("chunking", {})
    output = args.output or _project_path(args.config, str(chunking["output_path"]))
    strategy = str(chunking.get("strategy", "fixed_size"))
    common_options = {
        "max_words": int(chunking.get("max_words", 300)),
        "overlap_words": int(chunking.get("overlap_words", 50)),
    }
    if strategy == "fixed_size":
        chunker = FixedSizeChunker(**common_options)
    elif strategy == "structure_aware":
        chunker = StructureAwareChunker(
            **common_options,
            overlap_sentences=int(chunking.get("overlap_sentences", 2)),
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    pages_by_document = _group_pages(list(read_jsonl(input_path)))
    sections_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if strategy == "structure_aware":
        for section in read_jsonl(sections_path):
            sections_by_document[str(section["document_id"])].append(section)
        for document_sections in sections_by_document.values():
            document_sections.sort(key=lambda section: int(section["start_position"]))

    chunks: list[dict[str, object]] = []
    for document_id, document_pages in pages_by_document.items():
        document_text = _join_pages(document_pages)
        document_chunks = chunker.chunk(
            document_id,
            document_text,
            sections=sections_by_document.get(document_id),
        )
        chunks.extend(chunk.to_dict() for chunk in document_chunks)
    count = write_jsonl(output, chunks)
    LOGGER.info("Created %d %s chunk(s) in %s", count, strategy, output)


def _group_pages(pages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        grouped[str(page["document_id"])].append(page)
    for document_pages in grouped.values():
        document_pages.sort(key=lambda page: int(page["page_number"]))
    return dict(sorted(grouped.items()))


def _join_pages(pages: list[dict[str, Any]]) -> str:
    return "\n\n".join(str(page.get("clean_text", "")) for page in pages)


def _get(config: dict[str, Any], section: str, key: str) -> str:
    try:
        return str(config[section][key])
    except KeyError as error:
        raise KeyError(f"Missing required configuration value: {section}.{key}") from error


def _project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Config files live in <project>/configs by convention.
    return config_path.resolve().parent.parent / path

