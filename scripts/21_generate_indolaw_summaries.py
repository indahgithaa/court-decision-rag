"""Generate one auditable document fingerprint per Indo-Law judgment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.summarization import (
    QwenDocumentFingerprintGenerator,
    build_hierarchical_extractive_contexts,
    build_representative_document,
    build_structured_extractive_fingerprint,
)
from src.utils.io import read_jsonl, write_jsonl


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate Summary-Augmented Chunking document fingerprints."
    )
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed/indolaw_200")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/indolaw_200/document_summaries"),
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("development", "holdout"), default=("development",)
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--backend",
        choices=("extractive", "qwen"),
        default="extractive",
        help="Use the reproducible CPU-safe extractor or an optional local LLM.",
    )
    parser.add_argument("--target-chars", type=int, default=150)
    parser.add_argument("--tolerance-chars", type=int, default=20)
    parser.add_argument("--per-section-chars", type=int, default=1_600)
    parser.add_argument("--max-input-tokens", type=int, default=4_096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate only the first N documents for a smoke test.",
    )
    args = parser.parse_args(argv)

    generator = (
        QwenDocumentFingerprintGenerator(
            args.model,
            target_chars=args.target_chars,
            tolerance_chars=args.tolerance_chars,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        if args.backend == "qwen"
        else None
    )
    for split in args.splits:
        output = args.output_root / f"{split}.jsonl"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        sections = list(read_jsonl(args.processed_root / split / "sections.jsonl"))
        by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for section in sections:
            by_document[str(section["document_id"])].append(section)
        document_ids = sorted(by_document)
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("limit must be positive")
            document_ids = document_ids[: args.limit]

        records: list[dict[str, Any]] = []
        for index, document_id in enumerate(document_ids, start=1):
            source = build_representative_document(
                by_document[document_id], per_section_chars=args.per_section_chars
            )
            if generator is None:
                summary, metadata = build_structured_extractive_fingerprint(
                    by_document[document_id],
                    target_chars=args.target_chars,
                    tolerance_chars=args.tolerance_chars,
                )
            else:
                summary, metadata = generator.generate(source)
            document_context, section_contexts = (
                build_hierarchical_extractive_contexts(by_document[document_id])
            )
            records.append(
                {
                    "document_id": document_id,
                    "summary": summary,
                    "summary_chars": len(summary),
                    "document_context": document_context,
                    "section_contexts": section_contexts,
                    "source_policy": "section_balanced_head_tail_v1",
                    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    **metadata,
                }
            )
            print(f"{split}: {index}/{len(document_ids)} {document_id}: {summary}")
        count = write_jsonl(output, records)
        manifest = {
            "method": "Summary-Augmented Chunking document fingerprint",
            "paper": "Reuter et al. (NLLP 2025), DOI 10.18653/v1/2025.nllp-1.3",
            "adaptation": (
                "Deterministic structured extractive fingerprint; paper used "
                "GPT-4o-mini over each document."
                if args.backend == "extractive"
                else "Open-weight local instruction model and section-balanced input; "
                "paper used GPT-4o-mini over each document."
            ),
            "backend": args.backend,
            "split": split,
            "document_count": count,
            "model_name": args.model,
            "target_chars": args.target_chars,
            "tolerance_chars": args.tolerance_chars,
            "per_section_chars": args.per_section_chars,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "summaries_sha256": _sha256(output),
        }
        manifest_path = output.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {count} summaries and {manifest_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
