"""Run the controlled B1/B2/M1 hierarchical-SAC development experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from src.evaluation.ground_truth import join_clean_pages
from src.hierarchical_sac.chunking import (
    RecursiveCharacterChunker,
    assign_chunk_context,
    build_page_spans,
)
from src.hierarchical_sac.metrics import (
    classify_failures,
    evaluate_method,
    map_evidence_to_chunks,
    paired_cluster_comparisons,
    per_section_comparisons,
)
from src.hierarchical_sac.representations import (
    METHODS,
    apply_e5_token_budget,
    assert_aligned,
    build_embedding_representations,
)
from src.hierarchical_sac.summarization import (
    IndonesianT5SummaryGenerator,
    generate_summary_artifacts,
)
from src.utils.io import read_jsonl, write_jsonl


PAPER_URL = "https://aclanthology.org/2025.nllp-1.3/"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/hierarchical_sac_2026.yaml")
    )
    parser.add_argument(
        "--stage",
        choices=(
            "all", "audit", "summaries", "chunks", "representations",
            "embeddings", "evaluate", "report",
        ),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = _paths(config)
    for path in (paths["workspace"], paths["derived"], paths["indexes"], paths["results"]):
        path.mkdir(parents=True, exist_ok=True)

    stages = (
        ("audit", run_audit),
        ("summaries", run_summaries),
        ("chunks", run_chunks),
        ("representations", run_representations),
        ("embeddings", run_embeddings),
        ("evaluate", run_evaluation),
        ("report", run_report),
    )
    for name, function in stages:
        if args.stage in {"all", name}:
            print(f"\n[{datetime.now().isoformat(timespec='seconds')}] stage={name}", flush=True)
            function(config, paths, force=args.force)


def run_audit(config: dict[str, Any], paths: dict[str, Path], *, force: bool) -> None:
    del force
    data = config["data"]
    manifest = json.loads(Path(data["manifest"]).read_text(encoding="utf-8"))
    raw_dir = Path(data["raw_dir"])
    raw_files = sorted(raw_dir.glob("*.xml"))
    pdf_files = sorted(Path(".").glob("**/*.pdf"))
    expected = {Path(row["local_file"]).name: row for row in manifest["documents"]}
    integrity_errors: list[str] = []
    content_hashes: Counter[str] = Counter()
    empty_documents: list[str] = []
    malformed_xml: list[str] = []
    filename_issues: list[str] = []
    for path in raw_files:
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        content_hashes[digest] += 1
        record = expected.get(path.name)
        if record is None:
            integrity_errors.append(f"unexpected file: {path.as_posix()}")
        else:
            if digest != str(record["sha256"]) or len(payload) != int(record["bytes"]):
                integrity_errors.append(f"hash/size mismatch: {path.name}")
        if not re.fullmatch(r"[0-9a-f]{32}\.xml", path.name):
            filename_issues.append(path.name)
        try:
            root = ET.fromstring(payload)
            text = " ".join("".join(root.itertext()).split())
            if not text:
                empty_documents.append(path.name)
            if root.attrib.get("id", path.stem) != path.stem:
                integrity_errors.append(
                    f"root id differs from filename: {path.name}/{root.attrib.get('id')}"
                )
        except ET.ParseError:
            malformed_xml.append(path.name)
    missing_files = sorted(set(expected) - {path.name for path in raw_files})
    integrity_errors.extend(f"missing file: {name}" for name in missing_files)

    pages = list(read_jsonl(Path(data["pages"])))
    sections = list(read_jsonl(Path(data["sections"])))
    documents = join_clean_pages(pages)
    sections_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        sections_by_document[str(section["document_id"])].append(section)
    structure_errors: list[str] = []
    missing_section_counts: Counter[str] = Counter()
    required = {
        "identitas_terdakwa", "riwayat_penahanan", "riwayat_dakwaan",
        "fakta", "pertimbangan_hukum", "amar_putusan",
    }
    for document_id, text in documents.items():
        ordered = sorted(
            sections_by_document.get(document_id, ()),
            key=lambda row: int(row["start_position"]),
        )
        if "".join(str(row["section_text"]) for row in ordered) != text:
            structure_errors.append(f"section reconstruction mismatch: {document_id}")
        labels = {str(row["section_label"]) for row in ordered}
        for label in required - labels:
            missing_section_counts[label] += 1
        for row in ordered:
            start, end = int(row["start_position"]), int(row["end_position"])
            if text[start:end] != str(row["section_text"]):
                structure_errors.append(
                    f"section offset mismatch: {document_id}/{row['section_label']}"
                )

    questions = _read_questions(Path(data["questions"]))
    question_audit, included = _audit_questions(questions, documents, sections_by_document)
    write_jsonl(paths["benchmark"] / "included_questions.jsonl", included)
    _write_csv(paths["benchmark"] / "annotation_worksheet.csv", question_audit)
    _write_csv(paths["benchmark"] / "included_questions.csv", included)

    section_distribution = Counter(str(row["section_label"]) for row in sections)
    document_section_distribution = Counter(
        len(sections_by_document[document_id]) for document_id in documents
    )
    text_lengths = [len(text) for text in documents.values()]
    mojibake_counts = {
        marker: sum(text.count(marker) for text in documents.values())
        for marker in ("\ufffd", "Ã", "â€", "Â")
    }
    duplicate_groups = [count for count in content_hashes.values() if count > 1]
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_status_before_new_pipeline": "clean (observed before implementation)",
        "source_format": "Indo-Law normalized XML",
        "pdf_count": len(pdf_files),
        "xml_count": len(raw_files),
        "manifest_document_count": int(manifest["sample_size"]),
        "processed_document_count": len(documents),
        "valid_raw_document_count": len(raw_files) - len(malformed_xml) - len(empty_documents),
        "malformed_xml": malformed_xml,
        "empty_documents": empty_documents,
        "missing_files": missing_files,
        "integrity_errors": integrity_errors,
        "filename_issues": filename_issues,
        "duplicate_content_groups": len(duplicate_groups),
        "duplicate_content_documents": sum(duplicate_groups),
        "page_records": len(pages),
        "section_records": len(sections),
        "section_distribution": dict(sorted(section_distribution.items())),
        "sections_per_document_distribution": {
            str(key): value for key, value in sorted(document_section_distribution.items())
        },
        "missing_required_sections_by_label": dict(sorted(missing_section_counts.items())),
        "structure_errors": structure_errors,
        "document_text_characters": {
            "total": sum(text_lengths),
            "minimum": min(text_lengths),
            "maximum": max(text_lengths),
            "mean": sum(text_lengths) / len(text_lengths),
        },
        "mojibake_marker_counts": mojibake_counts,
        "ocr_assessment": "not applicable: active corpus is normalized XML, not OCR/PDF",
        "pdf_integrity_assessment": "not executable: repository contains zero PDFs",
        "qa": _question_audit_summary(question_audit, included),
        "historical_split": {
            "development_documents": int(manifest["split_counts"]["development"]),
            "holdout_documents": int(manifest["split_counts"]["holdout"]),
            "status": "previously opened/reused; not an independent confirmatory holdout",
        },
        "available_local_models": [
            "intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3",
            "Irvan14/t5-small-indonesian-summarization@65f2d2be6dac02a57444535aca909c5d4606de8a",
            "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
        ],
        "openai_api_key_available": bool(os.environ.get("OPENAI_API_KEY")),
        "corpus_sha256": _file_sha256(Path(data["pages"])),
        "sections_sha256": _file_sha256(Path(data["sections"])),
        "questions_sha256": _file_sha256(Path(data["questions"])),
    }
    _write_json(paths["workspace"] / "dataset_audit.json", audit)
    (paths["workspace"] / "dataset_audit.md").write_text(
        _render_audit_markdown(audit), encoding="utf-8"
    )
    print(
        f"audit: documents={len(documents)} XML={len(raw_files)} PDF={len(pdf_files)} "
        f"questions={len(questions)} included_exploratory={len(included)}",
        flush=True,
    )


def run_summaries(config: dict[str, Any], paths: dict[str, Path], *, force: bool) -> None:
    summarization = config["summarization"]
    sections = list(read_jsonl(Path(config["data"]["sections"])))
    generator = IndonesianT5SummaryGenerator(
        summarization["model_id"],
        revision=summarization["model_revision"],
        max_input_tokens=int(summarization["max_input_tokens"]),
        max_new_tokens=int(summarization["max_new_tokens"]),
        batch_size=int(summarization["batch_size"]),
        device=str(summarization["device"]),
        target_chars=int(summarization["target_characters"]),
        tolerance_chars=int(summarization["tolerance_characters"]),
    )
    documents, section_records, manifest = generate_summary_artifacts(
        sections,
        generator=generator,
        output_dir=paths["summaries"],
        short_section_chars=int(summarization["short_section_characters"]),
        document_section_chars=int(summarization["document_section_characters"]),
        force=force,
    )
    diagnostics = _summary_quality_report(documents, section_records)
    _write_json(paths["results"] / "summary_quality.json", diagnostics)
    sample = _summary_review_sample(documents, section_records, seed=int(config["experiment"]["seed"]))
    _write_csv(paths["results"] / "summary_manual_review_sample.csv", sample)
    print(
        f"summaries: documents={len(documents)} sections={len(section_records)} "
        f"flagged={diagnostics['overall']['flagged_count']} cache={manifest['schema_version']}",
        flush=True,
    )


def run_chunks(config: dict[str, Any], paths: dict[str, Path], *, force: bool) -> None:
    output = paths["canonical"] / "base_chunks.jsonl"
    manifest_path = paths["canonical"] / "manifest.json"
    source_hash = _file_sha256(Path(config["data"]["pages"]))
    expected = {
        "schema_version": "hsac-canonical-chunks-v1",
        "source_pages_sha256": source_hash,
        **config["chunking"],
    }
    if output.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Canonical chunk cache mismatch for {key}")
        print(f"chunks: cache hit count={manifest['chunk_count']}", flush=True)
        return

    pages = list(read_jsonl(Path(config["data"]["pages"])))
    sections = list(read_jsonl(Path(config["data"]["sections"])))
    documents = join_clean_pages(pages)
    page_spans = build_page_spans(pages)
    sections_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sections:
        sections_by_document[str(row["document_id"])].append(row)
    chunking = config["chunking"]
    chunker = RecursiveCharacterChunker(
        chunk_size=int(chunking["chunk_size_characters"]),
        chunk_overlap=int(chunking["chunk_overlap_characters"]),
        separators=chunking["separators"],
        preferred_fill_ratio=float(chunking["preferred_fill_ratio"]),
    )
    chunks: list[dict[str, Any]] = []
    for index, document_id in enumerate(sorted(documents), start=1):
        base = chunker.split(document_id, documents[document_id])
        chunks.extend(
            assign_chunk_context(
                base,
                sections=sections_by_document[document_id],
                page_spans=page_spans[document_id],
            )
        )
        if index % 25 == 0:
            print(f"chunks: {index}/{len(documents)} documents", flush=True)
    write_jsonl(output, chunks)
    assignment_counts = Counter(str(row["section_assignment_status"]) for row in chunks)
    manifest = {
        **expected,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks_sha256": _file_sha256(output),
        "assignment_status_counts": dict(sorted(assignment_counts.items())),
        "maximum_chunk_characters": max(len(str(row["original_text"])) for row in chunks),
        "minimum_chunk_characters": min(len(str(row["original_text"])) for row in chunks),
    }
    _write_json(manifest_path, manifest)
    print(f"chunks: wrote {len(chunks)} canonical chunks", flush=True)


def run_representations(
    config: dict[str, Any], paths: dict[str, Path], *, force: bool
) -> None:
    chunks = list(read_jsonl(paths["canonical"] / "base_chunks.jsonl"))
    documents = list(read_jsonl(paths["summaries"] / "document_summaries.jsonl"))
    sections = list(read_jsonl(paths["summaries"] / "section_summaries.jsonl"))
    document_map = {str(row["document_id"]): str(row["summary"]) for row in documents}
    section_map = {
        (str(row["document_id"]), str(row["section_label"])): str(row["summary"])
        for row in sections
    }
    records = build_embedding_representations(
        chunks, document_summaries=document_map, section_summaries=section_map
    )
    embedding = config["embedding"]
    from huggingface_hub import hf_hub_download

    from src.hierarchical_sac.embedding import load_e5_tokenizer

    tokenizer = load_e5_tokenizer(
        Path(
            hf_hub_download(
                str(embedding["model_id"]),
                "tokenizer.json",
                revision=str(embedding["model_revision"]),
            )
        )
    )
    manifests: dict[str, Any] = {}
    for method in METHODS:
        output = paths["representations"] / f"{method}.jsonl"
        if output.exists() and not force:
            budgeted = list(read_jsonl(output))
        else:
            budgeted = apply_e5_token_budget(
                records[method],
                tokenizer=tokenizer,
                max_tokens=int(embedding["max_sequence_tokens"]),
                passage_prefix=str(embedding["passage_prefix"]),
            )
            write_jsonl(output, budgeted)
        manifests[method] = {
            "chunk_count": len(budgeted),
            "sha256": _file_sha256(output),
            "token_budget_affected": sum(bool(row["summary_budget_applied"]) for row in budgeted),
            "maximum_embedding_tokens": max(int(row["embedding_tokens"]) for row in budgeted),
            "source_text_preserved": all(bool(row["source_text_preserved"]) for row in budgeted),
        }
    loaded = {
        method: list(read_jsonl(paths["representations"] / f"{method}.jsonl"))
        for method in METHODS
    }
    assert_aligned(loaded)
    _write_json(
        paths["representations"] / "manifest.json",
        {
            "schema_version": "hsac-representations-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_model_id": embedding["model_id"],
            "embedding_model_revision": embedding["model_revision"],
            "max_sequence_tokens": embedding["max_sequence_tokens"],
            "methods": manifests,
            "alignment_verified": True,
        },
    )
    print(f"representations: aligned {len(chunks)} chunks across B1/B2/M1", flush=True)


def run_embeddings(config: dict[str, Any], paths: dict[str, Path], *, force: bool) -> None:
    embedding = config["embedding"]
    torch = None
    backend = "sentence_transformers_pytorch"
    runtime_version: str | None = None
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            embedding["model_id"],
            revision=embedding["model_revision"],
            device=embedding["device"],
            local_files_only=True,
            trust_remote_code=False,
        )
        runtime_version = str(torch.__version__)
    except (ImportError, OSError) as error:
        if isinstance(error, OSError) and getattr(error, "winerror", None) != 4551:
            raise
        from huggingface_hub import hf_hub_download

        from src.hierarchical_sac.embedding import OnnxE5Encoder

        backend = "onnxruntime_official_fp32_export"
        export_revision = str(embedding["onnx_export_revision"])
        onnx_path = Path(
            hf_hub_download(
                str(embedding["model_id"]),
                "onnx/model.onnx",
                revision=export_revision,
            )
        )
        expected_onnx_hash = str(embedding["onnx_export_sha256"])
        if _file_sha256(onnx_path) != expected_onnx_hash:
            raise RuntimeError("Official E5 ONNX export checksum mismatch")
        tokenizer_path = Path(
            hf_hub_download(
                str(embedding["model_id"]),
                "tokenizer.json",
                revision=export_revision,
            )
        )
        model = OnnxE5Encoder(
            model_path=onnx_path,
            tokenizer_path=tokenizer_path,
            max_seq_length=int(embedding["max_sequence_tokens"]),
        )
        runtime_version = model.runtime_version
        print(
            f"embeddings: PyTorch unavailable ({type(error).__name__}); "
            "using verified official ONNX FP32 export",
            flush=True,
        )
    model.max_seq_length = int(embedding["max_sequence_tokens"])
    model.eval()

    if backend.startswith("onnxruntime"):
        reference_path = paths["indexes"] / "B1" / "embeddings.npy"
        reference_records_path = paths["representations"] / "B1.jsonl"
        if reference_path.exists() and reference_records_path.exists():
            probe_records = []
            for record in read_jsonl(reference_records_path):
                probe_records.append(record)
                if len(probe_records) == 8:
                    break
            probe = model.encode(
                [
                    str(embedding["passage_prefix"]) + str(row["embedding_text"])
                    for row in probe_records
                ],
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            reference = np.load(reference_path, mmap_mode="r", allow_pickle=False)[
                : len(probe_records)
            ]
            cosine = np.sum(probe * reference, axis=1)
            maximum_absolute_difference = float(np.max(np.abs(probe - reference)))
            equivalence = {
                "schema_version": "hsac-onnx-equivalence-v1",
                "probe_count": len(probe_records),
                "minimum_cosine": float(cosine.min()),
                "mean_cosine": float(cosine.mean()),
                "maximum_absolute_difference": maximum_absolute_difference,
                "required_minimum_cosine": 0.999999,
                "required_maximum_absolute_difference": 1e-5,
                "onnx_export_revision": embedding["onnx_export_revision"],
                "onnx_export_sha256": embedding["onnx_export_sha256"],
                "status": "passed",
            }
            if cosine.min() < 0.999999 or maximum_absolute_difference > 1e-5:
                equivalence["status"] = "failed"
                _write_json(paths["indexes"] / "onnx_equivalence.json", equivalence)
                raise RuntimeError(f"ONNX/PyTorch E5 equivalence failed: {equivalence}")
            _write_json(paths["indexes"] / "onnx_equivalence.json", equivalence)
            print(
                "embeddings: ONNX/PyTorch equivalence passed "
                f"min_cos={equivalence['minimum_cosine']:.9f} "
                f"max_abs={maximum_absolute_difference:.3e}",
                flush=True,
            )
    included = list(read_jsonl(paths["benchmark"] / "included_questions.jsonl"))
    query_path = paths["indexes"] / "query_embeddings.npy"
    query_manifest_path = paths["indexes"] / "query_manifest.json"
    query_source_hash = _records_sha256(included)
    if not query_path.exists() or force:
        queries = [str(row["question"]) for row in included]
        token_lengths = [
            len(model.tokenizer.encode(str(embedding["query_prefix"]) + query, add_special_tokens=True))
            for query in queries
        ]
        if max(token_lengths) > int(embedding["max_sequence_tokens"]):
            raise ValueError("A query exceeds the E5 token budget")
        query_embeddings = model.encode(
            [str(embedding["query_prefix"]) + query for query in queries],
            batch_size=int(embedding["batch_size"]),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        np.save(query_path, query_embeddings, allow_pickle=False)
        write_jsonl(
            paths["indexes"] / "queries.jsonl",
            ({"query_index": index, **row} for index, row in enumerate(included)),
        )
        _write_json(
            query_manifest_path,
            {
                "schema_version": "hsac-query-embeddings-v1",
                "model_id": embedding["model_id"],
                "model_revision": embedding["model_revision"],
                "source_sha256": query_source_hash,
                "query_count": len(included),
                "dimension": int(query_embeddings.shape[1]),
                "normalized": True,
                "maximum_tokens": max(token_lengths),
            },
        )
    else:
        query_manifest = json.loads(query_manifest_path.read_text(encoding="utf-8"))
        if query_manifest["source_sha256"] != query_source_hash:
            raise ValueError("Query embedding cache source mismatch")
        print("embeddings: query cache hit", flush=True)

    for method in METHODS:
        records_path = paths["representations"] / f"{method}.jsonl"
        index_dir = paths["indexes"] / method
        index_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = index_dir / "embeddings.npy"
        manifest_path = index_dir / "manifest.json"
        source_hash = _file_sha256(records_path)
        if matrix_path.exists() and manifest_path.exists() and not force:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source_sha256") != source_hash:
                raise ValueError(f"Embedding cache mismatch for {method}")
            print(f"embeddings: {method} cache hit", flush=True)
            continue
        records = list(read_jsonl(records_path))
        texts = [str(embedding["passage_prefix"]) + str(row["embedding_text"]) for row in records]
        batch_size = int(embedding["batch_size"])
        checkpoint_dir = index_dir / f"_parts_{source_hash[:12]}_b{batch_size}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_manifest_path = checkpoint_dir / "manifest.json"
        checkpoint_manifest = {
            "schema_version": "hsac-embedding-parts-v1",
            "method": method,
            "source_sha256": source_hash,
            "model_id": embedding["model_id"],
            "model_revision": embedding["model_revision"],
            "batch_size": batch_size,
            "vector_count": len(texts),
        }
        if checkpoint_manifest_path.exists() and not force:
            cached_checkpoint = json.loads(
                checkpoint_manifest_path.read_text(encoding="utf-8")
            )
            if cached_checkpoint != checkpoint_manifest:
                raise ValueError(f"Embedding checkpoint mismatch for {method}")
        else:
            _write_json(checkpoint_manifest_path, checkpoint_manifest)

        parts: list[np.ndarray] = []
        total_batches = math.ceil(len(texts) / batch_size)
        for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
            stop = min(start + batch_size, len(texts))
            part_path = checkpoint_dir / f"part_{batch_index - 1:05d}.npy"
            if part_path.exists() and not force:
                part = np.load(part_path, allow_pickle=False)
                if part.shape[0] != stop - start or not np.isfinite(part).all():
                    raise RuntimeError(
                        f"Invalid embedding checkpoint for {method} batch {batch_index}: "
                        f"{part.shape}"
                    )
            else:
                part = model.encode(
                    texts[start:stop],
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float32)
                temporary_path = part_path.with_suffix(".tmp.npy")
                np.save(temporary_path, part, allow_pickle=False)
                os.replace(temporary_path, part_path)
            parts.append(part)
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                print(
                    f"embeddings: {method} batches={batch_index}/{total_batches}",
                    flush=True,
                )
        matrix = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if matrix.shape[0] != len(records) or not np.isfinite(matrix).all():
            raise RuntimeError(f"Invalid embedding matrix for {method}: {matrix.shape}")
        np.save(matrix_path, matrix, allow_pickle=False)
        _write_json(
            manifest_path,
            {
                "schema_version": "hsac-index-v1",
                "method": method,
                "model_id": embedding["model_id"],
                "model_revision": embedding["model_revision"],
                "source_sha256": source_hash,
                "vector_count": int(matrix.shape[0]),
                "dimension": int(matrix.shape[1]),
                "normalized": True,
                "similarity": config["retrieval"]["similarity"],
                "device": str(embedding["device"]),
                "embedding_backend": backend,
                "runtime_version": runtime_version,
                "torch_version": str(torch.__version__) if torch is not None else None,
            },
        )
        print(f"embeddings: {method} matrix={matrix.shape}", flush=True)


def run_evaluation(config: dict[str, Any], paths: dict[str, Path], *, force: bool) -> None:
    del force
    questions = list(read_jsonl(paths["benchmark"] / "included_questions.jsonl"))
    chunks = list(read_jsonl(paths["canonical"] / "base_chunks.jsonl"))
    qrels, evidence_qrels, qrel_rows = map_evidence_to_chunks(questions, chunks)
    write_jsonl(paths["benchmark"] / "canonical_qrels.jsonl", qrel_rows)
    query_embeddings = np.load(paths["indexes"] / "query_embeddings.npy", allow_pickle=False)
    if query_embeddings.shape[0] != len(questions):
        raise ValueError("Query embedding count mismatch")
    top_k = int(config["retrieval"]["top_k_store"])
    ks = tuple(int(value) for value in config["retrieval"]["evaluation_k"])
    method_results: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    chunks_by_id = {str(row["chunk_id"]): row for row in chunks}
    for method in METHODS:
        records = list(read_jsonl(paths["representations"] / f"{method}.jsonl"))
        matrix = np.load(paths["indexes"] / method / "embeddings.npy", allow_pickle=False)
        runs = _retrieve_exact(questions, query_embeddings, records, matrix, top_k=top_k)
        write_jsonl(paths["results"] / f"run_{method}.jsonl", runs)
        method_results[method] = evaluate_method(
            questions, runs, qrels, evidence_qrels, ks=ks
        )
        failures[method] = classify_failures(
            questions, runs, qrels, chunks_by_id, k=5
        )
    statistics_config = config["statistics"]
    comparisons = paired_cluster_comparisons(
        method_results,
        seed=int(config["experiment"]["seed"]),
        bootstrap_samples=int(statistics_config["bootstrap_samples"]),
        permutation_samples=int(statistics_config["permutation_samples"]),
    )
    section_comparisons = per_section_comparisons(method_results)
    result = {
        "schema_version": "hsac-evaluation-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment"]["id"],
        "scientific_status": "exploratory development; no independent human-validated holdout",
        "primary_endpoint": config["experiment"]["primary_endpoint"],
        "methods": method_results,
        "comparisons": comparisons,
        "per_section_comparisons": section_comparisons,
        "failure_analysis": failures,
    }
    _write_json(paths["results"] / "evaluation.json", result)
    _write_json(paths["results"] / "failure_analysis.json", failures)
    _write_csv(paths["results"] / "aggregate_metrics.csv", _aggregate_metric_rows(method_results))
    _write_csv(paths["results"] / "per_section_metrics.csv", _section_metric_rows(method_results))
    _write_csv(paths["results"] / "paired_comparisons.csv", comparisons)
    _write_csv(paths["results"] / "per_section_comparisons.csv", section_comparisons)
    print(
        "evaluation: "
        + ", ".join(
            f"{method} R@5={method_results[method]['aggregate_micro']['recall@5']:.4f}"
            for method in METHODS
        ),
        flush=True,
    )


def run_report(config: dict[str, Any], paths: dict[str, Path], *, force: bool) -> None:
    del force
    audit = json.loads((paths["workspace"] / "dataset_audit.json").read_text(encoding="utf-8"))
    evaluation = json.loads((paths["results"] / "evaluation.json").read_text(encoding="utf-8"))
    summary_quality = json.loads((paths["results"] / "summary_quality.json").read_text(encoding="utf-8"))
    representation_manifest = json.loads(
        (paths["representations"] / "manifest.json").read_text(encoding="utf-8")
    )
    chunk_manifest = json.loads(
        (paths["canonical"] / "manifest.json").read_text(encoding="utf-8")
    )
    index_manifests = {
        method: json.loads(
            (paths["indexes"] / method / "manifest.json").read_text(encoding="utf-8")
        )
        for method in METHODS
    }
    equivalence_path = paths["indexes"] / "onnx_equivalence.json"
    onnx_equivalence = (
        json.loads(equivalence_path.read_text(encoding="utf-8"))
        if equivalence_path.exists()
        else None
    )
    _write_plots(evaluation, paths["figures"])
    experiment_manifest = {
        "schema_version": "hsac-experiment-manifest-v1",
        "experiment_id": config["experiment"]["id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": _file_sha256(Path("configs/hierarchical_sac_2026.yaml")),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "paper": PAPER_URL,
        "canonical_chunks": chunk_manifest,
        "representations": representation_manifest,
        "indexes": index_manifests,
        "onnx_pytorch_equivalence": onnx_equivalence,
        "summary_quality": summary_quality["overall"],
        "evaluation_sha256": _file_sha256(paths["results"] / "evaluation.json"),
        "claim_boundary": (
            "Development-only normalized-XML experiment. The old holdout was opened, "
            "and annotations were generated deterministically rather than validated by humans."
        ),
    }
    _write_json(paths["results"] / "experiment_manifest.json", experiment_manifest)
    (paths["workspace"] / "methodology.md").write_text(
        _render_methodology(
            config,
            chunk_manifest,
            representation_manifest,
            index_manifests,
            onnx_equivalence,
        ),
        encoding="utf-8",
    )
    (paths["workspace"] / "research_report_id.md").write_text(
        _render_research_report(
            config,
            audit,
            evaluation,
            summary_quality,
            chunk_manifest,
            representation_manifest,
            index_manifests,
            onnx_equivalence,
        ),
        encoding="utf-8",
    )
    (paths["workspace"] / "reproduction.md").write_text(
        _render_reproduction(config), encoding="utf-8"
    )
    print(f"report: {paths['workspace'] / 'research_report_id.md'}", flush=True)


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    data = config["data"]
    derived = Path(data["derived_root"])
    results = Path(data["results_root"])
    return {
        "workspace": Path(data["workspace"]),
        "derived": derived,
        "summaries": derived / "summaries",
        "canonical": derived / "canonical_chunks",
        "representations": derived / "representations",
        "benchmark": derived / "benchmark",
        "indexes": Path(data["index_root"]),
        "results": results,
        "figures": results / "figures",
    }


def _read_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            normalized = {str(key).lstrip("\ufeff"): value for key, value in row.items()}
            normalized["evidence_start_position"] = int(normalized["evidence_start_position"])
            normalized["evidence_end_position"] = int(normalized["evidence_end_position"])
            rows.append(normalized)
    return rows


def _audit_questions(
    questions: Sequence[Mapping[str, Any]],
    documents: Mapping[str, str],
    sections_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audited: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []
    id_counts = Counter(str(row["query_id"]) for row in questions)
    for question in questions:
        row = dict(question)
        query_id = str(row["query_id"])
        document_id = str(row["document_id"])
        flags: list[str] = []
        document = documents.get(document_id)
        start, end = int(row["evidence_start_position"]), int(row["evidence_end_position"])
        answer = " ".join(str(row["reference_answer"]).split())
        question_text = " ".join(str(row["question"]).split())
        if id_counts[query_id] > 1:
            flags.append("duplicate_query_id")
        if document is None:
            flags.append("unknown_document")
            evidence = ""
        elif not 0 <= start < end <= len(document):
            flags.append("invalid_evidence_offsets")
            evidence = ""
        else:
            evidence = document[start:end]
            if " ".join(evidence.split()).casefold() != answer.casefold():
                flags.append("answer_span_mismatch")
            overlaps = [
                section
                for section in sections_by_document.get(document_id, ())
                if max(
                    0,
                    min(end, int(section["end_position"]))
                    - max(start, int(section["start_position"])),
                )
                > 0
            ]
            labels = {str(section["section_label"]) for section in overlaps}
            if str(row["target_section_label"]) not in labels:
                flags.append("target_section_mismatch")
        if len(answer) < 2 or (answer.isdigit() and len(answer) <= 2):
            flags.append("implausibly_short_answer")
        if answer.casefold() in {"umur", "tanggal", "tempat", "nama"}:
            flags.append("field_label_as_answer")
        if len(question_text) > 300:
            flags.append("excessive_question_length")
        if "tidak dilahirkan" in question_text.casefold():
            flags.append("malformed_identity_question")
        structural_flags = {
            "duplicate_query_id", "unknown_document", "invalid_evidence_offsets",
            "answer_span_mismatch", "target_section_mismatch",
        }
        unresolved = bool(flags)
        status = "exclude_pending_human_review" if unresolved else "include_exploratory_only"
        audited_row = {
            **row,
            "evidence_text": evidence,
            "automatic_status": status,
            "automatic_flags": "|".join(flags),
            "human_validation_status": "not_reviewed",
            "human_reviewer": "",
            "human_notes": "",
            "confirmatory_eligible": False,
        }
        audited.append(audited_row)
        if status == "include_exploratory_only":
            included.append(
                {
                    **row,
                    "evidence_id": f"{query_id}:{start}:{end}",
                    "human_validation_status": "not_reviewed",
                    "evaluation_status": "exploratory_development",
                }
            )
    return audited, included


def _question_audit_summary(
    audited: Sequence[Mapping[str, Any]], included: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    flag_counts: Counter[str] = Counter()
    for row in audited:
        flag_counts.update(value for value in str(row["automatic_flags"]).split("|") if value)
    return {
        "question_count": len(audited),
        "included_exploratory_count": len(included),
        "excluded_pending_review_count": len(audited) - len(included),
        "human_validated_count": 0,
        "confirmatory_eligible_count": 0,
        "target_section_distribution_all": dict(
            sorted(Counter(str(row["target_section_label"]) for row in audited).items())
        ),
        "target_section_distribution_included": dict(
            sorted(Counter(str(row["target_section_label"]) for row in included).items())
        ),
        "automatic_flag_counts": dict(sorted(flag_counts.items())),
        "annotation_origin": "deterministic rules over normalized XML; not human gold",
    }


def _summary_quality_report(
    document_records: Sequence[Mapping[str, Any]],
    section_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_records = list(document_records) + list(section_records)
    flagged = [row for row in all_records if row["automated_quality_status"] == "flagged"]
    flags = Counter(flag for row in all_records for flag in row.get("quality_flags", ()))
    return {
        "overall": {
            "summary_count": len(all_records),
            "document_summary_count": len(document_records),
            "section_summary_count": len(section_records),
            "flagged_count": len(flagged),
            "flag_counts": dict(sorted(flags.items())),
            "empty_count": sum(not str(row["summary"]).strip() for row in all_records),
            "over_170_count": sum(len(str(row["summary"])) > 170 for row in all_records),
            "unsupported_number_count": sum(bool(row["unsupported_number_tokens"]) for row in all_records),
            "generation_status_counts": dict(
                sorted(Counter(str(row["generation_status"]) for row in all_records).items())
            ),
            "manual_review_status": "sample prepared; no human judgments recorded",
        },
        "document": _summary_group_stats(document_records),
        "section": _summary_group_stats(section_records),
        "by_section": {
            label: _summary_group_stats(
                [row for row in section_records if str(row["section_label"]) == label]
            )
            for label in sorted({str(row["section_label"]) for row in section_records})
        },
    }


def _summary_group_stats(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["summary_chars"]) for row in records]
    return {
        "count": len(records),
        "mean_chars": sum(lengths) / len(lengths) if lengths else None,
        "min_chars": min(lengths) if lengths else None,
        "max_chars": max(lengths) if lengths else None,
        "flagged_count": sum(row["automated_quality_status"] == "flagged" for row in records),
        "source_reduced_count": sum(bool(row["source_reduced_for_context"]) for row in records),
    }


def _summary_review_sample(
    documents: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sample: list[dict[str, Any]] = []
    doc_indices = rng.choice(len(documents), size=min(20, len(documents)), replace=False)
    for index in sorted(int(value) for value in doc_indices):
        row = documents[index]
        sample.append(_review_row(row))
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sections:
        by_label[str(row["section_label"])].append(row)
    for label in sorted(by_label):
        group = by_label[label]
        indices = rng.choice(len(group), size=min(3, len(group)), replace=False)
        for index in sorted(int(value) for value in indices):
            sample.append(_review_row(group[index]))
    return sample


def _review_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary_type": row["summary_type"],
        "document_id": row["document_id"],
        "section_label": row.get("section_label", ""),
        "source_sha256": row["source_sha256"],
        "summary": row["summary"],
        "summary_chars": row["summary_chars"],
        "automatic_flags": "|".join(row.get("quality_flags", ())),
        "grounded_human": "",
        "correct_association_human": "",
        "language_human": "",
        "notes_human": "",
    }


def _retrieve_exact(
    questions: Sequence[Mapping[str, Any]],
    query_embeddings: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    matrix: np.ndarray,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if matrix.shape[0] != len(records) or matrix.shape[1] != query_embeddings.shape[1]:
        raise ValueError("index/query shape mismatch")
    runs: list[dict[str, Any]] = []
    for batch_start in range(0, len(questions), 64):
        scores = query_embeddings[batch_start : batch_start + 64] @ matrix.T
        for local_index, row_scores in enumerate(scores):
            query_index = batch_start + local_index
            limit = min(top_k, len(records))
            candidate = np.argpartition(-row_scores, limit - 1)[:limit]
            ordered = sorted(candidate.tolist(), key=lambda index: (-float(row_scores[index]), index))
            ranking = [
                {
                    "rank": rank,
                    "chunk_id": str(records[index]["chunk_id"]),
                    "document_id": str(records[index]["document_id"]),
                    "score": float(row_scores[index]),
                }
                for rank, index in enumerate(ordered, start=1)
            ]
            runs.append(
                {
                    "query_id": str(questions[query_index]["query_id"]),
                    "ranking": ranking,
                }
            )
    return runs


def _aggregate_metric_rows(method_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for metric, value in method_results[method]["aggregate_micro"].items():
            name, k = metric.split("@")
            rows.append(
                {
                    "method": method,
                    "metric": name,
                    "k": int(k),
                    "value": value,
                    "query_count": method_results[method]["query_count"],
                }
            )
    return rows


def _section_metric_rows(method_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for label, values in method_results[method]["by_section"].items():
            for metric, value in values.items():
                if "@" not in metric:
                    continue
                name, k = metric.split("@")
                rows.append(
                    {
                        "method": method,
                        "section_label": label,
                        "query_count": values["query_count"],
                        "metric": name,
                        "k": int(k),
                        "value": value,
                    }
                )
    return rows


def _render_audit_markdown(audit: Mapping[str, Any]) -> str:
    qa = audit["qa"]
    section_rows = "\n".join(
        f"| `{label}` | {count} |" for label, count in audit["section_distribution"].items()
    )
    flag_rows = "\n".join(
        f"| `{label}` | {count} |" for label, count in qa["automatic_flag_counts"].items()
    ) or "| (tidak ada) | 0 |"
    return f"""# Audit dataset Hierarchical SAC 2026

## Status sumber

- Format aktif: **{audit['source_format']}**.
- Dokumen XML: **{audit['xml_count']}**; PDF: **{audit['pdf_count']}**.
- Dokumen terproses: **{audit['processed_document_count']}**.
- Error integritas: **{len(audit['integrity_errors'])}**; XML rusak: **{len(audit['malformed_xml'])}**; dokumen kosong: **{len(audit['empty_documents'])}**.
- Duplikat berbasis SHA-256: **{audit['duplicate_content_groups']} kelompok**.
- Penilaian PDF/OCR: {audit['pdf_integrity_assessment']} {audit['ocr_assessment']}

Corpus ini tidak mendukung klaim bahwa pipeline PDF telah diuji pada data utama.

## Struktur dokumen

| Label bagian | Jumlah |
|---|---:|
{section_rows}

Semua posisi bagian diverifikasi terhadap teks dokumen. Jumlah error rekonstruksi:
**{len(audit['structure_errors'])}**.

## Benchmark QA

- Pertanyaan tersedia: **{qa['question_count']}**.
- Lolos pemeriksaan otomatis untuk analisis development: **{qa['included_exploratory_count']}**.
- Ditahan untuk pemeriksaan manusia: **{qa['excluded_pending_review_count']}**.
- Anotasi tervalidasi manusia: **{qa['human_validated_count']}**.
- Layak konfirmatori: **{qa['confirmatory_eligible_count']}**.

| Flag otomatis | Jumlah |
|---|---:|
{flag_rows}

Label berasal dari aturan deterministik atas XML, bukan anotasi manusia. Split holdout
lama telah dibuka dan dipakai berulang, sehingga seluruh evaluasi baru wajib disebut
eksperimen development eksploratori.
"""


def _render_methodology(
    config: Mapping[str, Any],
    chunks: Mapping[str, Any],
    representations: Mapping[str, Any],
    index_manifests: Mapping[str, Mapping[str, Any]],
    onnx_equivalence: Mapping[str, Any] | None,
) -> str:
    backend_rows = "\n".join(
        f"- {method}: `{manifest.get('embedding_backend', 'sentence_transformers_pytorch')}` "
        f"(runtime `{manifest.get('runtime_version') or manifest.get('torch_version')}`)."
        for method, manifest in index_manifests.items()
    )
    equivalence_note = (
        "Uji kesetaraan ONNX/PyTorch pada "
        f"{onnx_equivalence['probe_count']} input lulus: cosine minimum "
        f"{onnx_equivalence['minimum_cosine']:.9f}, selisih absolut maksimum "
        f"{onnx_equivalence['maximum_absolute_difference']:.3e}."
        if onnx_equivalence
        else "Fallback ONNX tidak digunakan."
    )
    return f"""# Metodologi B1/B2/M1

## Posisi terhadap Reuter et al. (2025)

Reuter et al. memperkenalkan Summary-Augmented Chunking (SAC): satu ringkasan
dokumen yang sama ditambahkan ke setiap chunk. Mereka tidak mengimplementasikan
ringkasan bagian. Hierarki paragraf-bagian-dokumen hanya disebut sebagai pekerjaan
lanjutan. M1 di sini adalah adaptasi dua tingkat, dokumen dan bagian, bukan algoritme
yang telah divalidasi oleh penulis. Sumber primer: {PAPER_URL}

## Variabel eksperimen

- **B1**: teks chunk asli.
- **B2**: ringkasan dokumen + teks chunk asli.
- **M1**: ringkasan dokumen yang sama dengan B2 + ringkasan bagian + teks chunk asli.

Ketiga metode memakai **{chunks['chunk_count']}** chunk canonical yang sama dan ID,
offset, serta bukti sumber yang identik. Alignment terverifikasi:
**{representations['alignment_verified']}**.

## Chunking

Recursive character splitting versi `{config['chunking']['implementation']}` memakai
ukuran {config['chunking']['chunk_size_characters']} karakter, overlap
{config['chunking']['chunk_overlap_characters']}, dan separator berurutan
`{config['chunking']['separators']}`. Penetapan bagian memakai overlap karakter
maksimum; chunk lintas batas disimpan sebagai mixed-section.

## Ringkasan

Model lokal `{config['summarization']['model_id']}` revisi
`{config['summarization']['model_revision']}` menghasilkan ringkasan bahasa Indonesia
dengan decoding greedy deterministik. Target 150 karakter dengan toleransi 20.
Bagian yang panjang memakai representasi awal-akhir berbasis token; bagian sangat
pendek disalin secara deterministik. Ringkasan dokumen memakai sampel seimbang
awal-akhir dari enam bagian hukum penting. B2 dan M1 membaca cache ringkasan dokumen
yang sama.

## Embedding dan retrieval

Model `{config['embedding']['model_id']}` revisi `{config['embedding']['model_revision']}`
memakai prefix `query:`/`passage:`, panjang maksimum 512 token, dan normalisasi L2.
Jika representasi augmented melebihi anggaran, token ringkasan dikurangi terlebih
dahulu; teks asli tidak boleh terpotong. Retrieval memakai exact dense cosine search,
tanpa BM25, reranker, metadata-as-text, atau contextual embedding.

Backend aktual per matriks:

{backend_rows}

Smart App Control Windows mengaktifkan blokir DLL PyTorch setelah B1/B2 selesai.
M1 memakai ekspor ONNX FP32 resmi dengan bobot `model.safetensors` yang identik.
{equivalence_note}

## Evaluasi dan inferensi

Endpoint primer ditetapkan sebelum hasil dibuka: **Recall@5**. Recall memakai unit
bukti sebagai denominator; MRR memakai rank chunk relevan pertama; nDCG memakai
grade 2 untuk chunk yang memuat span lengkap dan grade 1 untuk overlap parsial; DRM
adalah proporsi chunk top-K dari luar dokumen sumber yang benar. Interval keyakinan
95% memakai {config['statistics']['bootstrap_samples']} bootstrap berpasangan pada
cluster dokumen. Uji tambahan memakai sign-flip cluster dengan
{config['statistics']['permutation_samples']} replikasi.
"""


def _render_research_report(
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    summary_quality: Mapping[str, Any],
    chunk_manifest: Mapping[str, Any],
    representation_manifest: Mapping[str, Any],
    index_manifests: Mapping[str, Mapping[str, Any]],
    onnx_equivalence: Mapping[str, Any] | None,
) -> str:
    del index_manifests
    methods = evaluation["methods"]
    aggregate_rows = []
    for method in METHODS:
        values = methods[method]["aggregate_micro"]
        aggregate_rows.append(
            f"| {method} | {values['recall@5']:.4f} | {values['mrr@5']:.4f} | "
            f"{values['ndcg@5']:.4f} | {values['drm@5']:.4f} |"
        )
    comparison_rows = []
    for row in evaluation["comparisons"]:
        if row["metric"] not in {"recall@5", "mrr@5", "ndcg@5", "drm@5"}:
            continue
        comparison_rows.append(
            f"| {row['baseline']}→{row['candidate']} | {row['metric']} | "
            f"{row['absolute_difference']:+.4f} | "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
            f"{row['paired_cluster_sign_flip_p']:.4f} |"
        )
    section_rows = []
    labels = sorted(methods["B1"]["by_section"])
    for label in labels:
        section_rows.append(
            f"| `{label}` | {methods['B1']['by_section'][label]['query_count']} | "
            f"{methods['B1']['by_section'][label]['recall@5']:.4f} | "
            f"{methods['B2']['by_section'][label]['recall@5']:.4f} | "
            f"{methods['M1']['by_section'][label]['recall@5']:.4f} |"
        )
    b2m1 = next(
        row for row in evaluation["comparisons"]
        if row["baseline"] == "B2" and row["candidate"] == "M1" and row["metric"] == "recall@5"
    )
    b1b2 = next(
        row for row in evaluation["comparisons"]
        if row["baseline"] == "B1" and row["candidate"] == "B2" and row["metric"] == "recall@5"
    )
    b2m1_mrr = next(
        row for row in evaluation["comparisons"]
        if row["baseline"] == "B2" and row["candidate"] == "M1" and row["metric"] == "mrr@5"
    )
    b2m1_ndcg = next(
        row for row in evaluation["comparisons"]
        if row["baseline"] == "B2" and row["candidate"] == "M1" and row["metric"] == "ndcg@5"
    )
    b2m1_drm = next(
        row for row in evaluation["comparisons"]
        if row["baseline"] == "B2" and row["candidate"] == "M1" and row["metric"] == "drm@5"
    )
    failure_rows = []
    failure_categories = sorted(
        set().union(
            *(set(evaluation["failure_analysis"][method]["category_counts"]) for method in METHODS)
        )
    )
    for category in failure_categories:
        failure_rows.append(
            f"| `{category}` | "
            + " | ".join(
                str(evaluation["failure_analysis"][method]["category_counts"].get(category, 0))
                for method in METHODS
            )
            + " |"
        )
    equivalence_note = (
        f"Uji pada {onnx_equivalence['probe_count']} input menghasilkan cosine minimum "
        f"{onnx_equivalence['minimum_cosine']:.9f} dan selisih absolut maksimum "
        f"{onnx_equivalence['maximum_absolute_difference']:.3e}; status "
        f"**{onnx_equivalence['status']}**."
        if onnx_equivalence
        else "Fallback ONNX tidak digunakan."
    )
    conclusion = _comparison_conclusion(b2m1)
    return f"""# Laporan Penelitian: Hierarchical Summary-Augmented Chunking

## 1. Tujuan penelitian

Penelitian ini menguji apakah penambahan ringkasan tingkat bagian meningkatkan
retrieval bukti pada putusan pengadilan Indonesia dibanding Recursive Chunking (B1)
dan Summary-Augmented Chunking tingkat dokumen (B2). Kontribusi M1 adalah implementasi
dan evaluasi adaptasi dua tingkat, bukan klaim algoritme baru.

## 2. Rujukan ilmiah

Rujukan utama adalah Reuter et al. (2025), *Towards Reliable Retrieval in RAG
Systems for Large Legal Datasets* ({PAPER_URL}). Studi asal memakai ringkasan dokumen
saja, chunk 500 karakter tanpa overlap, dan mengusulkan hierarki ringkasan sebagai
pekerjaan mendatang. Penelitian ini membatasi hierarki pada dokumen dan bagian.

## 3. Karakteristik dataset

Corpus aktif memuat **{audit['xml_count']} dokumen XML Indo-Law**, bukan PDF, dari
putusan pidana khusus narkotika/psikotropika. Tersedia {audit['qa']['question_count']}
pertanyaan hasil aturan deterministik. Setelah pemeriksaan otomatis,
**{audit['qa']['included_exploratory_count']}** dipakai untuk analisis dan
{audit['qa']['excluded_pending_review_count']} ditahan. Tidak ada pertanyaan yang telah
divalidasi manusia; holdout historis juga telah dibuka. Karena itu hasil berikut
adalah **development exploration**, bukan hasil holdout konfirmatori.

## 4. Metodologi

- B1 meng-embed teks asli.
- B2 menambahkan satu ringkasan dokumen yang dipakai ulang untuk seluruh chunk dokumen.
- M1 memakai ringkasan dokumen B2 yang sama dan menambahkan ringkasan bagian terkait.

Sebanyak **{chunk_manifest['chunk_count']}** chunk canonical dibuat satu kali. Semua
metode mempunyai ID dan batas chunk identik. Penetapan bagian menggunakan overlap
karakter maksimum terhadap tag bagian XML. Embedding menggunakan
`{config['embedding']['model_id']}` dan cosine similarity.

B1/B2 dihitung dengan PyTorch. Setelah Windows Smart App Control memblokir DLL
PyTorch yang tidak ditandatangani, M1 dilanjutkan memakai ekspor ONNX FP32 resmi dari
bobot model yang sama. {equivalence_note} Dengan demikian fallback dicatat dan diuji,
bukan diasumsikan ekuivalen.

## 5. Desain eksperimen dan metrik

Endpoint primer adalah Recall@5. Metrik pelengkap ialah MRR@K, nDCG@K, dan DRM@K
untuk K=1,3,5,10; K=20 dan 50 bersifat diagnostik. DRM lebih rendah lebih baik.
Perbandingan berpasangan memakai bootstrap cluster dokumen dan uji sign-flip.

## 6. Hasil retrieval

| Metode | Recall@5 | MRR@5 | nDCG@5 | DRM@5 ↓ |
|---|---:|---:|---:|---:|
{chr(10).join(aggregate_rows)}

## 7. Analisis statistik

| Perbandingan | Metrik | Selisih | 95% CI | p sign-flip |
|---|---|---:|---:|---:|
{chr(10).join(comparison_rows)}

Kesimpulan endpoint primer B2→M1: {conclusion}

B2−B1 pada Recall@5 adalah {b1b2['absolute_difference']:+.4f} dengan CI 95%
[{b1b2['ci95_low']:+.4f}, {b1b2['ci95_high']:+.4f}] dan p=
{b1b2['paired_cluster_sign_flip_p']:.4f}. M1−B2 pada MRR@5 adalah
{b2m1_mrr['absolute_difference']:+.4f}, pada nDCG@5
{b2m1_ndcg['absolute_difference']:+.4f}, dan pada DRM@5
{b2m1_drm['absolute_difference']:+.4f} (lebih rendah lebih baik). Nilai p masing-masing
{b2m1_mrr['paired_cluster_sign_flip_p']:.4f},
{b2m1_ndcg['paired_cluster_sign_flip_p']:.4f}, dan
{b2m1_drm['paired_cluster_sign_flip_p']:.4f}.

Nilai p bersifat pelengkap dan tidak mengubah status eksploratori dataset. Interval
yang melintasi nol tidak mendukung klaim peningkatan yang meyakinkan.

## 8. Hasil per bagian

| Bagian | N | B1 Recall@5 | B2 Recall@5 | M1 Recall@5 |
|---|---:|---:|---:|---:|
{chr(10).join(section_rows)}

Bagian yang tidak mempunyai pertanyaan valid tidak dinilai; tidak ada angka yang
diimputasi. Distribusi ini terutama menguji identitas, dakwaan, pertimbangan hukum,
amar, dan sedikit riwayat penahanan.

Efek M1 tidak seragam: terhadap B2, Recall@5 amar putusan naik dari
{methods['B2']['by_section']['amar_putusan']['recall@5']:.4f} menjadi
{methods['M1']['by_section']['amar_putusan']['recall@5']:.4f}, sedangkan riwayat
dakwaan turun dari {methods['B2']['by_section']['riwayat_dakwaan']['recall@5']:.4f}
menjadi {methods['M1']['by_section']['riwayat_dakwaan']['recall@5']:.4f}.

## 9. Diagnostik ringkasan dan kegagalan

Pipeline menghasilkan {summary_quality['overall']['document_summary_count']} ringkasan
dokumen dan {summary_quality['overall']['section_summary_count']} ringkasan bagian.
Sebanyak {summary_quality['overall']['flagged_count']} keluaran menerima sedikitnya
satu flag otomatis; sampel pemeriksaan manusia telah disiapkan tetapi belum dinilai.
Token-budget memengaruhi B2 pada
{representation_manifest['methods']['B2']['token_budget_affected']} chunk dan M1 pada
{representation_manifest['methods']['M1']['token_budget_affected']} chunk. Semua teks
bukti asli dipertahankan.

Rincian kategori salah dokumen, salah bagian, salah passage, dan bukti di luar top-5
tersedia pada `failure_analysis.json`. Kategori dihitung dari log retrieval, bukan
interpretasi yang dibuat tanpa bukti.

| Kategori top-5 | B1 | B2 | M1 |
|---|---:|---:|---:|
{chr(10).join(failure_rows)}

## 10. Diskusi

Hasil perlu dibaca sebagai efek representasi embedding pada corpus XML yang sangat
terstruktur. B2 dapat mengurangi salah dokumen dengan menambahkan identitas global,
tetapi ringkasan global juga dapat menutupi sinyal lokal. M1 menambah konteks bagian;
manfaatnya bergantung pada ketepatan asosiasi bagian dan kapasitas E5-small untuk
memadatkan tiga komponen ke satu vektor. Arah efek aktual dilaporkan apa adanya dan
tidak dipaksa positif.

Secara agregat, M1 memindahkan lebih banyak bukti relevan ke rank awal (Recall@1,
MRR@5, dan nDCG@5 meningkat), tetapi Recall@5 turun karena kerugian besar pada
pertanyaan dakwaan. Pada K=20/50, Recall M1 kembali melampaui B2. Pola ini mendukung
interpretasi bahwa ringkasan bagian mengubah urutan kandidat secara tajam, bukan
meningkatkan semua tipe bukti secara seragam.

## 11. Keterbatasan dan ancaman validitas

1. Tidak ada PDF dalam corpus aktif, sehingga ekstraksi, OCR, header/footer, dan page
   provenance PDF belum tervalidasi secara empiris untuk eksperimen utama.
2. Tag bagian berasal dari XML Indo-Law dan merupakan kondisi oracle relatif terhadap
   detector otomatis pada PDF.
3. Pertanyaan dan span bukti belum divalidasi manusia. Flag otomatis bukan pengganti
   adjudikasi ahli.
4. Holdout 160 dokumen lama telah dibuka dan dipakai pada eksperimen historis.
5. Model ringkasan lokal berbeda dari GPT-4o-mini pada Reuter et al.; E5-small juga
   berbeda dari GTE-large. Ini adalah substitusi tercatat, bukan reproduksi numerik.
6. Corpus hanya mencakup satu kategori perkara, sehingga generalisasi ke jenis perkara
   lain belum didukung.

## 12. Kesimpulan

Pipeline B1/B2/M1 telah dijalankan dengan chunk dan benchmark yang sama. Temuan saat
ini sah sebagai bukti development eksploratori. Klaim final skripsi tetap menunggu
holdout baru yang dibekukan dan pertanyaan bukti yang divalidasi manusia.

## 13. Reproduksi

Gunakan `configs/hierarchical_sac_2026.yaml` dan jalankan:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\27_run_hierarchical_sac.py --config configs\\hierarchical_sac_2026.yaml --stage all
```
"""


def _comparison_conclusion(row: Mapping[str, Any]) -> str:
    difference = float(row["absolute_difference"])
    low, high = float(row["ci95_low"]), float(row["ci95_high"])
    if difference > 0 and low > 0:
        return (
            f"M1 lebih tinggi {difference:+.4f}; CI 95% [{low:+.4f}, {high:+.4f}] "
            "tidak melintasi nol pada benchmark development ini."
        )
    if difference < 0 and high < 0:
        return (
            f"M1 lebih rendah {difference:+.4f}; CI 95% [{low:+.4f}, {high:+.4f}] "
            "tidak melintasi nol pada benchmark development ini."
        )
    return (
        f"selisih M1-B2 {difference:+.4f} dengan CI 95% [{low:+.4f}, {high:+.4f}] "
        "melintasi nol; tidak ada bukti peningkatan yang konklusif."
    )


def _render_reproduction(config: Mapping[str, Any]) -> str:
    return f"""# Reproduksi eksperimen

## Lingkungan

- Python 3.12
- Dependensi: `pip install -r requirements-research.txt`
- Model lokal: `{config['summarization']['model_id']}` revisi
  `{config['summarization']['model_revision']}` dan `{config['embedding']['model_id']}`
  revisi `{config['embedding']['model_revision']}`.
- API berbayar tidak digunakan.
- Jika PyTorch diblokir Windows Smart App Control, runner memakai ekspor ONNX FP32
  resmi revisi `{config['embedding']['onnx_export_revision']}` dengan checksum
  `{config['embedding']['onnx_export_sha256']}` dan mewajibkan uji equivalence terhadap
  cache PyTorch sebelum mencampur backend.

## Perintah

```powershell
.\\.venv\\Scripts\\python.exe scripts\\18_restore_indolaw_from_manifest.py
.\\.venv\\Scripts\\python.exe scripts\\12_prepare_indolaw_corpus.py
.\\.venv\\Scripts\\python.exe scripts\\27_run_hierarchical_sac.py --config configs\\hierarchical_sac_2026.yaml --stage all
.\\.venv\\Scripts\\python.exe -m pytest
```

Setiap tahap memakai hash sumber dan cache. Gunakan `--force` hanya jika memang ingin
menghasilkan ulang ringkasan/embedding dengan konfigurasi yang sama. Seed statistik
adalah {config['experiment']['seed']}.
"""


def _write_plots(evaluation: Mapping[str, Any], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = evaluation["methods"]
    for metric in ("recall", "mrr", "ndcg", "drm"):
        ks = (1, 3, 5, 10, 20, 50)
        series = {
            method: [float(methods[method]["aggregate_micro"][f"{metric}@{k}"]) for k in ks]
            for method in METHODS
        }
        (figure_dir / f"{metric}_at_k.svg").write_text(
            _line_chart_svg(
                title=f"{metric.upper()}@K - B1, B2, dan M1",
                x_values=ks,
                series=series,
                y_label=f"{metric.upper()} (proporsi)",
            ),
            encoding="utf-8",
        )
    labels = sorted(methods["B1"]["by_section"])
    section_series = {
        method: [float(methods[method]["by_section"][label]["recall@5"]) for label in labels]
        for method in METHODS
    }
    (figure_dir / "per_section_recall_at_5.svg").write_text(
        _bar_chart_svg("Recall@5 per bagian", labels, section_series), encoding="utf-8"
    )
    recall_comparisons = [
        row for row in evaluation["comparisons"] if row["metric"] == "recall@5"
    ]
    (figure_dir / "recall_at_5_uncertainty.svg").write_text(
        _interval_chart_svg("Selisih Recall@5 dengan CI 95%", recall_comparisons),
        encoding="utf-8",
    )


def _line_chart_svg(
    *, title: str, x_values: Sequence[int], series: Mapping[str, Sequence[float]], y_label: str
) -> str:
    width, height, left, right, top, bottom = 760, 440, 74, 24, 54, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(max(values) for values in series.values())
    y_max = min(1.0, max(0.1, math.ceil(maximum * 10 + 1) / 10))
    colors = {"B1": "#3366cc", "B2": "#dc7a00", "M1": "#198754"}
    x_positions = [left + index * plot_w / (len(x_values) - 1) for index in range(len(x_values))]
    parts = [_svg_header(width, height, title)]
    for tick in range(6):
        value = y_max * tick / 5
        y = top + plot_h - value / y_max * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8d8d8"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.2f}</text>')
    for x, value in zip(x_positions, x_values):
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle">{value}</text>')
    for method, values in series.items():
        points = " ".join(
            f"{x:.1f},{top + plot_h - value / y_max * plot_h:.1f}"
            for x, value in zip(x_positions, values)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[method]}" stroke-width="3"/>')
        for x, value in zip(x_positions, values):
            y = top + plot_h - value / y_max * plot_h
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[method]}"/>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-12}" text-anchor="middle">K</text>')
    parts.append(f'<text transform="translate(18 {top+plot_h/2}) rotate(-90)" text-anchor="middle">{y_label}</text>')
    for index, method in enumerate(METHODS):
        x = left + index * 92
        parts.append(f'<line x1="{x}" y1="34" x2="{x+24}" y2="34" stroke="{colors[method]}" stroke-width="3"/>')
        parts.append(f'<text x="{x+30}" y="38">{method}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _bar_chart_svg(
    title: str, labels: Sequence[str], series: Mapping[str, Sequence[float]]
) -> str:
    width = max(820, 150 * len(labels))
    height, left, right, top, bottom = 480, 70, 20, 58, 130
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(labels)
    bar_w = group_w * 0.22
    colors = {"B1": "#3366cc", "B2": "#dc7a00", "M1": "#198754"}
    parts = [_svg_header(width, height, title)]
    for tick in range(6):
        value = tick / 5
        y = top + plot_h - value * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d8d8d8"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{value:.1f}</text>')
    for label_index, label in enumerate(labels):
        center = left + (label_index + 0.5) * group_w
        for method_index, method in enumerate(METHODS):
            value = float(series[method][label_index])
            x = center + (method_index - 1) * bar_w - bar_w / 2
            y = top + plot_h - value * plot_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{value*plot_h:.1f}" fill="{colors[method]}"/>')
        escaped = label.replace("_", " ")
        parts.append(f'<text transform="translate({center:.1f} {height-bottom+18}) rotate(35)" text-anchor="start">{escaped}</text>')
    for index, method in enumerate(METHODS):
        x = left + index * 92
        parts.append(f'<rect x="{x}" y="28" width="20" height="12" fill="{colors[method]}"/>')
        parts.append(f'<text x="{x+28}" y="39">{method}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _interval_chart_svg(title: str, rows: Sequence[Mapping[str, Any]]) -> str:
    width, height, left, right, top, bottom = 760, 300, 140, 40, 54, 42
    values = [float(row[key]) for row in rows for key in ("ci95_low", "ci95_high")]
    extent = max(abs(min(values)), abs(max(values)), 0.02) * 1.15
    plot_w = width - left - right
    scale = lambda value: left + (value + extent) / (2 * extent) * plot_w
    parts = [_svg_header(width, height, title)]
    parts.append(f'<line x1="{scale(0):.1f}" y1="{top}" x2="{scale(0):.1f}" y2="{height-bottom}" stroke="#555" stroke-dasharray="4 4"/>')
    for index, row in enumerate(rows):
        y = top + 42 + index * 56
        label = f"{row['baseline']}→{row['candidate']}"
        low, high, point = (float(row[key]) for key in ("ci95_low", "ci95_high", "absolute_difference"))
        parts.append(f'<text x="{left-14}" y="{y+4}" text-anchor="end">{label}</text>')
        parts.append(f'<line x1="{scale(low):.1f}" y1="{y}" x2="{scale(high):.1f}" y2="{y}" stroke="#3366cc" stroke-width="3"/>')
        parts.append(f'<circle cx="{scale(point):.1f}" cy="{y}" r="6" fill="#198754"/>')
        parts.append(f'<text x="{scale(high)+8:.1f}" y="{y+4}">{point:+.3f}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-10}" text-anchor="middle">Selisih Recall@5 (candidate - baseline)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#222}'
        '.title{font-size:18px;font-weight:bold}</style>\n'
        f'<text class="title" x="{width/2}" y="24" text-anchor="middle">{title}</text>'
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(json.dumps(dict(row), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
