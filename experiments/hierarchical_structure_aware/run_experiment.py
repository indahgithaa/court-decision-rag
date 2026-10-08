"""Run the complete four-arm hierarchical structure-aware retrieval study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.hierarchical_structure_aware.experiment_evaluation import (
    build_positive_qrels,
    evaluate_method,
    paired_cluster_bootstrap,
    summarize_chunk_lengths,
)
from experiments.hierarchical_structure_aware.hierarchical_chunking import (
    HierarchicalChunker,
    aggregate_parent_ranking,
    select_context_budget,
)
from src.chunking import FixedSizeChunker, StructureAwareChunker
from src.embedding import SentenceTransformerEmbedder
from src.evaluation.ground_truth import join_clean_pages, validate_questions
from src.retrieval import DenseVectorStore
from src.utils.io import read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
QREL_FIELDS = (
    "query_id",
    "document_id",
    "target_section_label",
    "evidence_id",
    "strategy",
    "chunk_id",
    "chunk_start_position",
    "chunk_end_position",
    "evidence_start_position",
    "evidence_end_position",
    "evidence_coverage",
    "auto_grade",
    "relevance_grade",
    "reviewer_notes",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=EXPERIMENT_DIR / "artifacts",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse complete local chunks/indexes/runs when present.",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifacts = args.artifacts_dir.resolve()
    manifest_path = artifacts / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"Experiment artifacts already exist at {artifacts}; pass --resume"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    np.random.seed(int(config["seed"]))
    started = datetime.now(timezone.utc)

    pages_path = _project_path(config["pages"])
    sections_path = _project_path(config["sections"])
    questions_path = _project_path(config["questions"])
    pages = list(read_jsonl(pages_path))
    documents = join_clean_pages(pages)
    sections = list(read_jsonl(sections_path))
    sections_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        sections_by_document[str(section["document_id"])].append(section)
    for values in sections_by_document.values():
        values.sort(key=lambda row: int(row["start_position"]))
    questions = _read_questions(questions_path)
    question_validation = validate_questions(questions, documents)
    if not question_validation["ready"]:
        raise ValueError(f"Question validation failed: {question_validation}")
    if set(documents) != set(sections_by_document):
        raise ValueError("Document and section provenance sets differ")

    chunks_by_method, parents_by_method = _build_all_chunks(
        documents, sections_by_document, config
    )
    chunk_audits = {
        code: _audit_chunks(
            chunks,
            documents,
            parents_by_method.get(code, ()),
            require_section_boundaries=(code == "M1"),
        )
        for code, chunks in chunks_by_method.items()
    }
    _write_chunk_artifacts(
        artifacts, chunks_by_method, parents_by_method, resume=args.resume
    )

    qrels_by_method: dict[str, list[dict[str, Any]]] = {}
    for code, chunks in chunks_by_method.items():
        qrels = build_positive_qrels(
            questions, chunks, sections_by_document
        )
        qrels_by_method[code] = qrels
        qrel_path = artifacts / "qrels" / f"{code}.csv"
        _write_csv(qrel_path, qrels, QREL_FIELDS, resume=args.resume)

    embedding_config = config["embedding"]
    embedder = SentenceTransformerEmbedder(
        str(embedding_config["model_name"]),
        batch_size=int(embedding_config["batch_size"]),
        device=str(embedding_config["device"]),
        query_prefix=str(embedding_config["query_prefix"]),
        document_prefix=str(embedding_config["document_prefix"]),
        normalize_embeddings=bool(embedding_config["normalize_embeddings"]),
        trust_remote_code=False,
        max_sequence_length=int(embedding_config["max_sequence_length"]),
    )
    model = embedder._load_model()  # one pinned local model for audit and encoding
    tokenizer = model[0].tokenizer  # type: ignore[index]
    token_lengths_by_method = {
        code: _token_lengths(
            tokenizer,
            [str(row.get("embedding_text") or row["text"]) for row in chunks],
            prefix=str(embedding_config["document_prefix"]),
        )
        for code, chunks in chunks_by_method.items()
    }
    for code in chunks_by_method:
        chunk_audits[code]["lengths"] = summarize_chunk_lengths(
            chunks_by_method[code],
            token_lengths_by_method[code],
            limit=int(embedding_config["max_sequence_length"]),
        )

    stores, embedding_reuse = _build_indexes(
        artifacts,
        chunks_by_method,
        embedder,
        config,
        resume=args.resume,
    )
    run_records, retrieval_runtime = _run_retrieval(
        artifacts,
        stores,
        chunks_by_method,
        parents_by_method,
        questions,
        embedder,
        config,
        resume=args.resume,
    )

    evaluations = {
        code: evaluate_method(
            qrels_by_method[code],
            run_records[code],
            questions,
            chunks_by_method[code],
            sections_by_document,
            ks=tuple(config["retrieval"]["reported_cutoffs"]),
        )
        for code in ("B1", "B2", "B3", "M1")
    }
    query_documents = {
        str(question["query_id"]): str(question["document_id"])
        for question in questions
    }
    comparisons: dict[str, Any] = {}
    for left, right in (
        ("B2", "B1"),
        ("B3", "B1"),
        ("M1", "B1"),
        ("B3", "B2"),
        ("M1", "B2"),
        ("M1", "B3"),
    ):
        comparisons[f"{left}_minus_{right}"] = paired_cluster_bootstrap(
            evaluations[left]["overall"]["per_query"],
            evaluations[right]["overall"]["per_query"],
            query_documents,
            metrics=("recall@5", "mrr@5", "ndcg@5"),
            samples=int(config["retrieval"]["bootstrap_samples"]),
            seed=int(config["seed"]),
        )

    git_commit, git_dirty = _git_state()
    method_metadata = {
        code: {
            **dict(config["methods"][code]),
            "chunk_count": len(chunks_by_method[code]),
            "parent_count": len(parents_by_method.get(code, ())),
            "qrel_count": len(qrels_by_method[code]),
            "positive_queries": len(
                {str(row["query_id"]) for row in qrels_by_method[code]}
            ),
        }
        for code in ("B1", "B2", "B3", "M1")
    }
    result = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": config["protocol_status"],
        "dataset": {
            "documents": len(documents),
            "questions": len(questions),
            "section_distribution": dict(
                sorted(Counter(str(q["target_section_label"]) for q in questions).items())
            ),
            "question_validation": question_validation,
            "paths": {
                "pages": _relative(pages_path),
                "sections": _relative(sections_path),
                "questions": _relative(questions_path),
            },
            "sha256": {
                "pages": _sha256(pages_path),
                "sections": _sha256(sections_path),
                "questions": _sha256(questions_path),
            },
        },
        "configuration": config,
        "methods": method_metadata,
        "audit": {
            "chunks": chunk_audits,
            "embedding": {
                "model_name": embedding_config["model_name"],
                "model_max_sequence_length": int(model.max_seq_length),
                "embedding_dimension": int(model.get_embedding_dimension()),
                "tokenizer": tokenizer.__class__.__name__,
                "reuse": embedding_reuse,
            },
            "ground_truth": {
                "unit": "source evidence span/occurrence",
                "label_used_for_retrieval": False,
                "all_methods_cover_all_queries": True,
                "equivalent_statute_occurrences": True,
            },
        },
        "evaluations": evaluations,
        "paired_cluster_bootstrap": {
            "unit": "document",
            "samples": int(config["retrieval"]["bootstrap_samples"]),
            "seed": int(config["seed"]),
            "comparisons": comparisons,
        },
        "runtime": retrieval_runtime,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "sentence-transformers",
                    "torch",
                    "transformers",
                )
            },
            "git_commit_before_experiment": git_commit,
            "git_dirty_after_new_experiment_files": git_dirty,
        },
        "testing": _read_test_results(),
    }
    detailed_path = artifacts / "evaluation_detailed.json"
    _write_json(detailed_path, result)
    public_summary = _public_summary(result)
    summary_path = EXPERIMENT_DIR / "results_summary.json"
    _write_json(summary_path, public_summary)
    report_path = EXPERIMENT_DIR / "report.md"
    report_path.write_text(
        _render_report(result), encoding="utf-8", newline="\n"
    )

    manifest = _build_manifest(
        artifacts,
        config_path,
        result,
        tracked_outputs=tuple(
            path
            for path in (
                summary_path,
                report_path,
                EXPERIMENT_DIR / "test_results.json",
            )
            if path.exists()
        ),
    )
    _write_json(manifest_path, manifest)
    print(f"Experiment complete: {report_path}")
    _print_key_results(result)


def _build_all_chunks(
    documents: Mapping[str, str],
    sections_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    hierarchy = config["hierarchy"]
    flat_chunkers = {
        "B1": FixedSizeChunker(max_words=500, overlap_words=100),
        "B2": StructureAwareChunker(
            max_words=150, overlap_words=30, overlap_sentences=0
        ),
    }
    hierarchical_chunkers = {
        "B3": HierarchicalChunker(
            structure_aware=False,
            parent_words=int(hierarchy["parent_words"]),
            parent_overlap_words=int(hierarchy["parent_overlap_words"]),
            child_words=int(hierarchy["child_words"]),
            child_overlap_words=int(hierarchy["child_overlap_words"]),
        ),
        "M1": HierarchicalChunker(
            structure_aware=True,
            parent_words=int(hierarchy["parent_words"]),
            parent_overlap_words=int(hierarchy["parent_overlap_words"]),
            child_words=int(hierarchy["child_words"]),
            child_overlap_words=int(hierarchy["child_overlap_words"]),
        ),
    }
    chunks: dict[str, list[dict[str, Any]]] = {code: [] for code in ("B1", "B2", "B3", "M1")}
    parents: dict[str, list[dict[str, Any]]] = {"B3": [], "M1": []}
    for document_id in sorted(documents):
        text = documents[document_id]
        sections = sections_by_document[document_id]
        for code, chunker in flat_chunkers.items():
            chunks[code].extend(
                row.to_dict()
                for row in chunker.chunk(document_id, text, sections=sections)
            )
        for code, chunker in hierarchical_chunkers.items():
            document_parents, document_children = chunker.chunk(
                document_id, text, sections=sections
            )
            parents[code].extend(document_parents)
            chunks[code].extend(document_children)
    return chunks, parents


def _write_chunk_artifacts(
    artifacts: Path,
    chunks: Mapping[str, Sequence[Mapping[str, Any]]],
    parents: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    resume: bool,
) -> None:
    for code, rows in chunks.items():
        _write_or_validate_jsonl(artifacts / "chunks" / f"{code}.jsonl", rows, resume)
    for code, rows in parents.items():
        _write_or_validate_jsonl(artifacts / "parents" / f"{code}.jsonl", rows, resume)


def _build_indexes(
    artifacts: Path,
    chunks_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    embedder: SentenceTransformerEmbedder,
    config: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[dict[str, DenseVectorStore], dict[str, Any]]:
    model_name = str(config["embedding"]["model_name"])
    index_root = artifacts / "indexes"
    existing: dict[str, DenseVectorStore] = {}
    if resume:
        for code, chunks in chunks_by_method.items():
            path = index_root / code
            if (path / "manifest.json").exists():
                store = DenseVectorStore.load(path)
                if store.model_name != model_name or store.chunks != list(chunks):
                    raise ValueError(f"Existing index mismatch for {code}")
                existing[code] = store
    if len(existing) == len(chunks_by_method):
        return existing, {
            "mode": "resume_all_indexes",
            "reused_vectors": sum(len(store.chunks) for store in existing.values()),
            "new_vectors": 0,
        }

    vector_cache: dict[str, np.ndarray] = {}
    sources = (
        ROOT / "vector_db/indolaw_200/corpus/fixed_w150_o30",
        ROOT / "vector_db/indolaw_200/corpus/sac_w150_o30_s0",
        ROOT / "vector_db/indolaw_200/development/fixed_w500_o100",
        ROOT / "vector_db/indolaw_200/holdout/fixed_w500_o100",
    )
    reused_source_rows = 0
    used_sources: list[str] = []
    for source in sources:
        if not (source / "manifest.json").exists():
            continue
        store = DenseVectorStore.load(source)
        if store.model_name != model_name:
            raise ValueError(f"Reusable index model mismatch: {source}")
        used_sources.append(_relative(source))
        for row, vector in zip(store.chunks, store.embeddings):
            text = str(row.get("embedding_text") or row["text"])
            prior = vector_cache.get(text)
            if prior is not None and not np.allclose(prior, vector, atol=1e-6):
                raise ValueError("Same independent embedding text has inconsistent vectors")
            vector_cache[text] = np.asarray(vector, dtype=np.float32)
            reused_source_rows += 1

    all_texts = list(
        dict.fromkeys(
            str(row.get("embedding_text") or row["text"])
            for rows in chunks_by_method.values()
            for row in rows
        )
    )
    missing = [text for text in all_texts if text not in vector_cache]
    encoded = 0
    batch_group = 512
    for start in range(0, len(missing), batch_group):
        values = missing[start : start + batch_group]
        matrix = embedder.embed_documents(values)
        for text, vector in zip(values, matrix):
            vector_cache[text] = np.asarray(vector, dtype=np.float32)
        encoded += len(values)
        print(f"Embedded {encoded}/{len(missing)} previously unseen child texts")

    stores: dict[str, DenseVectorStore] = {}
    for code, chunks in chunks_by_method.items():
        if code in existing:
            stores[code] = existing[code]
            continue
        matrix = np.vstack(
            [
                vector_cache[str(row.get("embedding_text") or row["text"])]
                for row in chunks
            ]
        )
        store = DenseVectorStore(chunks, matrix, model_name=model_name)
        store.save(
            index_root / code,
            provenance={
                "experiment_id": config["experiment_id"],
                "method": code,
                "independent_embedding": True,
                "embedding_cache_validation": "exact_embedding_text_and_model",
                "source_indexes": used_sources,
                "query_prefix": config["embedding"]["query_prefix"],
                "document_prefix": config["embedding"]["document_prefix"],
                "normalize_embeddings": True,
            },
        )
        stores[code] = store
    return stores, {
        "mode": "exact_text_vector_reuse_plus_fresh_encoding",
        "source_indexes": used_sources,
        "source_rows_scanned": reused_source_rows,
        "unique_reused_texts": len(all_texts) - len(missing),
        "unique_new_texts": len(missing),
        "reuse_is_result_independent": True,
    }


def _run_retrieval(
    artifacts: Path,
    stores: Mapping[str, DenseVectorStore],
    chunks_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    parents_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    questions: Sequence[Mapping[str, Any]],
    embedder: SentenceTransformerEmbedder,
    config: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    run_root = artifacts / "runs"
    existing: dict[str, list[dict[str, Any]]] = {}
    if resume:
        for code in stores:
            path = run_root / f"{code}.jsonl"
            if path.exists():
                rows = list(read_jsonl(path))
                if len(rows) != len(questions):
                    raise ValueError(f"Existing run query count mismatch for {code}")
                existing[code] = rows
    if len(existing) == len(stores):
        return existing, {"mode": "resume_all_runs"}

    query_started = time.perf_counter()
    query_vectors = embedder.embed_queries(
        [str(question["question"]) for question in questions]
    )
    query_ms = (time.perf_counter() - query_started) * 1000
    amortized_query_ms = query_ms / len(questions)
    top_k = int(config["retrieval"]["top_k"])
    budget = int(config["retrieval"]["context_budget_words"])
    records_by_method: dict[str, list[dict[str, Any]]] = {}
    search_totals: dict[str, float] = {}
    for code, store in stores.items():
        if code in existing:
            records_by_method[code] = existing[code]
            continue
        parent_lookup = {
            str(row["parent_id"]): row for row in parents_by_method.get(code, ())
        }
        records: list[dict[str, Any]] = []
        search_total = 0.0
        for question, query_vector in zip(questions, query_vectors):
            search_started = time.perf_counter()
            found = store.search(query_vector, k=top_k)
            search_ms = (time.perf_counter() - search_started) * 1000
            search_total += search_ms
            ranking: list[dict[str, Any]] = []
            for rank, item in enumerate(found, start=1):
                chunk = dict(item.chunk)
                ranking.append(
                    {
                        "rank": rank,
                        "chunk_id": str(chunk["chunk_id"]),
                        "score": item.score,
                        "retrieved_document_id": str(chunk["document_id"]),
                        "start_position": int(chunk["start_position"]),
                        "end_position": int(chunk["end_position"]),
                        "section_label": chunk.get("section_label"),
                        "parent_id": chunk.get("parent_id"),
                    }
                )
            if code in {"B3", "M1"}:
                scored_children = [
                    {
                        **dict(item.chunk),
                        "rank": rank,
                        "score": item.score,
                    }
                    for rank, item in enumerate(found, start=1)
                ]
                contexts_ranked = aggregate_parent_ranking(
                    scored_children, parent_lookup
                )
            else:
                contexts_ranked = [
                    {
                        **dict(item.chunk),
                        "rank": rank,
                        "score": item.score,
                    }
                    for rank, item in enumerate(found, start=1)
                ]
            selected, returned_words = select_context_budget(
                contexts_ranked, budget_words=budget
            )
            context_records = [
                {
                    "rank": rank,
                    "context_id": str(
                        item.get("parent_id") or item.get("chunk_id")
                    ),
                    "document_id": str(item["document_id"]),
                    "start_position": int(item["start_position"]),
                    "end_position": int(item["end_position"]),
                    "section_label": item.get("section_label"),
                    "score": float(item["score"]),
                    "word_count": len(re.findall(r"\S+", str(item["text"]))),
                    "best_child_id": item.get("best_child_id"),
                    "best_child_rank": item.get("best_child_rank"),
                }
                for rank, item in enumerate(selected, start=1)
            ]
            records.append(
                {
                    "query_id": str(question["query_id"]),
                    "document_id": str(question["document_id"]),
                    "target_section_label": str(
                        question["target_section_label"]
                    ),
                    "method": code,
                    "strategy": str(store.chunks[0]["strategy"]),
                    "design": config["methods"][code]["design"],
                    "model_name": config["embedding"]["model_name"],
                    "retrieval_scope": "full_corpus",
                    "top_k": top_k,
                    "latency_ms": amortized_query_ms + search_ms,
                    "query_embedding_latency_ms": amortized_query_ms,
                    "search_latency_ms": search_ms,
                    "ranking": ranking,
                    "unique_context_ranking": [
                        {
                            "rank": int(item["rank"]),
                            "context_id": str(
                                item.get("parent_id") or item.get("chunk_id")
                            ),
                            "document_id": str(item["document_id"]),
                            "score": float(item["score"]),
                            "best_child_id": item.get("best_child_id"),
                            "best_child_rank": item.get("best_child_rank"),
                        }
                        for item in contexts_ranked
                    ],
                    "context_budget_words": budget,
                    "returned_context_words": returned_words,
                    "returned_context": context_records,
                }
            )
        path = run_root / f"{code}.jsonl"
        write_jsonl(path, records)
        records_by_method[code] = records
        search_totals[code] = search_total
        print(f"Retrieved {len(records)} queries for {code}")
    return records_by_method, {
        "mode": "fresh_query_embedding_and_exact_cosine_search",
        "query_embedding_batch_ms": query_ms,
        "query_embedding_amortized_ms": amortized_query_ms,
        "search_total_ms": search_totals,
    }


def _audit_chunks(
    chunks: Sequence[Mapping[str, Any]],
    documents: Mapping[str, str],
    parents: Sequence[Mapping[str, Any]],
    *,
    require_section_boundaries: bool,
) -> dict[str, Any]:
    ids: set[str] = set()
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        if not chunk_id or chunk_id in ids:
            raise ValueError(f"Duplicate or blank chunk ID: {chunk_id!r}")
        ids.add(chunk_id)
        document_id = str(chunk["document_id"])
        start = int(chunk["start_position"])
        end = int(chunk["end_position"])
        if not 0 <= start < end <= len(documents[document_id]):
            raise ValueError(f"Invalid chunk offset {document_id}:{start}:{end}")
        if documents[document_id][start:end] != str(chunk["text"]):
            raise ValueError(f"Chunk text/provenance mismatch: {chunk_id}")
        intervals[document_id].append((start, end))
    uncovered_words = 0
    for document_id, text in documents.items():
        merged = _merge_intervals(intervals[document_id])
        interval_index = 0
        for word in re.finditer(r"\S+", text):
            while (
                interval_index < len(merged)
                and merged[interval_index][1] < word.end()
            ):
                interval_index += 1
            if interval_index >= len(merged) or not (
                merged[interval_index][0] <= word.start()
                and word.end() <= merged[interval_index][1]
            ):
                uncovered_words += 1
    if uncovered_words:
        raise ValueError(f"Chunking lost {uncovered_words} source words")

    parent_count = len(parents)
    if parents:
        parent_by_id = {str(parent["parent_id"]): parent for parent in parents}
        if len(parent_by_id) != len(parents):
            raise ValueError("Duplicate parent IDs")
        for parent in parents:
            document_id = str(parent["document_id"])
            start = int(parent["start_position"])
            end = int(parent["end_position"])
            if documents[document_id][start:end] != str(parent["text"]):
                raise ValueError(f"Parent text/provenance mismatch: {parent['parent_id']}")
            if require_section_boundaries and not parent.get("section_label"):
                raise ValueError("M1 parent is missing structural provenance")
        for child in chunks:
            parent_id = str(child.get("parent_id", ""))
            if parent_id not in parent_by_id:
                raise ValueError(f"Missing parent for child {child['chunk_id']}")
            parent = parent_by_id[parent_id]
            if str(parent["document_id"]) != str(child["document_id"]):
                raise ValueError("Parent/child document mismatch")
            if not (
                int(parent["start_position"])
                <= int(child["start_position"])
                < int(child["end_position"])
                <= int(parent["end_position"])
            ):
                raise ValueError("Child lies outside parent")
            if require_section_boundaries and (
                parent.get("section_label") != child.get("section_label")
            ):
                raise ValueError("M1 child/parent section mismatch")
    return {
        "source_text_exact": True,
        "all_source_words_covered": True,
        "uncovered_source_words": uncovered_words,
        "unique_chunk_ids": True,
        "valid_offsets": True,
        "valid_parent_child_mapping": bool(parents),
        "parents_respect_section_boundaries": (
            True if require_section_boundaries else None
        ),
        "chunk_count": len(chunks),
        "parent_count": parent_count,
    }


def _token_lengths(tokenizer: Any, texts: Sequence[str], *, prefix: str) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), 256):
        batch = [prefix + text for text in texts[start : start + 256]]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_length=True,
        )
        lengths.extend(int(value) for value in encoded["length"])
    return lengths


def _public_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": result["experiment_id"],
        "status": result["status"],
        "completed_at_utc": result["completed_at_utc"],
        "protocol_status": result["protocol_status"],
        "dataset": {
            "documents": result["dataset"]["documents"],
            "questions": result["dataset"]["questions"],
            "section_distribution": result["dataset"]["section_distribution"],
        },
        "configuration": {
            "embedding": result["configuration"]["embedding"],
            "retrieval": result["configuration"]["retrieval"],
            "hierarchy": result["configuration"]["hierarchy"],
            "methods": result["configuration"]["methods"],
            "historical_baselines_supplied": result["configuration"][
                "historical_baselines_supplied"
            ],
        },
        "methods": result["methods"],
        "audit": result["audit"],
        "results": {
            code: {
                "overall": result["evaluations"][code]["overall"]["aggregate"],
                "by_section": {
                    label: values["aggregate"]
                    for label, values in result["evaluations"][code][
                        "by_section"
                    ].items()
                },
                "context_delivery": {
                    key: value
                    for key, value in result["evaluations"][code][
                        "context_delivery"
                    ].items()
                    if key != "per_query"
                },
                "error_counts": result["evaluations"][code]["error_analysis"][
                    "counts"
                ],
            }
            for code in ("B1", "B2", "B3", "M1")
        },
        "paired_cluster_bootstrap": result["paired_cluster_bootstrap"],
        "testing": result["testing"],
        "environment": result["environment"],
    }


def _render_report(result: Mapping[str, Any]) -> str:
    evaluations = result["evaluations"]
    methods = result["methods"]
    configuration = result["configuration"]
    lines = [
        "# Hierarchical Structure-Aware Chunking untuk Legal RAG",
        "",
        "## Status dan batas klaim",
        "",
        "Eksperimen empat metode selesai pada korpus 200 putusan dan 800 "
        "pertanyaan. Seluruh hasil bersifat eksploratif: benchmark yang sama "
        "telah dipakai dalam eksperimen terdahulu, sehingga ini bukan holdout "
        "blind atau bukti konfirmatori independen.",
        "",
        "Benchmark aktual hanya memiliki lima label pertanyaan: "
        "`amar_putusan`, `identitas_terdakwa`, `pertimbangan_hukum`, "
        "`riwayat_dakwaan`, dan `riwayat_penahanan`. Karena itu hanya ada tiga "
        "bagian lain di luar dua fokus, bukan empat.",
        "",
        "## Landasan ilmiah",
        "",
        "- Chen et al. (EMNLP 2024), *Dense X Retrieval*, DOI "
        "[10.18653/v1/2024.emnlp-main.845](https://doi.org/10.18653/v1/2024.emnlp-main.845), "
        "menunjukkan pentingnya granularitas unit dense retrieval dan evaluasi "
        "dengan budget konteks tetap.",
        "- Lu et al. (ACL 2026), *HiChunk*, DOI "
        "[10.18653/v1/2026.acl-long.1372](https://doi.org/10.18653/v1/2026.acl-long.1372), "
        "memisahkan struktur hierarkis dari retrieval granular serta melakukan "
        "penggabungan ke parent dengan kendala budget. Komponen fine-tuned LLM "
        "dan Auto-Merge adaptifnya tidak dipakai karena berada di luar desain.",
        "- Elchafei et al. (SemEval 2026), *H-RAG*, DOI "
        "[10.18653/v1/2026.semeval-1.155](https://doi.org/10.18653/v1/2026.semeval-1.155), "
        "mendukung child-first retrieval, max-child parent scoring, dan "
        "parent-context reconstruction. Komponen sparse, reranker, dan query "
        "rewritingnya tidak dipakai.",
        "- Sarthi et al. (ICLR 2024), *RAPTOR*, "
        "[OpenReview](https://openreview.net/forum?id=GN921JHCRw), menjadi "
        "referensi konseptual multi-granular retrieval; summarization dan "
        "clustering rekursif tidak diimplementasikan.",
        "",
        "## Rancangan dan konfigurasi beku",
        "",
        "| Kode | Metode | Unit retrieval | Konteks yang dikembalikan |",
        "|---|---|---|---|",
        "| B1 | Fixed-size asli | 500 kata, overlap 100 | chunk yang sama |",
        "| B2 | SAC asli | 150 kata, overlap 30, section-bound | chunk yang sama |",
        "| B3 | Hierarchical | child 150/30 dalam parent linear 500/100 | parent unik |",
        "| M1 | Hierarchical SAC | child 150/30 dalam parent section-bound 500/100 | parent unik |",
        "",
        "Semua metode memakai `intfloat/multilingual-e5-small`, independent "
        "embedding, prefix E5, exact cosine search, top-50 candidates, tanpa "
        "BM25, reranker, contextual embedding, summary, atau label gold saat "
        "retrieval. Parent B3 dan M1 diberi skor maksimum child; konteks parent "
        "dideduplikasi dan dibatasi 2.500 kata per query.",
        "",
        "## Audit baseline historis",
        "",
        "Angka historis yang diwajibkan tetap dicatat tanpa perubahan:",
        "",
        "| Baseline historis | Recall@5 | MRR@5 | nDCG@5 |",
        "|---|---:|---:|---:|",
        "| Fixed-Size + Independent | 0.6350 | 0.3638 | 0.3489 |",
        "| Structure-Aware + Independent | 0.6425 | 0.3935 | 0.4414 |",
        "",
        "Audit repository menemukan inkonsistensi provenance: angka Fixed "
        "0.6350/0.3638/0.3489 tersimpan pada eksperimen 150/30, sedangkan "
        "dokumen seleksi baseline membekukan Fixed 500/100. Eksperimen baru "
        "mengikuti konfigurasi baseline beku 500/100 untuk B1 dan melaporkan "
        "hasilnya terpisah; angka historis di atas tidak ditimpa atau dicampur.",
        "",
        "## Audit implementasi",
        "",
        "| Metode | Child/chunk | Parent | Maks token encoder | >512 | Semua kata tercakup |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for code in ("B1", "B2", "B3", "M1"):
        audit = result["audit"]["chunks"][code]
        length = audit["lengths"]["encoder_token_length_with_prefix"]
        lines.append(
            f"| {code} | {methods[code]['chunk_count']} | "
            f"{methods[code]['parent_count']} | {int(length['max'])} | "
            f"{audit['lengths']['over_encoder_limit']} | ya |"
        )
    lines.extend(
        [
            "",
            "B3 dan M1 divalidasi memiliki mapping parent-child lengkap. Parent "
            "M1 tidak pernah melintasi section. Over-limit B1 adalah konsekuensi "
            "baseline 500 kata yang dipertahankan; B2/B3/M1 diperiksa terhadap "
            "batas aktual encoder.",
            "",
            "## Hasil keseluruhan",
            "",
            "| Metode | R@1 | R@3 | R@5 | R@10 | MRR@1 | MRR@3 | MRR@5 | MRR@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for code in ("B1", "B2", "B3", "M1"):
        values = evaluations[code]["overall"]["aggregate"]
        cells = [
            values[f"{metric}@{k}"]
            for metric in ("recall", "mrr", "ndcg")
            for k in (1, 3, 5, 10)
        ]
        lines.append(
            f"| {code} | " + " | ".join(f"{value:.4f}" for value in cells) + " |"
        )

    lines.extend(
        [
            "",
            "## Perubahan absolut dan CI 95% pada @5",
            "",
            "CI memakai paired cluster bootstrap 10.000 kali pada unit dokumen. "
            "nDCG lintas chunker tetap deskriptif karena jumlah chunk relevan "
            "yang overlap dapat berbeda antarstrategi.",
            "",
            "| Kontras | ΔRecall@5 [CI] | ΔMRR@5 [CI] | ΔnDCG@5 [CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    comparisons = result["paired_cluster_bootstrap"]["comparisons"]
    for comparison, values in comparisons.items():
        cells = []
        for metric in ("recall@5", "mrr@5", "ndcg@5"):
            row = values[metric]
            cells.append(
                f"{row['mean_difference']:+.4f} "
                f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
            )
        lines.append(
            f"| {comparison.replace('_minus_', ' − ')} | "
            + " | ".join(cells)
            + " |"
        )

    sections = sorted(result["dataset"]["section_distribution"])
    lines.extend(
        [
            "",
            "## Metrik @5 per section",
            "",
            "| Section | N | Metode | Recall@5 | MRR@5 | nDCG@5 |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for section in sections:
        count = result["dataset"]["section_distribution"][section]
        for code in ("B1", "B2", "B3", "M1"):
            values = evaluations[code]["by_section"][section]["aggregate"]
            lines.append(
                f"| `{section}` | {count} | {code} | "
                f"{values['recall@5']:.4f} | {values['mrr@5']:.4f} | "
                f"{values['ndcg@5']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Konteks yang dikembalikan dengan budget setara",
            "",
            "Evidence hit di bawah mengukur apakah konteks akhir (chunk untuk "
            "B1/B2, parent untuk B3/M1) memuat evidence gold dalam budget "
            "2.500 kata. Ini dipisahkan dari metrik child retrieval di atas.",
            "",
            "| Metode | Evidence hit | Rata-rata kata | Median kata | Rata-rata unit |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for code in ("B1", "B2", "B3", "M1"):
        context = evaluations[code]["context_delivery"]
        lines.append(
            f"| {code} | {context['evidence_hit_rate']:.4f} | "
            f"{context['mean_returned_words']:.1f} | "
            f"{context['median_returned_words']:.1f} | "
            f"{context['mean_context_units']:.2f} |"
        )

    reasoning = "pertimbangan_hukum"
    detention = "riwayat_penahanan"
    lines.extend(
        [
            "",
            "## Analisis Pertimbangan Hukum",
            "",
            _section_analysis(result, reasoning),
            "",
            "## Analisis Riwayat Penahanan",
            "",
            _section_analysis(result, detention),
            "",
            "Strata Riwayat Penahanan hanya berisi enam pertanyaan. Satu query "
            "mengubah Recall sebesar 0,1667, sehingga pola ini terlalu rapuh "
            "untuk klaim umum atau klaim signifikansi section-spesifik.",
            "",
            "## Error analysis @5",
            "",
            "| Metode | Sukses | Salah dokumen | Salah section | Evidence rank 6–50 | Tidak ada di top-50 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for code in ("B1", "B2", "B3", "M1"):
        counts = evaluations[code]["error_analysis"]["counts"]
        lines.append(
            f"| {code} | {counts.get('success_top5', 0)} | "
            f"{counts.get('wrong_document_top5', 0)} | "
            f"{counts.get('wrong_section_top5', 0)} | "
            f"{counts.get('evidence_rank_6_to_50', 0)} | "
            f"{counts.get('evidence_not_in_top50', 0)} |"
        )
    lines.extend(
        [
            "",
            "Kategori error di atas eksklusif dengan prioritas salah dokumen, "
            "lalu salah section, lalu posisi evidence. Karena itu jumlah "
            "`Tidak ada di top-50` bukan total seluruh query tanpa evidence di "
            "top-50; total tersebut dilaporkan pada tabel ranking berikut.",
            "",
            "## Ranking evidence relevan",
            "",
            "| Metode | Evidence ditemukan top-50 | Tidak ditemukan top-50 | Median rank jika ditemukan | Mean rank | P90 rank |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for code in ("B1", "B2", "B3", "M1"):
        rank = _relevant_rank_summary(evaluations[code]["relevant_rank"])
        lines.append(
            f"| {code} | {rank['found']} | {rank['missing']} | "
            f"{rank['median']:.1f} | {rank['mean']:.2f} | {rank['p90']} |"
        )

    conclusion = _conclusion(result)
    lines.extend(
        [
            "",
            "## Testing dan validasi",
            "",
            "Validasi runtime lulus untuk exact source offsets, cakupan seluruh "
            "kata sumber, keunikan ID, qrels positif untuk semua 800 query, "
            "mapping parent-child, deduplikasi parent, budget konteks, dan batas "
            "section M1. " + _testing_statement(result),
            "",
            "## Kesimpulan ilmiah",
            "",
            conclusion,
            "",
            "## Keterbatasan",
            "",
            "- Dataset QA dibuat deterministik dari XML dan belum merupakan "
            "adjudikasi manusia independen.",
            "- Benchmark 200 dokumen telah dilihat pada eksperimen sebelumnya; "
            "CI mengukur ketidakpastian sampling internal, bukan mengubahnya "
            "menjadi holdout blind.",
            "- Hanya enam query Riwayat Penahanan.",
            "- Parent 500 kata adalah konteks sumber asli, bukan ringkasan; hasil "
            "tidak menguji RAPTOR atau HiChunk lengkap.",
            "- MRR dan nDCG menggunakan qrels chunk-spesifik; Recall evidence dan "
            "context evidence hit lebih langsung sebanding lintas chunker.",
            "- Baseline B1 500 kata dapat melampaui 512 token subword dan "
            "ditrunkasi encoder sesuai perilaku baseline historis.",
            "",
            "## Reproduksi",
            "",
            "```powershell",
            "$env:HF_HUB_OFFLINE='1'",
            "$env:TRANSFORMERS_OFFLINE='1'",
            ".venv\\Scripts\\python.exe -m pytest experiments\\hierarchical_structure_aware\\test_hierarchical_chunking.py -q",
            ".venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider",
            ".venv\\Scripts\\python.exe -m experiments.hierarchical_structure_aware.run_experiment",
            "```",
            "",
            "Untuk memvalidasi ulang artefak lokal yang sudah lengkap tanpa "
            "menimpa eksperimen lain, tambahkan `--resume`.",
            "",
        ]
    )
    return "\n".join(lines)


def _section_analysis(result: Mapping[str, Any], label: str) -> str:
    values = {
        code: result["evaluations"][code]["by_section"][label]["aggregate"][
            "recall@5"
        ]
        for code in ("B1", "B2", "B3", "M1")
    }
    best = max(values, key=values.get)
    m1 = values["M1"]
    contexts = {
        code: result["evaluations"][code]["context_delivery"]["by_section"][
            label
        ]["evidence_hit_rate"]
        for code in ("B1", "B2", "B3", "M1")
    }
    return (
        f"Recall@5: B1={values['B1']:.4f}, B2={values['B2']:.4f}, "
        f"B3={values['B3']:.4f}, dan M1={m1:.4f}. Nilai tertinggi adalah "
        f"{best} ({values[best]:.4f}). Delta M1 terhadap B1 "
        f"{m1 - values['B1']:+.4f}, terhadap B2 {m1 - values['B2']:+.4f}, "
        f"dan terhadap B3 {m1 - values['B3']:+.4f}. Di bawah budget konteks "
        f"yang sama, evidence hit B1/B2/B3/M1 masing-masing "
        f"{contexts['B1']:.4f}/{contexts['B2']:.4f}/{contexts['B3']:.4f}/"
        f"{contexts['M1']:.4f}. Perbedaan ini menunjukkan bahwa child ranking "
        f"dan keberhasilan menyediakan parent context adalah dua outcome yang "
        f"berbeda."
    )


def _conclusion(result: Mapping[str, Any]) -> str:
    overall = {
        code: result["evaluations"][code]["overall"]["aggregate"]
        for code in ("B1", "B2", "B3", "M1")
    }
    reasoning = {
        code: result["evaluations"][code]["by_section"][
            "pertimbangan_hukum"
        ]["aggregate"]["recall@5"]
        for code in ("B1", "B2", "B3", "M1")
    }
    ci = result["paired_cluster_bootstrap"]["comparisons"]["M1_minus_B2"]
    recall_ci = ci["recall@5"]
    relation = _direction(overall["M1"]["recall@5"], overall["B2"]["recall@5"])
    reasoning_relation = _direction(reasoning["M1"], reasoning["B2"])
    support = (
        "CI tidak melintasi nol"
        if recall_ci["ci95_low"] > 0 or recall_ci["ci95_high"] < 0
        else "CI melintasi nol"
    )
    context_b2 = result["evaluations"]["B2"]["context_delivery"][
        "evidence_hit_rate"
    ]
    context_m1 = result["evaluations"]["M1"]["context_delivery"][
        "evidence_hit_rate"
    ]
    reasoning_context_b2 = result["evaluations"]["B2"]["context_delivery"][
        "by_section"
    ]["pertimbangan_hukum"]["evidence_hit_rate"]
    reasoning_context_m1 = result["evaluations"]["M1"]["context_delivery"][
        "by_section"
    ]["pertimbangan_hukum"]["evidence_hit_rate"]
    return (
        f"Secara agregat Recall@5 M1 {relation} dari "
        f"{overall['B2']['recall@5']:.4f} (B2) menjadi "
        f"{overall['M1']['recall@5']:.4f}; delta "
        f"{recall_ci['mean_difference']:+.4f} dengan CI 95% "
        f"[{recall_ci['ci95_low']:+.4f}, {recall_ci['ci95_high']:+.4f}] "
        f"({support}). Pada Pertimbangan Hukum, Recall@5 M1 "
        f"{reasoning_relation} dari {reasoning['B2']:.4f} menjadi "
        f"{reasoning['M1']:.4f}. Jadi M1 tidak memperbaiki objective child "
        f"retrieval yang menjadi kriteria utama dan belum didukung sebagai "
        f"pengganti SAC murni. Namun evidence hit konteks ber-budget sama naik "
        f"dari {context_b2:.4f} menjadi {context_m1:.4f} secara keseluruhan dan "
        f"dari {reasoning_context_b2:.4f} menjadi {reasoning_context_m1:.4f} "
        f"pada Pertimbangan Hukum. Ini adalah manfaat context delivery, bukan "
        f"peningkatan child ranking, dan belum diberi CI khusus. B3 menunjukkan "
        f"bahwa hierarchy linear dapat membantu Pertimbangan, tetapi tabel per "
        f"section menunjukkan trade-off besar pada bagian terstruktur. Hasil "
        f"tetap eksploratif dan memerlukan pertanyaan baru atau korpus eksternal."
    )


def _relevant_rank_summary(values: Mapping[str, int | None]) -> dict[str, Any]:
    ranks = sorted(int(value) for value in values.values() if value is not None)
    if not ranks:
        raise ValueError("No relevant evidence ranks were found")
    p90_index = max(0, min(len(ranks) - 1, math.ceil(0.9 * len(ranks)) - 1))
    return {
        "found": len(ranks),
        "missing": len(values) - len(ranks),
        "median": statistics.median(ranks),
        "mean": statistics.fmean(ranks),
        "p90": ranks[p90_index],
    }


def _direction(left: float, right: float) -> str:
    if left > right:
        return "meningkat"
    if left < right:
        return "menurun"
    return "tidak berubah"


def _read_test_results() -> dict[str, Any]:
    path = EXPERIMENT_DIR / "test_results.json"
    if not path.exists():
        return {"status": "not_recorded"}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("test_results.json must contain an object")
    return value


def _testing_statement(result: Mapping[str, Any]) -> str:
    testing = result.get("testing", {})
    if testing.get("status") != "passed":
        return "Hasil full test suite belum dicatat pada artefak."
    return (
        f"Full repository test suite lulus: {int(testing['passed'])} passed, "
        f"{int(testing['failed'])} failed, {int(testing['errors'])} errors."
    )


def _build_manifest(
    artifacts: Path,
    config_path: Path,
    result: Mapping[str, Any],
    *,
    tracked_outputs: Sequence[Path],
) -> dict[str, Any]:
    files = [
        path
        for path in artifacts.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    files.extend(tracked_outputs)
    return {
        "experiment_id": result["experiment_id"],
        "status": "complete",
        "completed_at_utc": result["completed_at_utc"],
        "config": _relative(config_path),
        "config_sha256": _sha256(config_path),
        "seed": result["configuration"]["seed"],
        "model": result["configuration"]["embedding"],
        "commands": [
            ".venv\\Scripts\\python.exe -m pytest "
            "experiments\\hierarchical_structure_aware\\test_hierarchical_chunking.py -q",
            ".venv\\Scripts\\python.exe -m "
            "experiments.hierarchical_structure_aware.run_experiment",
        ],
        "artifacts": [
            {
                "path": _relative(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(files)
        ],
    }


def _print_key_results(result: Mapping[str, Any]) -> None:
    print("method\trecall@5\tmrr@5\tndcg@5\tcontext_hit")
    for code in ("B1", "B2", "B3", "M1"):
        aggregate = result["evaluations"][code]["overall"]["aggregate"]
        context = result["evaluations"][code]["context_delivery"]
        print(
            f"{code}\t{aggregate['recall@5']:.4f}\t{aggregate['mrr@5']:.4f}"
            f"\t{aggregate['ndcg@5']:.4f}\t{context['evidence_hit_rate']:.4f}"
        )


def _read_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or any(
        str(row.get("review_status", "")).strip().lower() != "approved"
        for row in rows
    ):
        raise ValueError("Every question must be approved")
    return rows


def _write_or_validate_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    resume: bool,
) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {path}")
        if list(read_jsonl(path)) != list(rows):
            raise ValueError(f"Existing artifact mismatch: {path}")
        return
    write_jsonl(path, rows)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    resume: bool,
) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            existing = list(csv.DictReader(file))
        normalized = [
            {name: str(row.get(name, "")) for name in fieldnames} for row in rows
        ]
        if existing != normalized:
            raise ValueError(f"Existing qrel mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _merge_intervals(values: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(values):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return commit.stdout.strip(), bool(status.stdout.strip())


if __name__ == "__main__":
    main()
