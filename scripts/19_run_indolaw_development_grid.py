"""Build and retrieve one Indo-Law split with one loaded model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.embedding import SentenceTransformerEmbedder
from src.retrieval import DenseVectorStore
from src.utils.io import read_jsonl, write_jsonl


CONFIGS = (
    "fixed_w150_o30",
    "fixed_w300_o60",
    "fixed_w500_o100",
    "sac_w150_o30_s0",
    "sac_w300_o60_s0",
    "sac_w500_o100_s0",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build indexes and top-k runs for one Indo-Law split."
    )
    parser.add_argument(
        "--split", choices=("development", "holdout"), default="development"
    )
    parser.add_argument("--configs", nargs="+")
    parser.add_argument(
        "--questions",
        type=Path,
    )
    parser.add_argument(
        "--chunks-root",
        type=Path,
    )
    parser.add_argument(
        "--indexes-root",
        type=Path,
    )
    parser.add_argument(
        "--runs-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("experiments/indolaw_200_manifest.json"),
    )
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete, validated index/run pairs; never overwrite them.",
    )
    args = parser.parse_args(argv)
    if args.configs is None:
        if args.split != "development":
            raise ValueError("Holdout requires explicit frozen --configs")
        args.configs = list(CONFIGS)
    args.questions = args.questions or Path(
        f"data/evaluation/indolaw_200/{args.split}_questions.csv"
    )
    args.chunks_root = args.chunks_root or Path(
        f"data/chunks/indolaw_200/{args.split}"
    )
    args.indexes_root = args.indexes_root or Path(
        f"vector_db/indolaw_200/{args.split}"
    )
    if args.batch_size <= 0 or args.top_k <= 0:
        raise ValueError("batch-size and top-k must be positive")
    if args.top_k != 50:
        raise ValueError("The frozen Indo-Law retrieval protocol requires top-k 50")
    if len(set(args.configs)) != len(args.configs):
        raise ValueError("configs must be unique")

    questions = _read_approved_questions(args.questions)
    run_paths = {
        name: args.runs_root / f"indolaw_200_{args.split}_{name}_top50.jsonl"
        for name in args.configs
    }
    index_paths = {name: args.indexes_root / name for name in args.configs}
    grid_manifest_path = (
        args.runs_root / f"indolaw_200_{args.split}_grid_run_manifest.json"
    )
    if grid_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {grid_manifest_path}")
    for name in args.configs:
        has_run = run_paths[name].exists()
        has_index = index_paths[name].exists() and any(index_paths[name].iterdir())
        if has_run != has_index:
            raise ValueError(
                f"Incomplete artifact pair for {name}; index={has_index}, run={has_run}"
            )
        if has_run and not args.resume:
            raise FileExistsError(
                f"Refusing to overwrite artifacts for {name}; pass --resume to validate and reuse"
            )

    embedder = SentenceTransformerEmbedder(
        args.model,
        batch_size=args.batch_size,
        device=args.device,
        query_prefix="query: ",
        document_prefix="passage: ",
        normalize_embeddings=True,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    query_started = time.perf_counter()
    query_embeddings = embedder.embed_queries(
        [question["question"] for question in questions]
    )
    query_embedding_batch_ms = (time.perf_counter() - query_started) * 1_000
    query_embedding_amortized_ms = query_embedding_batch_ms / len(questions)
    run_summaries: list[dict[str, Any]] = []
    for name in args.configs:
        chunks_path = args.chunks_root / name / "chunks.jsonl"
        chunks = list(read_jsonl(chunks_path))
        if run_paths[name].exists():
            store = DenseVectorStore.load(index_paths[name])
            records = list(read_jsonl(run_paths[name]))
            _validate_existing(
                name,
                chunks,
                store,
                records,
                args.model,
                args.top_k,
                len(questions),
            )
            run_summaries.append(
                _run_summary(
                    name,
                    chunks_path,
                    chunks,
                    records,
                    index_paths[name],
                    run_paths[name],
                    reused_existing=True,
                )
            )
            print(f"{name}: reused validated index and run")
            continue
        embeddings = embedder.embed_documents(
            [str(chunk["text"]) for chunk in chunks]
        )
        store = DenseVectorStore(chunks, embeddings, model_name=args.model)
        store.save(
            index_paths[name],
            provenance={
                "dataset": "indolaw_200",
                "split": args.split,
                "design": name,
                "chunks_sha256": _sha256(chunks_path),
                "batch_size": args.batch_size,
                "query_prefix": "query: ",
                "document_prefix": "passage: ",
                "normalize_embeddings": True,
                "device": args.device,
            },
        )

        records: list[dict[str, Any]] = []
        for question, query_embedding in zip(questions, query_embeddings):
            started = time.perf_counter()
            results = store.search(query_embedding, k=args.top_k)
            search_latency_ms = (time.perf_counter() - started) * 1_000
            records.append(
                {
                    "query_id": question["query_id"],
                    "document_id": question["document_id"],
                    "target_section_label": question["target_section_label"],
                    "strategy": str(store.chunks[0].get("strategy", "")),
                    "design": name,
                    "model_name": args.model,
                    "retrieval_scope": "full_corpus",
                    "top_k": args.top_k,
                    "latency_ms": (
                        query_embedding_amortized_ms + search_latency_ms
                    ),
                    "query_embedding_latency_ms": query_embedding_amortized_ms,
                    "search_latency_ms": search_latency_ms,
                    "latency_method": "batched_query_embedding_amortized_plus_search",
                    "ranking": [
                        {
                            "rank": rank,
                            "chunk_id": result.chunk_id,
                            "score": result.score,
                            "retrieved_document_id": str(
                                result.chunk["document_id"]
                            ),
                        }
                        for rank, result in enumerate(results, start=1)
                    ],
                }
            )
        count = write_jsonl(run_paths[name], records)
        run_summaries.append(
            _run_summary(
                name,
                chunks_path,
                chunks,
                records,
                index_paths[name],
                run_paths[name],
                reused_existing=False,
            )
        )
        print(f"{name}: indexed {len(chunks)} chunks; retrieved {count} queries")

    git_commit, git_dirty = _git_state()
    manifest = {
        "experiment_id": f"indolaw_200_{args.split}_grid_evidence_recall_v2",
        "metric_schema_version": "retrieval-v2-evidence-recall",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "code_files_sha256": {
            "runner": _sha256(Path(__file__)),
            "retrieval_metrics": _sha256(
                Path("src/evaluation/retrieval_metrics.py")
            ),
            "retrieval_report": _sha256(
                Path("src/evaluation/retrieval_report.py")
            ),
            "ground_truth": _sha256(Path("src/evaluation/ground_truth.py")),
        },
        "dataset_manifest": args.dataset_manifest.as_posix(),
        "dataset_manifest_sha256": _sha256(args.dataset_manifest),
        "split": args.split,
        "questions": args.questions.as_posix(),
        "questions_sha256": _sha256(args.questions),
        "seed": args.seed,
        "model_name": args.model,
        "batch_size": args.batch_size,
        "device": args.device,
        "top_k": args.top_k,
        "retrieval_scope": "full_corpus",
        "resume_mode": args.resume,
        "latency_method": "batched_query_embedding_amortized_plus_search",
        "query_embedding_batch_ms_this_process": query_embedding_batch_ms,
        "latency_use": "batched diagnostic only; not a final latency claim",
        "python_version": sys.version,
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
        "runs": run_summaries,
    }
    grid_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    grid_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote provenance manifest to {grid_manifest_path}")


def _read_approved_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or any(
        row.get("review_status", "").strip().lower() != "approved" for row in rows
    ):
        raise ValueError("All questions in the requested split must be approved")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_existing(
    design: str,
    source_chunks: list[dict[str, Any]],
    store: DenseVectorStore,
    records: list[dict[str, Any]],
    model_name: str,
    top_k: int,
    expected_queries: int,
) -> None:
    if store.model_name != model_name or store.chunks != source_chunks:
        raise ValueError(f"Existing index does not match source/model for {design}")
    if len(records) != expected_queries:
        raise ValueError(
            f"Existing run must contain {expected_queries} queries for {design}"
        )
    query_ids = {str(record.get("query_id", "")) for record in records}
    if len(query_ids) != len(records):
        raise ValueError(f"Existing run has duplicate/blank query IDs for {design}")
    for record in records:
        if (
            record.get("design") != design
            or record.get("model_name") != model_name
            or int(record.get("top_k", 0)) != top_k
            or record.get("retrieval_scope") != "full_corpus"
            or record.get("latency_method")
            != "batched_query_embedding_amortized_plus_search"
            or len(record.get("ranking", ())) != top_k
        ):
            raise ValueError(f"Existing run metadata mismatch for {design}")


def _run_summary(
    design: str,
    chunks_path: Path,
    chunks: list[dict[str, Any]],
    records: list[dict[str, Any]],
    index_path: Path,
    run_path: Path,
    *,
    reused_existing: bool,
) -> dict[str, Any]:
    return {
        "design": design,
        "strategy": records[0]["strategy"],
        "chunk_count": len(chunks),
        "query_count": len(records),
        "chunks_sha256": _sha256(chunks_path),
        "run_sha256": _sha256(run_path),
        "index_path": index_path.as_posix(),
        "run_path": run_path.as_posix(),
        "reused_existing": reused_existing,
    }


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return commit.stdout.strip(), bool(status.stdout.strip())


if __name__ == "__main__":
    main()
