"""Build fixed and SAC contextual indexes from one shared document encoding pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.embedding import ContextualSentenceTransformerEmbedder
from src.retrieval import DenseVectorStore
from src.utils.config import load_config
from src.utils.io import read_jsonl


SCHEMA_VERSION = 2


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-config",
        type=Path,
        default=Path("configs/indolaw_fixed_contextual.yaml"),
    )
    parser.add_argument(
        "--sac-config",
        type=Path,
        default=Path("configs/indolaw_sac_contextual.yaml"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("vector_db/indolaw_200/corpus/contextual_e5_v2_checkpoints"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "experiments/results/indolaw_200_contextual_index_manifest.json"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only model files already present in the Hugging Face cache.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    fixed = _read_design(args.fixed_config)
    sac = _read_design(args.sac_config)
    _validate_matched_embedding(fixed, sac)
    if fixed["output_path"] == sac["output_path"]:
        raise ValueError("Fixed and SAC contextual indexes need different outputs")
    for output in (fixed["output_path"], sac["output_path"]):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite contextual index {output}")
    if args.manifest_output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.manifest_output}")

    documents = _read_documents(fixed["documents_path"])
    fixed_chunks = list(read_jsonl(fixed["chunks_path"]))
    sac_chunks = list(read_jsonl(sac["chunks_path"]))
    checkpoint_manifest = {
        "schema_version": SCHEMA_VERSION,
        "contextual_algorithm_version": (
            "windowed_late_chunking_v2_strategy_specific_assignment"
        ),
        "embedder_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "src/embedding/embedder.py"
        ),
        "builder_sha256": _sha256(Path(__file__).resolve()),
        "model_name": fixed["model_name"],
        "documents_sha256": _sha256(fixed["documents_path"]),
        "fixed_chunks_sha256": _sha256(fixed["chunks_path"]),
        "sac_chunks_sha256": _sha256(sac["chunks_path"]),
        "embedding": fixed["embedding_signature"],
        "window_assignment": "strategy_specific_with_deduplicated_encoding",
    }
    _prepare_checkpoints(args.checkpoint_root, checkpoint_manifest, args.resume)

    fixed_by_document = _group_chunks(fixed_chunks)
    sac_by_document = _group_chunks(sac_chunks)
    expected_documents = set(documents)
    if set(fixed_by_document) != expected_documents:
        raise ValueError("Fixed chunks do not cover exactly the source documents")
    if set(sac_by_document) != expected_documents:
        raise ValueError("SAC chunks do not cover exactly the source documents")

    embedder = ContextualSentenceTransformerEmbedder(
        fixed["model_name"],
        batch_size=fixed["batch_size"],
        device=args.device,
        query_prefix=fixed["query_prefix"],
        document_prefix=fixed["document_prefix"],
        normalize_embeddings=fixed["normalize_embeddings"],
        trust_remote_code=fixed["trust_remote_code"],
        max_sequence_length=fixed["max_sequence_length"],
        context_window_tokens=fixed["context_window_tokens"],
        context_window_overlap_tokens=fixed["context_window_overlap_tokens"],
    )

    ordered_documents = sorted(documents)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    reused = 0
    for position, document_id in enumerate(ordered_documents, start=1):
        checkpoint = args.checkpoint_root / f"{document_id}.npz"
        if checkpoint.exists():
            _validate_checkpoint(
                checkpoint,
                fixed_by_document[document_id],
                sac_by_document[document_id],
            )
            reused += 1
        else:
            fixed_items = fixed_by_document[document_id]
            sac_items = sac_by_document[document_id]
            vectors = embedder.embed_document_chunk_groups(
                documents[document_id],
                {
                    "fixed": [chunk for _, chunk in fixed_items],
                    "sac": [chunk for _, chunk in sac_items],
                },
            )
            temporary = checkpoint.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                fixed_indices=np.asarray(
                    [index for index, _ in fixed_items], dtype=np.int64
                ),
                fixed_vectors=vectors["fixed"],
                sac_indices=np.asarray(
                    [index for index, _ in sac_items], dtype=np.int64
                ),
                sac_vectors=vectors["sac"],
            )
            os.replace(temporary, checkpoint)
        elapsed = time.perf_counter() - started
        rate = position / elapsed if elapsed else 0.0
        remaining = (len(ordered_documents) - position) / rate if rate else 0.0
        print(
            f"[{position:03d}/{len(ordered_documents)}] {document_id} "
            f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m "
            f"reused={reused}",
            flush=True,
        )

    fixed_matrix = _assemble_matrix(
        args.checkpoint_root,
        ordered_documents,
        key_prefix="fixed",
        row_count=len(fixed_chunks),
    )
    sac_matrix = _assemble_matrix(
        args.checkpoint_root,
        ordered_documents,
        key_prefix="sac",
        row_count=len(sac_chunks),
    )
    shared_provenance = {
        "embedding_method": "contextual",
        "documents_path": fixed["documents_path"].as_posix(),
        "documents_sha256": checkpoint_manifest["documents_sha256"],
        "joint_factorial_encoding": True,
        **fixed["embedding_signature"],
    }
    DenseVectorStore(
        fixed_chunks, fixed_matrix, model_name=fixed["model_name"]
    ).save(
        fixed["output_path"],
        provenance={
            **shared_provenance,
            "design": "fixed_contextual",
            "chunks_path": fixed["chunks_path"].as_posix(),
            "chunks_sha256": checkpoint_manifest["fixed_chunks_sha256"],
        },
    )
    DenseVectorStore(sac_chunks, sac_matrix, model_name=sac["model_name"]).save(
        sac["output_path"],
        provenance={
            **shared_provenance,
            "design": "sac_contextual",
            "chunks_path": sac["chunks_path"].as_posix(),
            "chunks_sha256": checkpoint_manifest["sac_chunks_sha256"],
        },
    )
    completed_at = datetime.now(timezone.utc).isoformat()
    elapsed_seconds = time.perf_counter() - started
    result = {
        **checkpoint_manifest,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "elapsed_seconds": elapsed_seconds,
        "document_count": len(documents),
        "fixed_chunk_count": len(fixed_chunks),
        "sac_chunk_count": len(sac_chunks),
        "fixed_index": fixed["output_path"].as_posix(),
        "sac_index": sac["output_path"].as_posix(),
        "checkpoint_root": args.checkpoint_root.as_posix(),
        "reused_checkpoint_count": reused,
        "device": args.device,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Built both contextual indexes in {elapsed_seconds / 60:.1f} minutes; "
        f"manifest={args.manifest_output}",
        flush=True,
    )


def _read_design(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    chunking = config["chunking"]
    embedding = config["embedding"]
    retrieval = config["retrieval"]
    if str(embedding.get("method")) != "contextual":
        raise ValueError(f"{config_path} must use embedding.method=contextual")
    documents_value = embedding.get("documents_path")
    if not documents_value:
        raise ValueError(f"{config_path} is missing embedding.documents_path")
    project_root = config_path.resolve().parent.parent

    def path(value: object) -> Path:
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else project_root / candidate

    max_length = embedding.get("max_sequence_length")
    signature = {
        "batch_size": int(embedding.get("batch_size", 16)),
        "query_prefix": str(embedding.get("query_prefix", "query: ")),
        "document_prefix": str(embedding.get("document_prefix", "passage: ")),
        "normalize_embeddings": bool(embedding.get("normalize_embeddings", True)),
        "trust_remote_code": bool(embedding.get("trust_remote_code", False)),
        "max_sequence_length": int(max_length) if max_length is not None else None,
        "context_window_tokens": int(embedding["context_window_tokens"]),
        "context_window_overlap_tokens": int(
            embedding["context_window_overlap_tokens"]
        ),
    }
    return {
        "config_path": config_path,
        "chunks_path": path(chunking["output_path"]),
        "documents_path": path(documents_value),
        "output_path": path(retrieval["index_path"]),
        "model_name": str(embedding["model_name"]),
        "embedding_signature": signature,
        **signature,
    }


def _validate_matched_embedding(fixed: dict[str, Any], sac: dict[str, Any]) -> None:
    for key in ("model_name", "documents_path", "embedding_signature"):
        if fixed[key] != sac[key]:
            raise ValueError(f"Contextual designs differ on {key}")


def _read_documents(path: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    for row in read_jsonl(path):
        document_id = str(row.get("document_id", ""))
        text = str(row.get("clean_text") or row.get("raw_text") or "")
        if not document_id or not text or document_id in documents:
            raise ValueError(f"Invalid or duplicate source document in {path}")
        documents[document_id] = text
    return documents


def _group_chunks(
    chunks: Sequence[dict[str, Any]],
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        grouped[str(chunk["document_id"])].append((index, chunk))
    return dict(grouped)


def _prepare_checkpoints(
    root: Path, expected_manifest: dict[str, Any], resume: bool
) -> None:
    manifest_path = root / "manifest.json"
    if root.exists():
        if not resume:
            raise FileExistsError(
                f"Checkpoint directory exists; pass --resume to use {root}"
            )
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected_manifest:
            raise ValueError("Checkpoint manifest does not match this experiment")
        return
    root.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(expected_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_checkpoint(
    path: Path,
    fixed_items: Sequence[tuple[int, dict[str, Any]]],
    sac_items: Sequence[tuple[int, dict[str, Any]]],
) -> None:
    with np.load(path, allow_pickle=False) as values:
        fixed_indices = values["fixed_indices"]
        sac_indices = values["sac_indices"]
        fixed_vectors = values["fixed_vectors"]
        sac_vectors = values["sac_vectors"]
    if fixed_indices.tolist() != [index for index, _ in fixed_items]:
        raise ValueError(f"Fixed checkpoint indices mismatch in {path}")
    if sac_indices.tolist() != [index for index, _ in sac_items]:
        raise ValueError(f"SAC checkpoint indices mismatch in {path}")
    if len(fixed_vectors) != len(fixed_items) or len(sac_vectors) != len(sac_items):
        raise ValueError(f"Checkpoint vector count mismatch in {path}")


def _assemble_matrix(
    root: Path,
    document_ids: Sequence[str],
    *,
    key_prefix: str,
    row_count: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    matrix: np.ndarray[Any, np.dtype[np.float32]] | None = None
    assigned = np.zeros(row_count, dtype=bool)
    for document_id in document_ids:
        with np.load(root / f"{document_id}.npz", allow_pickle=False) as values:
            indices = values[f"{key_prefix}_indices"]
            vectors = np.asarray(values[f"{key_prefix}_vectors"], dtype=np.float32)
        if matrix is None:
            matrix = np.empty((row_count, vectors.shape[1]), dtype=np.float32)
        if assigned[indices].any():
            raise ValueError(f"Duplicate {key_prefix} checkpoint indices")
        matrix[indices] = vectors
        assigned[indices] = True
    if matrix is None or not assigned.all():
        raise ValueError(f"Incomplete {key_prefix} checkpoint matrix")
    return matrix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
