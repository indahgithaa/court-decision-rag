"""Run both contextual factorial arms with one shared query-embedding pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.embedding import SentenceTransformerEmbedder
from src.retrieval import DenseVectorStore
from src.utils.config import load_config
from src.utils.io import write_jsonl


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
        "--questions",
        type=Path,
        default=Path("data/evaluation/indolaw_200/corpus_questions.csv"),
    )
    parser.add_argument(
        "--fixed-output",
        type=Path,
        default=Path(
            "experiments/results/"
            "indolaw_200_corpus_fixed_w150_o30_contextual_e5_top50.jsonl"
        ),
    )
    parser.add_argument(
        "--sac-output",
        type=Path,
        default=Path(
            "experiments/results/"
            "indolaw_200_corpus_sac_w150_o30_contextual_e5_top50.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "experiments/results/indolaw_200_contextual_retrieval_manifest.json"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for output in (args.fixed_output, args.sac_output, args.manifest_output):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")

    fixed = _read_design(args.fixed_config)
    sac = _read_design(args.sac_config)
    if fixed["embedding_signature"] != sac["embedding_signature"]:
        raise ValueError("Contextual arms must use identical embedding settings")
    stores = {
        "fixed_contextual": DenseVectorStore.load(fixed["index_path"]),
        "sac_contextual": DenseVectorStore.load(sac["index_path"]),
    }
    _validate_store(fixed["index_path"], stores["fixed_contextual"], fixed)
    _validate_store(sac["index_path"], stores["sac_contextual"], sac)

    questions = _read_approved_questions(args.questions)
    embedding = fixed["embedding"]
    embedder = SentenceTransformerEmbedder(
        fixed["model_name"],
        batch_size=int(embedding.get("batch_size", 16)),
        device=args.device,
        query_prefix=str(embedding.get("query_prefix", "query: ")),
        document_prefix=str(embedding.get("document_prefix", "passage: ")),
        normalize_embeddings=bool(embedding.get("normalize_embeddings", True)),
        trust_remote_code=bool(embedding.get("trust_remote_code", False)),
        max_sequence_length=_optional_int(embedding.get("max_sequence_length")),
    )
    started_at = datetime.now(timezone.utc).isoformat()
    query_started = time.perf_counter()
    query_embeddings = embedder.embed_queries(
        [question["question"] for question in questions]
    )
    query_batch_ms = (time.perf_counter() - query_started) * 1_000
    query_amortized_ms = query_batch_ms / len(questions)

    outputs = {
        "fixed_contextual": args.fixed_output,
        "sac_contextual": args.sac_output,
    }
    run_summaries: dict[str, Any] = {}
    for arm, store in stores.items():
        records: list[dict[str, Any]] = []
        search_total_ms = 0.0
        for question, query_embedding in zip(questions, query_embeddings):
            search_started = time.perf_counter()
            results = store.search(query_embedding, k=50)
            search_ms = (time.perf_counter() - search_started) * 1_000
            search_total_ms += search_ms
            records.append(
                {
                    "query_id": question["query_id"],
                    "document_id": question["document_id"],
                    "target_section_label": question["target_section_label"],
                    "strategy": str(store.chunks[0]["strategy"]),
                    "embedding_method": "contextual",
                    "design": arm,
                    "model_name": store.model_name,
                    "retrieval_scope": "full_corpus",
                    "top_k": 50,
                    "latency_ms": query_amortized_ms + search_ms,
                    "query_embedding_latency_ms": query_amortized_ms,
                    "search_latency_ms": search_ms,
                    "latency_method": (
                        "batched_query_embedding_amortized_plus_search"
                    ),
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
        count = write_jsonl(outputs[arm], records)
        run_summaries[arm] = {
            "output": outputs[arm].as_posix(),
            "sha256": _sha256(outputs[arm]),
            "query_count": count,
            "search_total_ms": search_total_ms,
        }
        print(f"{arm}: wrote {count} top-50 rankings to {outputs[arm]}")

    manifest = {
        "experiment": "indolaw_200_factorial_contextual_retrieval",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions": args.questions.as_posix(),
        "questions_sha256": _sha256(args.questions),
        "model_name": fixed["model_name"],
        "embedding_signature": fixed["embedding_signature"],
        "query_embedding_batch_ms": query_batch_ms,
        "query_embedding_amortized_ms": query_amortized_ms,
        "latency_method": "batched_query_embedding_amortized_plus_search",
        "runs": run_summaries,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote retrieval manifest to {args.manifest_output}")


def _read_design(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    embedding = config["embedding"]
    if str(embedding.get("method")) != "contextual":
        raise ValueError(f"{config_path} must be contextual")
    index_value = Path(str(config["retrieval"]["index_path"]))
    project_root = config_path.resolve().parent.parent
    index_path = (
        index_value if index_value.is_absolute() else project_root / index_value
    )
    signature = {
        key: embedding.get(key)
        for key in (
            "model_name",
            "batch_size",
            "query_prefix",
            "document_prefix",
            "normalize_embeddings",
            "trust_remote_code",
            "max_sequence_length",
            "context_window_tokens",
            "context_window_overlap_tokens",
        )
    }
    return {
        "index_path": index_path,
        "model_name": str(embedding["model_name"]),
        "embedding": embedding,
        "embedding_signature": signature,
    }


def _validate_store(
    index_path: Path, store: DenseVectorStore, design: dict[str, Any]
) -> None:
    manifest = json.loads(
        (index_path / "manifest.json").read_text(encoding="utf-8")
    )
    if store.model_name != design["model_name"]:
        raise ValueError(f"Index model mismatch in {index_path}")
    if manifest.get("provenance", {}).get("embedding_method") != "contextual":
        raise ValueError(f"Index is not a contextual index: {index_path}")


def _read_approved_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or any(
        row.get("review_status", "").strip().lower() != "approved" for row in rows
    ):
        raise ValueError("All questions must be approved")
    return rows


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
