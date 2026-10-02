"""Dependency-free BM25 reranking for dense retrieval sensitivity analysis."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def tokenize(text: str) -> list[str]:
    """Tokenize Indonesian legal text without dropping names or article numbers."""
    normalized = "".join(
        character if character.isalnum() else " " for character in text.casefold()
    )
    return normalized.split()


class BM25Index:
    """A small in-memory BM25 index over the complete chunk corpus."""

    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.k1 = float(k1)
        self.b = float(b)
        self.term_frequencies: dict[str, Counter[str]] = {}
        self.lengths: dict[str, int] = {}
        document_frequency: Counter[str] = Counter()
        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                raise ValueError("every chunk must have a chunk_id")
            if chunk_id in self.term_frequencies:
                raise ValueError(f"duplicate chunk_id: {chunk_id}")
            terms = tokenize(str(chunk.get("text", "")))
            frequencies = Counter(terms)
            self.term_frequencies[chunk_id] = frequencies
            self.lengths[chunk_id] = len(terms)
            document_frequency.update(frequencies)
        self.chunk_count = len(self.term_frequencies)
        self.average_length = sum(self.lengths.values()) / self.chunk_count
        if self.average_length == 0:
            raise ValueError("chunk corpus must contain at least one token")
        self.idf = {
            term: math.log(
                1 + (self.chunk_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str, chunk_ids: Sequence[str]) -> dict[str, float]:
        """Return BM25 scores for an explicit candidate set."""
        query_terms = Counter(tokenize(query))
        scores: dict[str, float] = {}
        for chunk_id in chunk_ids:
            if chunk_id not in self.term_frequencies:
                raise ValueError(f"unknown candidate chunk_id: {chunk_id}")
            frequencies = self.term_frequencies[chunk_id]
            length_ratio = self.lengths[chunk_id] / self.average_length
            score = 0.0
            for term, query_frequency in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * length_ratio
                )
                score += (
                    self.idf.get(term, 0.0)
                    * term_frequency
                    * (self.k1 + 1)
                    / denominator
                    * query_frequency
                )
            scores[chunk_id] = score
        return scores


def rerank_records(
    questions: Sequence[Mapping[str, Any]],
    run_records: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    *,
    dense_weight: float,
    rrf_constant: int = 60,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> list[dict[str, Any]]:
    """Rerank dense candidates with weighted dense/BM25 reciprocal ranks."""
    if not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight must be between zero and one")
    if rrf_constant < 0:
        raise ValueError("rrf_constant must be non-negative")
    question_by_id = _unique_by_id(questions, "query_id")
    bm25 = BM25Index(chunks, k1=bm25_k1, b=bm25_b)
    reranked_records: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for record in run_records:
        query_id = str(record.get("query_id", ""))
        if query_id in seen_queries:
            raise ValueError(f"duplicate run query_id: {query_id}")
        seen_queries.add(query_id)
        if query_id not in question_by_id:
            raise ValueError(f"run query_id is absent from questions: {query_id}")
        candidates = sorted(
            record.get("ranking", ()), key=lambda item: int(item["rank"])
        )
        candidate_ids = [str(item["chunk_id"]) for item in candidates]
        if not candidate_ids:
            raise ValueError(f"empty candidate ranking for {query_id}")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"duplicate candidate chunk for {query_id}")

        started = time.perf_counter()
        lexical_scores = bm25.score(
            str(question_by_id[query_id]["question"]), candidate_ids
        )
        lexical_order = sorted(
            range(len(candidates)),
            key=lambda index: (-lexical_scores[candidate_ids[index]], index),
        )
        lexical_ranks = {
            candidate_ids[index]: rank
            for rank, index in enumerate(lexical_order, start=1)
        }
        scored = []
        for dense_rank, item in enumerate(candidates, start=1):
            chunk_id = str(item["chunk_id"])
            bm25_rank = lexical_ranks[chunk_id]
            fused_score = dense_weight / (rrf_constant + dense_rank) + (
                1 - dense_weight
            ) / (rrf_constant + bm25_rank)
            scored.append(
                {
                    "chunk_id": chunk_id,
                    "retrieved_document_id": str(
                        item.get("retrieved_document_id", "")
                    ),
                    "dense_rank": dense_rank,
                    "dense_score": float(item.get("score", 0.0)),
                    "bm25_rank": bm25_rank,
                    "bm25_score": lexical_scores[chunk_id],
                    "fused_score": fused_score,
                }
            )
        scored.sort(key=lambda item: (-item["fused_score"], item["dense_rank"]))
        rerank_latency_ms = (time.perf_counter() - started) * 1000
        retrieval_latency_ms = float(record.get("latency_ms", 0.0))
        output = dict(record)
        output.update(
            {
                "retrieval_scope": str(
                    record.get("retrieval_scope", "full_corpus")
                ),
                "retrieval_latency_ms": retrieval_latency_ms,
                "rerank_latency_ms": rerank_latency_ms,
                "latency_ms": retrieval_latency_ms + rerank_latency_ms,
                "reranker": {
                    "method": "weighted_rrf_dense_bm25",
                    "dense_weight": dense_weight,
                    "bm25_weight": 1 - dense_weight,
                    "rrf_constant": rrf_constant,
                    "bm25_k1": bm25_k1,
                    "bm25_b": bm25_b,
                    "candidate_depth": len(candidates),
                },
                "ranking": [
                    {
                        **item,
                        "rank": rank,
                        "score": item["fused_score"],
                    }
                    for rank, item in enumerate(scored, start=1)
                ],
            }
        )
        reranked_records.append(output)
    if set(question_by_id) != seen_queries:
        missing = sorted(set(question_by_id) - seen_queries)
        raise ValueError(f"run is missing question IDs: {missing[:5]}")
    return reranked_records


def _unique_by_id(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            raise ValueError(f"every row must have {key}")
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    if not result:
        raise ValueError("rows must not be empty")
    return result
