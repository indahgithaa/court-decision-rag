import pytest

from src.retrieval import BM25Index, rerank_records


CHUNKS = [
    {"chunk_id": "dense-first", "document_id": "d1", "text": "perkara umum"},
    {
        "chunk_id": "lexical-first",
        "document_id": "d2",
        "text": "putusan narkotika terdakwa",
    },
]
QUESTIONS = [{"query_id": "q1", "question": "pidana narkotika"}]
RUN = [
    {
        "query_id": "q1",
        "strategy": "fixed_size",
        "target_section_label": "amar_putusan",
        "latency_ms": 10.0,
        "ranking": [
            {
                "rank": 1,
                "chunk_id": "dense-first",
                "retrieved_document_id": "d1",
                "score": 0.9,
            },
            {
                "rank": 2,
                "chunk_id": "lexical-first",
                "retrieved_document_id": "d2",
                "score": 0.8,
            },
        ],
    }
]


def test_bm25_scores_lexical_match_above_non_match() -> None:
    scores = BM25Index(CHUNKS).score(
        "pidana narkotika", ["dense-first", "lexical-first"]
    )

    assert scores["lexical-first"] > scores["dense-first"]


def test_dense_only_weight_preserves_dense_order() -> None:
    records = rerank_records(QUESTIONS, RUN, CHUNKS, dense_weight=1.0)

    assert [item["chunk_id"] for item in records[0]["ranking"]] == [
        "dense-first",
        "lexical-first",
    ]


def test_bm25_only_weight_promotes_lexical_match_and_records_provenance() -> None:
    records = rerank_records(QUESTIONS, RUN, CHUNKS, dense_weight=0.0)
    record = records[0]

    assert record["ranking"][0]["chunk_id"] == "lexical-first"
    assert record["reranker"]["method"] == "weighted_rrf_dense_bm25"
    assert record["reranker"]["candidate_depth"] == 2
    assert record["retrieval_scope"] == "full_corpus"
    assert record["latency_ms"] >= record["retrieval_latency_ms"]


def test_reranker_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match="dense_weight"):
        rerank_records(QUESTIONS, RUN, CHUNKS, dense_weight=1.1)
