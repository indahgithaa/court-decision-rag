from src.evaluation.error_analysis import analyze_retrieval_errors, render_error_markdown


def _run(query: str, strategy: str, ranking: list[tuple[str, str]]) -> dict:
    return {
        "query_id": query,
        "strategy": strategy,
        "ranking": [
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "retrieved_document_id": document_id,
                "score": 1 / rank,
            }
            for rank, (chunk_id, document_id) in enumerate(ranking, start=1)
        ],
    }


def test_error_analysis_separates_document_and_chunk_misses() -> None:
    questions = [
        {"query_id": "q1", "document_id": "d1", "target_section_label": "fakta", "question": "Q1?"},
        {"query_id": "q2", "document_id": "d2", "target_section_label": "amar_putusan", "question": "Q2?"},
    ]
    qrels = [
        {"query_id": query, "strategy": strategy, "chunk_id": f"{strategy}-{query}", "relevance_grade": "2"}
        for query in ("q1", "q2")
        for strategy in ("fixed_size", "structure_aware")
    ]
    runs = {
        "fixed_size": [
            _run("q1", "fixed_size", [("fixed_size-q1", "d1")]),
            _run("q2", "fixed_size", [("wrong-d2", "d2")]),
        ],
        "structure_aware": [
            _run("q1", "structure_aware", [("wrong-other", "d9")]),
            _run("q2", "structure_aware", [("structure_aware-q2", "d2")]),
        ],
    }
    chunks = {
        "fixed_size-q1": {"text": "relevant", "section_label": ""},
        "wrong-d2": {"text": "same document", "section_label": ""},
        "wrong-other": {"text": "wrong document", "section_label": ""},
        "structure_aware-q2": {"text": "relevant", "section_label": "amar_putusan"},
    }

    result = analyze_retrieval_errors(questions, qrels, runs, chunks, k=1)

    assert result["strategy_summary"]["fixed_size"]["status_counts"] == {
        "document_hit_chunk_miss": 1,
        "relevant_hit": 1,
    }
    assert result["strategy_summary"]["structure_aware"]["status_counts"] == {
        "document_miss": 1,
        "relevant_hit": 1,
    }
    assert result["paired_summary"] == {"fixed_only": 1, "structure_only": 1}
    assert "Hanya structure-aware" in render_error_markdown(result)
    assert "q1 - fixed-size" in render_error_markdown(result)
