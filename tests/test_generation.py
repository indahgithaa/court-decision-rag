"""Tests for fair, deterministic answer-generation inputs."""

import pytest

from src.generation.context import assemble_ranked_context, index_chunks
from src.generation.generator import balanced_generation_order, generate_answer
from src.generation.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt


def test_context_uses_exact_budget_and_truncates_final_chunk() -> None:
    chunks = index_chunks(
        [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "section_label": "fakta",
                "start_position": 10,
                "end_position": 29,
                "text": "satu dua tiga empat",
            },
            {
                "chunk_id": "c2",
                "document_id": "d2",
                "section_label": None,
                "section_heading": "Amar",
                "start_position": 100,
                "end_position": 126,
                "text": "lima enam tujuh delapan",
            },
        ]
    )
    result = assemble_ranked_context(
        [
            {"rank": 2, "chunk_id": "c2", "retrieved_document_id": "d2", "score": 0.7},
            {"rank": 1, "chunk_id": "c1", "retrieved_document_id": "d1", "score": 0.9},
        ],
        chunks,
        word_budget=6,
    )

    assert result["context_word_count"] == 6
    assert [item["chunk_id"] for item in result["selected_chunks"]] == ["c1", "c2"]
    assert result["selected_chunks"][1]["included_word_count"] == 2
    assert result["selected_chunks"][1]["truncated"] is True
    assert result["selected_chunks"][1]["included_end_position"] == 109
    assert result["context"].endswith("lima enam")
    assert "dokumen=" not in result["context"]
    assert "bagian=" not in result["context"]


def test_context_rejects_insufficient_ranked_text() -> None:
    chunks = index_chunks(
        [{"chunk_id": "c1", "document_id": "d1", "start_position": 0, "end_position": 8, "text": "dua kata"}]
    )
    with pytest.raises(ValueError, match="supplies only 2 words"):
        assemble_ranked_context(
            [{"rank": 1, "chunk_id": "c1", "score": 1.0}],
            chunks,
            word_budget=3,
        )


def test_context_rejects_document_mismatch() -> None:
    chunks = index_chunks(
        [{"chunk_id": "c1", "document_id": "d1", "start_position": 0, "end_position": 4, "text": "teks"}]
    )
    with pytest.raises(ValueError, match="Document mismatch"):
        assemble_ranked_context(
            [{"rank": 1, "chunk_id": "c1", "retrieved_document_id": "d2", "score": 1.0}],
            chunks,
            word_budget=1,
        )


def test_prompt_is_frozen_and_does_not_require_reference_answer() -> None:
    prompt = build_user_prompt(question="Apa putusannya?", context="Terdakwa dibebaskan.")

    assert PROMPT_VERSION == "legal_qa_id_v1"
    assert "hanya berdasarkan konteks" in SYSTEM_PROMPT
    assert "Apa putusannya?" in prompt
    assert "Terdakwa dibebaskan." in prompt
    assert prompt.endswith("Jawaban:")


@pytest.mark.parametrize("field", ("question", "context"))
def test_prompt_rejects_empty_input(field: str) -> None:
    values = {"question": "Pertanyaan", "context": "Konteks"}
    values[field] = " "
    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        build_user_prompt(**values)


def test_generate_answer_records_model_parameters_and_latency() -> None:
    class FakeGenerator:
        model_name = "fake/model-v1"
        parameters = {"temperature": 0}

        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            assert system_prompt == "sistem"
            assert user_prompt == "pengguna"
            return "  jawaban  "

    times = iter((10.0, 10.125))
    result = generate_answer(
        {"query_id": "q1", "system_prompt": "sistem", "user_prompt": "pengguna"},
        FakeGenerator(),
        clock=lambda: next(times),
    )

    assert result["answer"] == "jawaban"
    assert result["generation_model"] == "fake/model-v1"
    assert result["generation_parameters"] == {"temperature": 0}
    assert result["generation_latency_ms"] == pytest.approx(125.0)


def test_balanced_generation_order_is_reproducible_and_alternates_first_arm() -> None:
    records = [
        {"query_id": query_id, "strategy": strategy}
        for query_id in ("q1", "q2", "q3", "q4")
        for strategy in ("fixed_size", "structure_aware")
    ]

    ordered = balanced_generation_order(records, seed=7)
    repeated = balanced_generation_order(records, seed=7)
    first_strategies = [ordered[index]["strategy"] for index in range(0, 8, 2)]

    assert ordered == repeated
    assert first_strategies == [
        "fixed_size",
        "structure_aware",
        "fixed_size",
        "structure_aware",
    ]
    assert all(
        ordered[index]["query_id"] == ordered[index + 1]["query_id"]
        for index in range(0, 8, 2)
    )
