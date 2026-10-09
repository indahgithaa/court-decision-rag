"""Scientific-contract tests for the controlled B1/B2/M1 experiment."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from src.hierarchical_sac.chunking import (
    RecursiveCharacterChunker,
    assign_chunk_context,
)
from src.hierarchical_sac.metrics import (
    evaluate_method,
    map_evidence_to_chunks,
    paired_cluster_comparisons,
)
from src.hierarchical_sac.representations import (
    apply_e5_token_budget,
    assert_aligned,
    build_embedding_representations,
)
from src.hierarchical_sac.summarization import generate_summary_artifacts


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool, truncation: bool = False):
        del truncation
        values = [ord(char) for char in text]
        return ([1] + values + [2]) if add_special_tokens else values

    def decode(self, values, *, skip_special_tokens: bool = True):
        del skip_special_tokens
        return "".join(chr(value) for value in values if value > 2)


class FakeSummaryGenerator:
    model_id = "fake-local-summarizer"
    revision = "test-revision"
    target_chars = 150
    tolerance_chars = 20
    max_input_tokens = 512
    max_new_tokens = 48
    maximum_chars = 170

    def __init__(self) -> None:
        self.calls = 0

    def generate_many(self, requests):
        self.calls += 1
        return [
            {
                **dict(request),
                "summary": str(request["source_text"])[:150],
                "summary_chars": min(150, len(str(request["source_text"]))),
                "summary_sha256": "test",
                "generation_status": "generated",
                "source_reduced_for_context": False,
                "input_tokens": 1,
                "output_tokens": 1,
                "output_chars_before_limit": min(150, len(str(request["source_text"]))),
                "hard_truncated": False,
                "model_id": self.model_id,
                "model_revision": self.revision,
                "prompt_template": "summarize: {source_text}",
                "sampling": {},
                "api_cost_usd": 0.0,
                "unsupported_number_tokens": [],
                "lexical_support_ratio": 1.0,
                "unexpected_language": False,
                "quality_flags": [],
            }
            for request in requests
        ]


def test_recursive_chunks_are_deterministic_source_exact_and_bounded() -> None:
    text = ("Pertimbangan hukum menjelaskan unsur delik. " * 40).strip()
    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=0)

    first = chunker.split("doc-1", text)
    second = chunker.split("doc-1", text)

    assert first == second
    assert all(text[row["start_position"] : row["end_position"]] == row["original_text"] for row in first)
    assert all(len(row["original_text"]) <= 120 for row in first)
    assert len({row["chunk_id"] for row in first}) == len(first)


def test_section_assignment_uses_maximum_overlap_and_marks_mixed() -> None:
    chunks = [
        {
            "chunk_id": "c",
            "document_id": "d",
            "chunk_index": 0,
            "original_text": "x" * 100,
            "start_position": 50,
            "end_position": 150,
        }
    ]
    sections = [
        {"section_label": "fakta", "section_heading": "Fakta", "start_position": 0, "end_position": 80},
        {"section_label": "pertimbangan_hukum", "section_heading": "Menimbang", "start_position": 80, "end_position": 200},
    ]

    result = assign_chunk_context(
        chunks,
        sections=sections,
        page_spans=[{"page_number": 1, "start_position": 0, "end_position": 200}],
    )[0]

    assert result["section_label"] == "pertimbangan_hukum"
    assert result["section_assignment_status"] == "mixed_maximum_overlap"
    assert result["section_overlap_chars"] == 70
    assert result["page_numbers"] == [1]


def test_three_representations_keep_identical_chunk_identity_and_provenance() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "chunk_index": 0,
            "original_text": "bukti satu",
            "start_position": 0,
            "end_position": 10,
            "section_label": "fakta",
        },
        {
            "chunk_id": "c2",
            "document_id": "d2",
            "chunk_index": 0,
            "original_text": "bukti dua",
            "start_position": 0,
            "end_position": 9,
            "section_label": "amar_putusan",
        },
    ]
    representations = build_embedding_representations(
        chunks,
        document_summaries={"d1": "dokumen satu", "d2": "dokumen dua"},
        section_summaries={
            ("d1", "fakta"): "fakta satu",
            ("d2", "amar_putusan"): "amar dua",
        },
    )

    assert_aligned(representations)
    assert [row["chunk_id"] for row in representations["B1"]] == ["c1", "c2"]
    assert "dokumen satu" in representations["B2"][0]["embedding_text"]
    assert "dokumen dua" not in representations["B2"][0]["embedding_text"]
    assert "fakta satu" in representations["M1"][0]["embedding_text"]
    assert representations["B1"][0]["embedding_text"] == "bukti satu"


def test_summary_cache_preserves_source_hashes_and_document_isolation(tmp_path: Path) -> None:
    first_source = "fakta alfa " * 30
    second_source = "amar beta " * 30
    sections = [
        {
            "document_id": "d1",
            "section_label": "fakta_hukum",
            "section_text": first_source,
            "start_position": 0,
            "end_position": len(first_source),
        },
        {
            "document_id": "d2",
            "section_label": "amar_putusan",
            "section_text": second_source,
            "start_position": 0,
            "end_position": len(second_source),
        },
    ]
    generator = FakeSummaryGenerator()

    documents, section_records, manifest = generate_summary_artifacts(
        sections, generator=generator, output_dir=tmp_path
    )

    assert manifest["document_count"] == 2
    assert manifest["section_count"] == 2
    expected_hashes = {
        "d1": hashlib.sha256(first_source.strip().encode("utf-8")).hexdigest(),
        "d2": hashlib.sha256(second_source.strip().encode("utf-8")).hexdigest(),
    }
    assert {row["document_id"]: row["source_sha256"] for row in section_records} == expected_hashes
    assert "beta" not in next(row for row in documents if row["document_id"] == "d1")["source_text"]
    assert "alfa" not in next(row for row in documents if row["document_id"] == "d2")["source_text"]

    cached = generate_summary_artifacts(sections, generator=generator, output_dir=tmp_path)
    assert generator.calls == 2  # section batch + document batch; cache adds no calls
    assert cached[2] == manifest


def test_token_budget_removes_summary_before_original_evidence() -> None:
    record = {
        "chunk_id": "c1",
        "method": "M1",
        "original_text": "BUKTI-ASLI",
        "document_summary": "d" * 30,
        "section_summary": "s" * 30,
        "embedding_text": (
            "Ringkasan Dokumen:\n" + "d" * 30 + "\n\nRingkasan Bagian:\n"
            + "s" * 30 + "\n\nTeks Asli:\nBUKTI-ASLI"
        ),
    }

    result = apply_e5_token_budget(
        [record], tokenizer=CharacterTokenizer(), max_tokens=80, passage_prefix="p: "
    )[0]

    assert result["summary_budget_applied"] is True
    assert "BUKTI-ASLI" in result["embedding_text"]
    assert result["source_text_preserved"] is True
    assert result["embedding_tokens"] <= 80


def test_metrics_use_evidence_denominator_and_multi_document_drm() -> None:
    questions = [
        {
            "query_id": "q1",
            "document_id": "d1",
            "relevant_document_ids": ["d1", "d2"],
            "target_section_label": "fakta",
        }
    ]
    qrels = {"q1": {"c1": 2.0, "c2": 1.0}}
    evidence = {"q1": {"e1": ["c1"], "e2": ["c2"]}}
    runs = [
        {
            "query_id": "q1",
            "ranking": [
                {"chunk_id": "noise", "document_id": "d2", "rank": 1},
                {"chunk_id": "c1", "document_id": "d1", "rank": 2},
                {"chunk_id": "wrong", "document_id": "d3", "rank": 3},
            ],
        }
    ]

    result = evaluate_method(questions, runs, qrels, evidence, ks=(1, 3))
    row = result["per_query"]["q1"]

    assert row["recall@1"] == 0.0
    assert row["recall@3"] == 0.5
    assert row["mrr@3"] == 0.5
    assert row["drm@1"] == 0.0
    assert row["drm@3"] == pytest.approx(1 / 3)


def test_evidence_mapping_is_independent_of_method() -> None:
    chunks = [
        {"chunk_id": "c1", "document_id": "d", "start_position": 0, "end_position": 10},
        {"chunk_id": "c2", "document_id": "d", "start_position": 10, "end_position": 20},
    ]
    questions = [
        {
            "query_id": "q",
            "document_id": "d",
            "target_section_label": "fakta",
            "evidence_start_position": 8,
            "evidence_end_position": 12,
        }
    ]

    qrels, evidence, rows = map_evidence_to_chunks(questions, chunks)

    assert qrels == {"q": {"c1": 1.0, "c2": 1.0}}
    assert set(evidence["q"].values().__iter__().__next__()) == {"c1", "c2"}
    assert {row["chunk_id"] for row in rows} == {"c1", "c2"}


def test_cluster_bootstrap_is_reproducible() -> None:
    per_query_b1 = {
        "q1": {"document_id": "d1", "target_section_label": "f", "recall@5": 0.0, "mrr@5": 0.0, "ndcg@5": 0.0, "drm@5": 1.0},
        "q2": {"document_id": "d2", "target_section_label": "f", "recall@5": 1.0, "mrr@5": 1.0, "ndcg@5": 1.0, "drm@5": 0.0},
    }
    per_query_b2 = {
        "q1": {**per_query_b1["q1"], "recall@5": 1.0, "mrr@5": 1.0, "ndcg@5": 1.0, "drm@5": 0.0},
        "q2": dict(per_query_b1["q2"]),
    }
    results = {
        "B1": {"per_query": per_query_b1},
        "B2": {"per_query": per_query_b2},
        "M1": {"per_query": per_query_b2},
    }

    first = paired_cluster_comparisons(
        results, seed=7, bootstrap_samples=100, permutation_samples=100
    )
    second = paired_cluster_comparisons(
        results, seed=7, bootstrap_samples=100, permutation_samples=100
    )

    assert first == second


def test_historical_development_and_holdout_are_court_disjoint() -> None:
    manifest = json.loads(Path("experiments/indolaw_200_manifest.json").read_text(encoding="utf-8"))
    courts = {"development": set(), "holdout": set()}
    for row in manifest["documents"]:
        courts[str(row["split"])].add(str(row["court"]))

    assert courts["development"].isdisjoint(courts["holdout"])


def test_small_end_to_end_contract() -> None:
    text = "Identitas terdakwa. Fakta hukum yang relevan. Amar putusan."
    chunks = RecursiveCharacterChunker(chunk_size=35).split("d", text)
    enriched = assign_chunk_context(
        chunks,
        sections=[
            {"section_label": "fakta", "section_heading": "Fakta", "start_position": 0, "end_position": len(text)}
        ],
        page_spans=[{"page_number": 1, "start_position": 0, "end_position": len(text)}],
    )
    representations = build_embedding_representations(
        enriched,
        document_summaries={"d": "putusan pidana"},
        section_summaries={("d", "fakta"): "fakta hukum"},
    )

    assert len(representations["B1"]) == len(representations["B2"]) == len(representations["M1"])
    assert all(row["original_text"] in row["embedding_text"] for row in representations["M1"])
