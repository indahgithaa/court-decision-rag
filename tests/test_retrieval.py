from pathlib import Path

import numpy as np
import pytest

from src.retrieval import DenseRetriever, DenseVectorStore


class FakeEmbedder:
    model_name = "fake-asymmetric"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([_vector(text) for text in texts], dtype=np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray([_vector(text) for text in texts], dtype=np.float32)


def _vector(text: str) -> list[float]:
    return [1.0, 0.0] if "narkotika" in text.lower() else [0.0, 1.0]


def test_dense_store_round_trip_and_ranking(tmp_path: Path) -> None:
    chunks = [
        {
            "chunk_id": "c-law",
            "document_id": "d1",
            "strategy": "fixed_size",
            "text": "Narkotika",
        },
        {
            "chunk_id": "c-other",
            "document_id": "d2",
            "strategy": "fixed_size",
            "text": "Perdata",
        },
    ]
    embedder = FakeEmbedder()
    store = DenseVectorStore(
        chunks,
        embedder.embed_documents([chunk["text"] for chunk in chunks]),
        model_name=embedder.model_name,
    )
    store.save(tmp_path)

    loaded = DenseVectorStore.load(tmp_path)
    results = DenseRetriever(embedder, loaded).retrieve("perkara narkotika", k=2)

    assert [result.chunk_id for result in results] == ["c-law", "c-other"]
    assert results[0].score == pytest.approx(1.0)
    assert loaded.dimension == 2


def test_dense_store_saves_provenance_without_changing_load_contract(
    tmp_path: Path,
) -> None:
    store = DenseVectorStore(
        [{"chunk_id": "c1", "document_id": "d1", "text": "x"}],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        model_name="fake-asymmetric",
    )

    store.save(tmp_path, provenance={"design": "fixed_w500_o100"})
    loaded = DenseVectorStore.load(tmp_path)

    assert loaded.model_name == "fake-asymmetric"
    assert "fixed_w500_o100" in (tmp_path / "manifest.json").read_text(
        encoding="utf-8"
    )


def test_retriever_rejects_model_mismatch() -> None:
    store = DenseVectorStore(
        [{"chunk_id": "c1", "document_id": "d1", "text": "x"}],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        model_name="different-model",
    )

    with pytest.raises(ValueError, match="does not match"):
        DenseRetriever(FakeEmbedder(), store)


def test_dense_retriever_can_restrict_search_to_one_document() -> None:
    chunks = [
        {"chunk_id": "d1-low", "document_id": "d1", "text": "Perdata"},
        {"chunk_id": "d2-high", "document_id": "d2", "text": "Narkotika"},
    ]
    embedder = FakeEmbedder()
    store = DenseVectorStore(
        chunks,
        embedder.embed_documents([chunk["text"] for chunk in chunks]),
        model_name=embedder.model_name,
    )

    results = DenseRetriever(embedder, store).retrieve(
        "perkara narkotika",
        k=5,
        document_id="d1",
    )

    assert [result.chunk_id for result in results] == ["d1-low"]


def test_dense_store_rejects_unknown_document_filter() -> None:
    store = DenseVectorStore(
        [{"chunk_id": "c1", "document_id": "d1", "text": "x"}],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        model_name="fake-asymmetric",
    )

    with pytest.raises(ValueError, match="No chunks found"):
        store.search(np.asarray([1.0, 0.0]), document_id="missing")
