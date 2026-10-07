"""Tests for reproducible configuration and JSONL utilities."""

from pathlib import Path

from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl


def test_jsonl_round_trip_preserves_unicode(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "records.jsonl"
    records = [{"text": "putusan pengadilan"}, {"text": "terdakwa—bebas"}]

    assert write_jsonl(path, records) == 2
    assert list(read_jsonl(path)) == records


def test_experiment_config_inherits_default_values() -> None:
    config = load_config(Path("configs/structure_aware.yaml"))

    assert config["paths"]["processed_pages"] == "data/processed/pages.jsonl"
    assert config["chunking"]["strategy"] == "structure_aware"


def test_adaptive_structure_config_is_marked_post_hoc() -> None:
    config = load_config(Path("configs/structure_aware_adaptive.yaml"))

    assert config["experiment"]["status"] == "post_hoc_development_candidate"
    assert config["chunking"]["embedding_context"] == (
        "section_reasoning_document"
    )


def test_pure_sac_v2_config_has_no_contextual_embedding() -> None:
    config = load_config(Path("configs/structure_aware_pure_v2.yaml"))

    assert config["experiment"]["status"] == "post_hoc_development_candidate"
    assert config["chunking"]["boundary_mode"] == "legal_rhetorical"
    assert config["chunking"]["boundary_min_fill_ratio"] == 0.7
    assert config["chunking"]["embedding_context"] == "none"


def test_pure_sac_v2_has_a_matched_fixed_control() -> None:
    fixed = load_config(Path("configs/fixed_size_matched.yaml"))
    sac = load_config(Path("configs/structure_aware_pure_v2.yaml"))

    assert fixed["experiment"]["status"] == "post_hoc_development_control"
    assert fixed["chunking"]["strategy"] == "fixed_size"
    assert fixed["chunking"]["max_words"] == sac["chunking"]["max_words"]
    assert fixed["chunking"]["overlap_words"] == sac["chunking"]["overlap_words"]

