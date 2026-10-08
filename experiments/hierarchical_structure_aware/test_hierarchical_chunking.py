from __future__ import annotations

import re

from experiments.hierarchical_structure_aware.hierarchical_chunking import (
    HierarchicalChunker,
    aggregate_parent_ranking,
    select_context_budget,
)


def _text(words: int) -> str:
    return " ".join(f"w{index}" for index in range(words))


def test_b3_ignores_sections_and_maps_every_child_to_parent() -> None:
    text = _text(1200)
    fake_sections = [
        {
            "section_text": text[:20],
            "start_position": 0,
            "end_position": 20,
            "section_label": "gold_label_must_be_ignored",
        }
    ]
    chunker = HierarchicalChunker(structure_aware=False)
    parents_a, children_a = chunker.chunk("doc", text, sections=fake_sections)
    parents_b, children_b = chunker.chunk("doc", text, sections=None)
    assert parents_a == parents_b
    assert children_a == children_b
    by_id = {parent["parent_id"]: parent for parent in parents_a}
    for child in children_a:
        parent = by_id[child["parent_id"]]
        assert parent["start_position"] <= child["start_position"]
        assert child["end_position"] <= parent["end_position"]
        assert text[child["start_position"] : child["end_position"]] == child["text"]


def test_m1_never_crosses_section_boundaries_and_preserves_words() -> None:
    first = _text(700)
    second = " ".join(f"z{index}" for index in range(620))
    text = first + "\n\n" + second
    sections = [
        {
            "section_text": first,
            "start_position": 0,
            "end_position": len(first),
            "section_label": "pertimbangan_hukum",
            "section_heading": "menimbang",
        },
        {
            "section_text": second,
            "start_position": len(first) + 2,
            "end_position": len(text),
            "section_label": "amar_putusan",
            "section_heading": "mengadili",
        },
    ]
    parents, children = HierarchicalChunker(structure_aware=True).chunk(
        "doc", text, sections=sections
    )
    bounds = {
        "pertimbangan_hukum": (0, len(first)),
        "amar_putusan": (len(first) + 2, len(text)),
    }
    for row in [*parents, *children]:
        start, end = bounds[row["section_label"]]
        assert start <= row["start_position"] < row["end_position"] <= end
        assert text[row["start_position"] : row["end_position"]] == row["text"]
    covered = set()
    for child in children:
        for match in re.finditer(r"\S+", text):
            if child["start_position"] <= match.start() and match.end() <= child["end_position"]:
                covered.add(match.group(0))
    assert len(covered) == 1320


def test_parent_aggregation_uses_max_child_score_and_deduplicates() -> None:
    parents = {
        "p1": {"parent_id": "p1", "text": "one two", "document_id": "d"},
        "p2": {"parent_id": "p2", "text": "three", "document_id": "d"},
    }
    children = [
        {"chunk_id": "c1", "parent_id": "p1", "rank": 1, "score": 0.7},
        {"chunk_id": "c2", "parent_id": "p2", "rank": 2, "score": 0.6},
        {"chunk_id": "c3", "parent_id": "p1", "rank": 3, "score": 0.5},
    ]
    ranking = aggregate_parent_ranking(children, parents)
    assert [row["parent_id"] for row in ranking] == ["p1", "p2"]
    assert ranking[0]["score"] == 0.7
    assert ranking[0]["best_child_id"] == "c1"


def test_context_budget_is_ranked_unique_and_bounded() -> None:
    ranking = [
        {"parent_id": "p1", "text": _text(4)},
        {"parent_id": "p1", "text": _text(4)},
        {"parent_id": "p2", "text": _text(3)},
        {"parent_id": "p3", "text": _text(2)},
    ]
    selected, used = select_context_budget(ranking, budget_words=7)
    assert [row["parent_id"] for row in selected] == ["p1", "p2"]
    assert used == 7

