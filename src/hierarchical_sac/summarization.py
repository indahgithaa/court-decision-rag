"""Cached Indonesian document and section summarization for hierarchical SAC."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.summarization.document_fingerprint import truncate_at_word_boundary
from src.utils.io import read_jsonl, write_jsonl


PROMPT_TEMPLATE = "summarize: {source_text}"
DOCUMENT_SOURCE_LABELS = (
    "kepala_putusan",
    "identitas_terdakwa",
    "riwayat_dakwaan",
    "fakta_hukum",
    "pertimbangan_hukum",
    "amar_putusan",
)


class IndonesianT5SummaryGenerator:
    """Deterministic batched wrapper for an Indonesian T5 summarizer."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        max_input_tokens: int = 512,
        max_new_tokens: int = 48,
        batch_size: int = 32,
        device: str = "cpu",
        target_chars: int = 150,
        tolerance_chars: int = 20,
    ) -> None:
        if max_input_tokens <= 8 or max_new_tokens <= 0 or batch_size <= 0:
            raise ValueError("invalid model token or batch configuration")
        self.model_id = model_id
        self.revision = revision
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.device = device
        self.target_chars = target_chars
        self.tolerance_chars = tolerance_chars
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def maximum_chars(self) -> int:
        return self.target_chars + self.tolerance_chars

    def generate_many(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        tokenizer, model = self._load()
        prepared: list[dict[str, Any]] = []
        for request in requests:
            source = " ".join(str(request["source_text"]).split())
            bounded, source_was_reduced = self._bounded_source(source, tokenizer)
            prompt = PROMPT_TEMPLATE.format(source_text=bounded)
            prepared.append(
                {
                    **dict(request),
                    "normalized_source": source,
                    "bounded_source": bounded,
                    "source_was_reduced": source_was_reduced,
                    "prompt": prompt,
                    "input_tokens": len(
                        tokenizer.encode(prompt, add_special_tokens=True, truncation=False)
                    ),
                }
            )

        results: list[dict[str, Any]] = []
        for batch_start in range(0, len(prepared), self.batch_size):
            batch = prepared[batch_start : batch_start + self.batch_size]
            prompts = [str(item["prompt"]) for item in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=1,
                num_beams=1,
                do_sample=False,
                no_repeat_ngram_size=2,
                repetition_penalty=2.5,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for item, raw_summary, output_ids in zip(batch, decoded, outputs):
                normalized = " ".join(raw_summary.strip().split())
                status = "generated"
                if not normalized:
                    normalized = truncate_at_word_boundary(
                        str(item["normalized_source"]), self.maximum_chars
                    )
                    status = "fallback_extractive_empty_generation"
                before_limit = len(normalized)
                hard_truncated = before_limit > self.maximum_chars
                if hard_truncated:
                    normalized = truncate_at_word_boundary(normalized, self.maximum_chars)
                diagnostics = summary_diagnostics(
                    normalized,
                    str(item["normalized_source"]),
                    target_chars=self.target_chars,
                    tolerance_chars=self.tolerance_chars,
                )
                results.append(
                    {
                        **{key: value for key, value in item.items() if key not in {
                            "normalized_source", "bounded_source", "prompt"
                        }},
                        "summary": normalized,
                        "summary_chars": len(normalized),
                        "summary_sha256": _text_sha256(normalized),
                        "generation_status": status,
                        "source_reduced_for_context": bool(item["source_was_reduced"]),
                        "output_tokens": int((output_ids != tokenizer.pad_token_id).sum().item()),
                        "output_chars_before_limit": before_limit,
                        "hard_truncated": hard_truncated,
                        "model_id": self.model_id,
                        "model_revision": self.revision,
                        "prompt_template": PROMPT_TEMPLATE,
                        "sampling": {
                            "do_sample": False,
                            "num_beams": 1,
                            "max_new_tokens": self.max_new_tokens,
                            "no_repeat_ngram_size": 2,
                            "repetition_penalty": 2.5,
                        },
                        "api_cost_usd": 0.0,
                        **diagnostics,
                    }
                )
        return results

    def _bounded_source(self, source: str, tokenizer: Any) -> tuple[str, bool]:
        prefix = "summarize: "
        special = int(tokenizer.num_special_tokens_to_add(pair=False))
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        source_ids = tokenizer.encode(source, add_special_tokens=False)
        capacity = self.max_input_tokens - special - len(prefix_ids)
        if capacity <= 8:
            raise ValueError("summary input budget leaves no room for source text")
        if len(source_ids) <= capacity:
            return source, False
        marker_ids = tokenizer.encode(" [...] ", add_special_tokens=False)
        remaining = capacity - len(marker_ids)
        head = remaining // 2
        tail = remaining - head
        selected = source_ids[:head] + marker_ids + source_ids[-tail:]
        return tokenizer.decode(selected, skip_special_tokens=True).strip(), True

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Local summarization requires transformers") from error
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=True,
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            local_files_only=True,
        )
        self._model.to(self.device)
        self._model.eval()
        return self._tokenizer, self._model


def generate_summary_artifacts(
    sections: Sequence[Mapping[str, Any]],
    *,
    generator: IndonesianT5SummaryGenerator,
    output_dir: Path,
    short_section_chars: int = 170,
    document_section_chars: int = 320,
    force: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Generate or load one document summary and one summary per section."""
    document_path = output_dir / "document_summaries.jsonl"
    section_path = output_dir / "section_summaries.jsonl"
    manifest_path = output_dir / "manifest.json"
    source_hash = _records_sha256(sections)
    expected_config = {
        "schema_version": "hsac-summary-v1",
        "source_sha256": source_hash,
        "model_id": generator.model_id,
        "model_revision": generator.revision,
        "target_chars": generator.target_chars,
        "tolerance_chars": generator.tolerance_chars,
        "max_input_tokens": generator.max_input_tokens,
        "max_new_tokens": generator.max_new_tokens,
        "short_section_chars": short_section_chars,
        "document_section_chars": document_section_chars,
        "prompt_template": PROMPT_TEMPLATE,
    }
    if document_path.exists() and section_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in expected_config.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Summary cache mismatch for {key}: {manifest.get(key)!r} != {value!r}"
                )
        return list(read_jsonl(document_path)), list(read_jsonl(section_path)), manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    by_document: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for section in sections:
        by_document[str(section["document_id"])].append(section)

    section_records: list[dict[str, Any]] = []
    section_requests: list[dict[str, Any]] = []
    for document_id in sorted(by_document):
        ordered = sorted(
            by_document[document_id], key=lambda row: int(row["start_position"])
        )
        for section_index, section in enumerate(ordered):
            label = str(section.get("section_label") or "unknown")
            source = " ".join(str(section.get("section_text", "")).split())
            common = {
                "summary_type": "section",
                "document_id": document_id,
                "section_index": section_index,
                "section_label": label,
                "source_start_position": int(section["start_position"]),
                "source_end_position": int(section["end_position"]),
                "source_sha256": _text_sha256(source),
                "source_chars": len(source),
                "source_policy": "full_or_token_head_tail_v1",
            }
            if len(source) <= short_section_chars:
                summary = truncate_at_word_boundary(source, generator.maximum_chars)
                section_records.append(
                    {
                        **common,
                        "summary": summary,
                        "summary_chars": len(summary),
                        "summary_sha256": _text_sha256(summary),
                        "generation_status": "deterministic_short_section",
                        "source_reduced_for_context": False,
                        "input_tokens": None,
                        "output_tokens": None,
                        "output_chars_before_limit": len(summary),
                        "hard_truncated": False,
                        "model_id": "deterministic-source-copy",
                        "model_revision": "v1",
                        "prompt_template": None,
                        "sampling": None,
                        "api_cost_usd": 0.0,
                        **summary_diagnostics(
                            summary,
                            source,
                            target_chars=generator.target_chars,
                            tolerance_chars=generator.tolerance_chars,
                        ),
                    }
                )
            else:
                section_requests.append({**common, "source_text": f"[bagian {label}] {source}"})
    section_records.extend(generator.generate_many(section_requests))
    section_records.sort(
        key=lambda row: (str(row["document_id"]), int(row["section_index"]))
    )

    document_requests: list[dict[str, Any]] = []
    for document_id in sorted(by_document):
        source = _document_representative_source(
            by_document[document_id], per_section_chars=document_section_chars
        )
        document_requests.append(
            {
                "summary_type": "document",
                "document_id": document_id,
                "source_sha256": _text_sha256(source),
                "source_chars": len(source),
                "source_policy": "section_balanced_head_tail_v1",
                "source_text": source,
            }
        )
    document_records = generator.generate_many(document_requests)
    document_records.sort(key=lambda row: str(row["document_id"]))

    write_jsonl(document_path, document_records)
    write_jsonl(section_path, section_records)
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        **expected_config,
        "generated_at_utc": generated_at,
        "document_count": len(document_records),
        "section_count": len(section_records),
        "document_summaries_sha256": _file_sha256(document_path),
        "section_summaries_sha256": _file_sha256(section_path),
        "api_cost_usd": 0.0,
        "generation_backend": "local_transformers_seq2seq",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return document_records, section_records, manifest


def summary_diagnostics(
    summary: str,
    source: str,
    *,
    target_chars: int,
    tolerance_chars: int,
) -> dict[str, Any]:
    summary_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", summary))
    source_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", source))
    unsupported_numbers = sorted(summary_numbers - source_numbers)
    summary_words = set(re.findall(r"[a-zA-Z]+", summary.casefold()))
    source_words = set(re.findall(r"[a-zA-Z]+", source.casefold()))
    support_ratio = (
        len(summary_words & source_words) / len(summary_words) if summary_words else 1.0
    )
    indonesian_markers = {
        "yang", "dan", "dengan", "terdakwa", "putusan", "pidana", "pasal",
        "bahwa", "dalam", "kepada", "terhadap", "pengadilan", "menimbang",
    }
    unexpected_language = bool(summary_words) and not bool(summary_words & indonesian_markers) and support_ratio < 0.5
    flags = []
    if not summary.strip():
        flags.append("empty")
    if len(summary) > target_chars + tolerance_chars:
        flags.append("over_length")
    if len(summary) < max(20, target_chars // 3):
        flags.append("very_short")
    if unsupported_numbers:
        flags.append("unsupported_number")
    if support_ratio < 0.35:
        flags.append("low_lexical_support")
    if unexpected_language:
        flags.append("unexpected_language")
    return {
        "unsupported_number_tokens": unsupported_numbers,
        "lexical_support_ratio": round(support_ratio, 6),
        "unexpected_language": unexpected_language,
        "quality_flags": flags,
        "automated_quality_status": "flagged" if flags else "pass",
    }


def _document_representative_source(
    sections: Sequence[Mapping[str, Any]], *, per_section_chars: int
) -> str:
    by_label = {
        str(row.get("section_label") or "unknown"): " ".join(
            str(row.get("section_text", "")).split()
        )
        for row in sections
    }
    blocks: list[str] = []
    for label in DOCUMENT_SOURCE_LABELS:
        text = by_label.get(label, "")
        if not text:
            continue
        if len(text) > per_section_chars:
            head = per_section_chars // 2
            tail = per_section_chars - head
            first = truncate_at_word_boundary(text, head)
            last = text[-tail:]
            boundary = last.find(" ")
            if boundary >= 0:
                last = last[boundary + 1 :]
            text = f"{first} [...] {last.strip()}"
        blocks.append(f"[bagian {label}] {text}")
    if not blocks:
        raise ValueError("document has no supported sections for summarization")
    return " ".join(blocks)


def _records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
