"""Reproducible document fingerprints for Summary-Augmented Chunking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
import re
from typing import Any


SECTION_ORDER = (
    "kepala_putusan",
    "identitas_terdakwa",
    "riwayat_dakwaan",
    "fakta_hukum",
    "pertimbangan_hukum",
    "amar_putusan",
)


def build_representative_document(
    sections: Sequence[Mapping[str, object]],
    *,
    per_section_chars: int = 1_600,
) -> str:
    """Return a section-balanced input for a bounded-context summarizer.

    Court decisions can exceed a small local model's context window.  We retain
    both the beginning and end of every legally important section rather than
    silently truncating the complete document at its beginning.  The policy is
    deterministic and recorded in the summary artifact.
    """
    if per_section_chars < 200:
        raise ValueError("per_section_chars must be at least 200")
    by_label = {
        str(section.get("section_label", "")): " ".join(
            str(section.get("section_text", "")).split()
        )
        for section in sections
    }
    blocks: list[str] = []
    for label in SECTION_ORDER:
        text = by_label.get(label, "")
        if not text:
            continue
        if len(text) > per_section_chars:
            head_chars = per_section_chars // 2
            tail_chars = per_section_chars - head_chars
            text = (
                truncate_at_word_boundary(text, head_chars)
                + " [...] "
                + _take_tail_at_word_boundary(text, tail_chars)
            )
        blocks.append(f"[{label.replace('_', ' ')}] {text}")
    if not blocks:
        raise ValueError("No supported legal sections available for summarization")
    return "\n".join(blocks)


def build_structured_extractive_fingerprint(
    sections: Sequence[Mapping[str, object]],
    *,
    target_chars: int = 150,
    tolerance_chars: int = 20,
) -> tuple[str, dict[str, object]]:
    """Build a source-grounded legal synopsis without a generative model.

    This is a resource-conscious adaptation of Summary-Augmented Chunking.  It
    extracts the document identifiers and legal topics requested by Reuter et
    al.'s generic prompt.  Every output phrase occurs in the source document,
    avoiding hallucinated fingerprints on CPU-only research environments.
    """
    if target_chars <= 0 or tolerance_chars < 0:
        raise ValueError("summary character limits are invalid")
    by_label = {
        str(section.get("section_label", "")): " ".join(
            str(section.get("section_text", "")).split()
        )
        for section in sections
    }
    extracted = _extract_legal_components(by_label)
    components: list[str] = []
    case = extracted["case"]
    if case:
        components.append(f"perkara {case}")
    name = extracted["defendant"]
    if name:
        components.append(f"terdakwa {name}")
    statute = extracted["statute"]
    if statute:
        components.append(statute)
    if extracted["topic"]:
        components.append("perkara narkotika")
    disposition = extracted["disposition"]
    if disposition:
        components.append(truncate_at_word_boundary(disposition, 70))

    if not components:
        raise ValueError("Could not extract a document fingerprint")
    summary = truncate_at_word_boundary(
        "; ".join(dict.fromkeys(components)),
        target_chars + tolerance_chars,
    )
    return summary, {
        "model_name": "none",
        "prompt_version": "reuter-objective-structured-extractive-id-v4",
        "target_chars": target_chars,
        "tolerance_chars": tolerance_chars,
        "input_tokens": None,
        "output_chars_before_limit": len("; ".join(components)),
        "hard_truncated": len("; ".join(components)) > target_chars + tolerance_chars,
        "decoding": "deterministic_source_extraction",
    }


def build_hierarchical_extractive_contexts(
    sections: Sequence[Mapping[str, object]],
    *,
    document_max_chars: int = 120,
    section_max_chars: int = 120,
) -> tuple[str, dict[str, str]]:
    """Build source-grounded document and rhetorical-section fingerprints.

    The document context carries stable case identity.  Section contexts carry
    only information belonging to that rhetorical role, preventing a statute
    or disposition from making every chunk in the judgment look equivalent.
    """
    if document_max_chars <= 0 or section_max_chars <= 0:
        raise ValueError("context character limits must be positive")
    by_label = {
        str(section.get("section_label", "")): " ".join(
            str(section.get("section_text", "")).split()
        )
        for section in sections
    }
    extracted = _extract_legal_components(by_label)
    document_parts = []
    if extracted["case"]:
        document_parts.append(f"perkara {extracted['case']}")
    if extracted["defendant"]:
        document_parts.append(f"terdakwa {extracted['defendant']}")
    if extracted["topic"]:
        document_parts.append("perkara narkotika")
    if not document_parts:
        raise ValueError("Could not extract a document-level context")
    document_context = truncate_at_word_boundary(
        "; ".join(document_parts), document_max_chars
    )

    section_contexts: dict[str, str] = {}
    for label in by_label:
        parts = [f"bagian dokumen {label.replace('_', ' ')}"]
        if label == "identitas_terdakwa" and extracted["defendant"]:
            parts.append(f"identitas terdakwa {extracted['defendant']}")
        elif label == "riwayat_dakwaan" and extracted["charge"]:
            parts.append(f"dakwaan {extracted['charge']}")
        elif label == "pertimbangan_hukum" and extracted["statute"]:
            parts.append(f"ketentuan pidana {extracted['statute']}")
        elif label == "amar_putusan" and extracted["disposition"]:
            parts.append(extracted["disposition"])
        elif label == "riwayat_penahanan" and extracted["defendant"]:
            parts.append(f"penahanan terdakwa {extracted['defendant']}")
        section_contexts[label] = truncate_at_word_boundary(
            "; ".join(parts), section_max_chars
        )
    return document_context, section_contexts


def truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate normalized text without cutting a final word."""
    normalized = " ".join(str(text).split())
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(normalized) <= max_chars:
        return normalized
    candidate = normalized[: max_chars + 1]
    boundary = candidate.rfind(" ", 0, max_chars + 1)
    if boundary <= 0:
        return normalized[:max_chars].rstrip()
    return candidate[:boundary].rstrip(" ,;:-")


def _take_tail_at_word_boundary(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    candidate = normalized[-max_chars:]
    boundary = candidate.find(" ")
    return candidate[boundary + 1 :].lstrip(" ,;:-") if boundary >= 0 else candidate


def _first_group(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return " ".join(match.group(1).split()) if match else ""


def _defendant_name(by_label: Mapping[str, str]) -> str:
    sources = (
        (
            "riwayat_tuntutan",
            r"\bmenyatakan\s+terdakwa\s+(.+?)"
            r"(?=\s+(?:(?:secara\s+sah\s+dan\s+(?:meyakinkan|menyakinkan)|tidak|telah)\s+)*"
            r"(?:bersalah|terbukti)\b)",
        ),
        (
            "amar_putusan",
            r"\bmenyatakan\s+terdakwa\s+(.+?)"
            r"(?=\s+(?:tersebut\s+diatas\s+)?"
            r"(?:(?:secara\s+sah\s+dan\s+(?:meyakinkan|menyakinkan)|tidak|telah)\s+)*"
            r"(?:bersalah|terbukti)\b)",
        ),
        (
            "riwayat_dakwaan",
            r"\b(?:bahwa\s+)?(?:ia\s+)?terdakwa\s+(.+?)"
            r"(?=\s+(?:pada|bersama\s+sama|telah|yang)\b)",
        ),
        (
            "identitas_terdakwa",
            r"\b(?:\d+\s+)?nama(?:\s+lengkap)?\s+(.+?)"
            r"(?=\s+(?:\d+\s+)?tempat\s+lahir\b)",
        ),
        (
            "kepala_putusan",
            r"\bdalam\s+perkara\s+terdakwa(?:\s+sebagai\s+berikut)?\s+(.+)$",
        ),
    )
    for label, pattern in sources:
        candidate = _first_group(by_label.get(label, ""), pattern).strip(
            "0123456789 ,;:-"
        )
        if _plausible_person_name(candidate):
            return candidate
    return ""


def _extract_legal_components(by_label: Mapping[str, str]) -> dict[str, str]:
    heading = by_label.get("kepala_putusan", "")
    complete_text = " ".join(by_label.values())
    return {
        "case": _first_group(
            heading,
            r"\bputusan\s+nomor\s+(.{1,100}?)(?=\s+demi\s+keadilan\b)",
        ),
        "defendant": _defendant_name(by_label),
        "statute": _most_salient_statute(
            by_label.get("pertimbangan_hukum", "")
        ),
        "topic": (
            "narkotika"
            if re.search(r"\bnarkotika\b", complete_text, flags=re.IGNORECASE)
            else ""
        ),
        "disposition": _disposition(by_label.get("amar_putusan", "")),
        "charge": _charge(by_label.get("riwayat_dakwaan", "")),
    }


def _plausible_person_name(candidate: str) -> bool:
    words = candidate.casefold().split()
    if not 1 <= len(words) <= 12 or re.search(r"\d", candidate):
        return False
    non_name_words = {
        "umur",
        "tanggal",
        "lahir",
        "tahun",
        "januari",
        "februari",
        "maret",
        "april",
        "mei",
        "juni",
        "juli",
        "agustus",
        "september",
        "oktober",
        "nopember",
        "november",
        "desember",
        "jenis",
        "kelamin",
        "kebangsaan",
        "tempat",
        "tinggal",
    }
    return not any(word in non_name_words for word in words)


def _most_salient_statute(reasoning: str) -> str:
    article = (
        r"pasal\s+\d+(?:\s+ayat\s+(?:\(\s*)?\d+(?:\s*\))?)?"
        r"(?:\s+(?:huruf|butir)\s+[a-z])?"
    )
    compound_article = rf"{article}(?:\s+jo\s+{article})?"
    anchored = re.findall(
        rf"\b(?:memperhatikan|mengingat)\s+({compound_article})",
        reasoning,
        flags=re.IGNORECASE,
    )
    procedural_articles = {"pasal 22", "pasal 183", "pasal 193", "pasal 197", "pasal 222"}
    substantive_anchors = [
        value
        for value in anchored
        if not any(
            " ".join(value.casefold().split()).startswith(article)
            for article in procedural_articles
        )
    ]
    if substantive_anchors:
        return substantive_anchors[-1]

    statutes = re.findall(
        rf"\b{article}",
        reasoning,
        flags=re.IGNORECASE,
    )
    if not statutes:
        return ""
    normalized = [" ".join(value.casefold().replace("(", "").replace(")", "").split()) for value in statutes]
    substantive = [
        (index, key)
        for index, key in enumerate(normalized)
        if key not in {"pasal 1", "pasal 222", "pasal 197"}
    ]
    candidates = substantive or list(enumerate(normalized))
    counts = Counter(key for _, key in candidates)
    best_key = max(
        counts,
        key=lambda key: (counts[key], -next(index for index, value in candidates if value == key)),
    )
    first_index = next(index for index, key in candidates if key == best_key)
    return statutes[first_index]


def _disposition(ruling: str) -> str:
    patterns = (
        r"\b((?:menjatuhkan\s+pidana).{10,120})",
        r"\b((?:membebaskan\s+terdakwa).{10,120})",
        r"\b((?:menyatakan\s+terdakwa).{10,120})",
    )
    for pattern in patterns:
        value = _first_group(ruling, pattern)
        if value:
            return value
    return ""


def _charge(charge_history: str) -> str:
    patterns = (
        r"\b((?:tanpa\s+hak\s+atau\s+melawan\s+hukum).{10,120})",
        r"\b((?:menyalahgunakan\s+narkotika).{10,120})",
    )
    for pattern in patterns:
        value = _first_group(charge_history, pattern)
        if value:
            return truncate_at_word_boundary(value, 80)
    return ""


class QwenDocumentFingerprintGenerator:
    """Generate one concise legal-document summary with a local instruction LLM.

    Reuter et al. (2025) use GPT-4o-mini and target 150 characters.  This local,
    open-weight adaptation preserves their generic prompt objective and one-call
    per-document design while using a reproducible model available offline after
    its first download.
    """

    prompt_version = "reuter-generic-id-v1"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        target_chars: int = 150,
        tolerance_chars: int = 20,
        max_input_tokens: int = 4_096,
        max_new_tokens: int = 64,
        device: str = "cpu",
    ) -> None:
        if target_chars <= 0 or tolerance_chars < 0:
            raise ValueError("summary character limits are invalid")
        if max_input_tokens <= 0 or max_new_tokens <= 0:
            raise ValueError("token limits must be positive")
        self.model_name = model_name
        self.target_chars = target_chars
        self.tolerance_chars = tolerance_chars
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def maximum_chars(self) -> int:
        return self.target_chars + self.tolerance_chars

    def generate(self, document_text: str) -> tuple[str, dict[str, object]]:
        """Generate and length-normalize one fingerprint."""
        tokenizer, model = self._load()
        messages = [
            {
                "role": "system",
                "content": "Anda adalah peringkas ahli dokumen hukum Indonesia.",
            },
            {
                "role": "user",
                "content": (
                    "Ringkas teks putusan berikut. Utamakan entitas terpenting, "
                    "tujuan perkara, dan topik hukum utama. Ringkasan digunakan "
                    "sebagai konteks pembeda untuk potongan teks yang lebih kecil. "
                    f"Maksimal {self.target_chars} karakter. Keluarkan hanya "
                    "ringkasan tanpa judul atau penjelasan.\n\nDokumen:\n"
                    + document_text
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        input_length = int(inputs["input_ids"].shape[1])
        decoded = tokenizer.decode(
            output[0][input_length:],
            skip_special_tokens=True,
        )
        normalized = self._clean_output(decoded)
        if not normalized:
            raise RuntimeError("Summarization model returned a blank fingerprint")
        truncated = len(normalized) > self.maximum_chars
        if truncated:
            normalized = truncate_at_word_boundary(normalized, self.maximum_chars)
        metadata: dict[str, object] = {
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "target_chars": self.target_chars,
            "tolerance_chars": self.tolerance_chars,
            "input_tokens": input_length,
            "output_chars_before_limit": len(" ".join(decoded.split())),
            "hard_truncated": truncated,
            "decoding": "greedy",
        }
        return normalized, metadata

    @staticmethod
    def _clean_output(value: str) -> str:
        cleaned = " ".join(value.strip().strip('"').split())
        lowered = cleaned.casefold()
        for prefix in ("ringkasan:", "ringkasan dokumen:", "summary:"):
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break
        return cleaned

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Document fingerprint generation requires transformers."
            ) from error
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        return self._tokenizer, self._model
