"""Frozen prompt template for Indonesian legal question answering."""

from __future__ import annotations


PROMPT_VERSION = "legal_qa_id_v1"

SYSTEM_PROMPT = """Anda adalah sistem tanya jawab dokumen putusan pengadilan Indonesia.
Jawab hanya berdasarkan konteks yang diberikan. Jangan gunakan pengetahuan di luar konteks,
jangan menambah fakta, dan jangan memberikan nasihat hukum. Jika konteks tidak memuat jawaban,
jawab persis: Informasi tidak ditemukan dalam konteks. Berikan jawaban langsung dan ringkas
dalam bahasa Indonesia."""


def build_user_prompt(*, question: str, context: str) -> str:
    """Build a prompt without exposing the benchmark reference answer."""
    clean_question = question.strip()
    clean_context = context.strip()
    if not clean_question:
        raise ValueError("question must not be empty")
    if not clean_context:
        raise ValueError("context must not be empty")
    return f"""Konteks:
{clean_context}

Pertanyaan:
{clean_question}

Jawaban:"""

