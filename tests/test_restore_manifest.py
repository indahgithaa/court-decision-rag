import hashlib

import pytest


def _load_script_functions() -> dict:
    namespace: dict = {"__name__": "restore_manifest_test"}
    source = open(
        "scripts/18_restore_indolaw_from_manifest.py", encoding="utf-8"
    ).read()
    exec(compile(source, "scripts/18_restore_indolaw_from_manifest.py", "exec"), namespace)
    return namespace


def test_repository_slug_uses_pinned_github_repository() -> None:
    functions = _load_script_functions()

    assert (
        functions["_repository_slug"]("https://github.com/ir-nlp-csui/indo-law")
        == "ir-nlp-csui/indo-law"
    )


def test_verify_payload_rejects_hash_mismatch() -> None:
    functions = _load_script_functions()
    payload = b"frozen XML"

    functions["_verify_payload"](
        payload, hashlib.sha256(payload).hexdigest(), len(payload), "ok.xml"
    )
    with pytest.raises(ValueError, match="Integrity mismatch"):
        functions["_verify_payload"](
            payload, "0" * 64, len(payload), "changed.xml"
        )
