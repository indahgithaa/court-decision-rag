"""Restore the exact Indo-Law files recorded in a frozen research manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Restore pinned Indo-Law XML files and verify every SHA-256."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/indolaw_200_manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/indolaw_200")
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    repository = _repository_slug(str(manifest["source_repository"]))
    commit = str(manifest["source_commit"]).strip()
    documents = list(manifest["documents"])
    if len(documents) != int(manifest["sample_size"]):
        raise ValueError("Manifest sample_size does not match its document list")
    names = [Path(str(record["local_file"])).name for record in documents]
    if len(set(names)) != len(names):
        raise ValueError("Manifest contains colliding destination filenames")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "sac-rag-qa-legal-reproducibility"}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                lambda record: _restore_one(
                    record,
                    repository=repository,
                    commit=commit,
                    output_dir=args.output_dir,
                    headers=headers,
                ),
                documents,
            )
        )

    downloaded = sum(result == "downloaded" for result in results)
    verified_existing = sum(result == "verified_existing" for result in results)
    print(
        f"Verified {len(results)} files from {repository}@{commit}; "
        f"downloaded={downloaded}; existing={verified_existing}; "
        f"output={args.output_dir}"
    )


def _restore_one(
    record: Mapping[str, Any],
    *,
    repository: str,
    commit: str,
    output_dir: Path,
    headers: Mapping[str, str],
) -> str:
    source_path = str(record["source_path"])
    expected_sha256 = str(record["sha256"]).lower()
    expected_bytes = int(record["bytes"])
    destination = output_dir / Path(str(record["local_file"])).name
    if destination.exists():
        _verify_file(destination, expected_sha256, expected_bytes)
        return "verified_existing"

    url = (
        f"https://raw.githubusercontent.com/{repository}/{commit}/{source_path}"
    )
    payload = _download(url, headers)
    _verify_payload(payload, expected_sha256, expected_bytes, source_path)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "downloaded"


def _verify_file(path: Path, expected_sha256: str, expected_bytes: int) -> None:
    payload = path.read_bytes()
    _verify_payload(payload, expected_sha256, expected_bytes, path.as_posix())


def _verify_payload(
    payload: bytes, expected_sha256: str, expected_bytes: int, source: str
) -> None:
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_bytes or actual_sha256 != expected_sha256:
        raise ValueError(
            f"Integrity mismatch for {source}: bytes={len(payload)} "
            f"sha256={actual_sha256}"
        )


def _download(url: str, headers: Mapping[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=dict(headers))
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _repository_slug(url: str) -> str:
    prefix = "https://github.com/"
    if not url.startswith(prefix):
        raise ValueError(f"Unsupported source_repository: {url}")
    slug = url.removeprefix(prefix).strip("/")
    if slug.endswith(".git"):
        slug = slug[:-4]
    if slug.count("/") != 1:
        raise ValueError(f"Invalid GitHub repository slug: {slug}")
    return slug


if __name__ == "__main__":
    main()
