"""Acquire and freeze a 200-document Indo-Law research corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.data_sources.indolaw import is_holdout_candidate, parse_indolaw_xml


REPOSITORY = "ir-nlp-csui/indo-law"
DEFAULT_EXCLUDED_COURTS = (
    "pn-cirebon",
    "pn-gunungsitoli",
    "pn-klaten",
    "pn-lhokseumawe",
    "pn-mentok",
    "pn-palu",
    "pn-rangkasbitung",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Acquire a reproducible 200-document Indo-Law corpus."
    )
    parser.add_argument("--target-documents", type=int, default=200)
    parser.add_argument("--development-documents", type=int, default=40)
    parser.add_argument("--max-per-court", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--excluded-courts", nargs="+", default=DEFAULT_EXCLUDED_COURTS
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/indolaw_200")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/indolaw_200_manifest.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    _validate_args(args)
    if args.manifest.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace {args.manifest}; pass --overwrite if intentional"
        )

    headers = {"User-Agent": "sac-rag-qa-legal-research"}
    repository = _get_json(f"https://api.github.com/repos/{REPOSITORY}", headers)
    branch = repository["default_branch"]
    commit = _get_json(
        f"https://api.github.com/repos/{REPOSITORY}/commits/{branch}", headers
    )["sha"]
    tree = _get_json(
        f"https://api.github.com/repos/{REPOSITORY}/git/trees/{commit}?recursive=1",
        headers,
    )
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree response was truncated")
    paths = sorted(
        entry["path"]
        for entry in tree["tree"]
        if entry["type"] == "blob"
        and entry["path"].startswith("dataset/")
        and entry["path"].endswith(".xml")
    )
    random.Random(args.seed).shuffle(paths)

    excluded_courts = set(args.excluded_courts)
    court_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    inspected = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = max(args.workers * 4, 1)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for offset in range(0, len(paths), batch_size):
            batch = paths[offset : offset + batch_size]
            rows = list(
                executor.map(
                    lambda path: _fetch_candidate(path, commit, headers), batch
                )
            )
            for path, payload, metadata, error in rows:
                inspected += 1
                if error is not None:
                    errors.append({"source_path": path, "error": error})
                    continue
                assert payload is not None and metadata is not None
                if not is_holdout_candidate(
                    metadata, excluded_courts=excluded_courts
                ):
                    continue
                court = str(metadata["court"])
                if court_counts[court] >= args.max_per_court:
                    continue
                destination = args.output_dir / Path(path).name
                destination.write_bytes(payload)
                metadata.update(
                    {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                        "local_file": destination.as_posix(),
                    }
                )
                selected.append(metadata)
                court_counts[court] += 1
                if len(selected) >= args.target_documents:
                    break
            if len(selected) >= args.target_documents:
                break

    if len(selected) < args.target_documents:
        raise RuntimeError(
            f"Only {len(selected)} eligible documents found after inspecting {inspected}"
        )
    _assign_court_disjoint_splits(
        selected,
        development_documents=args.development_documents,
        seed=args.seed + 1,
    )
    split_counts = Counter(str(row["split"]) for row in selected)
    manifest = {
        "purpose": "Development and locked holdout for a from-scratch chunking comparison",
        "scope_note": "Normalized XML text study; PDF extraction is outside this corpus",
        "source_repository": f"https://github.com/{REPOSITORY}",
        "source_commit": commit,
        "source_license": f"https://github.com/{REPOSITORY}/blob/{commit}/LICENSE",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "method": "seeded path shuffle, frozen eligibility filters, court cap",
            "seed": args.seed,
            "target_documents": args.target_documents,
            "development_documents": args.development_documents,
            "holdout_documents": args.target_documents - args.development_documents,
            "split_group": "court",
            "max_documents_per_court": args.max_per_court,
            "excluded_prior_pilot_courts": sorted(excluded_courts),
            "classification": "pidana-khusus",
            "subclassification": "narkotika-dan-psikotropika",
            "inspected_documents": inspected,
        },
        "sample_size": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "court_count": len(court_counts),
        "year_distribution": dict(
            sorted(Counter(str(row["case_year"]) for row in selected).items())
        ),
        "court_distribution": dict(sorted(court_counts.items())),
        "fetch_error_count": len(errors),
        "fetch_errors": errors,
        "documents": selected,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Selected {len(selected)} documents from {len(court_counts)} courts after "
        f"inspecting {inspected}; splits={dict(split_counts)}; wrote {args.manifest}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.target_documents <= 0 or args.max_per_court <= 0:
        raise ValueError("target-documents and max-per-court must be positive")
    if not 0 < args.development_documents < args.target_documents:
        raise ValueError("development-documents must be between zero and target")
    if args.workers <= 0:
        raise ValueError("workers must be positive")


def _assign_court_disjoint_splits(
    rows: list[dict[str, Any]], *, development_documents: int, seed: int
) -> None:
    by_court: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_court[str(row["court"])].append(row)
    courts = sorted(by_court)
    random.Random(seed).shuffle(courts)
    development_courts: set[str] = set()
    count = 0
    for court in courts:
        size = len(by_court[court])
        if count + size <= development_documents:
            development_courts.add(court)
            count += size
        if count == development_documents:
            break
    if count != development_documents:
        raise RuntimeError(
            "Could not create an exact court-disjoint development split; "
            "change development-documents or max-per-court"
        )
    for row in rows:
        row["split"] = (
            "development" if row["court"] in development_courts else "holdout"
        )


def _fetch_candidate(
    path: str, commit: str, headers: dict[str, str]
) -> tuple[str, bytes | None, dict[str, Any] | None, str | None]:
    raw_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{path}"
    try:
        payload = _get_bytes(raw_url, headers)
        metadata = parse_indolaw_xml(payload.decode("utf-8"), source_path=path)
        return path, payload, metadata, None
    except Exception as error:
        return path, None, None, str(error)


def _get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    return json.loads(_get_bytes(url, headers).decode("utf-8"))


def _get_bytes(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers)
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


if __name__ == "__main__":
    main()
