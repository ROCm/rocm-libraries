#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check the published historical matrix and churn report against study data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATA = _ROOT / "historical_observations.json"
_DEFAULT_STUDY = _ROOT / "historical_study.md"
_DEFAULT_CHURN = _ROOT / "churn_report.md"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StudyError(RuntimeError):
    """Published study text disagrees with its machine-readable observations."""


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _published_matrix(path: Path) -> dict[str, dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    matrix: dict[str, dict[str, str]] = {}
    for index, line in enumerate(lines):
        if not line.startswith("| Feature probe |"):
            continue
        columns = _cells(line)[1:]
        row_index = index + 2
        while row_index < len(lines) and lines[row_index].startswith("|"):
            cells = _cells(lines[row_index])
            if len(cells) != len(columns) + 1:
                raise StudyError(f"malformed matrix row: {lines[row_index]}")
            label = cells[0]
            for column, value in zip(columns, cells[1:], strict=True):
                matrix.setdefault(column, {})[label] = value
            row_index += 1
    return matrix


def _published_boundaries(
    path: Path,
) -> list[tuple[str, str, tuple[str, ...]]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| W"):
            continue
        cells = _cells(line)
        match = re.fullmatch(r"(W[\w-]+)\s*→\s*(W[\w-]+)", cells[0])
        if match is not None:
            result.append(
                (
                    match.group(1),
                    match.group(2),
                    tuple(part.strip() for part in cells[1].split(";")),
                )
            )
    return result


def _observed_boundaries(
    rows: list[dict[str, object]],
) -> list[tuple[str, str, tuple[str, ...]]]:
    result = []
    previous = rows[0]
    for current in rows[1:]:
        previous_features = dict(previous["features"])
        current_features = dict(current["features"])
        previous_features.pop("Reported LLVM major", None)
        current_features.pop("Reported LLVM major", None)
        changed = tuple(
            name
            for name, value in previous_features.items()
            if current_features[name] != value
        )
        if changed:
            result.append((str(previous["id"]), str(current["id"]), changed))
        previous = current
    return result


def _repository_from_version(version_text: str) -> str | None:
    for url in re.findall(r"https?://[^\s)]+", version_text):
        normalized = url.removesuffix(".git")
        if normalized.rstrip("/").endswith("/llvm-project"):
            return normalized
    return None


def _source_revisions(version_text: str) -> tuple[str | None, str | None]:
    revisions = [
        value.lower() for value in re.findall(r"\b[0-9a-fA-F]{40}\b", version_text)
    ]
    patch = re.search(r"\+PATCHED:([0-9a-fA-F]{40})\b", version_text)
    patched = patch.group(1).lower() if patch else None
    primary = next((value for value in revisions if value != patched), None)
    return primary, patched


def check(data_path: Path, study_path: Path, churn_path: Path) -> None:
    """Raise ``StudyError`` when data, matrix, or churn claims disagree."""
    data = json.loads(data_path.read_text(encoding="utf-8"))
    weekly = data["weekly"]
    releases = data["releases"]
    if len(weekly) != 43 or len(releases) != 6:
        raise StudyError("study must contain 43 weekly columns and 6 releases")
    expected_sources = {
        "https://github.com/ROCm/TheRock/blob/main/docs/packaging/legacy_per_family_releases.md",
        "https://github.com/ROCm/TheRock/blob/main/docs/packaging/legacy_multi_arch_releases.md",
        "https://github.com/ROCm/TheRock/blob/main/RELEASES.md",
    }
    if set(data.get("public_sources", ())) != expected_sources:
        raise StudyError("public packaging sources are incomplete")

    ids = [str(row["id"]) for row in weekly]
    if len(ids) != len(set(ids)):
        raise StudyError("weekly identifiers must be unique")
    for row in (*weekly, *releases):
        compiler = row["compiler"]
        if not _SHA256_RE.fullmatch(compiler["executable_sha256"]):
            raise StudyError(f"{row['id']} has no compiler executable digest")
        if not _SHA256_RE.fullmatch(row["opt"]["sha256"]):
            raise StudyError(f"{row['id']} has no opt executable digest")
        serialized = json.dumps(row)
        if re.search(r"(?:/home/|/Users/|[A-Za-z]:\\\\)", serialized):
            raise StudyError(f"{row['id']} contains a machine-local path")
        version_text = compiler["version_text"]
        source_revision, patched_revision = _source_revisions(version_text)
        if compiler["source_revision"] != source_revision:
            raise StudyError(f"{row['id']} source revision disagrees with banner")
        if compiler["patched_revision"] != patched_revision:
            raise StudyError(f"{row['id']} patch revision disagrees with banner")
        if compiler["repository"] != _repository_from_version(version_text):
            raise StudyError(f"{row['id']} repository disagrees with banner")

    for row in weekly:
        if not row["index_url"].startswith("https://") or not row[
            "artifact_url"
        ].startswith("https://"):
            raise StudyError(f"{row['id']} artifact location is not public HTTPS")
        if row["artifact"] not in row["artifact_url"]:
            raise StudyError(f"{row['id']} artifact URL does not name its package")
        if not row["artifact_url"].startswith(row["index_url"]):
            raise StudyError(f"{row['id']} artifact URL is outside its public index")
    if any(
        not row.get("public_locator", "").startswith("https://") for row in releases
    ):
        raise StudyError("release baseline lacks a public locator")

    feature_names = tuple(weekly[0]["features"])
    if any(tuple(row["features"]) != feature_names for row in (*weekly, *releases)):
        raise StudyError("study rows do not share one feature vocabulary")

    published = _published_matrix(study_path)
    for row in (*weekly, *releases):
        identifier = str(row["id"])
        if published.get(identifier) != row["features"]:
            raise StudyError(f"published matrix differs for {identifier}")

    if not churn_path.exists():
        return

    observed_boundaries = _observed_boundaries(weekly)
    published_boundaries = _published_boundaries(churn_path)
    if observed_boundaries != published_boundaries:
        raise StudyError(
            "published churn boundaries differ: "
            f"observed={observed_boundaries}, published={published_boundaries}"
        )
    churn = churn_path.read_text(encoding="utf-8")
    if "43 weekly columns" not in churn or "42 adjacent boundaries" not in churn:
        raise StudyError("churn report column/boundary counts are stale")
    count_words = {7: "seven"}
    count = len(observed_boundaries)
    expected_count = count_words.get(count, str(count))
    if f"changes at {expected_count} boundaries" not in churn:
        raise StudyError("churn report changed-boundary count is stale")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--study", type=Path, default=_DEFAULT_STUDY)
    parser.add_argument("--churn", type=Path, default=_DEFAULT_CHURN)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        check(args.data, args.study, args.churn)
    except (OSError, json.JSONDecodeError, StudyError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print("RESULT: GREEN - historical study and churn report match observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
