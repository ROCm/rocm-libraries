#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate historical COMGR evidence and its published study/churn claims."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

if __package__:
    from . import probe
else:
    import sys

    _PLATFORM = Path(__file__).resolve().parents[2]
    _PYTHON = _PLATFORM / "python"
    if str(_PYTHON) not in sys.path:
        sys.path.insert(0, str(_PYTHON))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compiler_capabilities import probe

_ROOT = Path(__file__).resolve().parent
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCAL_PATH_RE = re.compile(r"(?:/home/|/Users/|/tmp/|/root/|/opt/|[A-Za-z]:\\)")
_HISTORICAL_PROCESSORS = ("gfx942", "gfx950", "gfx1151", "gfx1201", "gfx1250")


class StudyError(RuntimeError):
    """Historical evidence or a derived public claim is inconsistent."""


def _load(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(document, dict):
        raise StudyError(f"{path.name} must contain a JSON object")
    return document


def _structural_vector(
    document: dict[str, object], row: dict[str, object]
) -> dict[str, tuple[object, ...]]:
    vector = {}
    for processor, reference in row["reports"].items():
        vector[f"{processor}|effective_processor"] = (reference["effective_processor"],)
        capabilities = document["vectors"][reference["vector"]]
        for capability, observations in capabilities.items():
            for entry in observations:
                key = "|".join(
                    (
                        processor,
                        capability,
                        entry["family"],
                        entry["candidate"],
                    )
                )
                vector[key] = (entry["outcome"], entry.get("failed_stage"))
    return vector


def _boundaries(
    document: dict[str, object],
    rows: list[dict[str, object]],
) -> list[tuple[str, str, int, tuple[str, ...]]]:
    result = []
    for previous, current in itertools.pairwise(rows):
        before = _structural_vector(document, previous)
        after = _structural_vector(document, current)
        changed = [key for key, value in before.items() if after.get(key) != value]
        if not changed:
            continue
        capabilities = tuple(
            sorted(
                {
                    key.split("|", 3)[1]
                    for key in changed
                    if "|effective_processor" not in key
                }
            )
        )
        result.append((previous["id"], current["id"], len(changed), capabilities))
    return result


def _validate_inputs(
    inputs: dict[str, object], observations: dict[str, object]
) -> None:
    if inputs.get("format") != "rocke-comgr-historical-inputs/v1":
        raise StudyError("historical input format is stale")
    if observations.get("format") != "rocke-comgr-historical-observations/v3":
        raise StudyError("historical observation format is stale")
    public_sources = inputs.get("public_sources")
    if (
        not isinstance(public_sources, list)
        or len(public_sources) != 3
        or any(
            not isinstance(url, str) or not url.startswith("https://")
            for url in public_sources
        )
    ):
        raise StudyError(
            "historical input manifest must name three public HTTPS sources"
        )
    if observations.get("public_sources") != public_sources:
        raise StudyError("historical public sources differ from the input manifest")
    for group_name, expected_count in (("weekly", 9), ("releases", 3)):
        input_rows = inputs.get(group_name)
        rows = observations.get(group_name)
        if not isinstance(input_rows, list) or not isinstance(rows, list):
            raise StudyError(f"historical {group_name} rows must be lists")
        if len(input_rows) != expected_count or len(rows) != expected_count:
            raise StudyError(
                f"historical study must contain {expected_count} {group_name} rows"
            )
        input_identity = [
            (
                row["id"],
                row["kind"],
                row["artifact"],
                row["index_url"],
                row["artifact_url"],
            )
            for row in input_rows
        ]
        observed_identity = [
            (
                row["id"],
                row["kind"],
                row["artifact"],
                row["index_url"],
                row["artifact_url"],
            )
            for row in rows
        ]
        if observed_identity != input_identity:
            raise StudyError(
                f"historical {group_name} observations differ from selected inputs"
            )
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise StudyError(f"historical {group_name} identifiers must be unique")
        for row in rows:
            if not row["artifact_url"].startswith("https://"):
                raise StudyError(f"{row['id']} artifact URL is not public HTTPS")
            if not row["artifact_url"].startswith(row["index_url"]):
                raise StudyError(f"{row['id']} artifact is outside its public index")
            if row["artifact"] not in row["artifact_url"]:
                raise StudyError(f"{row['id']} artifact URL does not name its package")


def _validate_reports(observations: dict[str, object]) -> None:
    processors = observations.get("processors")
    if processors != list(_HISTORICAL_PROCESSORS):
        raise StudyError("historical processor set differs from the study contract")
    probe_info = observations.get("probe")
    vectors = observations.get("vectors")
    if not isinstance(probe_info, dict) or not isinstance(vectors, dict):
        raise StudyError("historical probe or vector store is malformed")
    expected_probe = {
        "implementation_sha256": probe._implementation_sha256(),
        "schema_sha256": probe._schema_sha256(),
    }
    for field, expected in expected_probe.items():
        value = str(probe_info.get(field, ""))
        if _SHA256_RE.fullmatch(value) is None:
            raise StudyError(f"historical study has no {field}")
        if value != expected:
            raise StudyError(
                f"historical {field} differs from the checked-in probe source"
            )
    for vector_id, capabilities in vectors.items():
        encoded = json.dumps(
            capabilities, sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != vector_id:
            raise StudyError(f"historical vector {vector_id} digest differs")
    if _LOCAL_PATH_RE.search(json.dumps(observations, sort_keys=True)):
        raise StudyError("historical observations contain a machine-local path")

    reports_by_url: dict[str, object] = {}
    for row in (*observations["weekly"], *observations["releases"]):
        reports = row.get("reports")
        compiler = row.get("compiler")
        if not isinstance(reports, dict) or sorted(reports) != sorted(processors):
            raise StudyError(f"{row['id']} does not contain every processor report")
        if not isinstance(compiler, dict):
            raise StudyError(f"{row['id']} compiler provenance is malformed")
        digest = compiler.get("sha256")
        if _SHA256_RE.fullmatch(str(digest or "")) is None:
            raise StudyError(f"{row['id']} has no COMGR digest")
        if not isinstance(compiler.get("version_source"), str) or not compiler.get(
            "version_source"
        ):
            raise StudyError(f"{row['id']} has no compiler version source")

        for processor in processors:
            reference = reports[processor]
            if not isinstance(reference, dict):
                raise StudyError(
                    f"{row['id']} {processor} report reference is malformed"
                )
            vector_id = reference.get("vector")
            capabilities = vectors.get(vector_id)
            if not isinstance(capabilities, dict):
                raise StudyError(
                    f"{row['id']} {processor} references no capability vector"
                )
            expected = probe._expected_observations(processor)
            if set(capabilities) != set(expected):
                raise StudyError(f"{row['id']} {processor} capability set differs")
            for name, corpus in expected.items():
                entries = capabilities[name]
                observed = [
                    (entry.get("family"), entry.get("candidate")) for entry in entries
                ]
                if observed != corpus:
                    raise StudyError(f"{row['id']} {processor} {name} corpus differs")
                for entry in entries:
                    if entry.get("outcome") not in {"accepted", "rejected", "crash"}:
                        raise StudyError(
                            f"{row['id']} {processor} {name} has incomplete outcome"
                        )
            effective = reference.get("effective_processor")
            allowed = {processor}
            if generic := probe._GENERIC_PROCESSORS.get(processor):
                allowed.add(generic)
            if effective not in allowed:
                raise StudyError(
                    f"{row['id']} {processor} effective processor is invalid"
                )
        evidence = (compiler, reports)
        prior = reports_by_url.setdefault(row["artifact_url"], evidence)
        if prior != evidence:
            raise StudyError(f"{row['id']} duplicate artifact evidence differs")


def _published_boundaries(path: Path) -> list[tuple[str, str, int]]:
    result = []
    pattern = re.compile(r"^\| (W[\w-]+) → (W[\w-]+) \| (\d+) \|")
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := pattern.match(line):
            result.append((match.group(1), match.group(2), int(match.group(3))))
    return result


def check(
    inputs_path: Path,
    observations_path: Path,
    study_path: Path,
    churn_path: Path,
) -> None:
    """Raise ``StudyError`` when evidence or derived claims disagree."""
    inputs = _load(inputs_path)
    observations = _load(observations_path)
    _validate_inputs(inputs, observations)
    _validate_reports(observations)
    rows = observations["weekly"]
    structural = _boundaries(observations, rows)
    published = _published_boundaries(study_path)
    expected_published = [(left, right, count) for left, right, count, _ in structural]
    if published != expected_published:
        raise StudyError(
            f"published structural boundaries differ: {published} != {expected_published}"
        )
    study = study_path.read_text(encoding="utf-8")
    churn = churn_path.read_text(encoding="utf-8")
    for document, claim in (
        (study, "nine selected weekly snapshots"),
        (churn, "nine selected weekly snapshots"),
    ):
        if claim not in re.sub(r"\s+", " ", document):
            raise StudyError(f"published scope claim is stale: {claim!r}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=_ROOT / "historical_inputs.json")
    parser.add_argument(
        "--observations",
        type=Path,
        default=_ROOT / "historical_observations.json",
    )
    parser.add_argument("--study", type=Path, default=_ROOT / "historical_study.md")
    parser.add_argument("--churn", type=Path, default=_ROOT / "churn_report.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    check(args.inputs, args.observations, args.study, args.churn)
    print("RESULT: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
