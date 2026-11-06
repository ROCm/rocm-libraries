# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What --resume reuses from the results of a previous sweep.

A finished case of the previous output is reused when the problem record of a
current case equals its own and it has no error. Everything is checked before
anything is built.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from sweep_skinny_gemm_config import SweepConfigError
from sweep_skinny_gemm_results import read_failures, read_results


@dataclass
class Resumed:
    """The reusable part of a previous output, keyed by the new case index."""

    meta: dict
    records: dict[int, dict] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    dropped: int = 0
    rerun: int = 0


def problem_key(problem: dict) -> tuple:
    return tuple(sorted(problem.items()))


def settings_differences(previous: dict, current: dict) -> list[str]:
    """The keys whose values differ, with both values."""
    missing = "<missing>"
    differences = []
    for key in sorted(set(previous) | set(current)):
        old, new = previous.get(key, missing), current.get(key, missing)
        if old != new:
            differences.append(f"{key} (previous {old}, current {new})")
    return differences


def check_compatible(meta: dict, one_shot: bool, settings: dict) -> None:
    """Raises unless the previous run was timed like this one."""
    if bool(meta.get("oneShot")) != one_shot:
        raise SweepConfigError(
            f"--resume: the previous results have oneShot {bool(meta.get('oneShot'))}"
            f", this run {one_shot}"
        )
    differences = settings_differences(meta.get("settings") or {}, settings)
    if differences:
        raise SweepConfigError(
            "--resume: the settings differ from the previous results: "
            + ", ".join(differences)
        )


def load_previous(path: str) -> dict:
    if not os.path.isfile(path):
        raise SweepConfigError(f"--resume: no results at {path}")
    document = read_results(path)
    valid = isinstance(document, dict) and isinstance(document.get("results"), list)
    if not valid or not isinstance(document.get("meta"), dict):
        raise SweepConfigError(f"--resume: {path} is not a sweep results file")
    return document


def load_resume(
    path: str, one_shot: bool, settings: dict, problems: list[dict]
) -> Resumed:
    """The reusable records of the results at path; problems are the problem
    records of the current cases, in order. settings is what the sweep writes
    in meta.settings."""
    document = load_previous(path)
    check_compatible(document["meta"], one_shot, settings)
    finished = {}
    for record in document["results"]:
        if "error" not in record:
            finished.setdefault(problem_key(record["problem"]), record)
    resumed = Resumed(document["meta"])
    for index, problem in enumerate(problems):
        record = finished.get(problem_key(problem))
        if record is None:
            resumed.rerun += 1
            continue
        resumed.records[index] = record
    kept = {problem_key(record["problem"]) for record in resumed.records.values()}
    current = {problem_key(problem) for problem in problems}
    keys = (problem_key(record["problem"]) for record in document["results"])
    resumed.dropped = sum(1 for key in keys if key not in current)
    resumed.failures = [
        group for group in read_failures(path) if problem_key(group["problem"]) in kept
    ]
    return resumed
