# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Case identity shared by enrolled cohorts and qualified bundle manifests.

Qualification names each manifest entry and payload directory by case id.
Verification matches entries by the complete recorded case contract instead,
so an authenticated bundle keeps replaying if an operation's id format changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any


def index_cases(cases: Iterable[Any], *, label: str) -> dict[str, Any]:
    """Index a cohort by id; colliding cases would share one manifest entry."""
    index = {}
    for case in cases:
        if case.id in index:
            raise ValueError(f"duplicate {label} case id: {case.id}")
        index[case.id] = case
    return index


@dataclass(frozen=True)
class CohortComparison:
    """How a manifest's recorded case contracts line up with an enrolled cohort.

    ``keys`` maps each enrolled case recorded exactly once to its manifest key.
    ``missing`` holds the contract of each enrolled case the manifest does not
    record, ``extra`` the recorded contract of each entry no enrolled case
    matches, and ``repeated`` the keys of each case recorded more than once.
    """

    keys: dict[str, str]
    missing: dict[str, dict]
    extra: dict[str, object]
    repeated: dict[str, list[str]]


def compare_manifest_cases(
    entries: Mapping[str, Mapping], cases: Iterable[Any], *, label: str
) -> CohortComparison:
    """Pair enrolled cases with manifest entries by complete case contract.

    Keys are not compared, so an entry written under an earlier id format still
    matches its unchanged contract.
    """
    unmatched = dict(entries)
    keys, missing, repeated = {}, {}, {}
    for case_id, case in index_cases(cases, label=label).items():
        contract = asdict(case)
        found = sorted(
            key for key, entry in entries.items() if entry.get("case") == contract
        )
        for key in found:
            unmatched.pop(key, None)
        if len(found) > 1:
            repeated[case_id] = found
        elif found:
            keys[case_id] = found[0]
        else:
            missing[case_id] = contract
    extra = {key: entry.get("case") for key, entry in unmatched.items()}
    return CohortComparison(keys, missing, extra, repeated)


def match_manifest_cases(
    entries: Mapping[str, Mapping], cases: Iterable[Any], *, label: str
) -> dict[str, str]:
    """Map each enrolled case id to the one manifest key recording its contract.

    The match is one-to-one: every enrolled case is recorded exactly once and
    the manifest records no other case. Keys are not compared, so an entry
    written under an earlier id format still matches its unchanged contract.
    """
    comparison = compare_manifest_cases(entries, cases, label=label)
    if comparison.repeated:
        case_id, found = next(iter(comparison.repeated.items()))
        raise ValueError(
            f"{label} bundle records case {case_id} more than once: " + ", ".join(found)
        )
    if comparison.missing or comparison.extra:
        missing = [describe_case(*item) for item in comparison.missing.items()]
        extra = [describe_case(*item) for item in comparison.extra.items()]
        raise ValueError(
            f"{label} bundle does not cover the complete enrolled cohort; "
            f"missing: {'; '.join(missing) or 'none'}; "
            f"extra: {'; '.join(extra) or 'none'}"
        )
    return comparison.keys


def describe_case(key: str, contract: object) -> str:
    """Name a case or manifest entry with its recorded contract fields."""
    if not isinstance(contract, Mapping):
        return f"{key} (no recorded case contract)"
    fields = ", ".join(f"{name}={value}" for name, value in sorted(contract.items()))
    return f"{key} ({fields})"
