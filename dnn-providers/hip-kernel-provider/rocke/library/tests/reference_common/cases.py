# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Case identity shared by enrolled cohorts and qualified bundle manifests.

Qualification names each manifest entry and payload directory by case id.
Verification matches entries by the complete recorded case contract instead,
so an authenticated bundle keeps replaying if an operation's id format changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Any


def index_cases(cases: Iterable[Any], *, label: str) -> dict[str, Any]:
    """Index a cohort by id; colliding cases would share one manifest entry."""
    index = {}
    for case in cases:
        if case.id in index:
            raise ValueError(f"duplicate {label} case id: {case.id}")
        index[case.id] = case
    return index


def match_manifest_cases(
    entries: Mapping[str, Mapping], cases: Iterable[Any], *, label: str
) -> dict[str, str]:
    """Map each enrolled case id to the one manifest key recording its contract.

    The match is one-to-one: every enrolled case is recorded exactly once and
    the manifest records no other case. Keys are not compared, so an entry
    written under an earlier id format still matches its unchanged contract.
    """
    unmatched = dict(entries)
    keys, missing = {}, []
    for case_id, case in index_cases(cases, label=label).items():
        contract = asdict(case)
        found = [key for key, entry in entries.items() if entry.get("case") == contract]
        if len(found) > 1:
            raise ValueError(
                f"{label} bundle records case {case_id} more than once: "
                + ", ".join(sorted(found))
            )
        if found:
            keys[case_id] = found[0]
            del unmatched[found[0]]
        else:
            missing.append(_describe(case_id, contract))
    if missing or unmatched:
        extra = [_describe(key, entry.get("case")) for key, entry in unmatched.items()]
        raise ValueError(
            f"{label} bundle does not cover the complete enrolled cohort; "
            f"missing: {'; '.join(missing) or 'none'}; "
            f"extra: {'; '.join(extra) or 'none'}"
        )
    return keys


def _describe(key: str, contract: object) -> str:
    if not isinstance(contract, Mapping):
        return f"{key} (no recorded case contract)"
    fields = ", ".join(f"{name}={value}" for name, value in sorted(contract.items()))
    return f"{key} ({fields})"
