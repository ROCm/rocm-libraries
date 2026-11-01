# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Comparator for flavor-keyed LLVM-IR golden fixtures.

A golden stores one sub-document per distinct emission profile under
``flavors``. ``aliases`` maps a compiler lane to another lane with the same
profile, avoiding duplicate fixtures while still lowering and checking the
aliased lane. Each sub-document holds a ``cases`` map and optionally an
``expected_failures`` map. Lowering takes the lane as an argument, so any host
can verify every qualified lane, not just the one it autodetects.

A golden is either one JSON file or a directory of shard files, one per family:
``<dir>/<family>.json`` holds the cases whose ``case_id`` starts with
``<family>/``. Each shard is a complete document (``schema``, ``aliases`` and
``flavors``), and :func:`load_golden` merges them. Sharding keeps unrelated
changes in separate files, so they merge without conflicts.

:data:`GOLDEN_FLAVORS` is :data:`~rocke.core.lower_llvm.LLVM_FLAVORS` itself.
Adding a lane there makes every golden checked through :func:`check_golden`
fail until it receives a profile document or an explicit shared-profile alias,
with no edit to the golden test. This module is
shipped so ``library/`` goldens can share it (the dependency stays
``library -> platform``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .llvm_profile import LLVM_EMISSION_ALIASES, LLVM_EMISSION_FLAVORS, LLVM_FLAVORS

#: Lanes every golden must represent directly or through an alias. Same object as
#: :data:`LLVM_FLAVORS`, never a copy.
GOLDEN_FLAVORS = LLVM_FLAVORS
GOLDEN_PROFILE_FLAVORS = LLVM_EMISSION_FLAVORS
GOLDEN_ALIASES = LLVM_EMISSION_ALIASES

__all__ = [
    "GOLDEN_ALIASES",
    "GOLDEN_FLAVORS",
    "GOLDEN_PROFILE_FLAVORS",
    "check_golden",
    "compare",
    "golden_shard",
    "golden_subdocument",
    "load_golden",
]

_SECTIONS = ("cases", "expected_failures")


def golden_shard(case_id: str) -> str:
    """Return the shard (family) a case is filed under: its first path segment."""
    return case_id.split("/", 1)[0]


def load_golden(path: Path) -> dict[str, Any]:
    """Read a golden file, or merge a directory of per-family shards.

    Raises ``ValueError`` if a shard is not valid JSON, if the shards disagree
    on ``schema`` or ``aliases``, or if a shard holds a case that belongs to
    another shard.
    """
    path = Path(path)
    if not path.is_dir():
        return json.loads(path.read_text())
    shards = sorted(path.glob("*.json"))
    if not shards:
        raise ValueError(f"golden directory {path} has no *.json shards")
    merged: dict[str, Any] = {}
    flavors: dict[str, dict[str, dict[str, Any]]] = {}
    for shard in shards:
        try:
            doc = json.loads(shard.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"golden shard {shard.name}: {e}") from e
        for key in ("schema", "aliases"):
            if shard is shards[0]:
                if key in doc:
                    merged[key] = doc[key]
            elif doc.get(key) != merged.get(key):
                raise ValueError(
                    f"golden shard {shard.name} has {key} {doc.get(key)!r}, "
                    f"{shards[0].name} has {merged.get(key)!r}; re-bless every shard"
                )
        for fl, sub in doc.get("flavors", {}).items():
            out = flavors.setdefault(fl, {s: {} for s in _SECTIONS})
            for section in _SECTIONS:
                for cid, rec in sub.get(section, {}).items():
                    if golden_shard(cid) != shard.stem:
                        raise ValueError(
                            f"golden shard {shard.name} holds {cid!r}, which "
                            f"belongs in {golden_shard(cid)}.json"
                        )
                    out[section][cid] = rec
    merged["flavors"] = flavors
    return merged


def golden_subdocument(doc: Mapping[str, Any], flavor: str) -> Mapping[str, Any] | None:
    """Return the baseline for ``flavor``, following one profile alias."""
    source = doc.get("aliases", {}).get(flavor, flavor)
    return doc.get("flavors", {}).get(source)


def compare(base: Mapping[str, Any], cur: Mapping[str, Any]) -> list[str]:
    """Diff one flavor sub-document against a fresh run. Empty list == OK."""
    errors = []
    for section in ("cases", "expected_failures"):
        bkeys = set(base.get(section, {}))
        ckeys = set(cur.get(section, {}))
        for missing in sorted(bkeys - ckeys):
            errors.append(f"{section}: missing current {missing}")
        for new in sorted(ckeys - bkeys):
            errors.append(f"{section}: new current {new}")
    for cid, brec in sorted(base.get("cases", {}).items()):
        crec = cur.get("cases", {}).get(cid)
        if not crec:
            continue
        if brec.get("sha256") != crec.get("sha256"):
            errors.append(f"{cid}: {brec.get('sha256')} -> {crec.get('sha256')}")
    for cid, brec in sorted(base.get("expected_failures", {}).items()):
        crec = cur.get("expected_failures", {}).get(cid)
        if not crec:
            continue
        if brec.get("type") != crec.get("type") or brec.get("message") != crec.get(
            "message"
        ):
            errors.append(f"{cid}: failure changed {brec} -> {crec}")
    return errors


def check_golden(
    golden_path: Path,
    run: Callable[[str], Mapping[str, Any]],
    flavor: str | None = None,
) -> list[str]:
    """Compare fresh runs against a golden's flavor sub-documents.

    ``run(flavor)`` lowers every case at ``flavor`` and returns a sub-document
    in the golden's shape. With no ``flavor``, every entry of
    :data:`GOLDEN_FLAVORS` is checked, and a sub-document for a flavor no
    longer in that tuple is reported as stale. Drift strings are prefixed
    with the flavor when more than one is checked. ``golden_path`` may be a
    file or a shard directory (see :func:`load_golden`). Empty list == OK.
    """
    try:
        doc = load_golden(golden_path)
    except ValueError as e:
        return [str(e)]
    have = doc.get("flavors", {})
    aliases = doc.get("aliases", {})
    wanted = [flavor] if flavor else list(GOLDEN_FLAVORS)
    errors: list[str] = []
    if not flavor:
        if aliases != GOLDEN_ALIASES:
            errors.append(
                f"golden aliases {aliases} do not match emission-profile aliases "
                f"{GOLDEN_ALIASES}; re-bless it"
            )
        if set(have) != set(GOLDEN_PROFILE_FLAVORS):
            errors.append(
                f"golden profile documents {sorted(have)} do not match "
                f"{list(GOLDEN_PROFILE_FLAVORS)}; re-bless it"
            )
        overlap = set(have) & set(aliases)
        if overlap:
            errors.append(
                f"golden lanes are both direct and aliased: {sorted(overlap)}; "
                "re-bless it"
            )
        if errors:
            return errors
    for fl in wanted:
        base = golden_subdocument(doc, fl)
        if base is None:
            errors.append(
                f"golden has no entry or valid alias for flavor {fl!r} "
                f"(have {sorted(have)}); re-bless it"
            )
            continue
        prefix = "" if len(wanted) == 1 else f"[{fl}] "
        errors.extend(prefix + e for e in compare(base, run(fl)))
    if not flavor:
        represented = set(have) | set(aliases)
        for fl in sorted(represented - set(GOLDEN_FLAVORS)):
            errors.append(
                f"golden has stale flavor {fl!r} not in LLVM_FLAVORS; re-bless it"
            )
        for fl, source in sorted(aliases.items()):
            if source not in have:
                errors.append(
                    f"golden alias {fl!r} names missing flavor {source!r}; re-bless it"
                )
    return errors
