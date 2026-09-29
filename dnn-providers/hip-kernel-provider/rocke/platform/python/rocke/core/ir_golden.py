# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Comparator for flavor-keyed LLVM-IR golden fixtures.

A golden stores one sub-document per LLVM flavor under ``flavors``, each
holding a ``cases`` map (``case_id -> {"sha256": ...}``) and optionally an
``expected_failures`` map (``case_id -> {"type", "message"}``). Lowering takes
the flavor as an argument, so any host can verify every sub-document, not just
the one its ROCm vintage autodetects.

:data:`GOLDEN_FLAVORS` is :data:`~rocke.core.lower_llvm.LLVM_FLAVORS` itself.
Adding a flavor there makes every golden checked through :func:`check_golden`
fail until it is re-blessed, with no edit to the golden test. This module is
shipped so ``library/`` goldens can share it (the dependency stays
``library -> platform``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional

from .lower_llvm import LLVM_FLAVORS

#: Flavors every golden must record. Deliberately the same object as
#: :data:`LLVM_FLAVORS`, never a copy.
GOLDEN_FLAVORS = LLVM_FLAVORS

__all__ = ["GOLDEN_FLAVORS", "check_golden", "compare"]


def compare(base: Mapping[str, Any], cur: Mapping[str, Any]) -> List[str]:
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
    flavor: Optional[str] = None,
) -> List[str]:
    """Compare fresh runs against a golden's flavor sub-documents.

    ``run(flavor)`` lowers every case at ``flavor`` and returns a sub-document
    in the golden's shape. With no ``flavor``, every entry of
    :data:`GOLDEN_FLAVORS` is checked, and a sub-document for a flavor no
    longer in that tuple is reported as stale. Drift strings are prefixed
    with the flavor when more than one is checked. Empty list == OK.
    """
    doc = json.loads(Path(golden_path).read_text())
    have = doc.get("flavors", {})
    wanted = [flavor] if flavor else list(GOLDEN_FLAVORS)
    errors: List[str] = []
    for fl in wanted:
        base = have.get(fl)
        if base is None:
            errors.append(
                f"golden has no entry for flavor {fl!r} (have {sorted(have)}); "
                "re-bless it"
            )
            continue
        prefix = "" if len(wanted) == 1 else f"[{fl}] "
        errors.extend(prefix + e for e in compare(base, run(fl)))
    if not flavor:
        for fl in sorted(set(have) - set(GOLDEN_FLAVORS)):
            errors.append(
                f"golden has stale flavor {fl!r} not in LLVM_FLAVORS; re-bless it"
            )
    return errors
