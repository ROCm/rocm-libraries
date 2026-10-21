#!/usr/bin/env python3
"""Pre-commit verifier for integration-test support-claim sidecars.

RFC 0015 §9.3 — validates that every .support.json in the bundle tree is
well-formed and consistent with the bundle it describes.  This is the first
slice of the RFC 0011 bundle verifier.

Checks
------
1. Schema: version == 1, claims is an object, arch -> array of tokens from
   {linux, windows}; sweep groups have non-empty cases + support.
   The validators live in support_claim_schema.py, shared with the
   support matrix collector.
2. enforcement_level: a graph with a non-empty claim must not declare an
   unrecognised enforcement_level (single-graph: X.meta.json; sweep:
   cases[].metadata.enforcement_level in sweep.json).  Absent is legal and
   means "full" -- see check_enforcement_level_single for why.
3. Sweep case ids: every claimed id must exist in the sibling sweep.json;
   no duplicate case id per engine.  The sweep.json itself must pass the
   harness's readSweepCaseIds() rules, or the harness registers none of it.
4. Orphaned sidecars: X.support.json without a sibling X.json; support.json
   in a directory that is not a sweep root.

Exit codes
----------
0  All checks passed (or no sidecars found).
1  One or more violations found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_discovery import (  # noqa: E402
    DEFAULT_BUNDLES_DIR,
    SWEEP_MANIFEST_NAME,
    SWEEP_SUPPORT_NAME,
    SWEEP_TEMPLATE_NAME,
    graph_for_sidecar,
    is_single_graph_sidecar,
    is_sweep_root,
    load_json,
    meta_for_graph,
    read_sweep_cases,
    walk_bundle_tree,
)
from support_claim_schema import (  # noqa: E402
    validate_single_graph_schema,
    validate_sweep_schema,
)

BUNDLE_ROOT = DEFAULT_BUNDLES_DIR
VALID_ENFORCEMENT_LEVELS = {"applicability", "buildable", "full"}


def _error(path: Path, message: str, errors: List[str]) -> None:
    errors.append(f"{path}: {message}")


def _has_non_empty_claims(data: object) -> bool:
    # A non-object sidecar has already been reported by the schema check.
    if not isinstance(data, dict):
        return False
    claims = data.get("claims", {})
    return isinstance(claims, dict) and len(claims) > 0


def check_enforcement_level_single(support_path: Path, errors: List[str]) -> None:
    """Reject a declared-but-unrecognised enforcement_level on a claimed bundle.

    An *absent* level is not an error.  BundleMetadata.hpp defines absence as
    EnforcementLevel::FULL -- the strictest rung -- so there is no ambiguity
    about how a claim on such a bundle is enforced, and requiring the field
    would mean writing "enforcement_level": "full" into every meta.json (and
    inventing a meta.json for every bundle that has none) purely to restate the
    default.  A *misspelt* level is a different matter: BundleMetadata.hpp
    rejects the whole metadata object for it, so the bundle would silently lose
    every other field it declares.  That is what this catches.
    """
    meta_path = meta_for_graph(graph_for_sidecar(support_path))
    if not meta_path.exists():
        return
    meta, error = load_json(meta_path)
    if error is not None:
        _error(meta_path, error, errors)
        return
    if not isinstance(meta, dict):
        # parseBundleMetadataJson() throws on this, losing every field.
        _error(meta_path, "is not a metadata JSON object", errors)
        return
    level = meta.get("enforcement_level")
    if level is not None and level not in VALID_ENFORCEMENT_LEVELS:
        _error(
            meta_path,
            f"invalid enforcement_level '{level}'"
            f" (expected one of {sorted(VALID_ENFORCEMENT_LEVELS)})",
            errors,
        )


def check_enforcement_level_sweep(
    sweep_path: Path, cases: List[dict], errors: List[str]
) -> None:
    """``cases`` comes from read_sweep_cases(), so every id is a string."""
    for case_entry in cases:
        metadata = case_entry.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        level = metadata.get("enforcement_level")
        if level is not None and level not in VALID_ENFORCEMENT_LEVELS:
            _error(
                sweep_path,
                f"case '{case_entry['id']}': invalid enforcement_level '{level}'",
                errors,
            )


def check_sweep_case_ids(
    support_path: Path, support_data: dict, cases: List[dict], errors: List[str]
) -> None:
    valid_ids = {case_entry["id"] for case_entry in cases}
    claims = support_data.get("claims", {})
    if not isinstance(claims, dict):
        return
    for engine_name, groups in claims.items():
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            for case_id in group.get("cases", []):
                if isinstance(case_id, str) and case_id not in valid_ids:
                    _error(
                        support_path,
                        f"engine '{engine_name}': case id '{case_id}'"
                        " not found in sweep.json",
                        errors,
                    )


def check_orphaned_single_graph(support_path: Path, errors: List[str]) -> None:
    bundle_path = graph_for_sidecar(support_path)
    if not bundle_path.exists():
        _error(
            support_path,
            f"orphaned sidecar: no sibling {bundle_path.name}",
            errors,
        )


def check_orphaned_sweep(support_path: Path, errors: List[str]) -> None:
    # Share the harness's sweep-root rule: a directory missing either control
    # file is never discovered, so its claims would be silently ignored.
    if not is_sweep_root(support_path.parent):
        _error(
            support_path,
            "orphaned support.json: directory is not a sweep root"
            f" (needs both {SWEEP_TEMPLATE_NAME} and {SWEEP_MANIFEST_NAME})",
            errors,
        )


def is_sweep_sidecar(path: Path) -> bool:
    return path.name == SWEEP_SUPPORT_NAME


def check_canonical_form(support_path: Path, data: object, errors: List[str]) -> None:
    canonical = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    raw = support_path.read_bytes().decode("utf-8")
    if raw != canonical:
        _error(support_path, "not in canonical form (re-run the writer)", errors)


def _load_sidecar(support_path: Path, errors: List[str]) -> tuple[object, bool]:
    data, error = load_json(support_path)
    if error is not None:
        _error(support_path, error, errors)
        return None, False
    return data, True


def check_sweep_manifest(
    support_path: Path, support_data: dict, errors: List[str]
) -> None:
    """Read the sibling sweep.json once and run both checks that need it."""
    sweep_path = support_path.parent / SWEEP_MANIFEST_NAME
    if not sweep_path.exists():
        # check_orphaned_sweep already reports this; no duplicate error.
        return
    cases, error = read_sweep_cases(sweep_path)
    if error is not None:
        _error(sweep_path, error, errors)
        return
    check_enforcement_level_sweep(sweep_path, cases, errors)
    check_sweep_case_ids(support_path, support_data, cases, errors)


def verify_all(bundle_root: Path) -> List[str]:
    errors: List[str] = []
    if not bundle_root.is_dir():
        return errors

    # Walk the tree the way the harness does, so a sidecar reached through a
    # directory symlink is checked rather than silently skipped.
    _, files = walk_bundle_tree(bundle_root)

    for support_path in (f for f in files if is_single_graph_sidecar(f)):
        data, loaded = _load_sidecar(support_path, errors)
        if not loaded:
            continue
        check_canonical_form(support_path, data, errors)
        validate_single_graph_schema(data, support_path, errors)
        check_orphaned_single_graph(support_path, errors)
        if _has_non_empty_claims(data):
            check_enforcement_level_single(support_path, errors)

    for support_path in (f for f in files if is_sweep_sidecar(f)):
        data, loaded = _load_sidecar(support_path, errors)
        if not loaded:
            continue
        check_canonical_form(support_path, data, errors)
        validate_sweep_schema(data, support_path, errors)
        check_orphaned_sweep(support_path, errors)
        if _has_non_empty_claims(data):
            check_sweep_manifest(support_path, data, errors)

    return errors


def main() -> int:
    errors = verify_all(BUNDLE_ROOT)
    if errors:
        print(
            f"verify_support_claims: {len(errors)} error(s) found:",
            file=sys.stderr,
        )
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
