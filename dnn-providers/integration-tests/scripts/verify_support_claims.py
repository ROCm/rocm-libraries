#!/usr/bin/env python3
"""Pre-commit verifier for integration-test support-claim sidecars.

RFC 0015 §9.3 — validates that every .support.json in the bundle tree is
well-formed and consistent with the bundle it describes.  This is the first
slice of the RFC 0011 bundle verifier.

Checks
------
1. Schema: version == 1, claims is an object, arch -> array of tokens from
   {linux, windows}; sweep groups have non-empty cases + support.
2. enforcement_level: a graph with a non-empty claim must not declare an
   unrecognised enforcement_level (single-graph: X.meta.json; sweep:
   cases[].metadata.enforcement_level in sweep.json).  Absent is legal and
   means "full" -- see check_enforcement_level_single for why.
3. Sweep case ids: every claimed id must exist in the sibling sweep.json;
   no duplicate case id per engine.
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

BUNDLE_ROOT = Path(__file__).resolve().parent.parent / "integration-test-bundles"
VALID_PLATFORMS = {"linux", "windows"}
VALID_ENFORCEMENT_LEVELS = {"applicability", "buildable", "full"}


def _error(path: Path, message: str, errors: List[str]) -> None:
    errors.append(f"{path}: {message}")


def _validate_arch_platform_map(
    data: dict, path: Path, context: str, errors: List[str]
) -> None:
    if not isinstance(data, dict):
        _error(path, f"{context}: support must be an object", errors)
        return
    for arch, platforms in data.items():
        if not isinstance(platforms, list):
            _error(
                path,
                f"{context}: arch '{arch}' value must be an array",
                errors,
            )
            continue
        for platform in platforms:
            if not isinstance(platform, str):
                _error(
                    path,
                    f"{context}: platform entry must be a string,"
                    f" got {type(platform).__name__}",
                    errors,
                )
            elif platform not in VALID_PLATFORMS:
                _error(
                    path,
                    f"{context}: invalid platform '{platform}'"
                    f" (expected one of {sorted(VALID_PLATFORMS)})",
                    errors,
                )


def validate_single_graph_schema(data: dict, path: Path, errors: List[str]) -> None:
    if not isinstance(data, dict):
        _error(path, "top-level value must be an object", errors)
        return
    if data.get("version") != 1:
        _error(path, f"version must be 1, got {data.get('version')!r}", errors)
    if "claims" not in data:
        return
    claims = data["claims"]
    if claims is None:
        _error(path, "'claims' must be an object (got null)", errors)
        return
    if not isinstance(claims, dict):
        _error(path, "claims must be an object", errors)
        return
    for engine_name, arch_map in claims.items():
        _validate_arch_platform_map(arch_map, path, f"claims.{engine_name}", errors)


def validate_sweep_schema(data: dict, path: Path, errors: List[str]) -> None:
    if not isinstance(data, dict):
        _error(path, "top-level value must be an object", errors)
        return
    if data.get("version") != 1:
        _error(path, f"version must be 1, got {data.get('version')!r}", errors)
    if "claims" not in data:
        return
    claims = data["claims"]
    if claims is None:
        _error(path, "'claims' must be an object (got null)", errors)
        return
    if not isinstance(claims, dict):
        _error(path, "claims must be an object", errors)
        return
    for engine_name, groups in claims.items():
        if not isinstance(groups, list):
            _error(
                path,
                f"claims.{engine_name} must be an array of groups",
                errors,
            )
            continue
        seen_case_ids = set()
        for i, group in enumerate(groups):
            ctx = f"claims.{engine_name}[{i}]"
            if not isinstance(group, dict):
                _error(path, f"{ctx}: group must be an object", errors)
                continue
            cases = group.get("cases")
            if not isinstance(cases, list) or len(cases) == 0:
                _error(path, f"{ctx}: must have a non-empty 'cases' array", errors)
            else:
                for case_id in cases:
                    if not isinstance(case_id, str):
                        _error(
                            path,
                            f"{ctx}: case id must be a string,"
                            f" got {type(case_id).__name__}",
                            errors,
                        )
                    elif case_id in seen_case_ids:
                        _error(
                            path,
                            f"{ctx}: duplicate case id '{case_id}'"
                            f" within engine '{engine_name}'",
                            errors,
                        )
                    else:
                        seen_case_ids.add(case_id)
            support = group.get("support")
            if support is None:
                _error(path, f"{ctx}: missing 'support' object", errors)
            else:
                _validate_arch_platform_map(support, path, f"{ctx}.support", errors)


def _has_non_empty_claims(data: dict) -> bool:
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
    # support_path is X.support.json => bundle is X.json => meta is X.meta.json
    bundle_stem = Path(support_path.stem).stem  # "X.support.json" -> "X"
    meta_path = support_path.parent / f"{bundle_stem}.meta.json"
    if not meta_path.exists():
        return
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _error(meta_path, f"cannot read metadata: {exc}", errors)
        return
    level = meta.get("enforcement_level")
    if level is not None and level not in VALID_ENFORCEMENT_LEVELS:
        _error(
            meta_path,
            f"invalid enforcement_level '{level}'"
            f" (expected one of {sorted(VALID_ENFORCEMENT_LEVELS)})",
            errors,
        )


def check_enforcement_level_sweep(support_path: Path, errors: List[str]) -> None:
    sweep_path = support_path.parent / "sweep.json"
    if not sweep_path.exists():
        return
    try:
        with open(sweep_path) as f:
            sweep = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _error(sweep_path, f"cannot read sweep: {exc}", errors)
        return
    cases = sweep.get("cases", [])
    if not isinstance(cases, list):
        return
    for case_entry in cases:
        if not isinstance(case_entry, dict):
            continue
        metadata = case_entry.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        level = metadata.get("enforcement_level")
        if level is not None and level not in VALID_ENFORCEMENT_LEVELS:
            case_id = case_entry.get("id", "<unknown>")
            _error(
                sweep_path,
                f"case '{case_id}': invalid enforcement_level '{level}'",
                errors,
            )


def check_sweep_case_ids(
    support_path: Path, support_data: dict, errors: List[str]
) -> None:
    sweep_path = support_path.parent / "sweep.json"
    if not sweep_path.exists():
        _error(
            support_path,
            "sweep support.json has no sibling sweep.json",
            errors,
        )
        return
    try:
        with open(sweep_path) as f:
            sweep = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _error(sweep_path, f"cannot read sweep: {exc}", errors)
        return
    valid_ids = set()
    for case_entry in sweep.get("cases", []):
        if isinstance(case_entry, dict) and isinstance(case_entry.get("id"), str):
            valid_ids.add(case_entry["id"])
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
    bundle_stem = Path(support_path.stem).stem  # "X.support" -> "X"
    bundle_path = support_path.parent / f"{bundle_stem}.json"
    if not bundle_path.exists():
        _error(
            support_path,
            f"orphaned sidecar: no sibling {bundle_path.name}",
            errors,
        )


def check_orphaned_sweep(support_path: Path, errors: List[str]) -> None:
    sweep_path = support_path.parent / "sweep.json"
    if not sweep_path.exists():
        _error(
            support_path,
            "orphaned support.json: directory is not a sweep root"
            " (no sweep.json found)",
            errors,
        )


def is_sweep_sidecar(path: Path) -> bool:
    return path.name == "support.json"


def check_canonical_form(support_path: Path, data: object, errors: List[str]) -> None:
    canonical = json.dumps(data, indent=2, sort_keys=True) + "\n"
    raw = support_path.read_text(encoding="utf-8")
    if raw != canonical:
        _error(support_path, "not in canonical form (re-run the writer)", errors)


def verify_all(bundle_root: Path) -> List[str]:
    errors: List[str] = []
    if not bundle_root.is_dir():
        return errors

    for support_path in sorted(bundle_root.rglob("*.support.json")):
        try:
            with open(support_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            _error(support_path, f"invalid JSON: {exc}", errors)
            continue
        except OSError as exc:
            _error(support_path, f"cannot read: {exc}", errors)
            continue
        check_canonical_form(support_path, data, errors)
        validate_single_graph_schema(data, support_path, errors)
        check_orphaned_single_graph(support_path, errors)
        if _has_non_empty_claims(data):
            check_enforcement_level_single(support_path, errors)

    for support_path in sorted(bundle_root.rglob("support.json")):
        try:
            with open(support_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            _error(support_path, f"invalid JSON: {exc}", errors)
            continue
        except OSError as exc:
            _error(support_path, f"cannot read: {exc}", errors)
            continue
        check_canonical_form(support_path, data, errors)
        validate_sweep_schema(data, support_path, errors)
        check_orphaned_sweep(support_path, errors)
        if _has_non_empty_claims(data):
            check_enforcement_level_sweep(support_path, errors)
            check_sweep_case_ids(support_path, data, errors)

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
