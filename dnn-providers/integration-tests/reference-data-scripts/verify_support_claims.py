#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pre-commit validator for RFC 0015 engine-support-claim sidecars.

Validates the two on-disk claim-sidecar shapes defined by RFC 0015
(`projects/hipdnn/docs/rfcs/0015_EngineSupportClaims.md`, §5, §5.4, §6, §9.3):

- A single-graph bundle's optional ``{Name}.support.json``, co-located with
  ``{Name}.json`` (and, when claim-bearing, requiring an ``enforcement_level``
  in the companion ``{Name}.meta.json``).
- A template-sweep bundle's optional bare ``support.json``, co-located with
  ``graph.template.json`` + ``sweep.json`` (and, when a case is claim-bearing,
  requiring an ``enforcement_level`` in that case's ``sweep.json`` metadata).

This script is scoped to RFC 0015 §9.3's support-claim checks only. It does
not perform RFC 0011's separate NaN/Inf, tier-folder, or size-budget checks;
those belong to a different verifier.

Usage:

    python3 verify_support_claims.py <root-dir> [<root-dir> ...]

Each root is typically a bundle tree such as
``dnn-providers/integration-tests/integration-test-bundles/quick``.
"""

import argparse
import json
import pathlib
import sys

# Companion-suffix kinds excluded from direct-bundle graph discovery, mirroring
# companionKinds() in src/harness/bundle/BundleDiscovery.hpp.
COMPANION_KINDS = {"meta", "support"}

VALID_PLATFORMS = {"linux", "windows"}
VALID_ENFORCEMENT_LEVELS = {"applicability", "buildable", "full"}

SWEEP_TEMPLATE_NAME = "graph.template.json"
SWEEP_MANIFEST_NAME = "sweep.json"
SWEEP_SUPPORT_NAME = "support.json"


def _load_json(path: pathlib.Path) -> tuple[object, list[str]]:
    """Parse ``path`` as JSON. Returns (data, errors); errors is non-empty on
    any I/O or parse failure, in which case data is None."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), []
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{path}: failed to parse JSON ({exc})"]


def _is_valid_version(value: object) -> bool:
    """RFC 0015 §5.1: version must be present and exactly the integer 1."""
    return isinstance(value, int) and not isinstance(value, bool) and value == 1


def _is_valid_enforcement_level(value: object) -> bool:
    return isinstance(value, str) and value in VALID_ENFORCEMENT_LEVELS


def _validate_platform_list(
    file_path: pathlib.Path, engine: str, arch: str, platforms: object
) -> list[str]:
    """RFC 0015 §5.1/§5.3: platform entries must be an array of exactly
    'linux'/'windows' tokens."""
    if not isinstance(platforms, list):
        return [
            f"{file_path}: engine '{engine}' arch '{arch}': platform value must be "
            f"an array of platform strings, got {type(platforms).__name__}"
        ]
    errors = []
    for platform in platforms:
        if platform not in VALID_PLATFORMS:
            errors.append(
                f"{file_path}: engine '{engine}' arch '{arch}': invalid platform "
                f"token {platform!r} (must be 'linux' or 'windows')"
            )
    return errors


# --------------------------------------------------------------------------
# Discovery (mirrors BundleDiscovery.hpp: isGraphFile / isSweepBundleRoot)
# --------------------------------------------------------------------------


def is_graph_file(json_path: pathlib.Path) -> bool:
    """Mirror isGraphFile() in BundleDiscovery.hpp: a .json file is a direct
    graph unless it is a sweep control file, or its stem (whole, or final
    dotted segment) is a companion kind ("meta"/"support")."""
    if json_path.suffix != ".json":
        return False
    if json_path.name in (SWEEP_TEMPLATE_NAME, SWEEP_MANIFEST_NAME):
        return False

    stem = json_path.stem
    if stem in COMPANION_KINDS:
        return False

    dot = stem.rfind(".")
    if dot == -1:
        return True
    return stem[dot + 1 :] not in COMPANION_KINDS


def is_sweep_root(directory: pathlib.Path) -> bool:
    """Mirror isSweepBundleRoot(): a directory directly containing both
    graph.template.json and sweep.json."""
    return (directory / SWEEP_TEMPLATE_NAME).is_file() and (
        directory / SWEEP_MANIFEST_NAME
    ).is_file()


def find_sweep_roots(root: pathlib.Path) -> list[pathlib.Path]:
    """Every directory under (or equal to) root that is a sweep root."""
    if not root.is_dir():
        return []

    roots = set()
    if is_sweep_root(root):
        roots.add(root)
    for entry in root.rglob("*"):
        if entry.is_dir() and is_sweep_root(entry):
            roots.add(entry)
    return sorted(roots)


def _is_descendant_of(path: pathlib.Path, ancestor: pathlib.Path) -> bool:
    try:
        path.relative_to(ancestor)
        return True
    except ValueError:
        return False


def find_graph_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Every direct-bundle graph .json under root: excludes companion
    sidecars, sweep control files, and anything under a sweep root."""
    if not root.is_dir():
        return []

    sweep_roots = find_sweep_roots(root)
    graphs = []
    for json_path in sorted(root.rglob("*.json")):
        if not json_path.is_file():
            continue
        if any(_is_descendant_of(json_path, sweep_root) for sweep_root in sweep_roots):
            continue
        if is_graph_file(json_path):
            graphs.append(json_path)
    return graphs


# --------------------------------------------------------------------------
# Single-graph bundle validation
# --------------------------------------------------------------------------


def _validate_single_support_schema(path: pathlib.Path, data: object) -> list[str]:
    """RFC 0015 §5, §5.1, §5.3 schema checks for a {Name}.support.json."""
    if not isinstance(data, dict):
        return [f"{path}: top-level JSON must be an object"]

    errors = []
    version = data.get("version")
    if not _is_valid_version(version):
        errors.append(f"{path}: 'version' must be exactly integer 1, got {version!r}")

    if "claims" not in data:
        return errors
    claims = data["claims"]
    if not isinstance(claims, dict):
        return errors + [
            f"{path}: 'claims' must be an object, got {type(claims).__name__}"
        ]

    for engine, arch_map in claims.items():
        if not isinstance(arch_map, dict):
            errors.append(
                f"{path}: engine '{engine}': value must be an object mapping "
                f"arch -> platform array, got {type(arch_map).__name__}"
            )
            continue
        for arch, platforms in arch_map.items():
            errors.extend(_validate_platform_list(path, engine, arch, platforms))

    return errors


def _validate_single_meta_enforcement(
    meta_path: pathlib.Path, support_path: pathlib.Path
) -> list[str]:
    """RFC 0015 §6, §6.2, §9.3: a claim-bearing single-graph bundle must have
    an explicit, valid enforcement_level in its companion meta.json."""
    if not meta_path.is_file():
        return [
            f"{meta_path}: missing (required: {support_path} carries a support "
            f"claim, so 'enforcement_level' must be one of "
            f"{sorted(VALID_ENFORCEMENT_LEVELS)})"
        ]

    data, parse_errors = _load_json(meta_path)
    if parse_errors:
        return parse_errors
    if not isinstance(data, dict):
        return [f"{meta_path}: top-level JSON must be an object"]

    level = data.get("enforcement_level")
    if not _is_valid_enforcement_level(level):
        return [
            f"{meta_path}: 'enforcement_level' must be one of "
            f"{sorted(VALID_ENFORCEMENT_LEVELS)}, got {level!r} (required: "
            f"{support_path} carries a support claim)"
        ]
    return []


def validate_single_graph_bundle(graph_json_path: pathlib.Path) -> list[str]:
    """Validate one direct-bundle graph's optional {Name}.support.json and,
    if claim-bearing, its companion {Name}.meta.json enforcement_level."""
    name = graph_json_path.stem
    support_path = graph_json_path.with_name(f"{name}.support.json")
    if not support_path.is_file():
        return []  # not claim-bearing; enforcement_level is out of scope.

    data, parse_errors = _load_json(support_path)
    if parse_errors:
        return parse_errors

    errors = _validate_single_support_schema(support_path, data)

    meta_path = graph_json_path.with_name(f"{name}.meta.json")
    errors.extend(_validate_single_meta_enforcement(meta_path, support_path))

    return errors


# --------------------------------------------------------------------------
# Template-sweep bundle validation
# --------------------------------------------------------------------------


def validate_sweep_bundle(sweep_dir: pathlib.Path) -> list[str]:
    """Validate one sweep root's optional bare support.json against its
    sibling sweep.json: schema, orphaned/ambiguous claims, and per-claimed-case
    enforcement_level."""
    sweep_json_path = sweep_dir / SWEEP_MANIFEST_NAME
    sweep_data, parse_errors = _load_json(sweep_json_path)
    if parse_errors:
        return parse_errors

    if not isinstance(sweep_data, dict) or not isinstance(
        sweep_data.get("cases"), list
    ):
        return [
            f"{sweep_json_path}: malformed sweep.json (expected a top-level "
            f"object with a 'cases' array)"
        ]

    case_ids = set()
    case_by_id = {}
    for case in sweep_data["cases"]:
        if isinstance(case, dict) and isinstance(case.get("id"), str):
            case_ids.add(case["id"])
            case_by_id[case["id"]] = case

    support_path = sweep_dir / SWEEP_SUPPORT_NAME
    if not support_path.is_file():
        return []  # no bare support.json; nothing claim-bearing here.

    data, parse_errors = _load_json(support_path)
    if parse_errors:
        return parse_errors
    if not isinstance(data, dict):
        return [f"{support_path}: top-level JSON must be an object"]

    errors = []
    version = data.get("version")
    if not _is_valid_version(version):
        errors.append(
            f"{support_path}: 'version' must be exactly integer 1, got {version!r}"
        )

    claims = data.get("claims", {})
    if "claims" in data and not isinstance(claims, dict):
        errors.append(
            f"{support_path}: 'claims' must be an object, got {type(claims).__name__}"
        )
        claims = {}

    claimed_case_ids = set()

    for engine, groups in claims.items():
        if not isinstance(groups, list):
            errors.append(
                f"{support_path}: engine '{engine}': value must be an array of "
                f"claim groups, got {type(groups).__name__}"
            )
            continue

        seen_in_engine = set()
        for group in groups:
            if not isinstance(group, dict):
                errors.append(
                    f"{support_path}: engine '{engine}': each claim group must be an object"
                )
                continue

            group_cases = group.get("cases")
            if (
                not isinstance(group_cases, list)
                or not group_cases
                or not all(isinstance(c, str) for c in group_cases)
            ):
                errors.append(
                    f"{support_path}: engine '{engine}': group 'cases' must be a "
                    f"non-empty array of strings"
                )
                group_cases = []

            support_map = group.get("support")
            if not isinstance(support_map, dict):
                errors.append(
                    f"{support_path}: engine '{engine}': group 'support' must be "
                    f"an object mapping arch -> platform array"
                )
            else:
                for arch, platforms in support_map.items():
                    errors.extend(
                        _validate_platform_list(support_path, engine, arch, platforms)
                    )

            for case_id in group_cases:
                if case_id not in case_ids:
                    errors.append(
                        f"{support_path}: engine '{engine}': orphaned claim - case "
                        f"id '{case_id}' not found in {sweep_json_path}"
                    )
                    continue
                if case_id in seen_in_engine:
                    errors.append(
                        f"{support_path}: engine '{engine}': ambiguous claim - case "
                        f"id '{case_id}' appears in more than one group"
                    )
                    continue
                seen_in_engine.add(case_id)
                claimed_case_ids.add(case_id)

    for case_id in sorted(claimed_case_ids):
        case = case_by_id.get(case_id, {})
        metadata = case.get("metadata", {}) if isinstance(case, dict) else {}
        level = (
            metadata.get("enforcement_level") if isinstance(metadata, dict) else None
        )
        if not _is_valid_enforcement_level(level):
            errors.append(
                f"{sweep_json_path}: case '{case_id}' in {sweep_dir}: "
                f"'enforcement_level' must be one of {sorted(VALID_ENFORCEMENT_LEVELS)}, "
                f"got {level!r} (required: claimed in {support_path})"
            )

    return errors


# --------------------------------------------------------------------------
# Directory-wide validation and CLI
# --------------------------------------------------------------------------


def validate_directory(root: pathlib.Path) -> list[str]:
    """Validate every direct-bundle graph and template-sweep bundle found
    recursively under root."""
    errors = []
    for graph_path in find_graph_files(root):
        errors.extend(validate_single_graph_bundle(graph_path))
    for sweep_root in find_sweep_roots(root):
        errors.extend(validate_sweep_bundle(sweep_root))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate RFC 0015 engine-support-claim sidecars (schema, "
            "cross-references, and enforcement_level) under one or more "
            "bundle root directories."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=pathlib.Path,
        metavar="root-dir",
        help="Directory to scan recursively for graph and template-sweep bundles.",
    )
    args = parser.parse_args(argv)

    all_errors = []
    for root in args.roots:
        all_errors.extend(validate_directory(root))

    for error in all_errors:
        print(error, file=sys.stderr)

    if all_errors:
        print(
            f"{len(all_errors)} support-claim error(s) found across "
            f"{len(args.roots)} root(s) scanned",
            file=sys.stderr,
        )
        return 1

    print("No support-claim errors found", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
