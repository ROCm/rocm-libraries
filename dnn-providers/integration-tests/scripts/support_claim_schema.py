#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The support-claim sidecar schema (RFC 0015 §9.3), as plain validators.

Kept apart from ``verify_support_claims.py`` so the tools that read sidecars
-- the verifier, the support matrix collector -- share one definition of a
well-formed sidecar without importing each other's command line.

Each validator appends ``"<path>: <message>"`` to ``errors`` for every
violation and never raises, so one call reports everything wrong with a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

# The "version" every sidecar carries.
SCHEMA_VERSION = 1
VALID_PLATFORMS = {"linux", "windows"}


VALID_PLATFORMS = {"linux", "windows"}


def _error(path: Path, message: str, errors: List[str]) -> None:
    errors.append(f"{path}: {message}")


def _validate_arch_platform_map(
    data: dict, path: Path, context: str, errors: List[str]
) -> None:
    if not isinstance(data, dict):
        _error(path, f"{context}: support must be an object", errors)
        return
    for arch, platforms in data.items():
        if not arch:
            _error(path, f"{context}: empty arch key", errors)
            continue
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
    if data.get("version") != SCHEMA_VERSION:
        _error(
            path,
            f"version must be {SCHEMA_VERSION}, got {data.get('version')!r}",
            errors,
        )
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
        if not engine_name:
            _error(path, "empty engine name in claims", errors)
            continue
        _validate_arch_platform_map(arch_map, path, f"claims.{engine_name}", errors)


def validate_sweep_schema(data: dict, path: Path, errors: List[str]) -> None:
    if not isinstance(data, dict):
        _error(path, "top-level value must be an object", errors)
        return
    if data.get("version") != SCHEMA_VERSION:
        _error(
            path,
            f"version must be {SCHEMA_VERSION}, got {data.get('version')!r}",
            errors,
        )
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
        if not engine_name:
            _error(path, "empty engine name in claims", errors)
            continue
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
                    elif not case_id:
                        _error(path, f"{ctx}: empty case id", errors)
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
