#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Discover ctest targets in a hipDNN superbuild.

Knows about the hip-kernel-provider naming inconsistency: that provider
registers ctest targets as bare `unit-check` / `check` / `integration-check`
under `dnn-providers/hip-kernel-provider/src/` rather than prefixed.
This helper falls back to the path-qualified target when the prefixed
form is not present.

Outputs one matching target per line on stdout, formatted as:
    <component>:<target>

Components: hipdnn, miopen, hipblaslt, hip-kernel, integration-tests
Scopes: unit, integration, all

Usage:
    discover_test_targets.py --build-dir <path> [--component <name>] [--scope <name>]
"""

import argparse
import subprocess
import sys


COMPONENT_PREFIXES = {
    "hipdnn": "hipdnn",
    "miopen": "miopen-provider",
    "hipblaslt": "hipblaslt-provider",
    "hip-kernel": "hip-kernel-provider",
    "integration-tests": "hipdnn-integration-tests",
}

# Fallback path fragments used when the prefixed top-level target doesn't exist.
COMPONENT_FALLBACK_PATHS = {
    "hip-kernel": "dnn-providers/hip-kernel-provider/src",
}

SCOPE_SUFFIXES = {
    "unit": "unit-check",
    "integration": "integration-check",
    "all": "check",
}


def list_ninja_targets(build_dir):
    r = subprocess.run(
        ["ninja", "-C", build_dir, "-t", "targets", "all"],
        capture_output=True,
        text=True,
        check=False,
    )
    targets = set()
    for line in r.stdout.splitlines():
        if ":" not in line:
            continue
        name = line.split(":", 1)[0].strip()
        if name:
            targets.add(name)
    return targets


def find_target(targets, component, scope_suffix):
    prefix = COMPONENT_PREFIXES[component]
    primary = f"{prefix}-{scope_suffix}"
    if primary in targets:
        return primary

    fallback_path = COMPONENT_FALLBACK_PATHS.get(component)
    if fallback_path:
        candidate = f"{fallback_path}/{scope_suffix}"
        if candidate in targets:
            return candidate

    return None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--build-dir",
        required=True,
        help="Path to superbuild build dir (contains build.ninja)",
    )
    p.add_argument(
        "--component", default="all", help="Component name or 'all' (default: all)"
    )
    p.add_argument(
        "--scope",
        default="unit",
        choices=["unit", "integration", "all"],
        help="Test scope (default: unit). 'all' returns every scope present.",
    )
    args = p.parse_args()

    if args.component != "all" and args.component not in COMPONENT_PREFIXES:
        print(
            f"ERROR: unknown component '{args.component}'. "
            f"Valid: {', '.join(COMPONENT_PREFIXES)}, all",
            file=sys.stderr,
        )
        return 2

    targets = list_ninja_targets(args.build_dir)
    if not targets:
        print(
            f"ERROR: no ninja targets discovered in {args.build_dir}", file=sys.stderr
        )
        return 1

    components = (
        list(COMPONENT_PREFIXES) if args.component == "all" else [args.component]
    )
    if args.scope == "all":
        scopes = ["unit-check", "integration-check"]
    else:
        scopes = [SCOPE_SUFFIXES[args.scope]]

    found_any = False
    for comp in components:
        for scope_suffix in scopes:
            t = find_target(targets, comp, scope_suffix)
            if t:
                print(f"{comp}:{t}")
                found_any = True

    return 0 if found_any else 1


if __name__ == "__main__":
    sys.exit(main())
