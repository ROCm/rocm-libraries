#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Discover ctest targets in a hipDNN superbuild.

Most components register prefixed top-level targets (e.g.
`<prefix>-unit-check`). Some register the bare `unit-check` / `check` /
`integration-check` targets under their source subdirectory instead of a
prefixed top-level target. This helper checks the prefixed form first, then
falls back to the path-qualified target for such components.
"""

import argparse
import os
import re
import subprocess
import sys


COMPONENT_PREFIXES = {
    "hipdnn": "hipdnn",
    "miopen": "miopen-provider",
    "hipblaslt": "hipblaslt-provider",
    "hip-kernel": "hip-kernel-provider",
    "integration-tests": "hipdnn-integration-tests",
}

COMPONENT_FALLBACK_PATHS = {
    "hip-kernel": "dnn-providers/hip-kernel-provider/src",
}

SCOPE_SUFFIXES = {
    "unit": "unit-check",
    "integration": "integration-check",
    "external-integration": "external-integration-check",
    "all": "check",
}

# Scope suffixes covered by the aggregate "all" scope, in report order.
ALL_SCOPE_SUFFIXES = ["unit-check", "integration-check", "external-integration-check"]

EXTERNAL_SCOPE_SUFFIX = "external-integration-check"
CTEST_FILENAME = "CTestTestfile.cmake"
_ADD_TEST_RE = re.compile(r"^add_test\(\s*(.*)\)\s*$")
# add_test() args are emitted as [=[bracketed]=], "quoted", or bare tokens.
_TOKEN_RE = re.compile(r"\[=\[(.*?)\]=\]|\"((?:[^\"\\]|\\.)*)\"|(\S+)")


def list_ninja_targets(build_dir):
    result = subprocess.run(
        ["ninja", "-C", build_dir, "-t", "targets", "all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "no diagnostic output"
        raise RuntimeError(f"ninja target discovery failed: {detail}")

    targets = set()
    for line in result.stdout.splitlines():
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


def _parse_add_tests(text):
    """Yield (test_name, argv) for each add_test() in a CTestTestfile.cmake.
    Generated files emit one add_test() per line with genex-resolved paths."""
    for line in text.splitlines():
        m = _ADD_TEST_RE.match(line.strip())
        if not m:
            continue
        tokens = []
        for tm in _TOKEN_RE.finditer(m.group(1)):
            bracket, quoted, bare = tm.groups()
            if bracket is not None:
                tokens.append(bracket)
            elif quoted is not None:
                tokens.append(quoted.replace('\\"', '"'))
            else:
                tokens.append(bare)
        if tokens:
            yield tokens[0], tokens[1:]


def collect_ctest_tests(build_dir):
    """Map test name -> resolved argv by scanning every CTestTestfile.cmake in
    the build tree. Empty when the build has no registered ctest tests."""
    tests = {}
    for root, _dirs, files in os.walk(build_dir):
        if CTEST_FILENAME not in files:
            continue
        try:
            text = open(
                os.path.join(root, CTEST_FILENAME), encoding="utf-8", errors="replace"
            ).read()
        except OSError:
            continue
        for name, argv in _parse_add_tests(text):
            tests.setdefault(name, argv)
    return tests


def _quote(token):
    return f'"{token}"' if (not token or " " in token) else token


def external_command(ctest_tests, component):
    """Ready-to-run command for a provider's external integration suite, taken
    from the first matching registered add_test() with its --gtest_filter
    stripped (so the caller can supply their own). None when absent."""
    key = f"{COMPONENT_PREFIXES[component]}-external-integration"
    for name in sorted(ctest_tests):
        if name.startswith(key):
            argv = [a for a in ctest_tests[name] if not a.startswith("--gtest_filter")]
            if argv:
                return " ".join(_quote(a) for a in argv)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--build-dir",
        required=True,
        help="Path to superbuild build dir containing build.ninja",
    )
    p.add_argument(
        "--component", default="all", help="Component name or all. Default: all."
    )
    p.add_argument(
        "--scope",
        default="unit",
        choices=["unit", "integration", "external-integration", "all"],
        help="Test scope. Default: unit. all returns every scope present. "
        "external-integration also emits the resolved cross-provider command line.",
    )
    args = p.parse_args()

    if args.component != "all" and args.component not in COMPONENT_PREFIXES:
        print(
            f"ERROR: unknown component '{args.component}'. "
            f"Valid: {', '.join(COMPONENT_PREFIXES)}, all",
            file=sys.stderr,
        )
        return 2

    try:
        targets = list_ninja_targets(args.build_dir)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not targets:
        print(
            f"ERROR: no ninja targets discovered in {args.build_dir}", file=sys.stderr
        )
        return 1

    components = (
        list(COMPONENT_PREFIXES) if args.component == "all" else [args.component]
    )
    scopes = (
        list(ALL_SCOPE_SUFFIXES)
        if args.scope == "all"
        else [SCOPE_SUFFIXES[args.scope]]
    )

    # The resolved external command is read from the generated CTestTestfile,
    # only needed when the external-integration scope is in play.
    ctest_tests = (
        collect_ctest_tests(args.build_dir) if EXTERNAL_SCOPE_SUFFIX in scopes else {}
    )

    found_any = False
    for comp in components:
        for scope_suffix in scopes:
            target = find_target(targets, comp, scope_suffix)
            if target:
                print(f"{comp}:{target}")
                found_any = True
            if scope_suffix == EXTERNAL_SCOPE_SUFFIX:
                command = external_command(ctest_tests, comp)
                if command:
                    print(f"{comp}:command:{command}")
                    found_any = True

    return 0 if found_any else 1


if __name__ == "__main__":
    sys.exit(main())
