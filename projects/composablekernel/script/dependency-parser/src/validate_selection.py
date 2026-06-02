#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validate a smart-build test selection against the real build/test namespaces.

The selector (selective_test_filter.py) emits a list of executables to build and
test. This tool asserts that every selected executable is a target that ninja
actually knows about, catching path/normalization drift between the dependency
analyzer's view and ninja's real target namespace *before* a build is attempted.

Why not `ninja -n`: CK uses CMake GLOB CONFIGURE_DEPENDS, so every ninja
invocation regenerates build.ninja and `ninja -n <target>` exits 0 for any
target (real or bogus) - it only performs the manifest regeneration. The
reliable oracle is the target list from `ninja -t targets all`, against which we
do plain set-membership here.

Usage:
  validate_selection.py <tests_to_run.json> --ninja-targets <ninja_targets.txt>
                        [--ctest <ctest_list.txt>] [--output smoke_result.json]
                        [--junit smoke_result.xml]

Exit code: 0 if the selection is valid (or empty), 1 otherwise.
"""

import argparse
import json
import os
import re
import sys
from xml.sax.saxutils import escape


def load_selected_executables(tests_json):
    """Return the list of selected executables from a select/tests_to_run JSON."""
    with open(tests_json) as f:
        data = json.load(f)
    # selective_test_filter.py writes both "executables" and "tests_to_run".
    return list(data.get("executables", data.get("tests_to_run", [])))


def load_ninja_targets(path):
    """Parse `ninja -t targets all` output into a set of target names.

    Each line has the form `target: rulename`; the target name is everything
    before the first colon.
    """
    targets = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name = line.split(":", 1)[0].strip()
            if name:
                targets.add(name)
    return targets


def load_ctest_tests(path):
    """Parse `ctest -N` output into a set of registered test names."""
    pattern = re.compile(r"^\s*Test\s+#\d+:\s*(.+)$")
    tests = set()
    with open(path) as f:
        for line in f:
            match = pattern.match(line)
            if match:
                tests.add(match.group(1).strip())
    return tests


def validate(selected, valid_targets, ctest_tests=None):
    """Validate selected executables against the known ninja targets.

    Returns a result dict. verdict is "pass" when every selected executable is a
    known target (an empty selection is a pass); "fail" otherwise. When
    ctest_tests is provided, also checks each executable's basename is a
    registered ctest test (secondary, may flip verdict to fail).
    """
    invalid_targets = [e for e in selected if e not in valid_targets]
    result = {
        "verdict": "pass" if not invalid_targets else "fail",
        "n_selected": len(selected),
        "n_known_targets": len(valid_targets),
        "n_invalid_targets": len(invalid_targets),
        "invalid_targets": invalid_targets,
    }
    if ctest_tests is not None:
        invalid_tests = [
            e for e in selected if os.path.basename(e) not in ctest_tests
        ]
        result["n_invalid_tests"] = len(invalid_tests)
        result["invalid_tests"] = invalid_tests
        if invalid_tests:
            result["verdict"] = "fail"
    return result


def render_junit(result):
    """Render a minimal JUnit XML report for the validation result."""
    failures = []
    for t in result.get("invalid_targets", []):
        failures.append(("not a ninja target", t))
    for t in result.get("invalid_tests", []):
        failures.append(("not a registered ctest test", t))

    _attr = {"'": "&apos;", '"': "&quot;"}
    cases = []
    if failures:
        for reason, name in failures:
            cases.append(
                f'    <testcase classname="smart-build.selection" '
                f'name="{escape(name, _attr)}">\n'
                f'      <failure message="{escape(reason, _attr)}"/>\n'
                f"    </testcase>"
            )
    else:
        cases.append(
            '    <testcase classname="smart-build.selection" '
            'name="selection-resolvable"/>'
        )

    n_failures = len(failures)
    body = "\n".join(cases)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite name="smart-build-selection" tests="{max(len(cases), 1)}" '
        f'failures="{n_failures}">\n'
        f"{body}\n"
        "</testsuite>\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate a smart-build selection against ninja/ctest namespaces"
    )
    parser.add_argument("tests_json", help="Path to tests_to_run.json from select")
    parser.add_argument(
        "--ninja-targets",
        required=True,
        help="Path to `ninja -t targets all` output",
    )
    parser.add_argument(
        "--ctest",
        help="Optional path to `ctest -N` output for a secondary test-name check",
    )
    parser.add_argument(
        "--output",
        default="smoke_result.json",
        help="Output JSON verdict file (default: smoke_result.json)",
    )
    parser.add_argument(
        "--junit",
        help="Optional path to write a JUnit XML report",
    )
    args = parser.parse_args()

    for path in [args.tests_json, args.ninja_targets] + (
        [args.ctest] if args.ctest else []
    ):
        if not os.path.exists(path):
            print(f"Error: file not found: {path}")
            sys.exit(2)

    selected = load_selected_executables(args.tests_json)
    valid_targets = load_ninja_targets(args.ninja_targets)
    ctest_tests = load_ctest_tests(args.ctest) if args.ctest else None

    result = validate(selected, valid_targets, ctest_tests)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    if args.junit:
        with open(args.junit, "w") as f:
            f.write(render_junit(result))

    # Human-readable summary
    print("=========================================")
    print("Smart-build selection validation")
    print("=========================================")
    print(f"Selected executables: {result['n_selected']}")
    print(f"Known ninja targets:  {result['n_known_targets']}")
    print(f"Invalid targets:      {result['n_invalid_targets']}")
    if result["invalid_targets"]:
        for t in result["invalid_targets"]:
            print(f"  ✗ {t}")
    if "invalid_tests" in result and result["invalid_tests"]:
        print(f"Invalid ctest names:  {result['n_invalid_tests']}")
        for t in result["invalid_tests"]:
            print(f"  ✗ {t}")
    print(f"Verdict: {result['verdict'].upper()}")
    print(f"Result written to: {args.output}")
    print("=========================================")

    sys.exit(0 if result["verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
