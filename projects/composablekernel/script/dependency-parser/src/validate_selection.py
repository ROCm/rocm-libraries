#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Validate a smart-build test selection against the real build/test namespaces.

PURPOSE
    Asserts that every selected executable is a target ninja actually knows about
    - plain set-membership against `ninja -t targets all`. With --ctest it also
    checks each selected basename is a registered ctest test. This catches
    path/normalization drift between the depmap's view and ninja's real target
    namespace *before* a build is attempted.

INPUTS (where each comes from)
    tests_to_run.json  <- main.py select
    ninja_targets.txt  <- ninja -t targets all > ninja_targets.txt
    ctest_list.txt     <- ctest -N > ctest_list.txt   (optional)

USAGE
    validate_selection.py <tests_to_run.json> --ninja-targets <ninja_targets.txt>
                          [--ctest <ctest_list.txt>] [--output smoke_result.json]
                          [--junit smoke_result.xml]

OUTPUT
    Writes smoke_result.json (+ optional JUnit XML). Exit codes:
      0  selection valid (or empty)
      1  selection invalid (a selected name is not a ninja target / ctest test)
      2  a required input file is missing

Caveat: `ninja -t targets all` is the oracle because CK's CMake
GLOB CONFIGURE_DEPENDS regenerates build.ninja on every call, so `ninja -n
<target>` exits 0 for any name and cannot be used to test target existence.

Terminology: see the Glossary in README.md.
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
    # Use `or` chaining so an explicit JSON null is treated as empty, not None.
    return list(data.get("executables") or data.get("tests_to_run") or [])


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


def validate(selected, valid_targets, ctest_tests=None, mode=None, label=None):
    """Check a selection against the known ninja targets (and optionally ctest).

    Runs up to two checks and returns a result dict:
      primary   - every selected executable exists in valid_targets;
      secondary - when ctest_tests is given, every selected basename is a
                  registered ctest test.
    The verdict is "pass" only if all applicable checks hold (an empty selection
    passes); otherwise "fail", with the offending names listed.

    `mode` (full|selective|none), when given, records whether this selection was
    actually used: on a full/none build it's an advisory "as-if" computation; only
    on a selective build does it drive what gets built/run. Tagged into the result
    and the JUnit so consumers don't conflate as-if with real selective runs.

    `label` (e.g. the GPU arch), when given, further tags the JUnit so the per-arch
    smoke results aren't published as indistinguishable duplicate rows in Jenkins.
    """
    invalid_targets = [e for e in selected if e not in valid_targets]
    result = {
        "verdict": "pass" if not invalid_targets else "fail",
        "n_selected": len(selected),
        "n_known_targets": len(valid_targets),
        "n_invalid_targets": len(invalid_targets),
        "invalid_targets": invalid_targets,
    }
    if mode is not None:
        result["mode"] = mode
        result["advisory"] = (mode != "selective")
    if label:
        result["label"] = label
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
    """Render a minimal JUnit XML report for the validation result.

    When result carries a `mode`, the suite/classname are tagged with it
    (e.g. smart-build.selection.full) so Jenkins' test-results trend keeps advisory
    as-if runs (full/none) distinct from real selective runs. A `label` (e.g. the
    GPU arch) is appended too, so per-arch publishes land as distinct rows
    (smart-build.selection.full.gfx942) instead of duplicate rows.
    """
    failures = []
    for t in result.get("invalid_targets", []):
        failures.append(("not a ninja target", t))
    for t in result.get("invalid_tests", []):
        failures.append(("not a registered ctest test", t))

    mode = result.get("mode")
    label = result.get("label")
    tag = (f".{mode}" if mode else "") + (f".{label}" if label else "")
    suite_tag = (f"-{mode}" if mode else "") + (f"-{label}" if label else "")
    classname = "smart-build.selection" + tag
    suite = "smart-build-selection" + suite_tag

    _attr = {"'": "&apos;", '"': "&quot;"}
    cases = []
    if failures:
        for reason, name in failures:
            cases.append(
                f'    <testcase classname="{classname}" '
                f'name="{escape(name, _attr)}">\n'
                f'      <failure message="{escape(reason, _attr)}"/>\n'
                f"    </testcase>"
            )
    else:
        # Include the label in the leaf case name too, so the per-arch publishes
        # read as distinct rows (all-selected-targets-exist (gfx942)) rather than
        # four identical entries in the Jenkins test list. The name states what the
        # pass asserts: every selected target resolves to a real ninja target.
        case_name = "all-selected-targets-exist" + (f" ({label})" if label else "")
        cases.append(
            f'    <testcase classname="{classname}" '
            f'name="{escape(case_name, _attr)}"/>'
        )

    props = [
        ("n_selected", result.get("n_selected", 0)),
        ("n_known_targets", result.get("n_known_targets", 0)),
        ("n_invalid_targets", result.get("n_invalid_targets", 0)),
    ]
    if "n_invalid_tests" in result:
        props.append(("n_invalid_tests", result["n_invalid_tests"]))
    if "advisory" in result:
        props.append(("advisory", str(result["advisory"]).lower()))
    props_xml = "\n".join(
        f'    <property name="{k}" value="{v}"/>' for k, v in props
    )

    n_failures = len(failures)
    body = "\n".join(cases)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite name="{suite}" tests="{max(len(cases), 1)}" '
        f'failures="{n_failures}">\n'
        f"  <properties>\n{props_xml}\n  </properties>\n"
        f"{body}\n"
        "</testsuite>\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate a smart-build selection: every selected executable "
        "must be a real `ninja -t targets all` target (and, with --ctest, a "
        "registered ctest test).",
        epilog=(
            "Inputs (where each comes from):\n"
            "  tests_to_run.json   main.py select\n"
            "  ninja_targets.txt   ninja -t targets all > ninja_targets.txt\n"
            "  ctest_list.txt      ctest -N > ctest_list.txt   (optional, --ctest)\n\n"
            "Example:\n"
            "  validate_selection.py tests_to_run.json \\\n"
            "    --ninja-targets ninja_targets.txt --ctest ctest_list.txt \\\n"
            "    --output smoke_result.json --junit smoke_result.xml\n\n"
            "Exit: 0 valid (or empty), 1 invalid, 2 missing input.\n"
            "Terminology: see the Glossary in README.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument(
        "--mode",
        choices=["full", "selective", "none"],
        help="Tag the result/JUnit with the build mode so advisory as-if runs "
        "(full/none) stay distinct from real selective runs",
    )
    parser.add_argument(
        "--label",
        help="Extra tag for the JUnit suite/classname (e.g. the GPU arch), so "
        "per-arch publishes land as distinct rows instead of duplicates",
    )
    args = parser.parse_args()

    for path in [args.tests_json, args.ninja_targets] + (
        [args.ctest] if args.ctest else []
    ):
        if not os.path.exists(path):
            print(f"Error: missing required input: {path}", file=sys.stderr)
            sys.exit(2)

    selected = load_selected_executables(args.tests_json)
    valid_targets = load_ninja_targets(args.ninja_targets)
    ctest_tests = load_ctest_tests(args.ctest) if args.ctest else None

    result = validate(selected, valid_targets, ctest_tests, mode=args.mode, label=args.label)

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
            print(f"  [X] {t}")
    if "invalid_tests" in result and result["invalid_tests"]:
        print(f"Invalid ctest names:  {result['n_invalid_tests']}")
        for t in result["invalid_tests"]:
            print(f"  [X] {t}")
    print(f"Verdict: {result['verdict'].upper()}")
    print(f"Result written to: {args.output}")
    print("=========================================")

    sys.exit(0 if result["verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
