#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prove that every declared hipDNN packaging probe ran and passed.

The packaging-probe CI jobs run ctest with --output-junit. ctest exits 0 for a
junit file that lists fewer tests than were declared, and a skipped test is not
a failure to it. This script compares the junit file against the manifest the
CMake configure wrote (one ctest test name per line) so that "declared N, ran
fewer", "skipped" and "ran nothing" are all red.

Usage:
    check_probe_junit.py --junit build/probe-junit.xml \
        --manifest build/hkp-probes/manifest.txt

Exit codes:
    0  every manifest name ran exactly once; no failure, error or skip
    1  one or more checks failed; each prints
       "check_probe_junit: FAIL <id>: <detail>"
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# One probe plus the "hkp-probe-tools" test.
MIN_MANIFEST_ENTRIES = 2


def read_manifest(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def check(junit_path: Path, manifest_path: Path) -> list[tuple[str, str]]:
    """Return (id, detail) for every failed check; empty means green."""
    try:
        names = read_manifest(manifest_path)
    except OSError as exc:
        return [("manifest-unreadable", f"{manifest_path}: {exc}")]

    failures: list[tuple[str, str]] = []
    if len(names) < MIN_MANIFEST_ENTRIES:
        failures.append(
            (
                "manifest-too-small",
                f"{manifest_path} lists {len(names)} test(s), expected at least "
                f"{MIN_MANIFEST_ENTRIES} (one probe plus hkp-probe-tools)",
            )
        )

    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError) as exc:
        failures.append(("junit-unreadable", f"{junit_path}: {exc}"))
        return failures

    cases = list(root.iter("testcase"))
    ran = [case.get("name", "") for case in cases]

    if len(cases) != len(names):
        failures.append(
            (
                "count-mismatch",
                f"junit has {len(cases)} testcase(s), manifest declares {len(names)}",
            )
        )
    for name in names:
        if name not in ran:
            failures.append(
                ("name-missing", f"declared test '{name}' is not in the junit")
            )

    for case in cases:
        name = case.get("name", "")
        if case.find("skipped") is not None or case.get("status") == "notrun":
            failures.append(("skipped", f"test '{name}' did not run"))
        if case.find("failure") is not None:
            failures.append(("failure", f"test '{name}' failed"))
        if case.find("error") is not None:
            failures.append(("error", f"test '{name}' errored"))

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--junit", required=True, type=Path, help="ctest --output-junit file"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="file listing one declared ctest test name per line",
    )
    args = parser.parse_args(argv)

    failures = check(args.junit, args.manifest)
    for check_id, detail in failures:
        print(f"check_probe_junit: FAIL {check_id}: {detail}")
    if failures:
        return 1
    count = len(read_manifest(args.manifest))
    print(f"check_probe_junit: OK {count} declared test(s) ran and passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
