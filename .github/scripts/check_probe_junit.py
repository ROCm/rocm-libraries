#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prove that every declared hipDNN packaging probe ran and passed.

The superbuild lanes run the whole ctest suite once with --output-junit. ctest
exits 0 when a test is disabled or not run, so this script reads that junit file
and checks it against the manifest the CMake configure wrote (one ctest test
name per line). Testcases not named in the manifest are ignored.

The manifest always names the probe tooling test "hkp-probe-tools", plus one entry
per declared probe. A lane whose GPU_TARGETS cover every architecture the
descriptors name declares no probe, so "hkp-probe-tools" alone is a valid manifest;
an empty one means the configure did not write it properly.

For every manifest name, the junit must hold exactly one testcase with that
name, and that testcase must have run (status "run" or "fail"; without a status
attribute, no <skipped> child) and must have no <failure> or <error> child.

Usage:
    check_probe_junit.py --junit build/ctest-junit.xml \
        --manifest build/hkp-probes/manifest.txt

Exit codes:
    0  every manifest name ran exactly once and passed
    1  one or more checks failed; each prints
       "check_probe_junit: FAIL <id>: <detail>", where <id> is one of
       manifest-unreadable, manifest-too-small, junit-unreadable,
       name-missing, name-duplicate, not-run, failure, error
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# The "hkp-probe-tools" test; probes are optional.
MIN_MANIFEST_ENTRIES = 1


def read_manifest(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def ran(case: ET.Element) -> bool:
    """Return whether ctest ran the testcase.

    ctest writes status "run", "fail", "disabled" or "notrun"; only the first
    two mean the test executed.
    """
    status = case.get("status")
    if status is None:
        return case.find("skipped") is None
    return status in ("run", "fail")


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
                f"{MIN_MANIFEST_ENTRIES} (hkp-probe-tools)",
            )
        )

    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError) as exc:
        failures.append(("junit-unreadable", f"{junit_path}: {exc}"))
        return failures

    cases: dict[str, list[ET.Element]] = {name: [] for name in names}
    for case in root.iter("testcase"):
        found = cases.get(case.get("name", ""))
        if found is not None:
            found.append(case)

    for name, found in cases.items():
        if not found:
            failures.append(
                ("name-missing", f"declared test '{name}' is not in the junit")
            )
            continue
        if len(found) > 1:
            failures.append(
                (
                    "name-duplicate",
                    f"declared test '{name}' appears {len(found)} times in the junit",
                )
            )
        for case in found:
            if not ran(case):
                status = case.get("status")
                failures.append(
                    ("not-run", f"test '{name}' did not run (status={status})")
                )
            if case.get("status") == "fail" or case.find("failure") is not None:
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
