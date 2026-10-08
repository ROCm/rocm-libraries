# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fail CI when a pytest suite skipped a test its allow-list does not name.

ctest reports a pytest invocation as passed even when every case inside it skipped,
so a gate that skips for want of an input (a compiler, a binary, an environment
variable) passes silently. The pytest junit file is the only place those skips are
visible. This reads one or more of them and fails on:

  unexpected-skip       a skipped (or xfailed) testcase no allow-list pattern matches
  stale-allow           an allow-list pattern that matched no skipped testcase, so the
                        entry outlived the skip it excused
  junit-unreadable      a junit file that is missing or is not valid XML
  junit-empty           a junit file with no testcase, i.e. the suite collected nothing
  allow-list-unreadable the allow-list file cannot be read
  allow-list-malformed  an allow-list line that is not ``<pattern> # <reason>``

Allow-list format: blank lines and lines starting with ``#`` are ignored; every other
line is ``<pattern> # <reason>``, split on the first ``#``, both parts non-empty. The
pattern is an fnmatch glob over ``<junit file stem>::<classname>::<name>``. Several
allow-lists may be given; their patterns are pooled, so a lane can add its own entries to
the shared ones.
"""

import argparse
import fnmatch
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PREFIX = "check_pytest_junit"


def _parse_allow_list(allow_path, failures):
    """Return the allow-list's patterns; append any problem to ``failures``."""
    try:
        text = Path(allow_path).read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(("allow-list-unreadable", f"{allow_path}: {exc}"))
        return []
    patterns = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, sep, reason = line.partition("#")
        pattern, reason = pattern.strip(), reason.strip()
        if not sep or not pattern or not reason:
            failures.append(
                (
                    "allow-list-malformed",
                    f"{allow_path}:{number}: expected '<pattern> # <reason>', got {line!r}",
                )
            )
            continue
        patterns.append(pattern)
    return patterns


def check(junit_paths, allow_paths):
    """Return ``(kind, detail)`` for every problem; empty means the check passed."""
    failures = []
    patterns = []
    for allow_path in allow_paths:
        patterns += _parse_allow_list(allow_path, failures)
    matched = set()
    for junit_path in map(Path, junit_paths):
        try:
            root = ET.parse(junit_path).getroot()
        except (OSError, ET.ParseError) as exc:
            failures.append(("junit-unreadable", f"{junit_path}: {exc}"))
            continue
        testcases = list(root.iter("testcase"))
        if not testcases:
            failures.append(("junit-empty", f"{junit_path}: no testcase recorded"))
            continue
        for case in testcases:
            skipped = case.find("skipped")
            if skipped is None:
                continue
            case_id = f"{junit_path.stem}::{case.get('classname', '')}::{case.get('name', '')}"
            hits = [p for p in patterns if fnmatch.fnmatchcase(case_id, p)]
            if hits:
                matched.update(hits)
            else:
                message = skipped.get("message") or (skipped.text or "").strip()
                failures.append(("unexpected-skip", f"{case_id}: {message}"))
    for pattern in patterns:
        if pattern not in matched:
            failures.append(("stale-allow", f"{pattern}: matched no skipped testcase"))
    return failures


def _count(junit_paths):
    """``(testcases, allow-listed skips)`` over the readable junit files."""
    testcases = skips = 0
    for junit_path in map(Path, junit_paths):
        try:
            root = ET.parse(junit_path).getroot()
        except (OSError, ET.ParseError):
            continue
        cases = list(root.iter("testcase"))
        testcases += len(cases)
        skips += sum(1 for case in cases if case.find("skipped") is not None)
    return testcases, skips


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--junit",
        action="append",
        required=True,
        metavar="PATH",
        help="pytest junit XML file; repeat for several suites",
    )
    parser.add_argument(
        "--allow-skips",
        action="append",
        required=True,
        metavar="PATH",
        help=(
            "allow-list of '<pattern> # <reason>' lines naming the skips to accept; "
            "repeat to pool several"
        ),
    )
    args = parser.parse_args(argv)

    failures = check(args.junit, args.allow_skips)
    if failures:
        for kind, detail in failures:
            print(f"{PREFIX}: FAIL {kind}: {detail}")
        return 1
    testcases, skips = _count(args.junit)
    print(
        f"{PREFIX}: OK {testcases} testcase(s) in {len(args.junit)} junit file(s), "
        f"{skips} allow-listed skip(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
