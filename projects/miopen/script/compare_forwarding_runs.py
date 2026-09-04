#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare two gtest JUnit XML runs that differ only in MIOPEN_HIPDNN_FORWARDING.

Exits non-zero if the set of tests differs, or if any test's outcome differs
between the two runs. Timing is ignored; only pass/fail/skip status is compared.

Both runs execute the same binary from the same build, so any divergence is a
behavioural difference introduced by forwarding.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET


def outcomes(path):
    root = ET.parse(path).getroot()
    result = {}
    for case in root.iter("testcase"):
        name = "{}.{}".format(case.get("classname"), case.get("name"))
        if case.find("failure") is not None or case.find("error") is not None:
            status = "failed"
        elif case.get("status") == "notrun" or case.get("result") == "skipped":
            status = "skipped"
        else:
            status = "passed"
        result[name] = status
    return result


def check_fresh(path, newer_than):
    """Reject an XML that is missing or left over from an earlier build.

    Two stale files compare just as cleanly as two fresh ones, so without this a
    run whose replays never executed is indistinguishable from one that agreed.
    """
    if not os.path.isfile(path):
        return "{} does not exist -- the replay that writes it did not run".format(path)
    if newer_than is None:
        return None
    if not os.path.isfile(newer_than):
        return "{} does not exist, so {} cannot be shown to be current".format(
            newer_than, path
        )
    if os.path.getmtime(path) < os.path.getmtime(newer_than):
        return "{} is older than {} -- it is left over from an earlier build".format(
            path, newer_than
        )
    return None


def main(disabled_xml, enabled_xml, newer_than=None):
    stale = [
        problem
        for problem in (
            check_fresh(disabled_xml, newer_than),
            check_fresh(enabled_xml, newer_than),
        )
        if problem
    ]
    if stale:
        sys.stderr.write("forwarding parity cannot be checked:\n")
        for problem in stale:
            sys.stderr.write("  {}\n".format(problem))
        return 1

    a, b = outcomes(disabled_xml), outcomes(enabled_xml)
    problems = []

    for name in sorted(set(a) - set(b)):
        problems.append("only in disabled run: {}".format(name))
    for name in sorted(set(b) - set(a)):
        problems.append("only in enabled run: {}".format(name))
    for name in sorted(set(a) & set(b)):
        if a[name] != b[name]:
            problems.append("{}: disabled={} enabled={}".format(name, a[name], b[name]))

    if not a and not b:
        problems.append("both runs reported zero tests")

    if problems:
        sys.stderr.write("forwarding parity failed:\n")
        for p in problems:
            sys.stderr.write("  {}\n".format(p))
        return 1

    print("forwarding parity OK: {} tests identical under both modes".format(len(a)))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disabled_xml", help="JUnit XML from the =disabled replay")
    parser.add_argument("enabled_xml", help="JUnit XML from the =enabled replay")
    parser.add_argument(
        "--newer-than",
        metavar="PATH",
        help="path to the test binary; both XML files must be newer than it, so "
        "that output left over from an earlier build cannot be compared instead",
    )
    args = parser.parse_args()
    sys.exit(main(args.disabled_xml, args.enabled_xml, args.newer_than))
