#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare two gtest JUnit XML runs that differ only in MIOPEN_HIPDNN_FORWARDING.

Exits non-zero if the set of tests differs, or if any test's outcome differs
between the two runs. Timing is ignored; pass/fail/skip status is compared, along
with any parity_* property a test recorded about which path it took -- so a test
that computed a result and one that was declined the problem do not both count as
a plain pass.

Divergences listed in --known-divergences are tolerated and printed rather than
failed, and a listed divergence that has stopped happening fails the comparison
so the list cannot go stale.

Both runs execute the same binary from the same build, so any divergence is a
behavioural difference introduced by forwarding.
"""

import argparse
import collections
import os
import sys
import xml.etree.ElementTree as ET

# A test says more about itself than pass/fail by recording a property under this prefix.
# Matched by prefix rather than by ignoring the attributes gtest writes itself: time and
# timestamp differ between any two runs, and a list of those would need revising whenever gtest
# adds another. Both shapes gtest has written properties in are read -- attributes on the
# testcase element, and property children of it -- so changing the bundled gtest cannot silently
# drop them and leave two runs looking like they agreed.
PROPERTY_PREFIX = "parity_"

KnownDivergence = collections.namedtuple(
    "KnownDivergence", "name disabled enabled reason"
)


class UnreadableRun(Exception):
    """An XML that cannot be turned into a set of outcomes at all."""


def outcomes(path):
    """Map each test name to the statuses recorded for it, in document order.

    A name maps to a list rather than a single status because a JUnit file can
    carry the same fully-qualified name more than once, and collapsing those
    into one entry would hide a divergence between them.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        # What a crashed or killed replay leaves behind. Say so, rather than
        # letting the traceback stand in for a diagnostic.
        raise UnreadableRun(
            "{} is not well-formed XML ({}) -- the replay that writes it "
            "most likely crashed partway through".format(path, exc)
        )
    except OSError as exc:
        raise UnreadableRun("{} could not be read ({})".format(path, exc))

    result = {}
    for case in root.iter("testcase"):
        name = "{}.{}".format(case.get("classname"), case.get("name"))
        if case.find("failure") is not None or case.find("error") is not None:
            status = "failed"
        elif case.get("status") == "notrun" or case.get("result") == "skipped":
            status = "skipped"
        else:
            status = "passed"
        recorded = list(case.attrib.items()) + [
            (prop.get("name"), prop.get("value")) for prop in case.iter("property")
        ]
        detail = sorted(
            "{}={}".format(key, value)
            for key, value in recorded
            if key and key.startswith(PROPERTY_PREFIX)
        )
        if detail:
            status = "{}[{}]".format(status, ", ".join(detail))
        result.setdefault(name, []).append(status)
    return result


def load_known_divergences(path):
    """Read the list of divergences that are known and accepted.

    Four fields per line, the two outcomes pinned rather than the test name alone:
    a line keyed on the name would also silence a different divergence appearing
    later in the same test, which is the way an allowlist usually goes wrong.
    """
    try:
        with open(path) as handle:
            text = handle.read()
    except OSError as exc:
        raise UnreadableRun("{} could not be read ({})".format(path, exc))

    known = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 4 or not all(fields):
            # Rejected rather than skipped: a line that does not parse is one
            # someone meant to have an effect, and skipping it silently would
            # leave them believing it had one.
            raise UnreadableRun(
                "{} line {} is not <test name> | <disabled outcome> | <enabled outcome> "
                "| <why it is accepted>: {}".format(path, number, line)
            )
        known.append(KnownDivergence(*fields))
    return known


def counts(run):
    """Return (total entries, entries that ran) for one run's outcome map."""
    statuses = [status for entry in run.values() for status in entry]
    return len(statuses), sum(1 for status in statuses if status != "skipped")


def describe(statuses):
    """Render one name's statuses, keeping the count visible when it is > 1."""
    if len(statuses) == 1:
        return statuses[0]
    return "{} ({} entries)".format(", ".join(statuses), len(statuses))


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


def main(disabled_xml, enabled_xml, newer_than=None, known_divergences=None):
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

    try:
        a, b = outcomes(disabled_xml), outcomes(enabled_xml)
        known = load_known_divergences(known_divergences) if known_divergences else []
    except UnreadableRun as exc:
        sys.stderr.write("forwarding parity cannot be checked:\n  {}\n".format(exc))
        return 1

    problems = []
    tolerated = []
    diverged = set()

    for name in sorted(set(a) - set(b)):
        problems.append("only in disabled run: {}".format(name))
    for name in sorted(set(b) - set(a)):
        problems.append("only in enabled run: {}".format(name))
    shared = set(a) & set(b)
    for name in sorted(shared):
        # Sorted, because the two runs agreeing on which outcomes occurred is the
        # claim; the order gtest happened to emit them in is not.
        if sorted(a[name]) == sorted(b[name]):
            continue
        diverged.add(name)
        left, right = describe(a[name]), describe(b[name])
        match = next(
            (
                entry
                for entry in known
                if entry.name == name
                and entry.disabled == left
                and entry.enabled == right
            ),
            None,
        )
        if match:
            tolerated.append(match)
        else:
            problems.append("{}: disabled={} enabled={}".format(name, left, right))

    # The list is policed in both directions. Without this, a line outlives the gap
    # it describes and goes on silencing a test that has started agreeing, so the
    # list only ever grows. A listed test missing from the run is left alone: the
    # harness is registered against several binaries and only one holds any given
    # test, which is absence rather than staleness.
    for entry in known:
        if entry.name in shared and entry.name not in diverged:
            problems.append(
                "{} no longer diverges (both runs report {}) -- remove its line from "
                "{}".format(entry.name, describe(a[entry.name]), known_divergences)
            )

    # Two runs that skipped everything agree perfectly and prove nothing, exactly
    # like two runs that reported nothing at all. Both are the same failure --
    # coverage that looks present and never ran -- so both are rejected here.
    total_a, ran_a = counts(a)
    total_b, ran_b = counts(b)
    if not ran_a and not ran_b:
        problems.append(
            "neither run executed a test ({} reported by the disabled run, {} by the "
            "enabled run, all skipped) -- agreement between two runs that did nothing "
            "is not evidence of parity".format(total_a, total_b)
        )

    if problems:
        sys.stderr.write("forwarding parity failed:\n")
        for p in problems:
            sys.stderr.write("  {}\n".format(p))
        return 1

    # Said out loud, because a tolerated divergence is a gap that is still open. A run
    # carrying one that printed the ordinary success line would read as two modes that
    # agreed, which is the thing the list exists to stop being said.
    if tolerated:
        print(
            "forwarding parity: {} known divergence{} tolerated, listed in {}".format(
                len(tolerated), "" if len(tolerated) == 1 else "s", known_divergences
            )
        )
        for entry in tolerated:
            print("  ! {}".format(entry.name))
            print("      disabled={} enabled={}".format(entry.disabled, entry.enabled))
            print("      {}".format(entry.reason))
        print(
            "  The modes do not agree here. Nothing above is being checked for parity."
        )
        print(
            "forwarding parity OK apart from {} known divergence{}: {} tests compared "
            "under both modes ({} executed, {} skipped)".format(
                len(tolerated),
                "" if len(tolerated) == 1 else "s",
                total_a,
                ran_a,
                total_a - ran_a,
            )
        )
        return 0

    print(
        "forwarding parity OK: {} tests identical under both modes "
        "({} executed, {} skipped)".format(total_a, ran_a, total_a - ran_a)
    )
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
    parser.add_argument(
        "--known-divergences",
        metavar="PATH",
        help="list of divergences that are known and accepted; without it every "
        "divergence fails the comparison",
    )
    args = parser.parse_args()
    sys.exit(
        main(
            args.disabled_xml,
            args.enabled_xml,
            args.newer_than,
            args.known_divergences,
        )
    )
