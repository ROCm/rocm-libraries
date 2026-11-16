#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check the wrapper checks are registered for every architecture with an enabled suite.

    check_wrapper_registration.py --labels "<ex_gpu_a;ex_gpu_b>" \\
        [--ctest-dir <build dir>] [--packaged <install_CTestTestfile.cmake>]

wrapper_abi_check and forwarding_failure_modes are registered once bare and once per
ex_gpu_<arch> label, each with the tier labels plus the arch label, because TheRock's
GPU jobs select tests with `-L ^<tier>$ -L ^ex_gpu_<arch>$`. If the per-arch entries are
dropped, no other test fails, so this check is the only thing that notices.

--labels is what forwarding_parity_gpu_labels.py prints for test_categories.yaml: the
architectures that have an enabled suite, the list CMake registers from. A label whose
suites are all DISABLED is in neither, so it is not checked. It is passed in rather than
read from the parity entries, so emptying the list in CMake cannot shrink what is expected.

The tier labels come from the bare parity entry (miopen_gtest_forwarding_parity), which
separate CMake registers, so a check that loses them is caught too. That entry exists only
in a MIOPEN_TEST_DISCRETE=OFF configure, and in a build tree only if the configure finds a
GPU or gets a MIOPEN_TEST_GFX* option, so the check cannot run on a tree configured
otherwise. --ctest-dir reads the build tree through `ctest --show-only=json-v1`;
--packaged reads the shipped test list. Give either or both.

Also reported: an entry no given label asks for, and a packaged entry added more than once
(the repeat would run the check again, and a mislabelled copy before the right one would be
hidden).
"""

import argparse
import json
import re
import subprocess
import sys

CHECKS = ("wrapper_abi_check", "forwarding_failure_modes")
PARITY = "miopen_gtest_forwarding_parity"

# The harness's own label, on the bare parity entry only.
PARITY_ONLY_LABEL = "forwarding_parity"


def entries_from_ctest_json(text):
    """Return {entry name: set of labels} from `ctest --show-only=json-v1` output."""
    entries = {}
    for test in json.loads(text).get("tests", []):
        labels = set()
        for prop in test.get("properties", []):
            if prop.get("name") == "LABELS":
                labels = set(prop.get("value", []))
        entries[test["name"]] = labels
    return entries


# The packaged list writes each entry's properties on one line.
ADD_TEST = re.compile(r"^add_test\((\S+) ", re.MULTILINE)
PROPERTIES = re.compile(
    r'^set_tests_properties\((\S+) PROPERTIES .*?\bLABELS "([^"]*)"', re.MULTILINE
)


def entries_from_packaged(text):
    """Return {entry name: set of labels} for entries of a packaged test list that
    are both added and given labels."""
    added = set(ADD_TEST.findall(text))
    return {
        name: set(labels.split(";"))
        for name, labels in PROPERTIES.findall(text)
        if name in added
    }


def repeated_entries(text):
    """Return the wrapper check entries that a packaged test list adds more than once."""
    counts = {}
    for name in ADD_TEST.findall(text):
        if any(name == check or name.startswith(f"{check}_") for check in CHECKS):
            counts[name] = counts.get(name, 0) + 1
    return sorted(name for name, count in counts.items() if count > 1)


def problems(entries, gpu_labels, where):
    """Return what is wrong with how the wrapper checks are registered in `entries`."""
    if not gpu_labels:
        return [
            f"{where}: no ex_gpu_* labels were given, so there is nothing to check "
            "the per-architecture entries against"
        ]
    if PARITY not in entries:
        return [
            f"{where}: the bare parity entry {PARITY} is not registered. The tier labels "
            "are read from it, so the tree must come from a MIOPEN_TEST_DISCRETE=OFF "
            "configure, and a build tree also needs a GPU or a MIOPEN_TEST_GFX* option "
            "for the configure to register it"
        ]
    tiers = entries[PARITY] - {PARITY_ONLY_LABEL}

    found = []
    for check in CHECKS:
        for gpu_label in ("", *gpu_labels):
            name = f"{check}_{gpu_label}" if gpu_label else check
            expected = tiers | {gpu_label} if gpu_label else tiers
            if name not in entries:
                found.append(f"{where}: {name} is not registered")
            elif entries[name] != expected:
                found.append(
                    f"{where}: {name} has labels {sorted(entries[name])}, "
                    f"expected {sorted(expected)}"
                )
    expected_names = {
        f"{check}_{gpu_label}" if gpu_label else check
        for check in CHECKS
        for gpu_label in ("", *gpu_labels)
    }
    for name in sorted(entries):
        if name not in expected_names and any(
            name.startswith(f"{check}_") for check in CHECKS
        ):
            found.append(
                f"{where}: {name} is registered but no given label asks for it"
            )
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        required=True,
        help="the ex_gpu_* labels as a ;-separated list, as "
        "forwarding_parity_gpu_labels.py prints them",
    )
    parser.add_argument("--ctest-dir", help="a configured build tree")
    parser.add_argument("--packaged", help="the packaged install_CTestTestfile.cmake")
    args = parser.parse_args(argv)
    if not (args.ctest_dir or args.packaged):
        parser.error("give --ctest-dir, --packaged or both")

    gpu_labels = [label for label in args.labels.split(";") if label]
    found = []
    if args.ctest_dir:
        listing = subprocess.run(
            ["ctest", "--test-dir", args.ctest_dir, "--show-only=json-v1"],
            capture_output=True,
            text=True,
        )
        if listing.returncode != 0:
            sys.stderr.write(listing.stderr)
            return listing.returncode
        found += problems(
            entries_from_ctest_json(listing.stdout), gpu_labels, "build tree"
        )
    if args.packaged:
        with open(args.packaged, encoding="utf-8") as packaged:
            text = packaged.read()
        found += problems(entries_from_packaged(text), gpu_labels, "packaged list")
        found += [
            f"packaged list: {name} is added more than once"
            for name in repeated_entries(text)
        ]

    for line in found:
        print(line)
    if not found:
        print(
            f"wrapper checks registered for {len(gpu_labels)} architecture labels "
            f"and bare: {', '.join(gpu_labels)}"
        )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
