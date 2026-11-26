#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check that a tier YAML sends every test of a GTest binary to exactly one tier.

The tier YAMLs (test_categories*.yaml) build one --gtest_filter per tier from name
patterns. A test that no tier selects never runs in any CTest suite, and a test that two
tiers select runs twice. Neither shows up as a failure, so this check lists the binary's
tests (--gtest_list_tests) and applies the YAML's patterns to them with GTest's own
matching rules (':'-free patterns, '*' wildcard, case-sensitive). It also applies the shared
parser's pattern rule (alphanumerics, '*', '.', '-', '_' and '/' only), so a pattern the
parser would reject, such as one with '?', fails here too.

It also checks that every test_patterns entry of the tiers named by --require-match
selects at least one test, which catches a renamed test or suite that an opt-in list
(the gpu-ref quick tier) still names.

--no-device-skips runs the named tiers' GTest filters with no visible device and fails if a
test skips for lack of one. It keeps the device tests out of a tier that has to run on a host
without a GPU (the unit-test quick tier).

Tiers are quick, standard, comprehensive and full; other categories (ffm-quick, ...) are
not tiers and are ignored.
"""

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# The tier YAMLs are validated by the shared parser at configure time; reuse its pattern rule.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared" / "ctest"))
from parse_test_categories import validate_gtest_pattern  # noqa: E402

TIERS = ("quick", "standard", "comprehensive", "full")

# A test that needs a device prints NO_DEVICE_MESSAGE and skips when HIP finds none.
NO_DEVICE_MESSAGE = "No devices available"
NO_DEVICE_ENVIRONMENT = {"HIP_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}
RUN_LINE = re.compile(r"^\[ RUN      \] (\S+)")


def list_tests(exe):
    result = subprocess.run(
        [exe, "--gtest_list_tests"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.exit(
            f"{exe} --gtest_list_tests exited with {result.returncode}:\n{result.stderr}"
        )
    names = []
    suite = None
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        body = line.split("#", 1)[0].rstrip()
        if not line.startswith(" "):
            if body.endswith("."):
                suite = body
            continue
        if suite is not None:
            names.append(suite + body.strip())
    return names


def matches(pattern, name):
    # '*' is the only wildcard the shared parser lets a pattern use; '?' and '[' stay literal.
    return fnmatch.fnmatchcase(name, pattern.replace("[", "[[]").replace("?", "[?]"))


def selects(category, name):
    return any(matches(p, name) for p in category.get("test_patterns", [])) and not any(
        matches(p, name) for p in category.get("exclude", [])
    )


def gtest_filter(category):
    """The --gtest_filter a tier's CTest suite uses: POSITIVE[-NEGATIVE]."""
    positive = ":".join(category.get("test_patterns", []))
    negative = ":".join(category.get("exclude", []))
    return f"{positive}-{negative}" if negative else positive


def tests_skipped_without_device(exe, filter_):
    """The tests of `filter_` that skip because no device is visible.

    The tests that need a device skip with NO_DEVICE_MESSAGE when HIP finds none, so a
    tier that has to run on a host without a GPU must select none of them.
    """
    env = {**os.environ, **NO_DEVICE_ENVIRONMENT}
    result = subprocess.run(
        [exe, f"--gtest_filter={filter_}"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if "[==========]" not in result.stdout:
        sys.exit(
            f"{exe} did not run the tests of the filter:\n{result.stdout}{result.stderr}"
        )
    skipped = set()
    current = None
    for line in result.stdout.splitlines():
        started = RUN_LINE.match(line)
        if started:
            current = started.group(1)
        elif NO_DEVICE_MESSAGE in line and current is not None:
            skipped.add(current)
    return sorted(skipped)


def check(yaml_path, exe, require_match, no_device_skips=()):
    with open(yaml_path, encoding="utf-8") as f:
        categories = yaml.safe_load(f)["test_categories"]
    tiers = {t: categories[t] for t in TIERS if t in categories}

    names = [n for n in list_tests(exe) if "DISABLED" not in n]
    if not names:
        sys.exit(
            f"{exe} listed no tests. On Windows the ROCm runtime directory must be on PATH."
        )

    # The same pattern rules as the shared parser that builds the CTest filters, so a pattern it
    # would reject at configure time (for example one with '?') fails here too.
    errors = []
    for category_name, category in categories.items():
        for key in ("test_patterns", "exclude"):
            for pattern in category.get(key, []) or []:
                problem = validate_gtest_pattern(pattern)
                if problem:
                    errors.append(f"{category_name} {key}: {problem}")

    for name in names:
        chosen = [t for t, c in tiers.items() if selects(c, name)]
        if len(chosen) != 1:
            errors.append(f"{name}: selected by {chosen or 'no tier'}")

    for tier in require_match:
        for pattern in tiers[tier].get("test_patterns", []):
            if not any(matches(pattern, n) for n in names):
                errors.append(f"{tier} test_patterns entry {pattern!r} selects no test")

    for tier in no_device_skips:
        for name in tests_skipped_without_device(exe, gtest_filter(tiers[tier])):
            errors.append(f"{name}: in {tier}, but skips when no device is visible")

    if errors:
        shown = errors[:50]
        sys.exit(
            f"{len(errors)} tier problem(s) in {yaml_path}:\n  "
            + "\n  ".join(shown)
            + ("\n  ..." if len(errors) > len(shown) else "")
        )
    counts = {t: sum(selects(c, n) for n in names) for t, c in tiers.items()}
    print(f"{len(names)} tests, each in exactly one tier: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--exe", required=True)
    parser.add_argument("--require-match", nargs="*", default=[], choices=TIERS)
    parser.add_argument("--no-device-skips", nargs="*", default=[], choices=TIERS)
    args = parser.parse_args()
    check(args.yaml, args.exe, args.require_match, args.no_device_skips)


if __name__ == "__main__":
    main()
