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

--no-device-skips runs the named tiers' GTest filters with no visible device. It fails when the
run exits non-zero (a crash or a failing test) or when a test skips for lack of a device, so it
keeps the device tests out of a tier that has to run on a host without a GPU (the unit-test
quick tier). As a control it also runs the first tier that was not named, still with no device,
and fails when no test skips: the device is then not hidden here (the variables are ignored),
and the check would pass without checking anything.

Tiers are quick, standard, comprehensive and full. Other categories (ffm-quick, ...) are not
tiers: their patterns are validated, but they are not partitioned.
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
# GTest's closing banner ("[==========] N tests from M test suites ran."), not the opening one.
COMPLETED_LINE = re.compile(
    r"^\[==========\] \d+ tests? from \d+ test suites? ran\.", re.M
)


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


def run_without_device(exe, filter_):
    env = {**os.environ, **NO_DEVICE_ENVIRONMENT}
    return subprocess.run(
        [exe, f"--gtest_filter={filter_}"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def device_skips(output):
    """The tests of a GTest run's output that skipped with NO_DEVICE_MESSAGE."""
    skipped = set()
    current = None
    for line in output.splitlines():
        started = RUN_LINE.match(line)
        if started:
            current = started.group(1)
        elif NO_DEVICE_MESSAGE in line and current is not None:
            skipped.add(current)
    return sorted(skipped)


def tests_skipped_without_device(exe, filter_):
    """The tests of `filter_` that skip because no device is visible.

    The tests that need a device skip with NO_DEVICE_MESSAGE when HIP finds none, so a
    tier that has to run on a host without a GPU must select none of them. The run itself
    must pass: a crash or a failing test would otherwise hide the tests that did not run.
    """
    result = run_without_device(exe, filter_)
    if result.returncode != 0 or not COMPLETED_LINE.search(result.stdout):
        sys.exit(
            f"{exe} --gtest_filter={filter_} with no visible device exited with "
            f"{result.returncode}; a tier that runs without a device must pass:\n"
            f"{result.stdout[-2000:]}{result.stderr[-2000:]}"
        )
    return device_skips(result.stdout)


def device_is_hidden(exe, filter_):
    """True when a run of `filter_` skips at least one test for lack of a device.

    The filter is one that selects device tests. The exit status is not read: only the
    skips matter, and the tier may hold tests that fail without their data.
    """
    return bool(device_skips(run_without_device(exe, filter_).stdout))


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
    control = next((t for t in tiers if t not in no_device_skips), None)
    if no_device_skips and control is not None:
        if not device_is_hidden(exe, gtest_filter(tiers[control])):
            errors.append(
                f"no test of the {control} tier skipped with HIP_VISIBLE_DEVICES and "
                "ROCR_VISIBLE_DEVICES set to -1: the device is not hidden here, so the "
                "no-device check proves nothing"
            )

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
    parser.add_argument("--yaml", required=True, help="the tier YAML to check")
    parser.add_argument("--exe", required=True, help="the GTest binary the YAML tiers")
    parser.add_argument(
        "--require-match",
        nargs="*",
        default=[],
        choices=TIERS,
        help="tiers whose test_patterns entries must each select at least one test",
    )
    parser.add_argument(
        "--no-device-skips",
        nargs="*",
        default=[],
        choices=TIERS,
        help="tiers that must pass with no visible device and skip no test for lack of one",
    )
    args = parser.parse_args()
    check(args.yaml, args.exe, args.require_match, args.no_device_skips)


if __name__ == "__main__":
    main()
