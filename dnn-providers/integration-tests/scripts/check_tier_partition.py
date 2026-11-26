#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check that a tier YAML sends every test of a GTest binary to exactly one tier.

The tier YAMLs (test_categories*.yaml) build one --gtest_filter per tier from name
patterns. A test that no tier selects never runs in any CTest suite, and a test that two
tiers select runs twice. Neither shows up as a failure, so this check lists the binary's
tests (--gtest_list_tests) and applies the YAML's patterns to them with GTest's own
matching rules (':'-free patterns, '*' and '?' wildcards, case-sensitive).

It also checks that every test_patterns entry of the tiers named by --require-match
selects at least one test, which catches a renamed test or suite that an opt-in list
(the gpu-ref quick tier) still names.

Tiers are quick, standard, comprehensive and full; other categories (ffm-quick, ...) are
not tiers and are ignored.
"""

import argparse
import fnmatch
import subprocess
import sys

import yaml

TIERS = ("quick", "standard", "comprehensive", "full")


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
    # GTest wildcards are '*' and '?'; '[' and ']' are literal characters in a filter.
    return fnmatch.fnmatchcase(name, pattern.replace("[", "[[]"))


def selects(category, name):
    return any(matches(p, name) for p in category.get("test_patterns", [])) and not any(
        matches(p, name) for p in category.get("exclude", [])
    )


def check(yaml_path, exe, require_match):
    with open(yaml_path, encoding="utf-8") as f:
        categories = yaml.safe_load(f)["test_categories"]
    tiers = {t: categories[t] for t in TIERS if t in categories}

    names = [n for n in list_tests(exe) if "DISABLED" not in n]
    if not names:
        sys.exit(
            f"{exe} listed no tests. On Windows the ROCm runtime directory must be on PATH."
        )

    errors = []
    for name in names:
        chosen = [t for t, c in tiers.items() if selects(c, name)]
        if len(chosen) != 1:
            errors.append(f"{name}: selected by {chosen or 'no tier'}")

    for tier in require_match:
        for pattern in tiers[tier].get("test_patterns", []):
            if not any(matches(pattern, n) for n in names):
                errors.append(f"{tier} test_patterns entry {pattern!r} selects no test")

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
    args = parser.parse_args()
    check(args.yaml, args.exe, args.require_match)


if __name__ == "__main__":
    main()
