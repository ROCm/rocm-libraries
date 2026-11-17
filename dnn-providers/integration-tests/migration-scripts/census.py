#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Census of the integration-test suite (ALMIOPEN-2221, migration phase 1).

Runs the integration-test binary's `--gtest_list_tests` and parses the full
parametrized expansion into a manifest. This is the *denominator* for the C++
graph -> bundle migration: every C++ graph test that must be accounted for, with
the seed and parameter string GTest reports for it.

Why the binary, not the source: GTest's own enumeration expands every
testing::Combine / ValuesIn / typed instantiation into the real set of
Suite.Test names — the same expansion used at run time, and the only source of
truth for "how many tests are there." The ~17 .cpp files expand into thousands
of cases.

The manifest classifies each case:

  * graph   — a C++ graph test (suite under a Smoke/ or Full/ instantiation,
              name starts with IntegrationGpu...). Its graph is built
              programmatically in C++; these are what the capture step turns
              into bundles. Their GetParam() carries the seed inline.
  * bundle  — an existing on-disk bundle test (GetParam() is a path to a
              .json bundle). Already in the target format; nothing to migrate.
  * other   — anything else (e.g. perf microbenchmarks). Reported so the count
              reconciles exactly and nothing is silently uncategorized.

Output is JSON on stdout (or --out FILE), plus a human summary on stderr.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict


# A GetParam() value that is a quoted path to a .json file marks an existing
# bundle test (the param IS the bundle on disk). Anything else is an inline-built
# C++ graph param (dims/seed/layout) or a non-parametrized case. Matched against
# the param text after the "GetParam() = " prefix has been stripped.
_BUNDLE_PARAM_RE = re.compile(r'^"(?P<path>.*\.json)"$')
_SEED_RE = re.compile(r"seed:(?P<seed>\d+)")


@dataclass
class Case:
    suite: str  # full gtest suite name, e.g. "Smoke/IntegrationGpuConvForwardFp16"
    name: str  # case name, e.g. "Correctness/0"
    full: str  # "{suite}.{name}" — the gtest identifier
    kind: str  # "graph" | "bundle" | "other"
    seed: int | None = None  # parsed from GetParam() for graph cases
    param: str = ""  # raw GetParam() text (after the '= '), if any


@dataclass
class Census:
    total: int = 0
    counts: dict = field(default_factory=dict)
    cases: list = field(default_factory=list)


def classify(suite: str, param: str) -> str:
    """Bucket a case from its suite name and raw GetParam() text."""
    if _BUNDLE_PARAM_RE.search(param):
        return "bundle"
    # Migration target: IntegrationGpu* suites build a graph and verify it
    # across GPU plugins — these are the tests whose graphs can be bundled
    # and re-executed with any plugin.
    leaf = suite.split("/", 1)[-1]
    if leaf.startswith("IntegrationGpu"):
        return "graph"
    return "other"


def parse_list_tests(text: str) -> Census:
    """Parse `gtest --gtest_list_tests` output into a Census.

    Format (two indentation levels):
        SuiteName.
          CaseName  # GetParam() = <param>
          CaseName
        OtherSuite.
          ...
    A line with no leading space that ends in '.' is a suite header; an indented
    line is a case under the most recent suite.
    """
    census = Census()
    suite = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if not raw.startswith(" "):
            # suite header line, e.g. "Smoke/IntegrationGpuConvForwardFp16."
            suite = raw.strip().rstrip(".")
            continue
        if suite is None:
            continue  # defensive: indented line before any suite header
        line = raw.strip()
        param = ""
        if "#" in line:
            case_name, _, comment = line.partition("#")
            case_name = case_name.strip()
            m = re.search(r"GetParam\(\)\s*=\s*(.*)$", comment.strip())
            param = m.group(1).strip() if m else ""
        else:
            case_name = line
        kind = classify(suite, param)
        seed = None
        if kind == "graph":
            sm = _SEED_RE.search(param)
            seed = int(sm.group("seed")) if sm else None
        census.cases.append(
            Case(
                suite=suite,
                name=case_name,
                full=f"{suite}.{case_name}",
                kind=kind,
                seed=seed,
                param=param,
            )
        )
    census.total = len(census.cases)
    for c in census.cases:
        census.counts[c.kind] = census.counts.get(c.kind, 0) + 1
    return census


def run_list_tests(binary: str) -> str:
    result = subprocess.run(
        [binary, "--gtest_list_tests"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "binary",
        help="path to the built hipdnn_integration_tests executable",
    )
    ap.add_argument(
        "--out",
        help="write the JSON manifest here (default: stdout)",
    )
    ap.add_argument(
        "--graph-only",
        action="store_true",
        help="emit only graph-kind cases (the migration denominator)",
    )
    args = ap.parse_args()

    try:
        text = run_list_tests(args.binary)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"census: failed to run {args.binary}: {exc}", file=sys.stderr)
        return 1

    census = parse_list_tests(text)

    # Flag graph cases missing a seed: the capture step relies on a known seed,
    # so a graph case without one is something to look at (none expected today).
    seedless = [c.full for c in census.cases if c.kind == "graph" and c.seed is None]

    print("== integration-test census ==", file=sys.stderr)
    print(f"  total cases: {census.total}", file=sys.stderr)
    for kind in sorted(census.counts):
        print(f"  {kind:7s}: {census.counts[kind]}", file=sys.stderr)
    suites = sorted({c.suite for c in census.cases if c.kind == "graph"})
    print(f"  graph suites to migrate: {len(suites)}", file=sys.stderr)
    if seedless:
        print(
            f"  WARNING: {len(seedless)} graph cases have no seed in GetParam():",
            file=sys.stderr,
        )
        for full in seedless[:10]:
            print(f"    {full}", file=sys.stderr)

    cases = census.cases
    if args.graph_only:
        cases = [c for c in cases if c.kind == "graph"]

    manifest = {
        "total": census.total,
        "counts": census.counts,
        "graph_suite_count": len(suites),
        "seedless_graph_cases": seedless,
        "cases": [asdict(c) for c in cases],
    }
    out_text = json.dumps(manifest, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out_text + "\n")
        print(f"  manifest written: {args.out}", file=sys.stderr)
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
