#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Layer 4 — differential coverage check.

Parses GTest JSON results from two runs:
  1. C++ integration suite  (--cpp)
  2. Bundle integration suite (--bundle)

Joins on ``reference_source`` metadata, then asserts::

    pass_set_bundle  ⊇  pass_set_cpp

Any case that PASSED as a C++ test but is SKIPPED or absent as a bundle is
a coverage regression: listed by name, non-zero exit.

This is the machine-checkable proof that turning off C++ integration tests
loses no coverage for graph+GPU-plugin tests.

Requires a GPU host (both runs exercise GPU-plugin execution).

Usage::

    diff_coverage.py --cpp /tmp/cpp.json --bundle /tmp/bundle.json \\
        [--bundle-dir integration-test-bundles/]
"""

import argparse
import json
import sys
from pathlib import Path


def _parse_gtest_json(path: Path) -> dict:
    """Parse GTest --gtest_output=json and return {test_name: status}.

    Status is 'PASS', 'FAIL', 'SKIP', or 'DISABLED'.
    GTest JSON schema: testsuites[].testsuite[].{name, status, result, ...}
    """
    with open(path) as f:
        data = json.load(f)

    results = {}
    for suite in data.get("testsuites", []):
        suite_name = suite.get("name", "")
        for case in suite.get("testsuite", []):
            case_name = case.get("name", "")
            full = f"{suite_name}.{case_name}"
            status = case.get("status", "")
            result = case.get("result", "")
            if status == "NOTRUN" or result in ("SUPPRESSED", "SKIPPED"):
                results[full] = "SKIP"
            elif case.get("failures"):
                results[full] = "FAIL"
            else:
                results[full] = "PASS"
    return results


def _sweep_suite_name(sweep_path: Path, bundle_dir: Path) -> str:
    """Derive the gtest suite name for a sweep from its path.

    Bundle tests register as ``<tier>_<Operation>_<Variant>`` (e.g.
    ``full_ConvolutionFwdPointwise_Default``), matching the
    ``<tier>/<Operation>/<Variant>/sweep.json`` layout on disk. The full gtest
    name of a case is then ``<suite>.<case_id>``.
    """
    rel = sweep_path.relative_to(bundle_dir).parts
    # rel == (tier, Operation, Variant, "sweep.json")
    if len(rel) >= 3:
        tier, operation, variant = rel[0], rel[1], rel[2]
        return f"{tier}_{operation}_{variant}"
    return ""


def _build_reference_source_map(bundle_dir: Path) -> dict:
    """Return {reference_source_key: full_test_name} by scanning sweep.json files.

    The value is the *fully-qualified* gtest name ``<suite>.<case_id>``, not the
    bare case id. Bare case ids collide across sibling suites (e.g. the same
    shape appears in both ``full_Batchnorm_Default`` and
    ``full_Batchnorm_Variant2``), so joining on case id alone lets a PASS in one
    suite falsely satisfy the coverage requirement for a C++ test whose bundle
    lives in — and may be SKIPPED in — the other. Qualifying by suite makes the
    join unambiguous.
    """
    mapping = {}
    if not bundle_dir or not bundle_dir.is_dir():
        return mapping
    for sweep_path in sorted(bundle_dir.rglob("sweep.json")):
        try:
            with open(sweep_path) as f:
                sweep = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        suite = _sweep_suite_name(sweep_path, bundle_dir)
        for case in sweep.get("cases", []):
            meta = case.get("metadata", {})
            key = meta.get("reference_source")
            cid = case.get("id", "")
            if key and suite and cid:
                mapping[key] = f"{suite}.{cid}"
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cpp", type=Path, required=True, help="GTest JSON from C++ suite run"
    )
    ap.add_argument(
        "--bundle", type=Path, required=True, help="GTest JSON from bundle suite run"
    )
    ap.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="bundle tree (for reference_source lookup, optional)",
    )
    args = ap.parse_args()

    cpp_results = _parse_gtest_json(args.cpp)
    bundle_results = _parse_gtest_json(args.bundle)

    cpp_pass = {k for k, v in cpp_results.items() if v == "PASS"}
    bundle_pass = {k for k, v in bundle_results.items() if v == "PASS"}

    print(f"  C++ total:    {len(cpp_results)}", file=sys.stderr)
    print(f"  C++ PASS:     {len(cpp_pass)}", file=sys.stderr)
    print(f"  bundle total: {len(bundle_results)}", file=sys.stderr)
    print(f"  bundle PASS:  {len(bundle_pass)}", file=sys.stderr)

    ported_map = _build_reference_source_map(args.bundle_dir) if args.bundle_dir else {}

    # The join key mismatch to bridge: gtest reports parametrized cases as
    # "Suite.Correctness/0" (slash), but reference_source stores the *sanitized*
    # case name "...Correctness_0" (underscore, as it appears on disk). Normalize
    # both sides — strip the "c++ integration suite: " prefix and unify slashes
    # to underscores — so the reverse lookup actually connects.
    def _norm(name: str) -> str:
        n = name.split("c++ integration suite:")[-1].strip()
        return n.replace("/", "_")

    # normalized C++ name -> fully-qualified bundle test name "<suite>.<case_id>"
    # (from reference_source metadata). Qualifying by suite is what makes the join
    # unambiguous — see _build_reference_source_map for why bare case ids collide.
    ported_by_norm = {_norm(key): full for key, full in ported_map.items()}

    regressions = []
    matched = 0

    for cpp_test in sorted(cpp_pass):
        if cpp_test in bundle_pass:
            matched += 1
            continue

        bundle_full = ported_by_norm.get(_norm(cpp_test))
        if bundle_full and bundle_full in bundle_pass:
            matched += 1
            continue

        bundle_status = bundle_results.get(cpp_test, "ABSENT")
        regressions.append((cpp_test, bundle_status))

    print(f"\n== diff_coverage ==", file=sys.stderr)
    print(f"  matched:      {matched}", file=sys.stderr)
    print(f"  regressions:  {len(regressions)}", file=sys.stderr)

    if regressions:
        print(
            f"\n  FAIL: {len(regressions)} C++ PASS tests not covered by bundles:",
            file=sys.stderr,
        )
        for test, status in regressions:
            print(f"    {test}  (bundle: {status})", file=sys.stderr)
        return 1

    print(f"\n  OK: pass_set_bundle ⊇ pass_set_cpp ({matched} tests)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
