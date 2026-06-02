#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Build-filter completeness oracle.

Validates the smart-build selection (the "filter") against the build system's
ground truth: which test executables actually recompile when a source file
changes. We don't run tests - the filter's job is to *build* the right tests, and
the compiler's #include resolution is the authority on what depends on a file.

Workflow (driven by an external sbatch wrapper that does the perturb+build):
  1. SEL(F)  = the filter's prediction: depmap[file] -> executables, intersected
               with the ctest-registered tests.
  2. TRUE(F) = ground truth: perturb F (e.g. append '#error'), run
               `ninja -k 0 <test targets>`, collect FAILED object files, and map
               them back to executables via build.ninja (same exe<-objects mapping
               the depmap uses, so the obj->exe step is not what's under test).
  3. FN(F)   = TRUE(F) \\ SEL(F)  -> depends on F but the filter would skip = a
               real false negative.
     FP(F)   = SEL(F) \\ TRUE(F)  -> over-selection (safe).

This catches gaps the smart-vs-legacy consistency check cannot: clang -MM
extraction failures, build-time generated headers, and (when configured in)
separate-build components like experimental/rocm_ck.
"""

import argparse
import json
import os
import re
import sys

# Reuse existing parsers.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from cmake_dependency_analyzer import NinjaTargetParser  # noqa: E402
from validate_selection import load_ctest_tests  # noqa: E402


def sel_for_file(file_to_executables, ctest_tests, file_key):
    """Filter's prediction for a changed file: dependents ∩ ctest tests."""
    exes = set(file_to_executables.get(file_key, []))
    if ctest_tests is not None:
        exes = {e for e in exes if os.path.basename(e) in ctest_tests}
    return exes


def parse_failed_objects(ninja_stderr_text):
    """Extract object paths from `ninja` FAILED: lines.

    A failed compile prints `FAILED: <output> [<output> ...]`. We keep tokens
    that look like object outputs (end in .o).
    """
    failed = set()
    for line in ninja_stderr_text.splitlines():
        line = line.strip()
        if not line.startswith("FAILED:"):
            continue
        for tok in line[len("FAILED:"):].split():
            if tok.endswith(".o"):
                failed.add(tok)
    return failed


def exes_for_objects(exe_to_objects, failed_objects, ctest_tests=None):
    """Map failed object files back to the executables that include them."""
    failed = set(failed_objects)
    hit = set()
    for exe, objs in exe_to_objects.items():
        if failed.intersection(objs):
            if ctest_tests is None or os.path.basename(exe) in ctest_tests:
                hit.add(exe)
    return hit


def evaluate(file_key, sel, true_set):
    fn = sorted(true_set - sel)
    fp = sorted(sel - true_set)
    return {
        "file": file_key,
        "n_sel": len(sel),
        "n_true": len(true_set),
        "n_fn": len(fn),
        "n_fp": len(fp),
        "false_negatives": fn,
        "false_positives": fp,
        "verdict": "pass" if not fn else "fail",
    }


def main():
    p = argparse.ArgumentParser(description="Build-filter completeness oracle")
    p.add_argument("--depmap", required=True, help="cmake_dependency_mapping.json")
    p.add_argument("--ninja", required=True, help="build.ninja")
    p.add_argument("--ctest", help="ctest -N output (optional intersection)")
    p.add_argument("--file", required=True, help="changed file, depmap-relative key")
    p.add_argument(
        "--failed-objects",
        required=True,
        help="file containing ninja stderr (FAILED: lines) from the perturbed build",
    )
    p.add_argument("--output", help="write per-probe result JSON here")
    args = p.parse_args()

    depmap = json.load(open(args.depmap))
    file_to_executables = depmap.get("file_to_executables", depmap)
    ctest_tests = load_ctest_tests(args.ctest) if args.ctest else None

    exe_to_objects = NinjaTargetParser(args.ninja).parse_executable_mappings()
    failed = parse_failed_objects(open(args.failed_objects, errors="replace").read())

    sel = sel_for_file(file_to_executables, ctest_tests, args.file)
    true_set = exes_for_objects(exe_to_objects, failed, ctest_tests)
    result = evaluate(args.file, sel, true_set)
    result["n_failed_objects"] = len(failed)

    if args.output:
        json.dump(result, open(args.output, "w"), indent=2)

    print(f"=== build-filter oracle: {args.file} ===")
    print(f"failed objects:   {result['n_failed_objects']}")
    print(f"SEL (filter):     {result['n_sel']}")
    print(f"TRUE (rebuild):   {result['n_true']}")
    print(f"false negatives:  {result['n_fn']}")
    for e in result["false_negatives"]:
        print(f"  FN ✗ {e}")
    print(f"false positives:  {result['n_fp']} (safe over-selection)")
    print(f"verdict: {result['verdict'].upper()}")
    sys.exit(0 if result["verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
