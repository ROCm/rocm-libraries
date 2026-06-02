#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Build-filter completeness oracle.

Validates the smart-build selection (the "filter") against the build system's
ground truth: which test executables actually recompile when a source file
changes. The filter's job is to *build* the right tests, so this verifies the
build - the compiler's #include resolution is the authority on what depends on a
file.

Workflow (driven by an external sbatch wrapper that does the perturb+build):
  1. SEL(F)  = the filter's prediction: depmap[file] -> executables, intersected
               with the ctest-registered tests.
  2. TRUE(F) = ground truth: perturb F (e.g. append '#error'), run
               `ninja -k 0 <test targets>`, collect FAILED object files, and map
               them back to executables via build.ninja (same exe<-objects mapping
               the depmap uses, so the obj->exe step is a shared given here, not
               the thing under test).
  3. FN(F)   = TRUE(F) \\ SEL(F)  -> depends on F but the filter would skip = a
               real false negative.
     FP(F)   = SEL(F) \\ TRUE(F)  -> over-selection (safe).

This catches gaps beyond the smart-vs-legacy consistency check: clang -MM
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
    """Filter's prediction for a changed file: dependents intersect ctest tests."""
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


def reachable_exe_basenames(depmap):
    """Executables that at least one file maps to (can be selected by the filter)."""
    e2f = depmap.get("executable_to_files", {})
    return set(os.path.basename(e) for e in e2f)


def classify_unreachable(depmap, ctest_tests, compiled=None, allow=None):
    """Split unreachable ctest tests into false negatives vs non-compiled.

    A ctest test backed by a compiled bin/ target but absent from the depmap is a
    real false negative (extraction gap). A ctest test with NO compiled bin/
    target (python scripts, try_compile negative tests) can never be mapped by a
    compile-dependency analysis - it's an always-run class, not an FN.

    Returns (false_negatives, non_compiled). When `compiled` is None (no
    build.ninja provided) everything unreachable is treated as a potential FN.
    """
    reach = reachable_exe_basenames(depmap)
    allow = allow or set()
    fn, non_compiled = [], []
    for t in sorted(ctest_tests):
        if t in reach or t in allow:
            continue
        if compiled is not None and t not in compiled:
            non_compiled.append(t)
        else:
            fn.append(t)
    return fn, non_compiled


def _run_probe(args):
    for path, label in [
        (args.depmap, "--depmap"),
        (args.ninja, "--ninja"),
        (args.failed_objects, "--failed-objects"),
    ] + ([(args.ctest, "--ctest")] if args.ctest else []):
        if not os.path.exists(path):
            print(f"Error: missing required input ({label}): {path}", file=sys.stderr)
            return 2

    with open(args.depmap) as f:
        depmap = json.load(f)
    file_to_executables = depmap.get("file_to_executables", depmap)
    ctest_tests = load_ctest_tests(args.ctest) if args.ctest else None

    exe_to_objects = NinjaTargetParser(args.ninja).parse_executable_mappings()
    with open(args.failed_objects, errors="replace") as f:
        failed = parse_failed_objects(f.read())

    sel = sel_for_file(file_to_executables, ctest_tests, args.file)
    true_set = exes_for_objects(exe_to_objects, failed, ctest_tests)
    result = evaluate(args.file, sel, true_set)
    result["n_failed_objects"] = len(failed)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    print(f"=== build-filter oracle: {args.file} ===")
    print(f"failed objects:   {result['n_failed_objects']}")
    print(f"SEL (filter):     {result['n_sel']}")
    print(f"TRUE (rebuild):   {result['n_true']}")
    print(f"false negatives:  {result['n_fn']}")
    for e in result["false_negatives"]:
        print(f"  FN [X] {e}")
    print(f"false positives:  {result['n_fp']} (safe over-selection)")
    print(f"verdict: {result['verdict'].upper()}")
    return 0 if result["verdict"] == "pass" else 1


def _run_reachability(args):
    for path, label in [(args.depmap, "--depmap"), (args.ctest, "--ctest")]:
        if not os.path.exists(path):
            print(f"Error: missing required input ({label}): {path}", file=sys.stderr)
            return 2

    with open(args.depmap) as f:
        depmap = json.load(f)
    ctest_tests = load_ctest_tests(args.ctest)
    allow = set()
    if args.allowlist and os.path.exists(args.allowlist):
        with open(args.allowlist) as f:
            allow = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    # If build.ninja is given, classify: only tests backed by a compiled bin/
    # target can be FNs; tests with no bin/ target are non-compiled (always-run).
    compiled = None
    if args.ninja and os.path.exists(args.ninja):
        compiled = set(
            os.path.basename(e)
            for e in NinjaTargetParser(args.ninja).parse_executable_mappings()
        )
    fn, non_compiled = classify_unreachable(depmap, ctest_tests, compiled, allow)
    result = {
        "n_ctest": len(ctest_tests),
        "n_reachable": len(reachable_exe_basenames(depmap)),
        "n_false_negatives": len(fn),
        "false_negatives": fn,
        "n_non_compiled": len(non_compiled),
        "non_compiled": non_compiled,
        "allowlisted": sorted(allow),
        "classified": compiled is not None,
        "verdict": "pass" if not fn else "fail",
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    print("=== reachability guardrail ===")
    print(f"ctest tests:      {result['n_ctest']}")
    print(f"reachable exes:   {result['n_reachable']}")
    print(f"false negatives:  {result['n_false_negatives']} "
          f"(compiled test, unreachable -> filter would skip)")
    for t in fn:
        print(f"  [X] {t}")
    if compiled is not None:
        print(f"non-compiled:     {result['n_non_compiled']} "
              f"(no bin/ target: python/try_compile; always-run class)")
    if allow:
        print(f"allowlisted:      {len(allow)}")
    print(f"verdict: {result['verdict'].upper()}")
    return 0 if result["verdict"] == "pass" else 1


def main():
    p = argparse.ArgumentParser(description="Build-filter completeness oracle")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("probe", help="Per-file FN/FP via perturb+rebuild ground truth")
    pr.add_argument("--depmap", required=True, help="cmake_dependency_mapping.json")
    pr.add_argument("--ninja", required=True, help="build.ninja")
    pr.add_argument("--ctest", help="ctest -N output (optional intersection)")
    pr.add_argument("--file", required=True, help="changed file, depmap-relative key")
    pr.add_argument(
        "--failed-objects",
        required=True,
        help="file with ninja stderr (FAILED: lines) from the perturbed build",
    )
    pr.add_argument("--output", help="write per-probe result JSON here")

    rc = sub.add_parser(
        "reachability",
        help="Guardrail: fail if any ctest test is unreachable in the depmap (no build)",
    )
    rc.add_argument("--depmap", required=True, help="cmake_dependency_mapping.json")
    rc.add_argument("--ctest", required=True, help="ctest -N output")
    rc.add_argument(
        "--ninja",
        help="build.ninja: classify unreachable tests into compiled (FN) vs "
        "non-compiled (no bin/ target -> always-run, not FN)",
    )
    rc.add_argument("--allowlist", help="file of known-acceptable unreachable test names")
    rc.add_argument("--output", help="write result JSON here")

    args = p.parse_args()
    if args.command == "probe":
        sys.exit(_run_probe(args))
    elif args.command == "reachability":
        sys.exit(_run_reachability(args))


if __name__ == "__main__":
    main()
