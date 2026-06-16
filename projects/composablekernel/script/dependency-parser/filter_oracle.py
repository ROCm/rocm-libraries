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
import fnmatch
import json
import os
import re
import sys

# Reuse existing parsers.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from cmake_dependency_analyzer import NinjaTargetParser  # noqa: E402
from validate_selection import load_ctest_tests  # noqa: E402


def load_codegen_globs(inventory_path):
    """Return the flat list of test_globs from a codegen_blindspots.json inventory."""
    with open(inventory_path) as f:
        data = json.load(f)
    globs = []
    for gen in data.get("generators", []):
        globs.extend(gen.get("test_globs", []))
    return globs


def expand_test_globs(globs, ctest_tests):
    """Match shell-style globs against ctest test names -> sorted list of names."""
    return sorted(
        name for name in ctest_tests
        if any(fnmatch.fnmatch(name, pat) for pat in globs)
    )


def sel_for_file(file_to_executables, ctest_tests, file_key):
    """Filter's prediction for a changed file: dependents intersect ctest tests."""
    exes = set(file_to_executables.get(file_key, []))
    if ctest_tests is not None:
        exes = {e for e in exes if os.path.basename(e) in ctest_tests}
    return exes


def compute_coverage(pre_f2e, post_f2e, ctest_tests=None):
    """Diff a pre-build depmap against the post-build ground-truth depmap.

    post_f2e is the real build's file->executables (from `ninja -t deps`); pre_f2e
    is the smart-build prediction (from cmake-parse). For each file, edges the real
    build proves but the prediction lacks (post[f] minus pre[f]) are false-negative
    candidates. With ctest_tests, only edges to registered tests count. Both maps
    must be keyed with the same workspace-root so paths match.
    """
    def _ctest_only(exes):
        if ctest_tests is None:
            return set(exes)
        return {e for e in exes if os.path.basename(e) in ctest_tests}

    false_negatives = {}
    n_edges_post = 0
    n_edges_covered = 0
    # Edge-level coverage is header-weighted (a file->test edge per included
    # header), so also track file-level (does a file resolve to ALL its tests?)
    # and test-level (are ALL of a test's source deps captured?) - the latter two
    # bracket the run-accuracy answer the edge metric overstates.
    n_files_with_edges = 0
    n_files_covered = 0
    tests_post = set()
    tests_with_fn = set()
    for f, post_exes in post_f2e.items():
        post_set = _ctest_only(post_exes)
        if not post_set:
            continue
        n_files_with_edges += 1
        pre_set = set(pre_f2e.get(f, []))
        missing = post_set - pre_set
        n_edges_post += len(post_set)
        n_edges_covered += len(post_set & pre_set)
        tests_post |= {os.path.basename(e) for e in post_set}
        if missing:
            false_negatives[f] = sorted(missing)
            tests_with_fn |= {os.path.basename(e) for e in missing}
        else:
            n_files_covered += 1

    def _frac(num, den):
        return round(num / den, 6) if den else 1.0

    n_tests = len(tests_post)
    n_tests_covered = n_tests - len(tests_with_fn)
    return {
        "n_files": len(post_f2e),
        # edge-level (header-weighted; the optimistic bound)
        "n_edges_post": n_edges_post,
        "n_edges_covered": n_edges_covered,
        "coverage": _frac(n_edges_covered, n_edges_post),
        # file-level: of source files with tests, how many resolve to all of them
        "n_files_with_edges": n_files_with_edges,
        "n_files_covered": n_files_covered,
        "file_coverage": _frac(n_files_covered, n_files_with_edges),
        # test-level: of tests, how many have every source dep captured (pessimistic)
        "n_tests": n_tests,
        "n_tests_covered": n_tests_covered,
        "test_coverage": _frac(n_tests_covered, n_tests),
        "tests_with_fn": sorted(tests_with_fn),
        "n_false_negatives": sum(len(v) for v in false_negatives.values()),
        "n_files_with_fn": len(false_negatives),
        "false_negatives": dict(sorted(false_negatives.items())),
        "verdict": "pass" if not false_negatives else "fail",
    }


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


def render_junit_reachability(result):
    """Render a JUnit XML report for the reachability guardrail result.

    Mirrors the validate_selection pattern: mode/label tagging on suite/classname
    so per-arch and per-mode publishes land as distinct rows in Jenkins. Pass =
    one testcase; fail = one testcase per false-negative test. A <properties>
    block carries the numeric counts so the detail page is self-contained.
    """
    from xml.sax.saxutils import escape

    mode = result.get("mode")
    label = result.get("label")
    tag = (f".{mode}" if mode else "") + (f".{label}" if label else "")
    suite_tag = (f"-{mode}" if mode else "") + (f"-{label}" if label else "")
    classname = "smart-build.reachability" + tag
    suite = "smart-build-reachability" + suite_tag

    _attr = {"'": "&apos;", '"': "&quot;"}
    fn = result.get("false_negatives", [])
    cases = []
    if fn:
        for name in fn:
            cases.append(
                f'    <testcase classname="{classname}" '
                f'name="{escape(name, _attr)}">\n'
                f'      <failure message="compiled test unreachable in depmap"/>\n'
                f"    </testcase>"
            )
    else:
        case_name = "all-compiled-tests-reachable" + (f" ({label})" if label else "")
        cases.append(
            f'    <testcase classname="{classname}" '
            f'name="{escape(case_name, _attr)}"/>'
        )

    props = [
        ("n_ctest", result.get("n_ctest", 0)),
        ("n_reachable", result.get("n_reachable", 0)),
        ("n_false_negatives", result.get("n_false_negatives", 0)),
        ("n_non_compiled", result.get("n_non_compiled", 0)),
        ("n_codegen_allowlisted", result.get("n_codegen_allowlisted", 0)),
        ("classified", str(result.get("classified", False)).lower()),
    ]
    if mode is not None:
        props.append(("advisory", str(mode != "selective").lower()))
    props_xml = "\n".join(
        f'    <property name="{k}" value="{v}"/>' for k, v in props
    )

    n_failures = len(fn)
    body = "\n".join(cases)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite name="{suite}" tests="{max(len(cases), 1)}" '
        f'failures="{n_failures}">\n'
        f"  <properties>\n{props_xml}\n  </properties>\n"
        f"{body}\n"
        "</testsuite>\n"
    )


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
    # Codegen-driven tests (from the codegen_blindspots.json inventory) look
    # unreachable in a pre-build depmap because their sources are generated at
    # build time. Mark them as a known codegen class so they don't read as FNs;
    # generator-input changes are handled separately by ci_safety_check.sh.
    codegen_allow = []
    if args.codegen_inventory and os.path.exists(args.codegen_inventory):
        codegen_allow = expand_test_globs(
            load_codegen_globs(args.codegen_inventory), ctest_tests
        )
    suppress = allow | set(codegen_allow)
    # If build.ninja is given, classify: only tests backed by a compiled bin/
    # target can be FNs; tests with no bin/ target are non-compiled (always-run).
    compiled = None
    if args.ninja and os.path.exists(args.ninja):
        compiled = set(
            os.path.basename(e)
            for e in NinjaTargetParser(args.ninja).parse_executable_mappings()
        )
    fn, non_compiled = classify_unreachable(depmap, ctest_tests, compiled, suppress)
    result = {
        "n_ctest": len(ctest_tests),
        "n_reachable": len(reachable_exe_basenames(depmap)),
        "n_false_negatives": len(fn),
        "false_negatives": fn,
        "n_non_compiled": len(non_compiled),
        "non_compiled": non_compiled,
        "allowlisted": sorted(allow),
        "n_codegen_allowlisted": len(codegen_allow),
        "codegen_allowlisted": codegen_allow,
        "classified": compiled is not None,
        "verdict": "pass" if not fn else "fail",
    }
    if getattr(args, "mode", None):
        result["mode"] = args.mode
    if getattr(args, "label", None):
        result["label"] = args.label
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    if getattr(args, "junit", None):
        with open(args.junit, "w") as f:
            f.write(render_junit_reachability(result))

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
    if codegen_allow:
        print(f"codegen class:    {len(codegen_allow)} "
              f"(generated at build time; tracked via codegen_blindspots.json)")
    print(f"verdict: {result['verdict'].upper()}")
    return 0 if result["verdict"] == "pass" else 1


def _run_codegen_allowlist(args):
    for path, label in [(args.inventory, "--inventory"), (args.ctest, "--ctest")]:
        if not os.path.exists(path):
            print(f"Error: missing required input ({label}): {path}", file=sys.stderr)
            return 2
    ctest_tests = load_ctest_tests(args.ctest)
    names = expand_test_globs(load_codegen_globs(args.inventory), ctest_tests)
    text = "".join(f"{n}\n" for n in names)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    print(f"codegen-driven tests matched: {len(names)}", file=sys.stderr)
    return 0


def _load_f2e(path):
    """Load a depmap JSON -> its file_to_executables dict."""
    with open(path) as f:
        data = json.load(f)
    return data.get("file_to_executables", data)


_MONOREPO_PREFIX = re.compile(r"^projects/[^/]+/")


def _canonical_key(key):
    """Drop a leading monorepo 'projects/<project>/' segment.

    cmake-parse keys are relative to its --workspace-root (repo root in CI ->
    'projects/composablekernel/...'), while enhanced_ninja_parser always truncates
    post-build keys to the project root ('include/...'). Canonicalizing both sides
    to the project root makes the pre/post diff valid regardless of the root each
    was generated with.
    """
    return _MONOREPO_PREFIX.sub("", key)


def _is_source_key(key):
    """True for project source a PR can edit.

    Excludes build outputs - generated sources and vendored deps (gtest, under
    build/_deps) live under 'build/'; system headers are absolute ('/usr', '/opt').
    The pre-build depmap never tracks those (clang -MM drops -isystem includes and
    is_project_file filters system paths), so counting them as misses would
    understate coverage. Generated-source changes are covered by the codegen
    backstop (ci_safety_check), not the depmap.
    """
    return not key.startswith("build/") and not key.startswith("/")


def _canon_f2e(f2e, source_only=False):
    """Canonicalize keys (and optionally drop non-source keys), merging collisions."""
    out = {}
    for k, v in f2e.items():
        ck = _canonical_key(k)
        if source_only and not _is_source_key(ck):
            continue
        out[ck] = sorted(set(out[ck]) | set(v)) if ck in out else list(v)
    return out


def _run_coverage(args):
    for path, label in [(args.pre, "--pre"), (args.post, "--post")]:
        if not os.path.exists(path):
            print(f"Error: missing required input ({label}): {path}", file=sys.stderr)
            return 2
    # Canonicalize both sides to the project root so a repo-root pre depmap and a
    # project-root post depmap compare correctly. Scope post to PR-editable source
    # unless --include-nonsource is given.
    source_only = not args.include_nonsource
    pre = _canon_f2e(_load_f2e(args.pre), source_only=source_only)
    post = _canon_f2e(_load_f2e(args.post), source_only=source_only)
    ctest_tests = load_ctest_tests(args.ctest) if args.ctest else None

    result = compute_coverage(pre, post, ctest_tests)
    result["scope"] = "source" if source_only else "all"
    if args.label:
        result["label"] = args.label

    # Backstop-credited view: exclude the codegen-class tests (their generated
    # sources are owned by the ci_safety_check full-build backstop, not the
    # depmap). Reported alongside the raw numbers, not instead of them.
    codegen = []
    if args.codegen_inventory and os.path.exists(args.codegen_inventory) and ctest_tests:
        codegen = expand_test_globs(
            load_codegen_globs(args.codegen_inventory), ctest_tests
        )
        credited = compute_coverage(pre, post, set(ctest_tests) - set(codegen))
        result["n_codegen_tests"] = len(codegen)
        result["codegen_credited"] = {
            k: credited[k] for k in (
                "coverage", "file_coverage", "test_coverage",
                "n_tests", "n_tests_covered", "n_files_covered",
                "n_files_with_edges", "n_false_negatives",
            )
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    print("=== filter coverage (pre-build depmap vs post-build ground truth) ===")
    print(f"scope:        {result['scope']} (PR-editable source"
          f"{'' if source_only else ' + build/system'})")
    print(f"edge coverage: {result['n_edges_covered']}/{result['n_edges_post']} "
          f"= {result['coverage']:.4f}  (header-weighted; optimistic bound)")
    print(f"file coverage: {result['n_files_covered']}/{result['n_files_with_edges']} "
          f"= {result['file_coverage']:.4f}  (files resolving to all their tests)")
    print(f"test coverage: {result['n_tests_covered']}/{result['n_tests']} "
          f"= {result['test_coverage']:.4f}  (tests with every source dep captured)")
    if "codegen_credited" in result:
        c = result["codegen_credited"]
        print(f"  [codegen-credited, {result['n_codegen_tests']} codegen tests excluded] "
              f"file={c['file_coverage']:.4f} test={c['test_coverage']:.4f} "
              f"FN={c['n_false_negatives']}")
    print(f"false negatives:  {result['n_false_negatives']} "
          f"(across {result['n_files_with_fn']} files; "
          f"{len(result['tests_with_fn'])} tests)")
    for f, exes in list(result["false_negatives"].items())[:20]:
        print(f"  FN {f} -> {', '.join(exes)}")
    print(f"verdict: {result['verdict'].upper()}")
    return 0 if result["verdict"] == "pass" else 1


def aggregate_coverage(results):
    """Merge per-arch coverage results into one cross-arch view.

    A test/edge missed on ANY arch is a real blind spot, so false negatives are
    unioned; the headline coverages are the worst (min) across arches. `results`
    is a list of per-arch coverage dicts (each tagged with `label`).
    """
    if not results:
        raise ValueError("aggregate_coverage requires at least one result")
    per_arch, union_fn, union_tests_fn = [], {}, set()
    worst = {"coverage": 1.0, "file_coverage": 1.0, "test_coverage": 1.0}
    any_fail = False
    for r in results:
        per_arch.append({
            "label": r.get("label", "?"),
            "coverage": r.get("coverage"),
            "file_coverage": r.get("file_coverage"),
            "test_coverage": r.get("test_coverage"),
            "n_false_negatives": r.get("n_false_negatives", 0),
            "verdict": r.get("verdict"),
        })
        for f, tests in (r.get("false_negatives") or {}).items():
            union_fn.setdefault(f, set()).update(tests)
        union_tests_fn.update(r.get("tests_with_fn") or [])
        for k in worst:
            v = r.get(k)
            if v is not None:
                worst[k] = min(worst[k], v)
        any_fail = any_fail or r.get("verdict") == "fail"
    return {
        "n_arches": len(results),
        "per_arch": per_arch,
        "worst_edge_coverage": worst["coverage"],
        "worst_file_coverage": worst["file_coverage"],
        "worst_test_coverage": worst["test_coverage"],
        "false_negatives": {f: sorted(t) for f, t in sorted(union_fn.items())},
        "n_false_negatives": sum(len(t) for t in union_fn.values()),
        "n_files_with_fn": len(union_fn),
        "tests_with_fn": sorted(union_tests_fn),
        "n_tests_with_fn": len(union_tests_fn),
        "verdict": "fail" if any_fail else "pass",
    }


def _run_coverage_aggregate(args):
    results = []
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"Warning: skipping missing input: {path}", file=sys.stderr)
            continue
        with open(path) as f:
            results.append(json.load(f))
    if not results:
        print("Error: no coverage result inputs found", file=sys.stderr)
        return 2
    agg = aggregate_coverage(results)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(agg, f, indent=2)
    print("=== filter coverage (aggregate across %d arch(es)) ===" % agg["n_arches"])
    for a in agg["per_arch"]:
        print("  %-12s edge=%.4f file=%.4f test=%.4f FN=%d" % (
            a["label"], a["coverage"] or 0, a["file_coverage"] or 0,
            a["test_coverage"] or 0, a["n_false_negatives"]))
    print("worst-case: edge=%.4f file=%.4f test=%.4f" % (
        agg["worst_edge_coverage"], agg["worst_file_coverage"],
        agg["worst_test_coverage"]))
    print("union false negatives: %d edges, %d files, %d tests" % (
        agg["n_false_negatives"], agg["n_files_with_fn"], agg["n_tests_with_fn"]))
    print("verdict: %s" % agg["verdict"].upper())
    return 0 if agg["verdict"] == "pass" else 1


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
    rc.add_argument(
        "--codegen-inventory",
        help="codegen_blindspots.json: mark codegen-driven tests (generated at "
        "build time) as a known class rather than false negatives",
    )
    rc.add_argument("--output", help="write result JSON here")
    rc.add_argument("--junit", help="write JUnit XML report here")
    rc.add_argument(
        "--mode",
        choices=["full", "selective", "none"],
        help="tag the result/JUnit with the build mode (mirrors validate --mode)",
    )
    rc.add_argument(
        "--label",
        help="extra tag for the JUnit suite/classname (e.g. GPU arch)",
    )

    ca = sub.add_parser(
        "codegen-allowlist",
        help="Expand codegen_blindspots.json test_globs against `ctest -N` -> test names",
    )
    ca.add_argument("--inventory", required=True, help="codegen_blindspots.json")
    ca.add_argument("--ctest", required=True, help="ctest -N output")
    ca.add_argument("--output", help="write matched test names here (default: stdout)")

    cov = sub.add_parser(
        "coverage",
        help="Diff a pre-build depmap vs the post-build ground truth (no build): "
        "report file->test edges the real build proves but the depmap lacks (FNs)",
        epilog="Keys are canonicalized to the project root on both sides, so --pre "
        "and --post may use different --workspace-roots. By default only PR-editable "
        "source counts (build/ outputs and system headers are excluded); pass "
        "--include-nonsource for the raw diff.",
    )
    cov.add_argument("--pre", required=True,
                     help="pre-build depmap JSON (cmake-parse output)")
    cov.add_argument("--post", required=True,
                     help="post-build depmap JSON (main.py parse / ninja -t deps)")
    cov.add_argument("--ctest", help="ctest -N output (count only registered-test edges)")
    cov.add_argument("--label",
                     help="tag the result with this label (e.g. GPU arch) so per-arch "
                     "results can be aggregated")
    cov.add_argument("--codegen-inventory",
                     help="codegen_blindspots.json; also report a backstop-credited "
                     "view with the codegen-class tests excluded")
    cov.add_argument("--include-nonsource", action="store_true",
                     help="also count build/ outputs and system headers the depmap "
                     "never tracks (default: PR-editable source only)")
    cov.add_argument("--output", help="write coverage result JSON here")

    agg = sub.add_parser(
        "coverage-aggregate",
        help="Merge per-arch coverage result JSONs into one cross-arch view "
        "(union of false negatives; worst-case coverages)",
    )
    agg.add_argument("inputs", nargs="+", help="per-arch coverage_result_*.json files")
    agg.add_argument("--output", help="write aggregate JSON here")

    args = p.parse_args()
    if args.command == "probe":
        sys.exit(_run_probe(args))
    elif args.command == "reachability":
        sys.exit(_run_reachability(args))
    elif args.command == "codegen-allowlist":
        sys.exit(_run_codegen_allowlist(args))
    elif args.command == "coverage":
        sys.exit(_run_coverage(args))
    elif args.command == "coverage-aggregate":
        sys.exit(_run_coverage_aggregate(args))


if __name__ == "__main__":
    main()
