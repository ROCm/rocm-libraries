#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Analyze the smart-build selection for a set of PRs against a dependency map.

For each PR, maps the changed files through the depmap, intersects with
ctest-registered tests, and emits a structured JSON analysis useful for
auditing false-negative / blind-spot coverage.

Usage (per-PR):
    python3 analyze_pr_selection.py \\
        enhanced_dependency_mapping.json \\
        ctest_list.txt \\
        pr_files.json \\
        output.json

    where pr_files.json is produced by:
        gh pr view <N> --json number,title,files \
            --jq '{number:.number,title:.title,files:[.files[].path]}' > pr_files.json

Output fields:
    pr, title
    n_changed_files, n_ck_files, n_code_files
    n_selected, selected           — ctest-registered test executables the filter picks
    n_expected_dependents          — raw executables before ctest intersection
    dropped_non_ctest              — executables in depmap but not in ctest
    files_outside_composablekernel — PR files outside the CK project root
    per_file                       — per-changed-file breakdown
    flags:
        code_files_with_no_dependents  — in depmap but no exe depends on it (dead header)
        code_files_not_in_depmap       — TU not extracted (potential FN source)
        noncode_files                  — cmake/yaml/docs etc., not mapped by compile deps
"""

import json
import os
import sys

# Reuse the canonical parser from validate_selection to avoid duplication.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from validate_selection import load_ctest_tests  # noqa: E402

CODE_EXT = {".hpp", ".h", ".hh", ".cpp", ".cc", ".cxx", ".c", ".cu", ".hip",
            ".inc", ".ipp", ".tpp"}
PROJ_PREFIX = "projects/composablekernel/"


def is_code_file(path):
    return os.path.splitext(path)[1].lower() in CODE_EXT


def analyze_pr(f2e, ctest_tests, pr):
    """Return the analysis dict for one PR dict (number, title, files)."""
    raw_files = pr["files"]
    ck_files, outside = [], []
    for f in raw_files:
        if f.startswith(PROJ_PREFIX):
            ck_files.append(f[len(PROJ_PREFIX):])
        else:
            outside.append(f)

    per_file = {}
    expected = set()
    code_no_deps, not_in_depmap, noncode = [], [], []
    for f in ck_files:
        deps = f2e.get(f)
        code = is_code_file(f)
        in_map = deps is not None
        deps = deps or []
        per_file[f] = {
            "is_code": code,
            "in_depmap": in_map,
            "n_deps": len(deps),
            "deps_sample": sorted(deps)[:8],
        }
        expected |= set(deps)
        if code and not in_map:
            not_in_depmap.append(f)
        elif code and in_map and not deps:
            code_no_deps.append(f)
        if not code:
            noncode.append(f)

    selected = sorted(e for e in expected if os.path.basename(e) in ctest_tests)
    dropped_non_ctest = sorted(expected - set(selected))

    return {
        "pr": pr["number"],
        "title": pr["title"],
        "n_changed_files": len(raw_files),
        "n_ck_files": len(ck_files),
        "n_code_files": sum(1 for f in ck_files if is_code_file(f)),
        "n_selected": len(selected),
        "selected": selected,
        "n_expected_dependents": len(expected),
        "dropped_non_ctest": dropped_non_ctest,
        "files_outside_composablekernel": outside,
        "per_file": per_file,
        "flags": {
            "code_files_with_no_dependents": code_no_deps,
            "code_files_not_in_depmap": not_in_depmap,
            "noncode_files": noncode,
        },
    }


def summary_line(result):
    fl = result["flags"]
    return (
        f"PR #{result['pr']:<5} sel={result['n_selected']:<3} "
        f"exp={result['n_expected_dependents']:<3} "
        f"code={result['n_code_files']:<2} "
        f"no_dep={len(fl['code_files_with_no_dependents'])} "
        f"not_in_map={len(fl['code_files_not_in_depmap'])} "
        f"dropped={len(result['dropped_non_ctest'])} "
        f":: {result['title'][:48]}"
    )


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) < 4:
        print(
            "Usage: analyze_pr_selection.py <depmap.json> <ctest_list.txt>"
            " <pr_files.json> <output.json>",
            file=sys.stderr,
        )
        sys.exit(2)
    depmap_path, ctest_path, pr_json_path, out_path = argv[:4]

    with open(depmap_path) as f:
        f2e = json.load(f)["file_to_executables"]
    ctest_tests = load_ctest_tests(ctest_path)
    with open(pr_json_path) as f:
        pr = json.load(f)

    result = analyze_pr(f2e, ctest_tests, pr)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(summary_line(result))


if __name__ == "__main__":
    main()
