#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Bulk audit of the smart-build test selection for a set of PRs.

PURPOSE
    For each PR, map its changed-file paths through a pre-built dependency map
    (depmap), intersect with the ctest-registered tests, and report what the
    smart-build filter *would* select - flagging false-negative / blind-spot
    risk. It only needs the changed *paths*, so it fetches them via `gh pr view`
    (a cheap metadata API call) and never checks out or builds anything.

WHEN TO USE (vs validate_pr.sh)
    This is the cheap, offline, many-PR audit. `validate_pr.sh` is the heavy,
    authoritative per-PR check (it checks out the PR, regenerates the depmap, and
    diffs the new selection against the legacy one). Use this to scan a corpus;
    use validate_pr.sh to certify a single PR.

PREREQUISITES (produced once from a configured build/ dir; see README)
    depmap : main.py cmake-parse compile_commands.json build.ninja \\
                 --workspace-root .. --output enhanced_dependency_mapping.json
    ctest  : ctest -N > ctest_list.txt

USAGE
    # Fetch PR file lists via gh (run inside the repo, or pass --repo):
    analyze_pr_selection.py 7964 7357 \\
        --depmap enhanced_dependency_mapping.json --ctest ctest_list.txt \\
        --output-dir pr_analysis --summary pr_analysis/summary.json

    # Offline (no gh): supply pre-fetched PR JSON instead of numbers:
    analyze_pr_selection.py --pr-files pr7964.json --depmap ... --ctest ...
    # (--pr-files JSON may be {number,title,files:[path,...]} or the raw
    #  `gh pr view --json number,title,files` shape with files:[{path,...}].)

OUTPUT (per PR)
    n_selected / selected          tests the filter would build+run (the answer)
    n_expected_dependents          executables before the ctest intersection
    dropped_non_ctest              expected exes that aren't ctest tests (skipped)
    files_outside_composablekernel PR files outside the CK project root
    per_file                       per-changed-file breakdown
    flags:
      code_files_not_in_depmap     code file the depmap never saw -> possible FN
      code_files_with_no_dependents code file no exe depends on (dead header)
      noncode_files                cmake/yaml/docs etc. (not compile-mapped)

GLOSSARY (one-liners; full definitions in README "Glossary")
    depmap            file -> dependent test executables, built pre-compile
    selection         expected dependents intersected with ctest-registered tests
    expected dependents  all exes a changed file maps to (before ctest filter)
    dropped_non_ctest    in the depmap but not a registered test (e.g. examples)
    dead header       a file no executable depends on
    FN / blind spot   a test that should be selected for a change but isn't
"""

import argparse
import json
import os
import subprocess
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


def _normalize_pr(data):
    """Normalize a PR dict to {number, title, files:[path,...]}.

    Accepts our own shape (files already a list of paths) or the raw
    `gh pr view --json ... files` shape (files a list of {path,...} objects).
    """
    files = data.get("files", [])
    norm = [f["path"] if isinstance(f, dict) else f for f in files]
    return {
        "number": data["number"],
        "title": data.get("title", ""),
        "files": norm,
    }


def fetch_pr(number, repo=None):
    """Fetch one PR's metadata via gh (number, title, changed-file paths).

    Uses `gh pr view <N> --json number,title,files`, a metadata-only call - no
    clone and no object transfer, unlike `git fetch`. Raises RuntimeError with a
    clear message if gh is missing/unauthenticated or the PR can't be read.
    """
    cmd = ["gh", "pr", "view", str(number), "--json", "number,title,files"]
    if repo:
        cmd += ["-R", repo]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except FileNotFoundError as e:
        raise RuntimeError("gh CLI not found on PATH (needed to fetch PRs)") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"gh failed for PR #{number}: {e.stderr.strip() or e}"
        ) from e
    return _normalize_pr(json.loads(out))


def load_pr_file(path):
    """Load and normalize a pre-fetched PR JSON file (offline path)."""
    with open(path) as f:
        return _normalize_pr(json.load(f))


def write_summary(results, path):
    """Write an aggregate JSON across all analyzed PRs."""
    def _flag(r, name):
        return r["flags"][name]
    rows = [
        {
            "pr": r["pr"],
            "n_selected": r["n_selected"],
            "n_expected_dependents": r["n_expected_dependents"],
            "n_code_files": r["n_code_files"],
            "n_files_not_in_depmap": len(_flag(r, "code_files_not_in_depmap")),
            "n_dead_headers": len(_flag(r, "code_files_with_no_dependents")),
        }
        for r in results
    ]
    summary = {
        "n_prs": len(results),
        "prs": rows,
        "prs_with_files_not_in_depmap": sorted(
            r["pr"] for r in results if _flag(r, "code_files_not_in_depmap")
        ),
        "prs_with_dead_headers": sorted(
            r["pr"] for r in results if _flag(r, "code_files_with_no_dependents")
        ),
        "totals": {
            "selected": sum(r["n_selected"] for r in results),
            "files_not_in_depmap": sum(
                len(_flag(r, "code_files_not_in_depmap")) for r in results
            ),
        },
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bulk audit of the smart-build test selection for a set of PRs "
        "(cheap, offline, many-PR; the authoritative per-PR check is validate_pr.sh).",
        epilog=(
            "Prerequisites (produced once from a configured build/ dir; see README):\n"
            "  depmap:  main.py cmake-parse compile_commands.json build.ninja \\\n"
            "             --workspace-root .. --output enhanced_dependency_mapping.json\n"
            "  ctest:   ctest -N > ctest_list.txt\n\n"
            "Example:\n"
            "  analyze_pr_selection.py 7964 7357 \\\n"
            "    --depmap enhanced_dependency_mapping.json --ctest ctest_list.txt \\\n"
            "    --output-dir pr_analysis --summary pr_analysis/summary.json\n\n"
            "Terminology: see the Glossary in README.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prs", nargs="*", help="PR numbers to fetch via gh")
    parser.add_argument(
        "--depmap",
        default="enhanced_dependency_mapping.json",
        help="dependency map JSON (default: enhanced_dependency_mapping.json)",
    )
    parser.add_argument(
        "--ctest",
        default="ctest_list.txt",
        help="`ctest -N` output (default: ctest_list.txt). Not auto-generated.",
    )
    parser.add_argument("--repo", help="OWNER/REPO for gh (default: inferred from cwd)")
    parser.add_argument(
        "--pr-files",
        nargs="+",
        default=[],
        metavar="JSON",
        help="offline: pre-fetched PR JSON file(s) instead of gh-fetching numbers",
    )
    parser.add_argument("--output-dir", help="write pr_<N>.json per PR into this dir")
    parser.add_argument("--summary", help="write an aggregate JSON across all PRs")
    args = parser.parse_args(argv)

    if not args.prs and not args.pr_files:
        parser.error("provide at least one PR number or --pr-files JSON")
    for path, label in [(args.depmap, "--depmap"), (args.ctest, "--ctest")]:
        if not os.path.exists(path):
            print(f"Error: file not found ({label}): {path}", file=sys.stderr)
            return 2

    with open(args.depmap) as f:
        f2e = json.load(f)["file_to_executables"]
    ctest_tests = load_ctest_tests(args.ctest)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    results = []
    failed = 0
    prs = [("file", p) for p in args.pr_files] + [("num", n) for n in args.prs]
    for kind, ref in prs:
        try:
            pr = load_pr_file(ref) if kind == "file" else fetch_pr(ref, args.repo)
            result = analyze_pr(f2e, ctest_tests, pr)
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            print(f"Error processing PR {ref}: {e}", file=sys.stderr)
            failed += 1
            continue
        results.append(result)
        print(summary_line(result))
        if args.output_dir:
            out = os.path.join(args.output_dir, f"pr_{result['pr']}.json")
            with open(out, "w") as f:
                json.dump(result, f, indent=2)

    if args.summary and results:
        write_summary(results, args.summary)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
