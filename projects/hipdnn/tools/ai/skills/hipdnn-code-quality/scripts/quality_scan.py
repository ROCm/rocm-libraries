#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Diff-scoped code-quality metrics for hipDNN reviews. In-harness only,
text-based - defers to the project's clang-tidy config (see `.clang-tidy`,
wired via `cmake/ClangTidy.cmake`) for anything an AST-based linter already
does better: function size, identifier casing, brace style, pass-by-value
params, and dozens of bugprone/modernize/performance checks. This script
covers what clang-tidy structurally cannot: cross-file and whole-tree
duplication, plus outlier file size. Emits candidate signals for a
human/agent to triage - every signal is a lead, not a verdict.

Usage:
    quality_scan.py --repo <root> --base <ref>            # diff vs merge-base
    quality_scan.py --repo <root> --files f1 f2 ...        # explicit changed set
    quality_scan.py --repo <root> --base <ref> --json

Design notes learned from running against the real tree:
  * Duplication uses normalized N-line windows - no parser, so it survives the
    template/macro-heavy files that ast-grep's C++ grammar fails to parse.
  * Changed files are the "needles"; the whole tree is the duplication
    "haystack" so a copy of existing code is found even if the original is
    untouched by the diff.
  * Long-function and naming detectors were removed from this script:
    clang-tidy's `readability-function-size` and `readability-identifier-naming`
    (both enabled, `WarningsAsErrors: "*"`) do this precisely via a real AST;
    a text heuristic here was strictly weaker and, for naming, actively wrong
    (the frontend intentionally mirrors cuDNN's snake_case API while the
    backend mandates camelCase - a repo-wide regex flagged 588 false
    positives in testing).
"""
import argparse, os, re, subprocess, sys, json
from collections import defaultdict

EXCLUDE = ("/build/", "_generated", "third_party", "/flatbuffers_sdk/")
SRC_EXT = (".cpp", ".cc", ".hpp", ".h")


def sh(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True).stdout


def changed_files(repo, base):
    mb = sh(["git", "merge-base", base, "HEAD"], repo).strip() or base
    out = sh(["git", "diff", "--name-only", f"{mb}...HEAD"], repo)
    return [l for l in out.splitlines() if l.strip()]


def is_src(p):
    return p.endswith(SRC_EXT) and not any(x in p for x in EXCLUDE)


def all_src(repo):
    files = []
    for r, _, fs in os.walk(repo):
        if any(x in r + "/" for x in EXCLUDE) or "/.git" in r:
            continue
        for f in fs:
            p = os.path.join(r, f)
            if is_src(p):
                files.append(p)
    return files


def read(path):
    try:
        return open(path, encoding="utf-8", errors="ignore").read().splitlines()
    except OSError:
        return []


# --- Duplication: normalized N-line windows across the tree ------------------
def norm(l):
    s = l.strip()
    # Boilerplate lines carry no design signal; blank them so a window made only
    # of includes/usings/braces falls under the triviality filter and is skipped.
    if (
        not s
        or s.startswith(
            (
                "#include",
                "#pragma",
                "#ifdef",
                "#ifndef",
                "#endif",
                "#else",
                "#define",
                "#if",
                "using ",
                "namespace",
                "//",
            )
        )
        or s in ("{", "}", "};", "public:", "private:", "protected:")
    ):
        return ""
    return re.sub(r"\s+", "", re.sub(r"//.*", "", l))


def build_index(files, w):
    idx = defaultdict(list)
    for f in files:
        nl = [norm(x) for x in read(f)]
        for i in range(len(nl) - w):
            win = nl[i : i + w]
            if sum(1 for x in win if len(x) > 3) < w:
                continue
            idx[hash("\n".join(win))].append((f, i + 1))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--file-threshold", type=int, default=800)
    ap.add_argument("--dup-window", type=int, default=6)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)

    rels = a.files if a.files else changed_files(repo, a.base or "origin/develop")
    changed = [os.path.join(repo, p) for p in rels if is_src(p)]
    changed = [p for p in changed if os.path.exists(p)]
    report = {
        "changed_src": [os.path.relpath(p, repo) for p in changed],
        "size": [],
        "duplication": [],
    }

    # size on changed files only; function-size and naming are clang-tidy's job
    for p in changed:
        rel = os.path.relpath(p, repo)
        lines = read(p)
        if len(lines) >= a.file_threshold:
            report["size"].append({"file": rel, "loc": len(lines)})

    # duplication: changed files' windows looked up in whole-tree index.
    # Collapse consecutive matching windows into a single region so a 40-line
    # copy reads as one finding, not (40 - window) adjacent ones.
    haystack = all_src(repo)
    idx = build_index(haystack, a.dup_window)
    for p in changed:
        nl = [norm(x) for x in read(p)]
        raw = []  # (start_i, other_file, other_line, occurrences)
        for i in range(len(nl) - a.dup_window):
            win = nl[i : i + a.dup_window]
            if sum(1 for x in win if len(x) > 3) < a.dup_window:
                raw.append(None)
                continue
            others = {
                (f, ln)
                for f, ln in idx.get(hash("\n".join(win)), [])
                if os.path.abspath(f) != p
            }
            raw.append(sorted(others)[0] + (len(others) + 1,) if others else None)
        i = 0
        while i < len(raw):
            if raw[i] is None:
                i += 1
                continue
            f2, l2, occ = raw[i]
            j = i
            while j + 1 < len(raw) and raw[j + 1] is not None:
                j += 1
            report["duplication"].append(
                {
                    "file": os.path.relpath(p, repo),
                    "line": i + 1,
                    "span": (j - i) + a.dup_window,
                    "also_in": f"{os.path.relpath(f2, repo)}:{l2}",
                    "occurrences": occ,
                }
            )
            i = j + 1

    report["duplication"].sort(key=lambda d: (-d["span"], -d["occurrences"]))

    if a.json:
        print(json.dumps(report, indent=2))
        return
    print(f"# Quality scan: {len(changed)} changed source file(s)\n")
    if report["size"]:
        print(f"## Large files (>= {a.file_threshold} LOC)")
        for d in report["size"]:
            print(f"  {d['file']}  {d['loc']} LOC")
        print()
    if report["duplication"]:
        print(
            f"## Duplicated regions (>= {a.dup_window} contiguous lines, seen elsewhere)"
        )
        for d in report["duplication"][:20]:
            print(
                f"  {d['occurrences']}x  {d['file']}:{d['line']} (~{d['span']} lines)  also {d['also_in']}"
            )
        print()
    total = sum(len(report[k]) for k in ("size", "duplication"))
    if total == 0:
        print("No metric signals on changed files.")


if __name__ == "__main__":
    main()
