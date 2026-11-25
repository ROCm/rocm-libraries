#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Develop's tests that CI never ran with the PR: the test files develop added or changed after the develop commit the
PR's CI merged with, for the checks' "Develop's tests changed since CI's merge" section. CI tested the PR merged with
the develop of its last push; a test develop changed since then meets the PR's change for the first time at landing
(lessons L-596), so the checks run it on today's merge and compare each failure with develop (tools/pytest_vs_base.py).

CI's develop is the PR's base commit as GitHub records it (`base.sha`, the base branch's head when the PR's head was
last pushed; fetch_pr.py stages it in pr/meta.md); without it, the merge base of the head and develop. A commit that
is not an ancestor of develop counts from its merge base with develop. The PR's own test files and the files another
section runs already (`--exclude`, `--exclude-list`) are left out, and so are files the merged tree does not have. The
newest first; at most `--cap` run, and the rest are named as not run.

Usage: python3 -I devtests.py --git REPO --rocke K --tree MERGED_ROCKE_ROOT --since SHA --develop SHA --list FILE
           [--map JSON] [--cap 12] [--exclude FILE ...] [--exclude-list FILE]
Prints the selection; writes the chosen files (relative to the rocKE root, space-separated) to FILE, and with --map
each chosen file's newest develop commit to JSON.
       python3 -I devtests.py --fails RUN_DIR/tests_pr_failed.txt --map JSON
After the run: one `DEVTESTS_FAIL <file> (develop <sha10>)` line per file that fails on the merge only (develop passes
it), naming the develop commit that last changed it.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from importer_tests import failed_files  # noqa: E402

CAP = 12
SHOWN = 10
TEST_FILE = re.compile(r"(^|/)test_[^/]*\.py$")


def git(repo: str, *args: str) -> str:
    p = subprocess.run(
        ["git", "-C", repo, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return p.stdout.strip() if p.returncode == 0 else ""


def since_base(repo: str, since: str, develop: str) -> str:
    """The develop commit to count from: `since` when develop contains it, else its merge base with develop."""
    s = git(repo, "rev-parse", "-q", "--verify", f"{since}^{{commit}}")
    d = git(repo, "rev-parse", "-q", "--verify", f"{develop}^{{commit}}")
    if not (s and d):
        return ""
    return s if git(repo, "merge-base", s, d) == s else git(repo, "merge-base", s, d)


def select(
    repo: str,
    rocke: str,
    tree: pathlib.Path,
    since: str,
    develop: str,
    exclude: set[str],
    cap: int = CAP,
) -> dict:
    """{"since": sha, "develop": sha, "chosen": [...], "rest": [...], "by_file": {file: newest develop sha}}; "since"
    is "" when a commit cannot be read."""
    base = since_base(repo, since, develop)
    out = {"since": base, "develop": develop, "chosen": [], "rest": [], "by_file": {}}
    if not base:
        return out
    k = rocke.strip("/") + "/"
    log = git(
        repo,
        "log",
        "--format=@%H",
        "--name-only",
        "--diff-filter=AMR",
        f"{base}..{develop}",
        "--",
        k,
    )
    by_file: dict[str, str] = {}
    sha = ""
    for ln in log.splitlines():
        if ln.startswith("@"):
            sha = ln[1:]
        elif ln.startswith(k) and TEST_FILE.search(ln):
            rel = ln[len(k) :]
            if rel not in by_file:  # the log runs newest first
                by_file[rel] = sha
    order = [
        f
        for f in by_file
        if f not in exclude and (tree / f).is_file() and "/." not in "/" + f
    ]
    out.update(
        chosen=order[:cap], rest=order[cap:], by_file={f: by_file[f] for f in order}
    )
    return out


def report(sel: dict, cap: int) -> list[str]:
    if not sel["since"]:
        return ["CI's develop commit or develop's could not be read: nothing selected"]
    total = len(sel["chosen"]) + len(sel["rest"])
    out = [
        f"develop since CI's merge: {sel['since'][:10]}..{sel['develop'][:10]}; test files it adds or changes that "
        f"no other section runs: {total}; run {len(sel['chosen'])} (cap {cap}), the newest first"
    ]
    for f in sel["chosen"]:
        out.append(f"  {f} (develop {sel['by_file'][f][:10]})")
    if sel["rest"]:
        out.append(
            f"NOT RUN (over the cap, {len(sel['rest'])}): "
            + ", ".join(sel["rest"][:SHOWN])
            + (
                f" (+{len(sel['rest']) - SHOWN} more)"
                if len(sel["rest"]) > SHOWN
                else ""
            )
        )
    return out


def fail_lines(failed: pathlib.Path, by_file: dict[str, str]) -> list[str]:
    return [
        f"DEVTESTS_FAIL {f} (develop {(by_file.get(f) or 'unknown')[:10]})"
        for f in failed_files(failed)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--git")
    ap.add_argument("--rocke")
    ap.add_argument("--tree")
    ap.add_argument("--since", default="")
    ap.add_argument("--develop", default="")
    ap.add_argument("--list")
    ap.add_argument("--map")
    ap.add_argument("--fails")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--exclude-list", default="")
    a = ap.parse_args(argv)
    if a.fails:
        try:
            by_file = (
                json.loads(pathlib.Path(a.map).read_text(encoding="utf-8"))
                if a.map
                else {}
            )
        except (OSError, ValueError):
            by_file = {}
        lines = fail_lines(pathlib.Path(a.fails), by_file)
        if lines:
            print("\n".join(lines))
        return 0
    if not (a.git and a.rocke and a.tree and a.list):
        ap.error("--git, --rocke, --tree and --list are needed (or --fails)")
    exclude = set(a.exclude)
    if a.exclude_list:
        try:
            exclude |= set(pathlib.Path(a.exclude_list).read_text().split())
        except OSError:
            pass
    sel = select(
        a.git, a.rocke, pathlib.Path(a.tree), a.since, a.develop, exclude, a.cap
    )
    pathlib.Path(a.list).write_text(" ".join(sel["chosen"]) + "\n", encoding="utf-8")
    if a.map:
        pathlib.Path(a.map).write_text(
            json.dumps({f: sel["by_file"][f] for f in sel["chosen"]}, indent=1) + "\n",
            encoding="utf-8",
        )
    print("\n".join(report(sel, a.cap)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
