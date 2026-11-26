#!/usr/bin/env python3
"""Render the Phase 6 proposal pieces from the plan files. Read-only.

usage: render_plan.py --run RUN --prefix users/<name>/ [--repo REPO]
                      [--table] [--transitions] [--states]

  --table        the stack table: exactly the columns `#`, `type`, `branch suffix`, `description`,
                 preceded by a note with the common branch prefix (default if nothing else is given)
  --transitions  the planned transition code (what, added in, removed in, why)
  --states       every non-`final` state (`stored:`, `subtractive:`, `commit:`) as a before/after diff
                 (needs --repo)
"""
import argparse
import difflib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splitlib import (
    FILE_STATE_KINDS,
    CatFile,
    SplitError,
    contained,
    die,
    expand_steps,
    load_assign,
    load_groups,
    load_ledger,
    load_transitions,
    split_lines,
    toplevel,
)


def esc(s):
    return s.replace("|", "\\|").replace("\n", " ")


def table(run, prefix):
    groups = load_groups(run)
    print("All branches are under `%s`." % prefix)
    print()
    print("| # | type | branch suffix | description |")
    print("|---|---|---|---|")
    for g in groups:
        print(
            "| %s | %s | %s | %s |"
            % (g["NN"], g["type"], esc(g["suffix"]), esc(g["summary"]))
        )


def transitions(run):
    ts = load_transitions(run)
    if not ts:
        print("No transition code planned.")
        return
    print("| id | file | added in | removed in | why |")
    print("|---|---|---|---|---|")
    for t in ts:
        print(
            "| %s | %s | %s | %s | %s |"
            % (
                t["id"],
                esc(t["path"]),
                t["added_in"],
                t["removed_in"],
                esc(t["reason"]),
            )
        )


def states(run, repo):
    meta, rows = load_ledger(run)
    groups = load_groups(run)
    gindex = {g["NN"]: i for i, g in enumerate(groups)}
    steps = expand_steps(rows, load_assign(run), gindex)
    by_id = {r["id"]: r for r in rows}
    cat = CatFile(toplevel(repo))
    shown = 0
    try:
        for rid, st in sorted(steps.items()):
            r = by_id[rid]
            if all(s == "final" for _, s in st):
                continue
            if r["kind"] != "H":
                prev = split_lines(
                    cat.blob("%s:%s" % (meta["base"], r["old_path"] or r["path"]))
                    or b""
                )
                for gi, s_ in st:
                    kind, _, arg = s_.partition(":")
                    if kind in FILE_STATE_KINDS:
                        with open(contained(run, arg), "rb") as f:
                            data = f.read()
                    elif kind == "commit":
                        data = cat.blob("%s:%s" % (arg, r["path"])) or b""
                    else:
                        data = cat.blob("%s:%s" % (meta["tip"], r["path"])) or b""
                    cur = split_lines(data)
                    print(
                        "### %s %s -> branch %s: state %s"
                        % (rid, r["path"], groups[gi]["NN"], s_)
                    )
                    if b"\0" in data[:8000] or any(b"\0" in x for x in prev[:50]):
                        print("(binary content not shown)")
                    else:
                        for ln in difflib.unified_diff(
                            [x.decode("utf-8", "replace") for x in prev],
                            [x.decode("utf-8", "replace") for x in cur],
                            "before",
                            "after",
                            lineterm="",
                        ):
                            print(ln.rstrip("\n"))
                    prev = cur
                shown += 1
                continue
            a, b, c, d = r["hunk_t"]
            base_lines = split_lines(
                cat.blob("%s:%s" % (meta["base"], r["path"])) or b""
            )
            tip_lines = split_lines(cat.blob("%s:%s" % (meta["tip"], r["path"])) or b"")
            prev = base_lines[(a - 1) if b > 0 else a : (a - 1 + b) if b > 0 else a]
            for gi, s in st:
                if s == "final":
                    cur = tip_lines[c - 1 : c - 1 + d] if d > 0 else []
                else:
                    with open(contained(run, s.partition(":")[2]), "rb") as f:
                        cur = split_lines(f.read())
                print(
                    "### %s %s (base lines %d-%d) -> branch %s: state %s"
                    % (rid, r["path"], a, a + max(b - 1, 0), groups[gi]["NN"], s)
                )
                for ln in difflib.unified_diff(
                    [x.decode("utf-8", "replace") for x in prev],
                    [x.decode("utf-8", "replace") for x in cur],
                    "before",
                    "after",
                    lineterm="",
                ):
                    print(ln.rstrip("\n"))
                prev = cur
            shown += 1
    finally:
        cat.close()
    if not shown:
        print("No stored intermediate states.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--repo")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--transitions", action="store_true")
    ap.add_argument("--states", action="store_true")
    a = ap.parse_args()
    if not (a.table or a.transitions or a.states):
        a.table = True
    if a.table:
        table(a.run, a.prefix)
    if a.transitions:
        if a.table:
            print()
        transitions(a.run)
    if a.states:
        if not a.repo:
            die("--states needs --repo")
        if a.table or a.transitions:
            print()
        states(a.run, a.repo)


if __name__ == "__main__":
    try:
        main()
    except SplitError as e:
        die(str(e))
