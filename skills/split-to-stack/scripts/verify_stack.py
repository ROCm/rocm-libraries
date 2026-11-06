#!/usr/bin/env python3
"""Independent verification of a built stack. Read-only. Exit 0 = PASS.

usage: verify_stack.py --repo REPO --run RUN --prefix users/<name>/ [--tip REV] [--refs] [--no-content-check]

Does not use the composition code of build_stack.py. Checks:
  - the branches in stack.tsv are exactly the groups of groups.tsv (count, order, names under --prefix)
  - the bottom commit's parent is BASE; each commit has exactly one parent, the previous commit; no merges
  - recorded trees match the commits; tree(top) == tree(TIP) (TIP from the ledger header, or --tip)
  - between consecutive branches only paths the plan names change, and a group with assigned entries
    changes something
  - content (unless --no-content-check), for paths whose entries are all single-step `final` and that no
    transition touches: a text file's added/deleted line counts against BASE equal the sums of the hunks
    assigned to groups up to that branch (catches a hunk that leaked into a lower branch or was dropped);
    a whole-file entry is the TIP version from its group on and the BASE version before it
  - with --refs every branch exists and points at its recorded sha
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splitlib import (
    CatFile,
    SplitError,
    dec,
    die,
    expand_steps,
    git,
    git_ok,
    git_text,
    is_mode_row,
    is_whole_row,
    load_assign,
    load_groups,
    load_ledger,
    load_transitions,
    read_tsv,
    toplevel,
)


def numstat(repo, base, sha):
    out = git(
        repo,
        "diff",
        "--numstat",
        "-z",
        "--no-renames",
        "-l0",
        "--no-relative",
        base,
        sha,
    )
    res = {}
    for tok in out.split(b"\0"):
        if not tok:
            continue
        x, y, p = tok.decode("utf-8", "surrogateescape").split("\t", 2)
        if x != "-":
            res[dec(p.encode("utf-8", "surrogateescape"))] = (int(x), int(y))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tip")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--no-content-check", action="store_true")
    a = ap.parse_args()
    repo = toplevel(a.repo)
    meta, rows = load_ledger(a.run)
    base = meta["base"]
    tip = a.tip or meta["tip"]
    groups = load_groups(a.run)
    gindex = {g["NN"]: i for i, g in enumerate(groups)}
    steps = expand_steps(rows, load_assign(a.run), gindex)
    trans = load_transitions(a.run)
    _, stack = read_tsv(
        os.path.join(a.run, "stack.tsv"), ["NN", "branch", "sha", "tree"]
    )
    errs = []
    if not stack:
        die("stack.tsv is empty")

    want = [(g["NN"], a.prefix + g["suffix"]) for g in groups]
    have = [(s["NN"], s["branch"]) for s in stack]
    if want != have:
        errs.append("stack.tsv branches %s differ from groups.tsv %s" % (have, want))

    expected = [set() for _ in groups]
    has_entries = [False] * len(groups)
    for r in rows:
        for gi, _ in steps.get(r["id"], []):
            expected[gi].add(r["path"])
            if r["status"] == "R" and r["old_path"]:
                expected[gi].add(r["old_path"])
            has_entries[gi] = True
    trans_paths = set()
    for t in trans:
        trans_paths.add(t["path"])
        for g in (t["added_in"], t["removed_in"]):
            if g in gindex:
                expected[gindex[g]].add(t["path"])

    # content oracle set-up: paths whose rows are all single-step `final`
    by_path = {}
    for r in rows:
        by_path.setdefault(r["path"], []).append(r)
    old_paths = {r["old_path"] for r in rows if r["status"] == "R" and r["old_path"]}
    hpaths, wpaths = {}, {}
    for p, rs in by_path.items():
        if p in trans_paths or p in old_paths:
            continue
        simple = all(
            len(steps.get(r["id"], [])) == 1 and steps[r["id"]][0][1] == "final"
            for r in rs
        )
        if not simple:
            continue
        if all(r["kind"] == "H" or is_mode_row(r) for r in rs):
            hs = [r for r in rs if r["kind"] == "H"]
            if hs:
                hpaths[p] = hs
        elif len(rs) == 1 and is_whole_row(rs[0]) and rs[0]["status"] != "R":
            wpaths[p] = rs[0]
    cat = CatFile(repo)
    cat.prefetch(base, list(wpaths))
    cat.prefetch(tip, list(wpaths))

    prev_sha = base
    for i, s in enumerate(stack):
        parents = git_text(repo, "rev-list", "--parents", "-n", "1", s["sha"]).split()[
            1:
        ]
        if parents != [prev_sha]:
            errs.append(
                "%s (%s): parents %s, expected [%s]"
                % (s["NN"], s["sha"][:10], [p[:10] for p in parents], prev_sha[:10])
            )
        tree = git_text(repo, "rev-parse", s["sha"] + "^{tree}")
        if tree != s["tree"]:
            errs.append("%s: recorded tree does not match commit" % s["NN"])
        if not git_ok(repo, "check-ref-format", "--branch", s["branch"]):
            errs.append("%s: invalid branch name %s" % (s["NN"], s["branch"]))
        if i < len(expected):
            out = git(
                repo,
                "diff-tree",
                "-r",
                "--no-renames",
                "--name-only",
                "-z",
                prev_sha,
                s["sha"],
            )
            changed = {dec(x) for x in out.split(b"\0") if x}
            extra = changed - expected[i]
            if extra:
                errs.append(
                    "%s: paths changed that the plan does not assign to this group: %s"
                    % (s["NN"], ", ".join(sorted(extra)[:8]))
                )
            if has_entries[i] and not changed:
                errs.append(
                    "%s: the group has assigned entries but the commit changes nothing"
                    % s["NN"]
                )
            if not a.no_content_check and i < len(groups):
                ns = numstat(repo, base, s["sha"])
                bad = []
                for p, hs in hpaths.items():
                    exp = (
                        sum(int(r["plus"]) for r in hs if steps[r["id"]][0][0] <= i),
                        sum(int(r["minus"]) for r in hs if steps[r["id"]][0][0] <= i),
                    )
                    if ns.get(p, (0, 0)) != exp:
                        bad.append(
                            "%s (has +%d/-%d, plan says +%d/-%d)"
                            % (p, *ns.get(p, (0, 0)), *exp)
                        )
                if bad:
                    errs.append(
                        "%s: text content does not match the plan's hunks for: %s"
                        % (s["NN"], "; ".join(bad[:6]))
                    )
                cat.prefetch(s["sha"], list(wpaths))
                bad = []
                for p, r in wpaths.items():
                    reached = steps[r["id"]][0][0] <= i
                    if reached:
                        exp = None if r["status"] == "D" else cat.entry(tip, p)
                    else:
                        exp = cat.entry(base, p)
                    if cat.entry(s["sha"], p) != exp:
                        bad.append(p)
                if bad:
                    errs.append(
                        "%s: whole-file content does not match the plan for: %s"
                        % (s["NN"], ", ".join(bad[:8]))
                    )
        if a.refs:
            cur = (
                git(
                    repo,
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    "refs/heads/" + s["branch"],
                    check=False,
                )
                .decode()
                .strip()
            )
            if cur != s["sha"]:
                errs.append(
                    "%s: branch %s is %s, expected %s"
                    % (s["NN"], s["branch"], cur[:10] or "missing", s["sha"][:10])
                )
        prev_sha = s["sha"]
    cat.close()
    merges = git_text(repo, "rev-list", "--merges", "%s..%s" % (base, stack[-1]["sha"]))
    if merges:
        errs.append("merge commits in the stack: " + merges.replace("\n", " "))
    top_tree = stack[-1]["tree"]
    tip_tree = git_text(repo, "rev-parse", tip + "^{tree}")
    if top_tree != tip_tree:
        diff = git_text(repo, "diff", "--name-only", top_tree, tip_tree)
        errs.append(
            "tree(top) %s != tree(tip) %s; differing files:\n    %s"
            % (top_tree[:10], tip_tree[:10], diff.replace("\n", "\n    "))
        )
    if errs:
        print("STACK VERIFY FAIL")
        for e in errs:
            print("  - " + e)
        sys.exit(1)
    print(
        "STACK VERIFY PASS: %d branches, linear on %s, tree(top) == tree(tip) == %s"
        % (len(stack), base[:10], top_tree)
    )


if __name__ == "__main__":
    try:
        main()
    except SplitError as e:
        die(str(e))
