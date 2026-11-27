#!/usr/bin/env python3
"""Write the stored state of a hunk entry (`H<n>`) as it was at a source commit. Read-only except --out.

usage: extract_state.py --repo REPO --run RUN --entry H124 --commit <full sha> --out RUN/states/H124_g01.txt

The ledger hunk replaces base lines [a, a+b) with L's lines. At COMMIT the same region of the file has some
intermediate text: the base lines of the region with the changes of `git diff B COMMIT -- path` that fall into
the region applied. That text is what `plan_assign.tsv` references as `stored:<path>` for an earlier step.
Fails (write the state by hand) if a change at COMMIT crosses the region boundary, or the file is missing there.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splitlib import (
    DIFF_OPTS,
    HUNK_OPT,
    CatFile,
    SplitError,
    contained,
    die,
    git,
    load_ledger,
    parse_hunk,
    split_lines,
    toplevel,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = toplevel(a.repo)
    meta, rows = load_ledger(a.run)
    base = meta["base"]
    row = next((r for r in rows if r["id"] == a.entry), None)
    if row is None or row["kind"] != "H":
        raise SplitError("%s is not a hunk entry of the ledger" % a.entry)
    path = row["path"]
    ha, hb, _, _ = row["hunk_t"]
    s = (ha - 1) if hb > 0 else ha
    e = s + hb
    cat = CatFile(repo)
    try:
        base_blob = cat.blob("%s:%s" % (base, path))
        com_blob = cat.blob("%s:%s" % (a.commit, path))
    finally:
        cat.close()
    if base_blob is None or com_blob is None:
        raise SplitError("%s is missing at the base or at %s" % (path, a.commit[:10]))
    base_lines, com_lines = split_lines(base_blob), split_lines(com_blob)
    diff = git(
        repo,
        "-c",
        "core.attributesFile=/dev/null",
        "diff",
        HUNK_OPT,
        *DIFF_OPTS,
        base,
        a.commit,
        "--",
        path,
        env={"GIT_ATTR_SOURCE": a.commit}
    ).decode("utf-8", "surrogateescape")
    picked = []
    for ln in diff.split("\n"):
        if not ln.startswith("@@"):
            continue
        h = parse_hunk(ln)
        s2 = (h[0] - 1) if h[1] > 0 else h[0]
        e2 = s2 + h[1]
        if hb == 0:
            touches = (s2 == s and e2 == s) or (s2 < s < e2)
        else:
            touches = (s2 < e and e2 > s) or (h[1] == 0 and s <= s2 <= e)
        if not touches:
            continue
        if s2 < s or e2 > e:
            raise SplitError(
                "a change at %s (base lines %d-%d) crosses the region of %s (base lines %d-%d); "
                "write this state by hand"
                % (a.commit[:10], s2 + 1, e2, a.entry, s + 1, e)
            )
        picked.append(
            (s2, e2, com_lines[h[2] - 1 : h[2] - 1 + h[3]] if h[3] > 0 else [])
        )
    picked.sort(key=lambda x: (x[0], x[1]))
    out, pos = [], s
    for s2, e2, new in picked:
        out.extend(base_lines[pos:s2])
        out.extend(new)
        pos = e2
    out.extend(base_lines[pos:e])
    full = contained(
        a.run,
        (
            a.out
            if not os.path.isabs(a.out)
            else os.path.relpath(a.out, os.path.realpath(a.run))
        ),
    )
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(b"".join(out))
    print("wrote %s (%d lines) for %s at %s" % (full, len(out), a.entry, a.commit[:10]))


if __name__ == "__main__":
    try:
        main()
    except SplitError as e:
        die(str(e))
