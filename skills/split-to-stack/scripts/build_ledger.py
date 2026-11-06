#!/usr/bin/env python3
"""Build ledger.tsv: one row per piece of change in `git diff BASE TIP`.

Read-only against the repository. Used by the Inventory agent; check_plan.py re-runs
`generate()` and requires the same bytes.

usage: build_ledger.py --repo REPO --base BASE --tip TIP --out RUN/ledger.tsv
                       [--generated-set NAME=GLOB ...] [--attr-generated] [--exclude-generated GLOB ...]

Row ids: H<n> a hunk inside a modified text file, F<n> a whole-file change (add, delete,
rename, type change, binary, submodule, or a mode-only change), G<n> one file of a
generated set. A set is taken from --generated-set (first matching glob wins; Python
fnmatch semantics, '*' also matches '/') or, with --attr-generated, from the
linguist-generated / -diff attributes of TIP (set name 'attr-generated').

Only added (A) and modified (M) paths become G rows. A deleted, renamed or type-changed path that
matches a set is an ordinary F row whose `generated` column still names the set (information and size
accounting only; it is not part of `@set`). A path matching --exclude-generated is never generated:
it is an ordinary row with `generated` empty (for generated test fixtures, or files that must land
with their code). The exclusions are recorded in the ledger header.

Exit status: 0 ok, 2 error, 3 the diff is empty (nothing to split).
"""
import argparse
import fnmatch
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splitlib import (
    DIFF_OPTS,
    DIFF_OPTS_TEXT,
    GENERATED_STATUSES,
    HUNK_OPT,
    LEDGER_COLUMNS,
    SplitError,
    dec,
    die,
    git,
    hunk_text,
    parse_hunk,
    render_tsv,
    rev_parse,
    toplevel,
)


def parse_raw(data):
    """Parse `git diff --raw -z` output into dicts."""
    toks = data.split(b"\0")
    i = 0
    out = []
    while i < len(toks) and toks[i]:
        meta = toks[i].decode().lstrip(":").split()
        oldmode, newmode, oldsha, newsha, status = meta[:5]
        code = status[0]
        if code in ("R", "C"):
            old, new = dec(toks[i + 1]), dec(toks[i + 2])
            i += 3
        else:
            old, new = None, dec(toks[i + 1])
            i += 2
        out.append(
            dict(
                oldmode=oldmode,
                newmode=newmode,
                oldsha=oldsha,
                newsha=newsha,
                status=code,
                old=old,
                path=new,
            )
        )
    return out


def parse_numstat(data):
    toks = data.split(b"\0")
    i = 0
    out = {}
    while i < len(toks) and toks[i]:
        a, d, p = toks[i].decode("utf-8", "surrogateescape").split("\t", 2)
        if p == "":
            new = dec(toks[i + 2])
            i += 3
        else:
            new = dec(p.encode("utf-8", "surrogateescape"))
            i += 1
        out[new] = (a, d)
    return out


def commit_map(repo, base, tip):
    """path -> full commit ids (oldest first) touching it in base..tip (no merges)."""
    data = git(
        repo,
        "log",
        "-z",
        "--reverse",
        "--no-merges",
        "--no-renames",
        "--name-only",
        "--format=%x01%H",
        "%s..%s" % (base, tip),
    )
    m = {}
    cur = None
    for tok in data.split(b"\0"):
        if not tok:
            continue
        if tok.startswith(b"\x01"):
            cur = tok[1:].decode()
            continue
        if tok.startswith(b"\n"):
            tok = tok[1:]
        path = dec(tok)
        if cur:
            lst = m.setdefault(path, [])
            if cur not in lst:
                lst.append(cur)
    return m


def attr_generated(repo, tip, paths):
    """Subset of paths with linguist-generated set or `diff` unset, read from TIP's attributes."""
    if not paths:
        return set()
    try:
        data = git(
            repo,
            "check-attr",
            "--source",
            tip,
            "-z",
            "--stdin",
            "linguist-generated",
            "diff",
            input=b"\0".join(os.fsencode(p) for p in paths) + b"\0",
        )
    except SplitError as e:
        print("WARNING: --attr-generated unavailable (%s)" % e, file=sys.stderr)
        return set()
    toks = data.split(b"\0")
    gen = set()
    for i in range(0, len(toks) - 2, 3):
        path, attr, val = dec(toks[i]), toks[i + 1].decode(), toks[i + 2].decode()
        if attr == "linguist-generated" and val in ("set", "true"):
            gen.add(path)
        if attr == "diff" and val == "unset":
            gen.add(path)
    return gen


def generate(repo, base, tip, sets, use_attr, excludes=()):
    """Return (meta_lines, rows, counters). `sets` is a list of (name, glob); `excludes` a list of globs."""
    # Attributes come from TIP's .gitattributes (git >= 2.40), not from the checkout, a global attributes
    # file or the system file. $GIT_DIR/info/attributes is still read by git: warn if it exists.
    denv = {"GIT_ATTR_SOURCE": tip}
    pin = ["-c", "core.attributesFile=/dev/null"]
    info = (
        git(repo, "rev-parse", "--git-path", "info/attributes", check=False)
        .decode()
        .strip()
    )
    if info:
        info = info if os.path.isabs(info) else os.path.join(repo, info)
        if os.path.isfile(info) and os.path.getsize(info) > 0:
            print(
                "WARNING: %s is not empty; it can change the ledger on this machine only"
                % info,
                file=sys.stderr,
            )
    raw = parse_raw(
        git(
            repo,
            *pin,
            "diff",
            "--raw",
            "-z",
            "--no-abbrev",
            *DIFF_OPTS,
            base,
            tip,
            env=denv
        )
    )
    num = parse_numstat(
        git(repo, *pin, "diff", "--numstat", "-z", *DIFF_OPTS, base, tip, env=denv)
    )
    raw.sort(key=lambda e: os.fsencode(e["path"]))

    attr_set = set()
    if use_attr:
        attr_set = attr_generated(repo, tip, [e["path"] for e in raw])

    def set_of(path):
        for g in excludes:
            if fnmatch.fnmatchcase(path, g):
                return ""
        for n, g in sets:
            if fnmatch.fnmatchcase(path, g):
                return n
        if path in attr_set:
            return "attr-generated"
        return ""

    cmap = commit_map(repo, base, tip)
    rows = []
    counters = {"H": 0, "F": 0, "G": 0}

    def new_id(k):
        counters[k] += 1
        return "%s%d" % (k, counters[k])

    def commits_for(*paths):
        seen = []
        for p in paths:
            for c in cmap.get(p, []):
                if c not in seen:
                    seen.append(c)
        return ",".join(seen)

    for e in raw:
        path, status = e["path"], e["status"]
        if status == "C":
            raise SplitError(
                "copy detected for %s; the pinned options do not enable copies" % path
            )
        for ch in ("\t", "\n"):
            if ch in path or (e["old"] and ch in e["old"]):
                raise SplitError("path contains a tab or newline: %r" % path)
        plus, minus = num.get(path, ("0", "0"))
        binary = "no"
        if plus == "-":
            binary = "yes"
            plus = minus = "0"
        if e["newmode"] == "160000" or e["oldmode"] == "160000":
            binary = "sub"
        modechg = (
            e["oldmode"] != e["newmode"]
            and e["oldmode"] != "000000"
            and e["newmode"] != "000000"
        )
        newmode = e["newmode"] if (modechg or status in ("A", "T")) else ""
        gset = set_of(path)
        common = dict(
            path=path,
            old_path=e["old"] or "",
            status=status,
            plus=plus,
            minus=minus,
            binary=binary,
            mode=newmode,
            generated=gset,
            commits=commits_for(path, *([e["old"]] if e["old"] else [])),
        )
        if gset and status in GENERATED_STATUSES:
            rows.append(dict(common, id=new_id("G"), hunk=""))
            continue
        if status in ("A", "D", "R", "T"):
            rows.append(dict(common, id=new_id("F"), hunk=""))
            continue
        # status M
        if binary in ("yes", "sub"):
            rows.append(dict(common, id=new_id("F"), hunk=""))
            continue
        if e["oldsha"] != e["newsha"]:
            diff = git(
                repo,
                *pin,
                "diff",
                HUNK_OPT,
                *DIFF_OPTS,
                base,
                tip,
                "--",
                path,
                env=denv
            )
            hunks = []
            for ln in diff.decode("utf-8", "surrogateescape").split("\n"):
                if ln.startswith("@@"):
                    h = parse_hunk(ln)
                    if h is None:
                        raise SplitError("cannot parse hunk header: %r" % ln)
                    hunks.append(h)
            if not hunks:
                raise SplitError(
                    "%s differs between base and tip but git reports no hunks "
                    "(attributes or config hide the change); fix the cause" % path
                )
            for h in hunks:
                rows.append(
                    dict(
                        common,
                        id=new_id("H"),
                        hunk=hunk_text(h),
                        plus=str(h[3]),
                        minus=str(h[1]),
                        mode="",
                        generated="",
                    )
                )
        if modechg:
            # a text file keeps content and mode separate; whole-file rows carry the mode themselves
            rows.append(dict(common, id=new_id("F"), hunk="", plus="0", minus="0"))

    meta = [
        "split-to-stack ledger v1",
        "base=%s" % base,
        "tip=%s" % tip,
        "diff-options=%s" % DIFF_OPTS_TEXT,
        "generated-sets=%s" % ";".join("%s=%s" % s for s in sets),
        "attr-generated=%s" % ("yes" if use_attr else "no"),
    ]
    if excludes:
        meta.append("generated-exclude=%s" % ";".join(excludes))
    return meta, rows, counters


def parse_excludes(items):
    out = []
    for g in items:
        if not g or any(c in g for c in ";\t\n"):
            raise SplitError(
                "--exclude-generated glob must be non-empty and contain no ';', tab or newline: %r"
                % g
            )
        out.append(g)
    return out


def parse_sets(items):
    sets = []
    for s in items:
        if "=" not in s:
            raise SplitError("--generated-set needs NAME=GLOB, got %r" % s)
        n, g = s.split("=", 1)
        if not re.match(r"^[A-Za-z0-9_.-]+$", n):
            raise SplitError("generated set name must match [A-Za-z0-9_.-]+: %r" % n)
        if not g or any(c in g for c in ";\t\n"):
            raise SplitError(
                "generated set glob must be non-empty and contain no ';', tab or newline: %r"
                % g
            )
        sets.append((n, g))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generated-set", action="append", default=[], metavar="NAME=GLOB")
    ap.add_argument("--attr-generated", action="store_true")
    ap.add_argument("--exclude-generated", action="append", default=[], metavar="GLOB")
    a = ap.parse_args()

    repo = toplevel(a.repo)
    base = rev_parse(repo, a.base)
    tip = rev_parse(repo, a.tip)
    meta, rows, counters = generate(
        repo,
        base,
        tip,
        parse_sets(a.generated_set),
        a.attr_generated,
        parse_excludes(a.exclude_generated),
    )
    if not rows:
        print(
            "EMPTY: git diff %s %s is empty; nothing to split" % (base[:10], tip[:10]),
            file=sys.stderr,
        )
        sys.exit(3)
    with open(a.out, "wb") as f:
        f.write(render_tsv(LEDGER_COLUMNS, rows, meta))
    hand = sum(int(r["plus"]) + int(r["minus"]) for r in rows if not r["generated"])
    gen = sum(int(r["plus"]) + int(r["minus"]) for r in rows if r["generated"])
    files = len({r["path"] for r in rows})
    print(
        "ledger: %d entries over %d files (H=%d F=%d G=%d); hand-written changed lines %d, generated %d"
        % (len(rows), files, counters["H"], counters["F"], counters["G"], hand, gen)
    )
    print("base=%s tip=%s" % (base, tip))


if __name__ == "__main__":
    try:
        main()
    except SplitError as e:
        die(str(e))
