#!/usr/bin/env python3
"""Completeness gate for the plan files. Read-only. Exit 0 = PASS.

usage: check_plan.py --repo REPO --run RUN --prefix users/<name>/ [--check-remote REMOTE] [--no-regen]

Checks (every failure is listed):
  1. ledger header options match; the ledger is regenerated from its header (base, tip, sets, exclusions)
     and must be byte-identical (--no-regen: only the line totals are compared with `git diff --numstat`)
  2. every ledger entry is assigned; no unknown entry, group or set; generated sets are not split
  3. steps are 1..n, nsteps agrees, groups strictly increase, the last step is `final`; states are valid
     (`commit:<full sha>` only for whole-file entries, and the path must exist at that commit; a mode-only
     entry takes only `final`; `stored:<path>` and `subtractive:<path>` exist under RUN, are not used for a
     submodule and, for hunk entries, end with a newline unless the file has none at that point)
  3b. every `subtractive:<path>` state is the final text with lines removed: its lines (exact bytes) are an
     ordered subsequence of the final text (an H entry: L's lines of the hunk; an F entry: the whole text
     file at L), and of the next step's text when that step is also `subtractive:`. Only text hunks and
     added, renamed or type-changed regular text files can take it.
  4. groups: NN is 01..n, suffixes unique and not `/`-prefixes of each other, type in the closed list,
     author looks like `Name <email>`, depends_on names lower groups, none empty
  5. transitions: groups exist and added_in < removed_in, stored file under RUN, path is a file (not a
     directory) where it exists, windows on one path do not overlap, no deletion/rename/type change of
     the path inside the window; entries of the same path inside a window are listed as warnings
  6. branch names are valid, and do not already exist locally (or on --check-remote) unless this run
     published them (RUN/refs.tsv)
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ledger import generate, parse_excludes, parse_sets
from splitlib import (
    AUTHOR_RE,
    DIFF_OPTS,
    FILE_STATE_KINDS,
    LEDGER_COLUMNS,
    TYPES,
    SplitError,
    contained,
    die,
    first_unmatched,
    git,
    git_ok,
    is_mode_row,
    is_whole_row,
    load_assign,
    load_groups,
    load_ledger,
    load_transitions,
    read_tsv,
    render_tsv,
    split_lines,
    toplevel,
)


def subtractive_target(repo, tip, r):
    """The final text (list of lines) a `subtractive:` state of ledger row r is checked against, or None
    if the row cannot take a subtractive state."""
    if r["kind"] == "H":
        _, _, c, d = r["hunk_t"]
        blob = git(repo, "cat-file", "blob", "%s:%s" % (tip, r["path"]))
        return split_lines(blob)[c - 1 : c - 1 + d] if d > 0 else []
    if r["kind"] == "F" and r["status"] in ("A", "R", "T") and r["binary"] == "no":
        ent = git(repo, "ls-tree", tip, "--", r["path"], check=False).decode().split()
        if ent and ent[0] in ("100644", "100755"):
            return split_lines(
                git(repo, "cat-file", "blob", "%s:%s" % (tip, r["path"]))
            )
    return None


def show_line(b):
    return b[:80].decode("utf-8", "replace").rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--check-remote", metavar="REMOTE")
    ap.add_argument("--no-regen", action="store_true")
    a = ap.parse_args()
    repo = toplevel(a.repo)
    errs, warns = [], []
    if a.prefix and not a.prefix.endswith("/"):
        raise SplitError("--prefix must end with '/' (for example users/<you>/)")

    def err(m):
        errs.append(m)

    meta, rows = load_ledger(a.run)
    base, tip = meta["base"], meta["tip"]
    groups = load_groups(a.run)
    gnames = [g["NN"] for g in groups]
    gindex = {n: i for i, n in enumerate(gnames)}
    if gnames != ["%02d" % i for i in range(1, len(gnames) + 1)]:
        err(
            "groups.tsv: NN must be 01..%02d without gaps or duplicates, got %s"
            % (len(gnames), gnames)
        )
    sufs = [g["suffix"] for g in groups]
    for i, x in enumerate(sufs):
        for j, y in enumerate(sufs):
            if i != j and (x == y or y.startswith(x + "/")):
                err("suffixes %r and %r cannot both exist as branches" % (x, y))
    for g in groups:
        if g["type"] not in TYPES:
            err("group %s: type %r not in %s" % (g["NN"], g["type"], TYPES))
        if not g["subject"].strip():
            err("group %s: empty subject" % g["NN"])
        if not g["summary"].strip():
            err("group %s: empty summary" % g["NN"])
        if g["author"] and not AUTHOR_RE.match(g["author"]):
            err(
                "group %s: author must look like 'Name <email>': %r"
                % (g["NN"], g["author"])
            )
        for d in [
            x.strip() for x in g["depends_on"].replace(";", ",").split(",") if x.strip()
        ]:
            if d not in gindex:
                err("group %s: depends_on unknown group %s" % (g["NN"], d))
            elif gindex[d] >= gindex[g["NN"]]:
                err("group %s depends on %s, which is not below it" % (g["NN"], d))

    # 1. ledger regeneration
    if not a.no_regen:
        try:
            sets = parse_sets(
                [s for s in meta.get("generated-sets", "").split(";") if s]
            )
            excludes = parse_excludes(
                [s for s in meta.get("generated-exclude", "").split(";") if s]
            )
            m2, rows2, _ = generate(
                repo, base, tip, sets, meta.get("attr-generated") == "yes", excludes
            )
            with open(os.path.join(a.run, "ledger.tsv"), "rb") as f:
                on_disk = f.read()
            if render_tsv(LEDGER_COLUMNS, rows2, m2) != on_disk:
                err(
                    "ledger.tsv differs from a fresh run of build_ledger.py with the same header "
                    "(edited, incomplete, or the repository changed); regenerate it"
                )
        except SplitError as e:
            err("cannot regenerate the ledger: %s" % e)
    else:
        num = git(repo, "diff", "--numstat", "-z", *DIFF_OPTS, base, tip).split(b"\0")
        tot_add = tot_del = 0
        i = 0
        while i < len(num) and num[i]:
            x, y, p = num[i].decode("utf-8", "surrogateescape").split("\t", 2)
            i += 3 if p == "" else 1
            if x != "-":
                tot_add += int(x)
                tot_del += int(y)
        led_add = sum(int(r["plus"]) for r in rows if r["binary"] != "yes")
        led_del = sum(int(r["minus"]) for r in rows if r["binary"] != "yes")
        if (led_add, led_del) != (tot_add, tot_del):
            err(
                "ledger totals +%d/-%d differ from git diff --numstat +%d/-%d"
                % (led_add, led_del, tot_add, tot_del)
            )
        per = {}
        for r in rows:
            if r["binary"] != "yes":
                x = per.setdefault(r["path"], [0, 0])
                x[0] += int(r["plus"])
                x[1] += int(r["minus"])
        num2 = {}
        i = 0
        while i < len(num) and num[i]:
            x, y, p = num[i].decode("utf-8", "surrogateescape").split("\t", 2)
            if p == "":
                p = num[i + 2].decode("utf-8", "surrogateescape")
                i += 3
            else:
                i += 1
            if x != "-":
                num2[p] = [int(x), int(y)]
        if per != num2:
            bad = sorted(
                set(per) ^ set(num2)
                | {p for p in per if p in num2 and per[p] != num2[p]}
            )
            err(
                "ledger per-file totals differ from git diff --numstat for: %s"
                % ", ".join(bad[:8])
            )

    assign = load_assign(a.run)
    ids = {r["id"]: r for r in rows}
    sets = {}
    for r in rows:
        if r["kind"] == "G":
            sets.setdefault(r["generated"], []).append(r["id"])

    per_entry = {}
    for x in assign:
        if x["group"] not in gindex:
            err("plan_assign: unknown group %r (entry %s)" % (x["group"], x["entry"]))
            continue
        if x["entry"].startswith("@"):
            if x["entry"][1:] not in sets:
                err("plan_assign: unknown generated set %s" % x["entry"])
                continue
            targets = sets[x["entry"][1:]]
            if int(x["nsteps"]) != 1 or int(x["step"]) != 1:
                err("plan_assign: generated set %s must be a single step" % x["entry"])
        else:
            if x["entry"] not in ids:
                err("plan_assign: unknown entry %s" % x["entry"])
                continue
            targets = [x["entry"]]
        for t in targets:
            per_entry.setdefault(t, []).append(x)

    n_subtractive = 0
    for rid, r in ids.items():
        xs = per_entry.get(rid)
        if not xs:
            err("entry %s (%s) is not assigned" % (rid, r["path"]))
            continue
        xs = sorted(xs, key=lambda x: int(x["step"]))
        n = len(xs)
        if [int(x["step"]) for x in xs] != list(range(1, n + 1)):
            err("entry %s: steps are not 1..%d" % (rid, n))
        for x in xs:
            if int(x["nsteps"]) != n:
                err("entry %s: nsteps %s != %d rows" % (rid, x["nsteps"], n))
        gi = [gindex[x["group"]] for x in xs if x["group"] in gindex]
        if gi != sorted(set(gi)):
            err("entry %s: groups must strictly increase with step" % rid)
        if xs[-1]["state"] != "final":
            err("entry %s: the last step must be `final`" % rid)
        if r["kind"] == "G" and n != 1:
            err("generated entry %s cannot have several steps" % rid)
        for xi, x in enumerate(xs):
            st = x["state"]
            kind, _, arg = st.partition(":")
            if st == "final":
                continue
            if is_mode_row(r):
                err(
                    "entry %s: a mode-only entry has no content to stage; its only valid state is `final`"
                    % rid
                )
                continue
            if kind == "commit":
                if not is_whole_row(r):
                    err(
                        "entry %s: `commit:` state is only valid for whole-file entries"
                        % rid
                    )
                elif not re.match(r"^[0-9a-f]{40}([0-9a-f]{24})?$", arg):
                    err(
                        "entry %s: `commit:` needs a full object id, got %r"
                        % (rid, arg)
                    )
                elif not git_ok(repo, "cat-file", "-e", arg + "^{commit}"):
                    err("entry %s: commit %s not found" % (rid, arg))
                elif not git(
                    repo, "ls-tree", arg, "--", r["path"], check=False
                ).strip():
                    err(
                        "entry %s: %s does not exist at commit %s"
                        % (rid, r["path"], arg[:10])
                    )
            elif kind in FILE_STATE_KINDS:
                if r["binary"] == "sub":
                    err(
                        "entry %s: a submodule entry cannot use a %s state"
                        % (rid, kind)
                    )
                    continue
                try:
                    full = contained(a.run, arg)
                except SplitError as e:
                    err("entry %s: %s" % (rid, e))
                    continue
                if not os.path.isfile(full):
                    err("entry %s: %s state %s not found" % (rid, kind, arg))
                    continue
                if r["kind"] == "H":
                    data = open(full, "rb").read()
                    if data and not data.endswith(b"\n"):
                        ha, hb = r["hunk_t"][0], r["hunk_t"][1]
                        blob = git(
                            repo,
                            "cat-file",
                            "blob",
                            "%s:%s" % (base, r["path"]),
                            check=False,
                        )
                        nl = blob.count(b"\n") + (
                            0 if blob.endswith(b"\n") or not blob else 1
                        )
                        reaches_eof = (
                            hb > 0 and (ha - 1 + hb) == nl and not blob.endswith(b"\n")
                        )
                        if not reaches_eof:
                            err(
                                "entry %s: stored state %s must end with a newline (it would glue itself "
                                "to the next line)" % (rid, arg)
                            )
                if kind == "subtractive":
                    final = subtractive_target(repo, tip, r)
                    if final is None:
                        err(
                            "entry %s: a `subtractive:` state needs a text hunk or an added, renamed or "
                            "type-changed regular text file (%s %s)"
                            % (rid, r["status"], r["path"])
                        )
                        continue
                    n_subtractive += 1
                    mine = split_lines(open(full, "rb").read())
                    k = first_unmatched(mine, final)
                    if k is not None:
                        err(
                            "entry %s: subtractive state %s is not the final text with lines removed: its line "
                            "%d %r is not in the final text (in this order)"
                            % (rid, arg, k + 1, show_line(mine[k]))
                        )
                    nxt = xs[xi + 1] if xi + 1 < len(xs) else None
                    nkind, _, narg = (
                        nxt["state"].partition(":") if nxt else ("", "", "")
                    )
                    if nkind == "subtractive":
                        try:
                            nfull = contained(a.run, narg)
                        except SplitError:
                            nfull = None
                        if nfull and os.path.isfile(nfull):
                            k = first_unmatched(
                                mine, split_lines(open(nfull, "rb").read())
                            )
                            if k is not None:
                                err(
                                    "entry %s: subtractive state %s is not contained in the next step's state %s: "
                                    "its line %d %r is missing there (the text must only grow toward `final`)"
                                    % (rid, arg, narg, k + 1, show_line(mine[k]))
                                )
            else:
                err("entry %s: bad state %r" % (rid, st))
    for sname, members in sets.items():
        gs = {x["group"] for m in members for x in per_entry.get(m, [])}
        if len(gs) > 1:
            err("generated set %s is split over groups %s" % (sname, sorted(gs)))

    used = {x["group"] for xs in per_entry.values() for x in xs}
    trans = load_transitions(a.run)
    for t in trans:
        used.add(t["added_in"])
    for g in groups:
        if g["NN"] not in used:
            err("group %s has no assigned entries" % g["NN"])

    # 5. transitions
    windows = {}
    for t in trans:
        if t["added_in"] not in gindex or t["removed_in"] not in gindex:
            err("transition %s: unknown group" % t["id"])
            continue
        ai, ri = gindex[t["added_in"]], gindex[t["removed_in"]]
        if ai >= ri:
            err("transition %s: removed_in must be above added_in" % t["id"])
        if not t["reason"].strip():
            err("transition %s: reason is required" % t["id"])
        try:
            full = contained(a.run, t["stored"])
            if not os.path.isfile(full):
                err("transition %s: stored file %s not found" % (t["id"], t["stored"]))
        except SplitError as e:
            err("transition %s: %s" % (t["id"], e))
        for rev in (base, tip):
            typ = (
                git(repo, "cat-file", "-t", "%s:%s" % (rev, t["path"]), check=False)
                .decode()
                .strip()
            )
            if typ and typ != "blob":
                err(
                    "transition %s: %s is a %s at %s, not a file"
                    % (t["id"], t["path"], typ, rev[:10])
                )
        parts = t["path"].split("/")
        for i in range(1, len(parts)):
            anc = "/".join(parts[:i])
            if any(
                git(repo, "cat-file", "-t", "%s:%s" % (rev, anc), check=False)
                .decode()
                .strip()
                == "blob"
                for rev in (base, tip)
            ) or any(r["path"] == anc for r in rows):
                err(
                    "transition %s: %s lies under %s, which is a file at base or tip or a ledger path"
                    % (t["id"], t["path"], anc)
                )
        for r in rows:
            if r["path"].startswith(t["path"] + "/"):
                err(
                    "transition %s: %s is a file but the ledger has %s under it"
                    % (t["id"], t["path"], r["path"])
                )
        for oa, orr, oid in windows.get(t["path"], []):
            if ai < orr and oa < ri:
                err("transitions %s and %s overlap on %s" % (oid, t["id"], t["path"]))
        windows.setdefault(t["path"], []).append((ai, ri, t["id"]))
        for r in rows:
            touches = r["path"] == t["path"] or (
                r["status"] == "R" and r["old_path"] == t["path"]
            )
            if not touches:
                continue
            for x in per_entry.get(r["id"], []):
                gx = gindex.get(x["group"])
                if gx is None or not (ai <= gx < ri):
                    continue
                if r["status"] in ("D", "R", "T") or r["old_path"] == t["path"]:
                    err(
                        "transition %s: entry %s (%s %s) lands in group %s inside the window"
                        % (t["id"], r["id"], r["status"], r["path"], x["group"])
                    )
                else:
                    warns.append(
                        "transition %s: entry %s of %s lands in group %s inside the window; "
                        "the stored file must already contain that change"
                        % (t["id"], r["id"], t["path"], x["group"])
                    )

    # 6. branch names
    pub = {}
    pf = os.path.join(a.run, "refs.tsv")
    if os.path.exists(pf):
        pub = {r["branch"]: r["sha"] for r in read_tsv(pf)[1]}
    remote_heads = set()
    if a.check_remote:
        try:
            out = git(repo, "ls-remote", "--heads", a.check_remote).decode()
        except SplitError as e:
            err("cannot list heads on %s: %s" % (a.check_remote, e))
            out = ""
        remote_heads = {ln.split("\t", 1)[1] for ln in out.split("\n") if "\t" in ln}
    for g in groups:
        full = a.prefix + g["suffix"]
        if not git_ok(repo, "check-ref-format", "--branch", full):
            err("group %s: %r is not a valid branch name" % (g["NN"], full))
            continue
        live = (
            git(
                repo,
                "rev-parse",
                "--verify",
                "--quiet",
                "refs/heads/" + full,
                check=False,
            )
            .decode()
            .strip()
        )
        if live and pub.get(full) != live:
            err(
                "group %s: branch %s already exists locally and was not published by this run"
                % (g["NN"], full)
            )
        parts = full.split("/")
        for i in range(1, len(parts)):
            anc = "/".join(parts[:i])
            if git(
                repo,
                "rev-parse",
                "--verify",
                "--quiet",
                "refs/heads/" + anc,
                check=False,
            ).strip():
                err(
                    "group %s: branch %s exists, so %s cannot be created"
                    % (g["NN"], anc, full)
                )
        below = (
            git(
                repo,
                "for-each-ref",
                "--format=%(refname:short)",
                "refs/heads/" + full + "/",
                check=False,
            )
            .decode()
            .split()
        )
        if below:
            err(
                "group %s: branches exist below %s (%s), so it cannot be created"
                % (g["NN"], full, below[0])
            )
        if "refs/heads/" + full in remote_heads:
            err(
                "group %s: branch %s already exists on %s"
                % (g["NN"], full, a.check_remote)
            )

    stats = {n: [0, 0, 0, 0] for n in gnames}
    for rid, xs in per_entry.items():
        r = ids[rid]
        last = sorted(xs, key=lambda x: int(x["step"]))[-1]
        if last["group"] not in stats:
            continue
        s = stats[last["group"]]
        if r["generated"]:
            s[2] += int(r["plus"]) + int(r["minus"])
            s[3] += 1
        else:
            s[0] += int(r["plus"]) + int(r["minus"])
            s[1] += 1

    if errs:
        print("PLAN CHECK FAIL (%d problems)" % len(errs))
        for e in errs[:200]:
            print("  - " + e)
        for w in warns:
            print("  warning: " + w)
        sys.exit(1)
    print(
        "PLAN CHECK PASS: %d entries assigned over %d groups%s"
        % (
            len(rows),
            len(groups),
            "" if a.no_regen else "; ledger reproduces byte for byte",
        )
    )
    if n_subtractive:
        print(
            "  %d subtractive states checked: each is the final text with lines removed"
            % n_subtractive
        )
    for n in gnames:
        s = stats[n]
        print(
            "  group %s: %d entries / %d hand-written changed lines; %d generated files / %d lines"
            % (n, s[1], s[0], s[3], s[2])
        )
    for w in warns:
        print("  warning: " + w)


if __name__ == "__main__":
    try:
        main()
    except SplitError as e:
        die(str(e))
