#!/usr/bin/env python3
"""Build the stack of branches from the plan files. No worktree, no checkout.

usage: build_stack.py --repo REPO --run RUN --prefix users/<name>/ [--refs none|create|update]
                      [--date ISO8601]

Reads RUN/ledger.tsv, groups.tsv, plan_assign.tsv, transitions.tsv (optional), states/.
For group k the content of every file is computed from the BASE blob plus the ledger entries
assigned to groups <= k (hunks placed at their recorded BASE line numbers; whole-file entries
taken from TIP or a stored state). Trees are written with a private index file; one commit is
created per group, chained from BASE. Writes RUN/stack.tsv (NN, branch, sha, tree).
Exit 0 only if tree(top) == tree(TIP).
"""
import argparse
import datetime
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splitlib import (
    AUTHOR_RE,
    FILE_STATE_KINDS,
    NULL_SHA,
    CatFile,
    SplitError,
    contained,
    die,
    enc,
    expand_steps,
    git,
    git_text,
    is_mode_row,
    is_whole_row,
    load_assign,
    load_groups,
    load_ledger,
    load_transitions,
    read_tsv,
    split_lines,
    split_trailers,
    toplevel,
    write_tsv,
)


class Builder:
    def __init__(self, repo, run):
        self.repo, self.run = repo, run
        self.meta, self.rows = load_ledger(run)
        self.base, self.tip = self.meta["base"], self.meta["tip"]
        self.groups = load_groups(run)
        self.gindex = {g["NN"]: i for i, g in enumerate(self.groups)}
        self.steps = expand_steps(self.rows, load_assign(run), self.gindex)
        missing = [r["id"] for r in self.rows if r["id"] not in self.steps]
        if missing:
            raise SplitError(
                "%d ledger entries are not assigned (run check_plan.py): %s"
                % (len(missing), ", ".join(missing[:10]))
            )
        self.trans = []
        for t in load_transitions(run):
            if t["added_in"] not in self.gindex or t["removed_in"] not in self.gindex:
                raise SplitError("transition %s: unknown group" % t["id"])
            self.trans.append(
                dict(t, ai=self.gindex[t["added_in"]], ri=self.gindex[t["removed_in"]])
            )
        self.by_id = {r["id"]: r for r in self.rows}
        self.rows_by_path = {}
        self.renamed_from = {}
        for r in self.rows:
            self.rows_by_path.setdefault(r["path"], []).append(r)
            if r["status"] == "R" and r["old_path"]:
                self.renamed_from.setdefault(r["old_path"], []).append(r)
        self.cat = CatFile(repo)
        self._lines = {}
        self._base_entry = {}
        self._tip_entry = {}
        wanted = (
            set(self.rows_by_path)
            | set(self.renamed_from)
            | {t["path"] for t in self.trans}
        )
        self.cat.prefetch(self.base, sorted(wanted))
        self.cat.prefetch(self.tip, sorted(wanted))

    # ---- object access -------------------------------------------------
    def base_entry(self, p):
        if p not in self._base_entry:
            self._base_entry[p] = self.cat.entry(self.base, p)
        return self._base_entry[p]

    def tip_entry(self, p):
        if p not in self._tip_entry:
            self._tip_entry[p] = self.cat.entry(self.tip, p)
        return self._tip_entry[p]

    def lines(self, rev, p):
        key = (rev, p)
        if key not in self._lines:
            data = self.cat.blob("%s:%s" % (rev, p))
            if data is None:
                raise SplitError("%s is missing at %s" % (p, rev[:10]))
            self._lines[key] = split_lines(data)
        return self._lines[key]

    def hash_blob(self, data):
        return (
            git(self.repo, "hash-object", "-w", "--stdin", "--no-filters", input=data)
            .decode()
            .strip()
        )

    def stored(self, rel):
        full = contained(self.run, rel)
        if not os.path.isfile(full):
            raise SplitError("stored state not found: %s" % full)
        with open(full, "rb") as f:
            return f.read()

    # ---- step lookup ---------------------------------------------------
    def latest(self, rid, k):
        st = None
        for gi, state in self.steps[rid]:
            if gi <= k:
                st = state
        return st

    # ---- content computation -------------------------------------------
    def whole_entry(self, r, state, p):
        tipe = self.tip_entry(p)
        if state == "final":
            return tipe
        kind, _, arg = state.partition(":")
        if kind in FILE_STATE_KINDS and r["binary"] == "sub":
            raise SplitError(
                "%s: a submodule entry cannot use a %s state (use commit:<sha>)"
                % (r["id"], kind)
            )
        if kind == "commit":
            e = self.cat.entry(arg, p)
            if e is None:
                raise SplitError("%s does not exist at commit %s" % (p, arg))
            return e
        if kind in FILE_STATE_KINDS:
            mode = (tipe or self.base_entry(p) or ("100644", None))[0]
            return (mode, self.hash_blob(self.stored(arg)))
        raise SplitError("bad state %r for %s" % (state, r["id"]))

    def hunk_new_lines(self, r, state, p):
        a, b, c, d = r["hunk_t"]
        if state == "final":
            return self.lines(self.tip, p)[c - 1 : c - 1 + d] if d > 0 else []
        kind, _, arg = state.partition(":")
        if kind in FILE_STATE_KINDS:
            return split_lines(self.stored(arg))
        raise SplitError("state %r is not valid for hunk entry %s" % (state, r["id"]))

    def entry_at(self, p, k):
        for r in self.renamed_from.get(p, []):
            if self.latest(r["id"], k) is not None:
                return None
        rows = self.rows_by_path.get(p, [])
        whole = [r for r in rows if is_whole_row(r)]
        if whole:
            r = whole[0]
            st = self.latest(r["id"], k)
            if st is None:
                return self.base_entry(p)
            if r["status"] == "D" and st == "final":
                return None
            return self.whole_entry(r, st, p)
        be = self.base_entry(p)
        if be is None:
            if not rows:
                return None  # a path only a transition ever creates
            raise SplitError(
                "%s: modified-file entries but the file is missing at base" % p
            )
        mode, sha = be
        hunks = [r for r in rows if r["kind"] == "H"]
        if hunks:
            base_lines = self.lines(self.base, p)
            applied = []
            for r in hunks:
                a, b, c, d = r["hunk_t"]
                st = self.latest(r["id"], k)
                if st is not None:
                    applied.append(((a - 1) if b > 0 else a, b, st, r))
            if applied:
                applied.sort(key=lambda x: x[0])
                out, pos = [], 0
                for start, b, st, r in applied:
                    if start < pos:
                        raise SplitError("%s: overlapping hunks (%s)" % (p, r["id"]))
                    out.extend(base_lines[pos:start])
                    pos = start + b
                    out.extend(self.hunk_new_lines(r, st, p))
                out.extend(base_lines[pos:])
                sha = self.hash_blob(b"".join(out))
        for r in rows:
            if is_mode_row(r):
                st = self.latest(r["id"], k)
                if st is not None and st != "final":
                    raise SplitError(
                        "%s: a mode-only entry has no content to stage; its state must be final"
                        % r["id"]
                    )
                if st is not None:
                    mode = self.tip_entry(p)[0]
        return (mode, sha)

    def entry_final(self, p, k):
        e = self.entry_at(p, k)
        for t in self.trans:
            if t["path"] == p and t["ai"] <= k < t["ri"]:
                fallback = (
                    e or self.tip_entry(p) or self.base_entry(p) or ("100644", None)
                )
                return (fallback[0], self.hash_blob(self.stored(t["stored"])))
        return e

    def affected(self, k):
        paths = set()
        for r in self.rows:
            if any(gi == k for gi, _ in self.steps[r["id"]]):
                paths.add(r["path"])
                if r["status"] == "R" and r["old_path"]:
                    paths.add(r["old_path"])
        for t in self.trans:
            if t["ai"] == k or t["ri"] == k:
                paths.add(t["path"])
        return paths


def parse_author(s):
    if not AUTHOR_RE.match(s.strip()):
        raise SplitError("author must look like 'Name <email>': %r" % s)
    name, _, rest = s.strip().partition(" <")
    return name.strip(), rest.rstrip(">")


def worktree_branches(repo):
    out = git_text(repo, "worktree", "list", "--porcelain")
    return {ln.split(" ", 1)[1] for ln in out.split("\n") if ln.startswith("branch ")}


def index_count(repo, env):
    return len([x for x in git(repo, "ls-files", "-z", env=env).split(b"\0") if x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. users/alvasile/")
    ap.add_argument("--refs", choices=["none", "create", "update"], default="none")
    ap.add_argument(
        "--date", help="fixed commit date (ISO 8601); default: RUN/build_date"
    )
    a = ap.parse_args()
    repo, run = toplevel(a.repo), a.run

    date_file = os.path.join(run, "build_date")
    date = a.date
    if not date and os.path.exists(date_file):
        date = open(date_file).read().strip()
    if not date:
        date = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S+0000"
        )

    b = Builder(repo, run)
    tmpdir = tempfile.mkdtemp(prefix="split-index-")
    index = os.path.join(tmpdir, "index")
    ienv = {"GIT_INDEX_FILE": index}
    git(repo, "read-tree", b.base, env=ienv)
    count = index_count(repo, ienv)

    cur = {}
    parent = b.base
    parent_tree = git_text(repo, "rev-parse", b.base + "^{tree}")
    stack = []
    try:
        for k, g in enumerate(b.groups):
            changes = []
            for p in sorted(b.affected(k)):
                new = b.entry_final(p, k)
                old = cur[p] if p in cur else b.base_entry(p)
                cur[p] = new
                if new != old:
                    changes.append((p, new, old))
            changes.sort(key=lambda x: (x[1] is not None, x[0]))
            if changes:
                info = b"".join(
                    (
                        ("%s %s\t" % (e[0], e[1])).encode()
                        if e
                        else ("0 %s\t" % NULL_SHA).encode()
                    )
                    + enc(p)
                    + b"\n"
                    for p, e, _ in changes
                )
                git(repo, "update-index", "--index-info", input=info, env=ienv)
                count += sum(
                    (1 if new is not None else 0) - (1 if old is not None else 0)
                    for _, new, old in changes
                )
                if index_count(repo, ienv) != count:
                    raise SplitError(
                        "group %s: a file and a directory of the same name would exist at once "
                        "(git silently drops one of them); the plan cannot be built in this order. "
                        "Paths changed in this group: %s"
                        % (g["NN"], ", ".join(p for p, _, _ in changes[:8]))
                    )
            tree = git_text(repo, "write-tree", env=ienv)
            if tree == parent_tree:
                raise SplitError(
                    "group %s would be an empty commit: none of its entries changes any file "
                    "(its changes are masked by a transition window, or a step repeats the "
                    "previous state). Fix the plan." % g["NN"]
                )
            parent_tree = tree
            head = (
                "%s(%s): %s" % (g["type"], g["scope"], g["subject"])
                if g["scope"]
                else "%s: %s" % (g["type"], g["subject"])
            )
            msg = head + "\n\n" + g["summary"] + "\n"
            trailers = split_trailers(g["trailers"])
            if trailers:
                msg += "\n" + "\n".join(trailers) + "\n"
            env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
            if g["author"]:
                n, e = parse_author(g["author"])
                env.update(GIT_AUTHOR_NAME=n, GIT_AUTHOR_EMAIL=e)
            sha = git_text(
                repo, "commit-tree", tree, "-p", parent, input=msg.encode(), env=env
            )
            stack.append(
                dict(NN=g["NN"], branch=a.prefix + g["suffix"], sha=sha, tree=tree)
            )
            parent = sha
    finally:
        b.cat.close()
        try:
            os.remove(index)
            os.rmdir(tmpdir)
        except OSError:
            pass

    top_tree, tip_tree = stack[-1]["tree"], git_text(
        repo, "rev-parse", b.tip + "^{tree}"
    )
    for s in stack:
        print("%s\t%s\t%s" % (s["NN"], s["branch"], s["sha"]))
    if top_tree != tip_tree:
        names = git_text(repo, "diff", "--name-only", top_tree, tip_tree)
        print(
            "FAIL: tree(top) != tree(tip). Nothing was recorded or moved. Differing files:\n"
            + names,
            file=sys.stderr,
        )
        sys.exit(1)

    # Only a successful build is recorded.
    with open(date_file, "w") as f:
        f.write(date + "\n")
    write_tsv(os.path.join(run, "stack.tsv"), ["NN", "branch", "sha", "tree"], stack)

    if a.refs != "none":
        pub_file = os.path.join(run, "refs.tsv")
        published = {}
        if os.path.exists(pub_file):
            _, prows = read_tsv(pub_file)
            published = {r["branch"]: r["sha"] for r in prows}
        busy = worktree_branches(repo)
        lines = ["start"]
        for s in stack:
            ref = "refs/heads/" + s["branch"]
            if ref in busy:
                die("%s is checked out in a worktree; detach it first" % ref)
            live = (
                git(repo, "rev-parse", "--verify", "--quiet", ref, check=False)
                .decode()
                .strip()
            )
            if not live:
                lines.append("create %s %s" % (ref, s["sha"]))
            elif a.refs == "create":
                die("%s already exists" % ref)
            else:
                old = published.get(s["branch"])
                if old != live:
                    die(
                        "%s is at %s, but this run last published %s; someone else moved it (or it was "
                        "not created by this run)"
                        % (ref, live[:10], (old or "nothing")[:10])
                    )
                lines.append("update %s %s %s" % (ref, s["sha"], live))
        lines += ["prepare", "commit"]
        git(repo, "update-ref", "--stdin", input=("\n".join(lines) + "\n").encode())
        for s in stack:
            published[s["branch"]] = s["sha"]
        write_tsv(
            pub_file,
            ["branch", "sha"],
            [dict(branch=k, sha=v) for k, v in sorted(published.items())],
        )
        stale = [k for k in published if k not in {s["branch"] for s in stack}]
        if stale:
            print(
                "NOTE: branches recorded in refs.tsv but no longer in the plan (not touched): "
                + ", ".join(stale)
            )
    print("OK: tree(top) == tree(tip) == %s" % top_tree)


if __name__ == "__main__":
    try:
        main()
    except SplitError as e:
        die(str(e))
