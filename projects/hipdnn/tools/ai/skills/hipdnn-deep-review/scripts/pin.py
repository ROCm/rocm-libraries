#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pin and intake for hipdnn-deep-review (RUNBOOK "Pin and intake" and "Iterate").

  pin.py key <text>
  pin.py init --evidence DIR --scratch DIR --round N --owner-repo O/R
              (--pr N | --branch NAME [--target BR] | --range BASE..HEAD)
              [--clone DIR] [--remote-url URL] [--skill DIR]
  pin.py cleanup --evidence DIR --round N

Each subcommand prints terse results and exits non-zero with a one-line reason.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PIN_VARS = "SKILL|EVIDENCE|SCRATCH|CLONE|KEY|TARGET|HEAD|HEAD_SHORT|BASE|REPO|REPO_BASE|KNOWLEDGE"

# Loads PIN.md values as quoted exports; eval-based, so it works in a persistent shell.
ENV_SH = (
    r"""eval "$(awk '/^(""" + PIN_VARS + r"""):/{v=$0; sub(/^[A-Z_]+: */,"",v); k=$1; sub(/:$/,"",k); gsub(/\047/,"\047\\\047\047",v); print "export " k "=\047" v "\047"}' "$(dirname "${BASH_SOURCE[0]}")/PIN.md")"
: "${EVIDENCE:?}" "${SCRATCH:?}" "${CLONE:?}" "${KEY:?}"; cd "$(dirname "${BASH_SOURCE[0]}")"
"""
)

MAX_GH_FILES = 3000


def die(msg):
    sys.exit(f"pin: {msg}")


def git(clone, *args, cwd=None, stdin=None, check=True):
    """Run git (-C clone); return stdout without the trailing newline."""
    cmd = ["git"] + (["-C", str(clone)] if clone else []) + list(args)
    r = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True)
    if check and r.returncode != 0:
        lines = r.stderr.strip().splitlines()
        die(f"git {args[0]} failed: {lines[-1] if lines else r.returncode}")
    return r.stdout.rstrip("\n")


def gh(*args):
    """Run gh and return stdout; the only place the GitHub CLI is invoked."""
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        lines = r.stderr.strip().splitlines()
        die(f"gh {args[0]} failed: {lines[-1] if lines else r.returncode}")
    return r.stdout


def gh_paginated_list(endpoint):
    """Concatenate the JSON arrays printed by `gh api --paginate`."""
    text, dec, pos, out = gh("api", "--paginate", endpoint), json.JSONDecoder(), 0, []
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        arr, pos = dec.raw_decode(text, pos)
        out.extend(arr)
    return out


def derive_key(text):
    """Sanitize text into a review key and validate it as a ref component."""
    key = re.sub(r"[^A-Za-z0-9._-]", "-", text)
    key = re.sub(r"\.{2,}", "-", key)
    key = re.sub(r"^\.", "", key)
    key = re.sub(r"\.lock$", "", key)
    ok = subprocess.run(
        ["git", "check-ref-format", f"refs/review/{key}/target"], capture_output=True
    ).returncode
    if ok != 0:
        die(f"unusable review key: {key}")
    return key


def cmd_key(args):
    """Print the review key for a branch name or <base>..<head> text."""
    print(derive_key(args.text))


def sparse_set(files):
    """Sparse-checkout directories for changed paths (RUNBOOK sparse-set rules)."""
    out = set()
    for f in files:
        parts = f.split("/")
        if len(parts) == 1:
            continue
        if parts[0] in ("projects", "dnn-providers", "shared"):
            if len(parts) > 2:
                out.add(f"{parts[0]}/{parts[1]}")
            continue
        out.add("/".join(parts[:-1]))
    if any(re.match(r"projects/[^/]+/tools/", f) for f in files):
        out.add("tools")
    return sorted(out)


def pin_lines(values):
    """Format `NAME: value` PIN.md lines."""
    return "".join(f"{k}: {v}\n" for k, v in values)


def is_sha(text):
    return re.fullmatch(r"[0-9a-f]{7,40}", text) is not None


def cmd_init(args):
    """Pin one review round: PIN.md, env.sh, refs, resolved SHAs, sparse worktrees, Gate checks."""
    targets = [bool(args.pr), bool(args.branch), bool(args.range)]
    if sum(targets) != 1:
        die("exactly one of --pr, --branch, --range is required")
    n = args.round
    evidence_dir, scratch_dir = Path(args.evidence), Path(args.scratch)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence, scratch = evidence_dir.resolve(), scratch_dir.resolve()
    skill = Path(args.skill).resolve() if args.skill else Path(__file__).resolve().parent.parent

    if args.pr:
        key = f"pr{args.pr}"
    else:
        key = derive_key(args.branch or args.range)
    clone = Path(args.clone or os.environ.get("CLONE") or scratch / "clone")
    repo, repo_base = scratch / f"r{n}" / "head", scratch / f"r{n}" / "base"
    rdir = evidence / f"r{n}"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "PIN.md").write_text(
        pin_lines(
            [("SKILL", skill), ("EVIDENCE", evidence), ("SCRATCH", scratch),
             ("CLONE", clone), ("KEY", key), ("REPO", repo), ("REPO_BASE", repo_base)],
        )
    )
    (rdir / "env.sh").write_text(ENV_SH)

    url = args.remote_url or f"https://github.com/{args.owner_repo}.git"
    if subprocess.run(["git", "-C", str(clone), "rev-parse", "--git-dir"],
                      capture_output=True).returncode != 0:
        git(None, "clone", "--filter=blob:none", "--no-checkout", url, str(clone))
    git(clone, "config", "gc.auto", "0")

    ref_t, ref_h = f"refs/review/{key}/target", f"refs/review/{key}/r{n}"
    specs, shas = [], []
    if args.pr:
        target = gh("pr", "view", args.pr, "-R", args.owner_repo,
                    "--json", "baseRefName", "--jq", ".baseRefName").strip()
        specs = [f"+refs/heads/{target}:{ref_t}", f"+refs/pull/{args.pr}/head:{ref_h}"]
    elif args.branch:
        target = args.target or "develop"
        specs = [f"+refs/heads/{target}:{ref_t}", f"+refs/heads/{args.branch}:{ref_h}"]
    else:
        if args.range.count("..") != 1 or "..." in args.range:
            die(f"range must be <base>..<head>: {args.range}")
        target, head_side = args.range.split("..")
        if not target or not head_side:
            die(f"range must be <base>..<head>: {args.range}")
        for side, ref in ((target, ref_t), (head_side, ref_h)):
            if is_sha(side):
                shas.append((ref, side))
            else:
                specs.append(f"+refs/heads/{side}:{ref}")
    if specs:
        git(clone, "fetch", "origin", *specs)
    for ref, sha in shas:
        git(clone, "update-ref", ref, git(clone, "rev-parse", "--verify", f"{sha}^{{commit}}"))

    head = git(clone, "rev-parse", f"{ref_h}^{{commit}}")
    if args.range:
        base = git(clone, "rev-parse", f"{ref_t}^{{commit}}")
    else:
        base = git(clone, "merge-base", ref_t, head)
    with open(rdir / "PIN.md", "a") as f:
        f.write(pin_lines([("TARGET", target), ("HEAD", head),
                                 ("HEAD_SHORT", head[:12]), ("BASE", base)]))
    if base == head:
        print("empty diff: PR already merged by merge commit?", file=sys.stderr)

    files = git(clone, "diff", "--name-only", "--no-renames", base, head).splitlines()
    (rdir / "files.txt").write_text("".join(f + "\n" for f in files))
    (rdir / "sparse.txt").write_text("".join(d + "\n" for d in sparse_set(files)))
    (rdir / "numstat.txt").write_text(
        git(clone, "diff", "--numstat", "--no-renames", base, head) + "\n")
    (rdir / "commits.txt").write_text(git(clone, "log", "--format=%H %s", f"{base}..{head}") + "\n")
    sparse_text = (rdir / "sparse.txt").read_text()
    for w, sha in ((repo, head), (repo_base, base)):
        git(clone, "worktree", "add", "--no-checkout", "--detach", str(w), sha)
        git(w, "sparse-checkout", "set", "--cone", "--stdin", stdin=sparse_text)
        git(w, "checkout", "HEAD")

    for w, sha in ((repo, head), (repo_base, base)):
        if git(w, "rev-parse", "HEAD") != sha:
            die(f"worktree {w} is not at {sha}")
    if args.pr:
        pr_gate(args, rdir, clone, base, head)
    print(f"KEY={key} TARGET={target} HEAD={head[:12]} BASE={base[:12]} files={len(files)}")


def pr_gate(args, rdir, clone, base, head):
    """Save pr/ artifacts and check headRefOid and the file-count Gate."""
    pr, o = args.pr, args.owner_repo
    pr_dir = rdir / "pr"
    pr_dir.mkdir(exist_ok=True)
    fields = "title,body,baseRefName,headRefName,headRefOid,files,comments,reviews"
    meta = gh("pr", "view", pr, "-R", o, "--json", fields)
    (pr_dir / "pr.json").write_text(meta)
    info = json.loads(meta)
    (pr_dir / "body.md").write_text(info.get("body") or "")
    inline = gh_paginated_list(f"repos/{o}/pulls/{pr}/comments")
    (pr_dir / "inline.json").write_text(json.dumps(inline))
    pr_files = gh_paginated_list(f"repos/{o}/pulls/{pr}/files")
    (pr_dir / "files.json").write_text(json.dumps(pr_files))
    (pr_dir / "checks.tsv").write_text(
        gh("api", "--paginate", f"repos/{o}/commits/{head}/check-runs",
           "--jq", r'.check_runs[] | [.name,.conclusion,.html_url] | @tsv'))

    if info.get("headRefOid") != head:
        die(f"PR head moved since pin: {info.get('headRefOid')} != {head}")
    if len(pr_files) >= MAX_GH_FILES:
        with open(rdir / "PIN.md", "a") as f:
            f.write(f"Files: unverifiable (>{MAX_GH_FILES})\n")
        return
    local = git(clone, "diff", "--name-only", "-M", base, head).splitlines()
    if len(local) == len(pr_files):
        return
    both = set(git(clone, "diff", "--name-only", "--no-renames", base, head).splitlines())
    theirs = {x["filename"] for x in pr_files}
    theirs |= {x["previous_filename"] for x in pr_files if x.get("previous_filename")}
    if both != theirs:
        die(f"file lists differ: git {len(local)} files, GitHub {len(pr_files)} files; "
            f"only git: {sorted(both - theirs)[:5]}, only GitHub: {sorted(theirs - both)[:5]}")


def read_pin(rdir):
    pin = {}
    for line in (rdir / "PIN.md").read_text().splitlines():
        m = re.match(r"([A-Z_]+): ?(.*)", line)
        if m:
            pin[m.group(1)] = m.group(2)
    return pin


def cmd_cleanup(args):
    """Remove review worktrees under $SCRATCH (except $CLONE) and delete refs/review/$KEY/*."""
    pin = read_pin(Path(args.evidence) / f"r{args.round}")
    clone, scratch, key = pin["CLONE"], pin["SCRATCH"].rstrip("/") + "/", pin["KEY"]
    n = 0
    for line in git(clone, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            w = line[len("worktree "):]
            if w != clone and w.startswith(scratch):
                git(clone, "worktree", "remove", "--force", w)
                n += 1
    refs = git(clone, "for-each-ref", "--format=%(refname)", f"refs/review/{key}/").splitlines()
    if refs:
        git(clone, "update-ref", "--stdin", stdin="".join(f"delete {r}\n" for r in refs))
    print(f"removed {n} worktrees, {len(refs)} refs")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("key", help=cmd_key.__doc__)
    s.add_argument("text")
    s.set_defaults(fn=cmd_key)
    s = sub.add_parser("init", help=cmd_init.__doc__)
    for a in ("--evidence", "--scratch", "--owner-repo"):
        s.add_argument(a, required=True)
    s.add_argument("--round", type=int, required=True)
    for a in ("--pr", "--branch", "--range", "--target", "--clone", "--remote-url", "--skill"):
        s.add_argument(a)
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("cleanup", help=cmd_cleanup.__doc__)
    s.add_argument("--evidence", required=True)
    s.add_argument("--round", type=int, required=True)
    s.set_defaults(fn=cmd_cleanup)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
