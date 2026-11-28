#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Usage: mkwt.py <PR number>

Fetches the PR head and develop, creates sparse worktrees wt<N> (head) and wt<N>m (head merged with develop),
refreshes wtdev to develop (made on first use), and lists files that conflict with develop.
RKMON_BASE (optional): the PR's base branch when it is not develop (a feature branch, the PR's base.ref). It is
fetched too, and wt<N>m merges it instead of develop; the log's develop=, behind/ahead and "conflicting files vs
develop:" lines are then about it, and base=<branch> ends the first line.
SPARSE_EXTRA (optional, space-separated paths): also check out paths outside rocKE's provider folder, for PRs that
change other projects. RKMON_SPARSE is rocKE's provider folder and .github.
With the RAM disk on (RKMON_RAMDISK, config ramdisk), the trees live there and $D/wt<N> links to each, so every path
that names $D/wt<N> works as before; a link whose tree is gone (the RAM disk emptied by a restart) is made again.
Each worktree is added into a folder made first for a tree PR code reads (perms.tree_folder: in the container the PR
runners' group, pr_user.group(), reads it and none of them writes it; on a host a plain folder).
Paths and settings come from the job's RKMON_* environment (rkenv.py). A failing step ends it with that step's exit
code. The clone is shared: the whole staging holds its lock (steps.clone_lock; the toolkit's fetches take the same),
and a fetch that still meets the shallow-file race (another git changed the clone's shallow history) runs once more.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import perms  # noqa: E402
import pr_user  # noqa: E402
import rkenv  # noqa: E402
from steps import (
    capture,
    clone_lock,
    echo,
    link,
    make_dirs,
    must,
    must_fetch,
    remove,
    run,
    words,
    write_raw,
)  # noqa: E402


def git(*args) -> list:
    return ["git", *args]


def add_tree(R: str, path: str, ref: str) -> None:
    """The worktree of `ref` added at `path`, into a folder made first for a tree PR code reads; a folder that cannot
    be made so ends the script (exit 1), since PR code could not read the tree."""
    why = perms.tree_folder(path, pr_user.group())
    if why:
        echo(f"!! {why}")
        sys.exit(1)
    must(git("-C", R, "worktree", "add", "-q", "--no-checkout", "--detach", path, ref))


def main(argv: list) -> int:
    rkenv.load()
    D, R = os.environ["RKMON_WORKING_DIR"], os.environ["RKMON_REPO_DIR"]
    N = argv[1] if len(argv) > 1 else ""
    if not N:
        echo("usage: mkwt.py <PR number>")
        return 2
    # One staging at a time on the shared clone (its fetches and worktrees), and the toolkit's fetches wait for it.
    with clone_lock(R):
        return stage(D, R, N)


def stage(D: str, R: str, N: str) -> int:
    since = rkenv.clone_since(R)
    B = os.environ.get("RKMON_BASE") or "develop"
    BR = f"origin/{B}"
    extra = [] if B == "develop" else [f"+refs/heads/{B}:refs/remotes/origin/{B}"]
    must_fetch(
        git(
            "-C",
            R,
            "fetch",
            "-q",
            "--filter=blob:none",
            *([f"--shallow-since={since}"] if since else []),
            "origin",
            f"+refs/pull/{N}/head:refs/remotes/origin/pr{N}",
            "+develop:refs/remotes/origin/develop",
            *extra,
        )
    )
    RD = os.environ.get("RKMON_RAMDISK", "")
    sparse = os.environ.get("RKMON_SPARSE", "")
    sparse_extra = os.environ.get("SPARSE_EXTRA", "")
    quiet = {"stderr": subprocess.DEVNULL}
    for W in (f"wt{N}", f"wt{N}m"):
        tree = f"{D}/{W}"
        if RD and os.path.islink(tree) and not os.path.isdir(tree):
            remove(tree)
            must(git("-C", R, "worktree", "prune"))
        if not os.path.isdir(tree):
            if RD:
                if make_dirs(RD):
                    return 1
                if os.path.lexists(f"{RD}/{W}"):
                    remove(f"{RD}/{W}")
                    must(git("-C", R, "worktree", "prune"))
                add_tree(R, f"{RD}/{W}", f"origin/pr{N}")
                link(f"{RD}/{W}", tree)
            else:
                add_tree(R, tree, f"origin/pr{N}")
            must(
                git(
                    "-C",
                    tree,
                    "sparse-checkout",
                    "set",
                    "--no-cone",
                    *words(sparse),
                    *words(sparse_extra),
                )
            )
        elif sparse_extra:
            must(
                git(
                    "-C",
                    tree,
                    "sparse-checkout",
                    "set",
                    "--no-cone",
                    *words(sparse),
                    *words(sparse_extra),
                )
            )
        # Isolation: no leftovers from an earlier review of this PR (an unfinished merge, edits, untracked or
        # ignored files) survive into this one.
        run(git("-C", tree, "merge", "--abort"), **quiet)
        run(git("-C", tree, "reset", "-q", "--hard"), **quiet)
        run(git("-C", tree, "clean", "-fdxq"), **quiet)
        must(git("-C", tree, "checkout", "-q", "--detach", f"origin/pr{N}"))
    if run(
        git(
            "-C",
            f"{D}/wt{N}m",
            "-c",
            "user.name=r",
            "-c",
            "user.email=r@x",
            "merge",
            "-q",
            "--no-ff",
            "--no-edit",
            BR,
        )
    ):
        echo(f"!! merge with {B} has conflicts in wt{N}m")
    if not os.path.isdir(f"{D}/wtdev"):
        add_tree(R, f"{D}/wtdev", "origin/develop")
        must(
            git(
                "-C",
                f"{D}/wtdev",
                "sparse-checkout",
                "set",
                "--no-cone",
                *words(sparse),
            )
        )
    must(git("-C", f"{D}/wtdev", "checkout", "-q", "--detach", "origin/develop"))
    rc, mb = capture(git("-C", R, "merge-base", f"origin/pr{N}", BR))
    if rc:
        return rc
    on = "" if B == "develop" else f" base={B}"
    head = capture(git("-C", R, "rev-parse", "--short=10", f"origin/pr{N}"))[1]
    dev = capture(git("-C", R, "rev-parse", "--short=10", BR))[1]
    echo(f"head={head} develop={dev} merge-base={mb[:10]}{on}")
    echo(
        f"behind/ahead: {capture(git('-C', R, 'rev-list', '--left-right', '--count', f'{BR}...origin/pr{N}'))[1]}"
    )
    echo("conflicting files vs develop:")
    # merge-tree's lines 2 to 200 (the tree id first), up to and with the first empty line.
    sys.stdout.flush()
    raw = subprocess.run(
        git("-C", R, "merge-tree", "--write-tree", "--name-only", BR, f"origin/pr{N}"),
        stdout=subprocess.PIPE,
    ).stdout
    write_raw(conflict_lines(raw))
    return 0


def conflict_lines(raw: bytes) -> bytes:
    """`sed -n '2,200p' | sed '/^$/q'` of merge-tree's output."""
    lines = raw.split(b"\n")
    lines = [x + b"\n" for x in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    out = []
    for line in lines[1:200]:
        if not line.endswith(b"\n"):
            line += b"\n"  # sed ends its last line too
        out.append(line)
        if line == b"\n":
            break
    return b"".join(out)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
