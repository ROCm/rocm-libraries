# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Local git queries on the review clone (partial clone; blobs are fetched on demand).

Only fetches and read-only plumbing: no checkouts, no merges in worktrees. Merge simulation uses
`git merge-tree --write-tree`, which writes objects but never touches a ref or a working tree.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import time


class GitError(RuntimeError):
    pass


# A partial clone asks the remote for any object it lacks; with this set, a missing object is an error instead. For reads
# the tick must keep cheap (review.merge_only: a pruned old head must not cost a network round trip).
NO_LAZY_FETCH = {"GIT_NO_LAZY_FETCH": "1"}


def lazy_fetch_off(env=None) -> bool:
    """Lazy fetches are off in `env` (default this process's environment): GIT_NO_LAZY_FETCH holds a value git reads as
    true. So it is in a PR runner's slot (docker/rk_prcode.py), whose git fetches nothing.
    """
    value = (os.environ if env is None else env).get("GIT_NO_LAZY_FETCH", "")
    value = value.strip().lower()
    if value in ("", "false", "no", "off"):
        return False
    try:
        return int(value) != 0
    except ValueError:
        return True  # a value git refuses: it fetches nothing either


# The review clone is shared with the stagings (worker/mkwt.py): a fetch while another git changes its shallow history
# fails ("shallow file has changed since we read it"). A fetch holds the clone's lock, the file worker/steps.py's
# clone_lock takes, waiting at most CLONE_LOCK_WAIT_S for a staging to finish (then it goes on without it).
CLONE_LOCK = "rkmon-clone.lock"
CLONE_LOCK_WAIT_S = 120


def clone_lock_path(repo: str) -> str | None:
    """The lock file in the clone's git folder (`.git`, or the folder itself when bare); None for no repository."""
    git_dir = os.path.join(repo, ".git")
    if os.path.isdir(git_dir):
        return os.path.join(git_dir, CLONE_LOCK)
    if os.path.isdir(os.path.join(repo, "objects")):
        return os.path.join(repo, CLONE_LOCK)
    return None


@contextlib.contextmanager
def clone_lock(repo: str, wait_s: float = CLONE_LOCK_WAIT_S):
    path = clone_lock_path(repo)
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664) if path else None
    except OSError:
        fd = None
    if fd is None:  # no clone there (yet), or not writable: nothing to share
        yield False
        return
    held, end = False, time.monotonic() + wait_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except BlockingIOError:
                if time.monotonic() >= end:
                    break
                time.sleep(0.5)
        yield held
    finally:
        os.close(fd)


def git(
    repo: str, *args: str, check: bool = True, timeout: int = 300, lazy: bool = True
) -> str:
    """`lazy=False`: never fetch a missing object on demand (NO_LAZY_FETCH). A fetch holds the clone's lock."""
    with clone_lock(repo) if args[:1] == ("fetch",) else contextlib.nullcontext():
        p = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=None if lazy else {**os.environ, **NO_LAZY_FETCH},
        )
    if check and p.returncode:
        raise GitError(f"git {' '.join(args[:3])}: {p.stderr.strip()[:300]}")
    return p.stdout


def fetch_develop(repo: str) -> str:
    git(
        repo,
        "fetch",
        "-q",
        "--filter=blob:none",
        "origin",
        "+develop:refs/remotes/origin/develop",
    )
    return git(repo, "rev-parse", "refs/remotes/origin/develop").strip()


DEVELOP_REF = "refs/remotes/origin/develop"


def is_develop(base: str | None) -> bool:
    """A PR's base branch is develop (or not known: develop is the default)."""
    return base in (None, "", "develop")


def base_ref(base: str | None) -> str:
    """The mirror's remote-tracking ref of a PR's base branch (`base.ref`): develop's for develop or none."""
    return DEVELOP_REF if is_develop(base) else f"refs/remotes/origin/{base}"


def fetch_base(repo: str, base: str | None) -> str:
    """Fetch a base branch other than develop into its remote-tracking ref (develop is the mirror's, as the tick fetches
    it); returns the ref."""
    if not is_develop(base):
        git(
            repo,
            "fetch",
            "-q",
            "--filter=blob:none",
            "origin",
            f"+refs/heads/{base}:{base_ref(base)}",
        )
    return base_ref(base)


def fetch_pr(repo: str, n: int) -> str:
    git(
        repo,
        "fetch",
        "-q",
        "--filter=blob:none",
        "origin",
        f"+refs/pull/{n}/head:refs/remotes/origin/pr{n}",
    )
    return git(repo, "rev-parse", f"refs/remotes/origin/pr{n}").strip()


def has_commit(repo: str, sha: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", repo, "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
        ).returncode
        == 0
    )


def first_parent_since(
    repo: str, old: str, new: str, path: str | None = None
) -> list[tuple[str, str]]:
    """(sha, subject) of first-parent commits in old..new, newest first, optionally touching path."""
    args = ["log", "--first-parent", "--format=%H%x09%s", f"{old}..{new}"]
    if path:
        args += ["--", path]
    out = git(repo, *args)
    return [tuple(line.split("\t", 1)) for line in out.splitlines() if "\t" in line]


def first_parent_recent(repo: str, sha: str, n: int) -> list[tuple[str, str]]:
    """(sha, subject) of the newest n first-parent commits up to sha, newest first. Unlike a `sha~n..sha` range it
    works however short the history is (a new repository, or a shallow clone's boundary).
    """
    out = git(repo, "log", "--first-parent", f"-n{n}", "--format=%H%x09%s", sha)
    return [tuple(line.split("\t", 1)) for line in out.splitlines() if "\t" in line]


def changed_files(repo: str, base_ref: str, head_ref: str) -> list[str]:
    """Files the PR changes relative to its merge base with base_ref."""
    mb = git(repo, "merge-base", base_ref, head_ref).strip()
    return git(repo, "diff", "--name-only", mb, head_ref).splitlines()


def merge_tree(
    repo: str, a: str, b: str, base: str | None = None, timeout: int = 600
) -> tuple[str | None, list[str] | None]:
    """A 3-way merge of a and b, written as objects only: (tree id, conflicted paths). The merge base is their own, or
    `base` (which also merges two trees, e.g. two PRs each merged with develop, on develop). ([] = clean; (None, None)
    when git could not decide.) A result always starts with the tree id: git exits 1, the conflict code, with no output
    when it cannot read an object ("not something we can merge"), and that is no clean merge.
    """
    p = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "merge-tree",
            "--write-tree",
            "--name-only",
            "--no-messages",
            *([f"--merge-base={base}"] if base else []),
            a,
            b,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    lines = p.stdout.splitlines()
    tree = (
        lines[0]
        if lines
        and len(lines[0]) in (40, 64)
        and all(c in "0123456789abcdef" for c in lines[0])
        else None
    )
    if p.returncode not in (0, 1) or not tree:
        return None, None
    return tree, (
        [line for line in lines[1:] if line.strip()] if p.returncode == 1 else []
    )


def merge_conflicts(repo: str, a: str, b: str, timeout: int = 600) -> list[str] | None:
    """Files that conflict when merging a and b (their own merge base). [] = clean; None = could not decide."""
    return merge_tree(repo, a, b, timeout=timeout)[1]


def tree_blobs(repo: str, rev: str, paths: list[str]) -> dict[str, str]:
    """Blob ids of `paths` at `rev`, read from its trees alone (a partial clone fetches no blob for it); paths absent
    at `rev` are left out."""
    out = {}
    for i in range(
        0, len(paths), 400
    ):  # a few hundred paths per call keep the command line short
        raw = git(
            repo,
            "--literal-pathspecs",
            "ls-tree",
            "-r",
            "-z",
            rev,
            "--",
            *paths[i : i + 400],
        )
        for entry in raw.split("\0"):
            meta, _, path = entry.partition("\t")
            parts = meta.split()
            if len(parts) == 3 and parts[1] == "blob":
                out[path] = parts[2]
    return out


def cat_blobs(repo: str, specs: list[str], timeout: int = 300) -> dict[str, str | None]:
    """The text of each `rev:path` spec (a tree id works as rev; None when there is no such blob), in one
    `git cat-file --batch`."""
    if not specs:
        return {}
    p = subprocess.run(
        ["git", "-C", repo, "cat-file", "--batch"],
        input="".join(s + "\n" for s in specs).encode(),
        capture_output=True,
        timeout=timeout,
    )
    out, buf, i = {}, p.stdout, 0
    for s in specs:
        nl = buf.find(b"\n", i)
        if nl < 0:
            out[s] = None
            continue
        head = buf[i:nl].split(b" ")
        i = nl + 1
        if len(head) != 3 or not head[2].isdigit():  # "<spec> missing" (or ambiguous)
            out[s] = None
            continue
        size = int(head[2])
        out[s] = (
            buf[i : i + size].decode("utf-8", "replace") if head[1] == b"blob" else None
        )
        i += size + 1  # the content, then a newline
    return out


def is_ancestor(repo: str, a: str, b: str, lazy: bool = True) -> bool:
    return (
        subprocess.run(
            ["git", "-C", repo, "merge-base", "--is-ancestor", a, b],
            capture_output=True,
            timeout=120,
            env=None if lazy else {**os.environ, **NO_LAZY_FETCH},
        ).returncode
        == 0
    )


def expand(repo: str, shorts: list[str]) -> dict[str, str]:
    """Full ids of the abbreviated commit ids the clone has, in one `git cat-file --batch-check`; missing and ambiguous
    ones are left out. No lazy fetch: a prefix cannot be fetched."""
    if not shorts:
        return {}
    p = subprocess.run(
        ["git", "-C", repo, "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        input="".join(s + "\n" for s in shorts),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    out = {}
    for s, line in zip(shorts, p.stdout.splitlines()):
        parts = line.split()
        if len(parts) == 2 and parts[1] == "commit" and parts[0].startswith(s):
            out[s] = parts[0]
    return out
