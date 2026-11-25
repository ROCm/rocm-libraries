# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A PR's two trees, for the checks that compare its head with its merge base (tools/ctest_labels.py,
tools/moved_refs.py, tools/import_probe.py, and the mutation sweep's scratch copy): the merge base, what the PR adds,
changes, deletes and moves (git's rename detection), a commit's files written to a scratch directory, and the module
names rocKE's Python is imported by. Git only: reads the mirror and fetches the blobs the clone lacks (a read of
GitHub); never writes a ref or a worktree.

Nothing is fetched where lazy fetches are off (GIT_NO_LAZY_FETCH): a PR runner's slot, whose git cannot write the
clone. A check that runs there lists the blobs it reads for its `--fetch` (`fetch_main`), which the monitor runs before
the check's job starts (rkmon_lib/prcode.py, `prf` and `fetch_for`); the check then reads them from the clone.

The checks take the folder of the repository they look at, rocKE's (K), from --rocke-path: `folder_argument` is its
type, and asks the running configuration for its folder (paths.repository_folder) when none is given. No check keeps
a folder of its own.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import tarfile

from . import gitops

DEVELOP = "refs/remotes/origin/develop"
# rocKE's import roots, relative to its tree: the checks' PYTHONPATH (platform/python, then library).
PY_ROOTS = ("platform/python", "library")


class NotRun(Exception):
    """What a check could not read; its section reads NOT RUN with the reason."""


def folder_argument(value: str) -> str:
    """A check's --rocke-path, as argparse's type with the default "": VALUE with no slash at either end; when it is
    blank, the folder the running configuration names (paths.repository_folder). argparse.ArgumentTypeError when
    neither names one."""
    folder = (value or "").strip("/")
    if folder:
        return folder
    # call-time imports: the configuration is read only when no folder is given
    from . import config, paths, store

    try:
        folder = paths.repository_folder(store.load_config())
    except config.ConfigError as exc:
        raise argparse.ArgumentTypeError(
            f"no folder given, and the configuration cannot be read: {exc}"
        ) from None
    if not folder:
        raise argparse.ArgumentTypeError(
            "no folder given, and the configuration names none (repository.folder)"
        )
    return folder


def merge_base(repo: str, base: str, head: str) -> str:
    """merge-base(base, head); NotRun when the head is not in the mirror or the merge base is beyond its history."""
    if not gitops.has_commit(repo, head):
        raise NotRun(f"the head {head[:10]} is not in the mirror")
    try:
        mb = gitops.git(repo, "merge-base", base, head, lazy=False).strip()
    except gitops.GitError:
        mb = ""
    if not mb:
        raise NotRun(
            f"no merge base with {base} in the mirror (beyond its shallow history)"
        )
    return mb


def changes(
    repo: str, a: str, b: str, lazy: bool = True, timeout: int = 600
) -> list[dict]:
    """Each file `b` changes against `a`, with git's rename detection: {"status": "A" (added), "M" (changed, a type
    change too), "D" (deleted) or "R" (moved), "old": its path at `a` (None when added), "new": its path at `b` (None
    when deleted)}. A copy reads as added. `lazy`: the blobs rename detection needs and the clone lacks are fetched
    (git's diff asks for them in one request); otherwise a missing one is an error (gitops.GitError).
    """
    out = gitops.git(
        repo,
        "-c",
        "core.quotepath=false",
        "-c",
        "diff.renameLimit=20000",
        "diff",
        "--name-status",
        "-z",
        "-M",
        "--no-color",
        a,
        b,
        timeout=timeout,
        lazy=lazy,
    ).split("\0")
    got, i = [], 0
    while i < len(out) - 1:
        st = out[i]
        if st[:1] in ("R", "C"):
            got.append(
                {"status": "R", "old": out[i + 1], "new": out[i + 2]}
                if st[0] == "R"
                else {"status": "A", "old": None, "new": out[i + 2]}
            )
            i += 3
            continue
        if st:
            path = out[i + 1]
            got.append(
                {"status": "A", "old": None, "new": path}
                if st[0] == "A"
                else (
                    {"status": "D", "old": path, "new": None}
                    if st[0] == "D"
                    else {"status": "M", "old": path, "new": path}
                )
            )
        i += 2
    return got


def present(repo: str, rev: str, paths: list[str]) -> list[str]:
    """The `paths` (files or directories) that exist at `rev`, read from its trees alone: no blob is fetched."""
    if not paths:
        return []
    out = gitops.git(
        repo,
        "-c",
        "core.quotepath=false",
        "ls-tree",
        "--name-only",
        "-z",
        rev,
        "--",
        *paths,
        lazy=False,
    )
    have = set(out.split("\0"))
    return [p for p in paths if p.rstrip("/") in have]


def blobs(repo: str, rev: str, paths: list[str]) -> list[str]:
    """The ids of the blobs under `paths` (files or directories) at `rev`, read from its trees alone: no blob is
    fetched. No path, no blob."""
    if not paths:
        return []
    listing = gitops.git(repo, "ls-tree", "-r", "-z", rev, "--", *paths, lazy=False)
    return [
        e.split("\t", 1)[0].split()[2]
        for e in listing.split("\0")
        if e and e.split()[1:2] == ["blob"]
    ]


def diff_blobs(repo: str, a: str, b: str, paths: list[str] | None = None) -> list[str]:
    """The ids of the blobs on both sides of each file `b` adds, changes or deletes against `a` (under `paths`, else in
    the whole tree), read from the trees alone: what a diff of the two reads, its rename detection included.
    """
    out = gitops.git(
        repo,
        "diff-tree",
        "-r",
        "-z",
        "--no-renames",
        a,
        b,
        *(["--", *paths] if paths else []),
        lazy=False,
    ).split("\0")
    ids = []
    for meta in out[0::2]:  # each entry: its modes, ids and status, then its path
        parts = meta.lstrip(":").split()
        if len(parts) < 4:
            continue
        for mode, oid in ((parts[0], parts[2]), (parts[1], parts[3])):
            if mode not in (
                "000000",
                "160000",
            ):  # no file on that side; a submodule's commit
                ids.append(oid)
    return ids


def missing(repo: str, ids) -> list[str]:
    """The blobs of `ids` the partial clone lacks, read without fetching anything. GitError when git cannot tell."""
    want = sorted(set(ids))
    if not want:
        return []
    p = subprocess.run(
        ["git", "-C", repo, "cat-file", "--batch-check"],
        input="\n".join(want) + "\n",
        capture_output=True,
        text=True,
        env={**os.environ, **gitops.NO_LAZY_FETCH},
        timeout=120,
    )
    if p.returncode:
        raise gitops.GitError(f"git cat-file --batch-check: {p.stderr.strip()[:300]}")
    return [ln.split()[0] for ln in p.stdout.splitlines() if ln.endswith(" missing")]


def missing_blobs(repo: str, rev: str, paths: list[str]) -> list[str]:
    """The blobs of `paths` at `rev` the partial clone lacks (read without fetching anything)."""
    return missing(repo, blobs(repo, rev, paths))


def fetch(repo: str, ids, timeout: float = 600) -> list[str]:
    """Fetch the blobs of `ids` the partial clone lacks, in one request, holding the clone's lock; the ones still
    missing after it. Where lazy fetches are off (gitops.lazy_fetch_off: a PR runner's slot) nothing is fetched: all
    the missing ones. GitError when git cannot fetch."""
    want = missing(repo, ids)
    if not want or gitops.lazy_fetch_off():
        return want
    with gitops.clone_lock(repo):
        p = subprocess.run(
            [
                "git",
                "-C",
                repo,
                "-c",
                "fetch.negotiationAlgorithm=noop",
                "fetch",
                "-q",
                "--no-tags",
                "--no-write-fetch-head",
                "--recurse-submodules=no",
                "--filter=blob:none",
                "--stdin",
                "origin",
            ],
            input="\n".join(want) + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    if p.returncode:
        raise gitops.GitError(
            f"git fetch of {len(want)} blob(s): {p.stderr.strip()[:300]}"
        )
    return missing(repo, want)


def prefetch(repo: str, rev: str, paths: list[str], timeout: float = 600) -> int:
    """Fetch the blobs of `paths` at `rev` the partial clone lacks in one request (`fetch`), so `git archive` reads
    them locally instead of asking GitHub for one blob after another (slow under load). Best effort: on any failure
    the archive fetches what it needs itself, where lazy fetches are on. How many were missing.
    """
    try:
        want = missing_blobs(repo, rev, paths)
        if want:
            fetch(repo, want, timeout)
        return len(want)
    except (gitops.GitError, OSError, subprocess.SubprocessError, IndexError):
        return 0


FETCH_HELP = (
    "fetch the blobs of the clone the check reads, then stop: the monitor runs it before the check's job runs in a PR "
    "runner's slot, where git fetches nothing"
)


def fetch_main(repo: str, need) -> int:
    """A check's `--fetch`: the blobs `need()` lists fetched into the clone (`fetch`). Prints nothing and exits 0 when
    the clone has them all; else a line that says what it lacks, and exit 1."""
    try:
        left = fetch(repo, need())
    except (
        NotRun,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        IndexError,
    ) as exc:
        print(f"fetch: the blobs this check reads were not fetched: {str(exc)[:300]}")
        return 1
    if left:
        why = (
            "lazy fetches are off here"
            if gitops.lazy_fetch_off()
            else "the remote did not send them"
        )
        print(
            f"fetch: {len(left)} blob(s) this check reads are not in the clone ({why})"
        )
        return 1
    return 0


def extract(
    repo: str, rev: str, paths: list[str], into: pathlib.Path, timeout: int = 900
) -> None:
    """Write `paths` (files or directories) of commit `rev` under `into`, as `git archive` holds them (blobs the clone
    lacks are fetched: first all at once, prefetch, then any left by the archive itself; none where lazy fetches are
    off, and a blob the clone lacks fails it there). The archive reads the blobs of `paths` alone: it takes the
    attributes of the clone's work tree, which has none (no checkout), never the tree's own (with them, git reads
    every blob of the commit's tree first). RuntimeError when git cannot."""
    import time

    into.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    prefetch(repo, rev, paths, timeout=max(10.0, timeout * 0.6))
    timeout = max(10.0, timeout - (time.monotonic() - t0))
    tar = into / ".rkmon_archive.tar"
    try:
        with open(tar, "wb") as fh:
            p = subprocess.run(
                [
                    "git",
                    "-C",
                    repo,
                    "archive",
                    "--worktree-attributes",
                    "--format=tar",
                    rev,
                    "--",
                    *paths,
                ],
                stdout=fh,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        if p.returncode:
            raise RuntimeError(
                f"git archive of {rev[:10]} failed: {p.stderr.decode(errors='replace').strip()[:200]}"
            )
        with tarfile.open(tar) as t:
            t.extractall(into, filter="data")
    finally:
        tar.unlink(missing_ok=True)


def module_name(path: str, K: str) -> str | None:
    """The dotted name rocKE imports a Python file by (`rocke.core.backend`, `kernels.conv.engine_bridge`): its path
    under the first of PY_ROOTS that holds it, a package's `__init__.py` by the package; None outside them or when a
    part is no identifier (a script directory such as `gfx950-tools`)."""
    for root in PY_ROOTS:
        pre = f"{K}/{root}/"
        if path.startswith(pre) and path.endswith(".py"):
            parts = path[len(pre) : -3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if parts and all(p.isidentifier() for p in parts):
                return ".".join(parts)
    return None
