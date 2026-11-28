# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A tiny local stand-in for <ROCKE_REPO> (no network): a repository with a develop branch, rocKE's folder at
`a/b/rocke`, a CODEOWNERS file, a project outside rocKE's provider, and PR 7 at refs/pull/7/head. It serves blobless,
shallow fetches over file:// as GitHub does. `pull_request` adds an open PR to it, with GitHub's view of it.
"""

import os
import pathlib
import subprocess

ROCKE_PATH = "a/b/rocke"
SINCE = "2001-01-01"
ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_AUTHOR_DATE": "2002-02-02T00:00:00Z",
    "GIT_COMMITTER_DATE": "2002-02-02T00:00:00Z",
}


def git(repo, *args, at: str = "") -> str:
    """git ARGS in REPO, its commits by ENV's author, dated ENV's date, or AT when given."""
    dates = {"GIT_AUTHOR_DATE": at, "GIT_COMMITTER_DATE": at} if at else {}
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **ENV, **dates},
    ).stdout.strip()


def make(base) -> str:
    """Build the repository under `base`; return its file:// URL."""
    origin = pathlib.Path(base) / "origin"
    (origin / ROCKE_PATH / "platform").mkdir(parents=True)
    (origin / ".github").mkdir()
    (origin / "other").mkdir()
    git(origin, "init", "-q", "-b", "develop")
    (origin / ROCKE_PATH / "platform" / "x.txt").write_text("x\n")
    (origin / ROCKE_PATH / "library").mkdir()
    (origin / ROCKE_PATH / "library" / "y.txt").write_text("y\n")
    (origin / "other" / "z.txt").write_text("z\n")
    (origin / ".github" / "CODEOWNERS").write_text(
        f"/{ROCKE_PATH}/           @acme/team-p @acme/team-l\n/{ROCKE_PATH}/platform/  @acme/team-p\n"
        f"/{ROCKE_PATH}/style/     @acme/team-p\n/{ROCKE_PATH}/library/   @acme/team-l\n"
    )
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "develop: first")
    git(origin, "checkout", "-qb", "pr7")
    (origin / ROCKE_PATH / "platform" / "x.txt").write_text("x2\n")
    git(origin, "commit", "-qam", "pr: change x")
    git(origin, "update-ref", "refs/pull/7/head", "pr7")
    git(origin, "checkout", "-q", "develop")
    (origin / "other" / "z.txt").write_text("z2\n")
    git(origin, "commit", "-qam", "develop: change z (#1)")
    git(origin, "config", "uploadpack.allowFilter", "true")
    git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    return origin.resolve().as_uri()


def today(origin) -> str:
    """develop one commit further on in ORIGIN, the repository `make` builds, a commit made now: a clone that starts
    from a recent date, as the monitor's does (deploy.default_since), finds it. develop's head.
    """
    now = {k: v for k, v in ENV.items() if not k.endswith("_DATE")}
    subprocess.run(
        [
            "git",
            "-C",
            str(origin),
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "develop: today",
        ],
        check=True,
        capture_output=True,
        env=dict(os.environ, **now),
    )
    return git(origin, "rev-parse", "refs/heads/develop")


def pull_request(
    origin, n: int, lines: int = 3, at: str = ENV["GIT_COMMITTER_DATE"], pushes: int = 1
) -> dict:
    """Open PR N in ORIGIN, the repository `make` builds: a branch pr<N> from develop with PUSHES commits, each of
    which writes LINES lines of one file under rocKE's folder, dated AT, its head at refs/pull/N/head. GitHub's view of
    it, as the fake GitHub serves it (tests/fakes/fake_servers.py GitHub), with "pushes", each commit's SHA in order.
    """
    base = git(origin, "rev-parse", "develop")
    path = f"{ROCKE_PATH}/platform/change_{n}.txt"
    git(origin, "checkout", "-qb", f"pr{n}", base)
    shas = []
    for k in range(pushes):
        again = f", push {k + 1}" if k else ""
        (pathlib.Path(origin) / path).write_text(
            f"change {n}{again}\n" * lines, encoding="utf-8"
        )
        git(origin, "add", "-A")
        git(origin, "commit", "-qm", f"rocKE: change {n}{again}", at=at)
        shas.append(git(origin, "rev-parse", "HEAD"))
    git(origin, "update-ref", f"refs/pull/{n}/head", f"pr{n}")
    git(origin, "checkout", "-q", "develop")
    diff = subprocess.run(
        ["git", "-C", str(origin), "diff", f"{base}..{shas[-1]}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        "number": n,
        "title": f"rocKE: change {n}",
        "author": "dev-a",
        "labels": ["rocke"],
        "head": shas[-1],
        "head_ref": f"pr{n}",
        "base_sha": base,
        "committed_at": at,
        "additions": lines,
        "deletions": 0,
        "diff": diff,
        "files": [{"path": path, "additions": lines, "deletions": 0}],
        "pushes": shas,
    }
