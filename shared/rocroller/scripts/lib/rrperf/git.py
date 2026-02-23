# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Git utilities."""

import subprocess
from pathlib import Path


def top(loc: str = None) -> Path:
    path_arg = ["-C", loc] if loc is not None else []
    command = ["git"] + path_arg + ["rev-parse", "--show-toplevel"]
    p = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(p.stdout.strip()).resolve()


def clone(remote: str | Path, repo: Path) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            str(remote),
            str(repo),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "init", "--cone"], cwd=str(repo), check=True
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "shared/rocroller", "shared/mxdatagenerator"],
        cwd=str(repo),
        check=True,
    )


def checkout(repo: Path, commit: str) -> None:
    subprocess.run(["git", "checkout", commit], cwd=str(repo), check=True)


def is_dirty(repo: Path) -> bool:
    p = subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD"], check=False, cwd=str(repo)
    )
    return p.returncode != 0


def branch(repo: Path) -> str:
    p = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )
    return p.stdout.strip()


def short_hash(repo: Path, commit: str = "HEAD") -> str:
    p = subprocess.run(
        ["git", "rev-parse", "--short", commit],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )
    return p.stdout.strip()


def full_hash(repo: Path) -> str:
    p = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )
    return p.stdout.strip()


def rev_list(repo: Path, old_commit: str, new_commit: str) -> list[str]:
    """
    Gets a list of commits, starting with the newest commit and ending
    with the oldest commit, with every commit in between.
    Returns an empty list if there is no path.
    """
    p = subprocess.run(
        [
            "git",
            "rev-list",
            "--ancestry-path",
            f"{old_commit}~..{new_commit}",
        ],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        encoding="ascii",
        check=True,
    )
    return p.stdout.strip().split("\n")


def ls_tree(repo: Path = None):
    """
    Returns a list of all files committed into Git in this repo.
    """

    if repo is None:
        repo = Path.cwd()

    p = subprocess.run(
        [
            "git",
            "ls-tree",
            "--full-tree",
            "--full-name",
            "-r",
            "--name-only",
            "HEAD:shared/rocroller",
        ],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        check=True,
    )

    return p.stdout.decode().strip().split("\n")
