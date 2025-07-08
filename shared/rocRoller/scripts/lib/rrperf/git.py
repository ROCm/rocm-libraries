################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

"""Git utilities."""

import subprocess
from pathlib import Path
from typing import List, Union


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


def clone(remote: Union[str, Path], repo: Path) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--recurse-submodules",
            str(remote),
            str(repo),
        ],
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


def rev_list(repo: Path, old_commit: str, new_commit: str) -> List[str]:
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
        ["git", "ls-tree", "--full-tree", "--full-name", "-r", "--name-only", "HEAD"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        check=True,
    )

    return p.stdout.decode().strip().split("\n")
