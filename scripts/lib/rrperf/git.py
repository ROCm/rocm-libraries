"""Git utilities."""

import subprocess
from pathlib import Path
from typing import List, Union


def top() -> Path:
    p = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
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
