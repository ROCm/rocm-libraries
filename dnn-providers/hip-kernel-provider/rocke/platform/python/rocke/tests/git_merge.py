# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: does git merge two changes to the same files cleanly?

Builds a throwaway repository in ``workdir`` with a base commit, one branch per
change, and merges the second into the first, exactly as two pull requests
landing one after the other would meet::

    clean = merges_cleanly(tmp_path, base, left, right)

``base``, ``left`` and ``right`` map a file path (relative to the repository) to
its text; ``left`` and ``right`` hold only the files each change rewrites. Skips
the test when no ``git`` executable is on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

__all__ = ["merges_cleanly"]

# The repository is throwaway: commits need an identity, and the user's signing
# and hook settings must not apply to it.
_SETTINGS = (
    "-c",
    "user.name=rocke-test",
    "-c",
    "user.email=rocke-test@invalid",
    "-c",
    "commit.gpgsign=false",
    "-c",
    f"core.hooksPath={os.devnull}",
)


def _git(repository: Path, *arguments: str, may_conflict: bool = False) -> str:
    """Run git in ``repository``; return its output. Any failure raises, except
    a conflicted merge when ``may_conflict`` (then the caller reads the status)."""
    result = subprocess.run(
        ["git", *_SETTINGS, "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode and not (may_conflict and "CONFLICT" in output):
        raise RuntimeError(f"git {' '.join(arguments)} failed: {output}")
    return output


def _commit(repository: Path, files: Mapping[str, str], message: str) -> None:
    for name, text in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "-m", message)


def merges_cleanly(
    workdir: Path,
    base: Mapping[str, str],
    left: Mapping[str, str],
    right: Mapping[str, str],
) -> bool:
    """Whether git merges the ``right`` change into the ``left`` one, both made
    on ``base``, without a conflict."""
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    repository = Path(workdir) / "repository"
    repository.mkdir(parents=True)
    for arguments in (("init", "--quiet"), ("checkout", "--quiet", "-b", "base")):
        _git(repository, *arguments)
    _commit(repository, base, "base")
    for branch, change in (("right", right), ("left", left)):
        _git(repository, "checkout", "--quiet", "-b", branch, "base")
        _commit(repository, change, branch)
    output = _git(repository, "merge", "--no-edit", "right", may_conflict=True)
    return "CONFLICT" not in output
