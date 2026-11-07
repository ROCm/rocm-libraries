# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Optional git input for `check --base`: which files and lines changed, and renames.

Git is never required. Without --base the tool compares every file to the baseline,
which is what an installed copy (TheRock) does.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    try:
        p = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True
        )
    except FileNotFoundError as e:
        raise GitError("git is not installed") from e
    if p.returncode:
        raise GitError(p.stderr.strip() or f"git {' '.join(args)} failed")
    return p.stdout


def toplevel(root: Path) -> Path:
    return Path(_git(root, "rev-parse", "--show-toplevel").strip())


def changes(root: Path, base: str) -> dict:
    """Changes between `base` and the working tree, for files under `root`.

    Returns {"changed": {rel: set(new-side line numbers)}, "renamed": {new_rel: old_rel}}.
    Paths are relative to `root` (the rocke folder), not to the git top level.
    """
    top = toplevel(root)
    prefix = root.resolve().relative_to(top.resolve()).as_posix()
    prefix = "" if prefix == "." else prefix + "/"

    def rel(p: str) -> str | None:
        return p[len(prefix) :] if p.startswith(prefix) else None

    renamed = {}
    for line in _git(root, "diff", "-M", "--name-status", base, "--", ".").splitlines():
        parts = line.split("\t")
        if parts[0].startswith("R") and len(parts) == 3:
            old, new = rel(parts[1]), rel(parts[2])
            if old and new:
                renamed[new] = old

    changed: dict[str, set[int]] = {}
    current = None
    for line in _git(
        root, "diff", "-M", "-U0", "--no-color", base, "--", "."
    ).splitlines():
        if line.startswith("+++ "):
            path = line[4:]
            current = rel(path[2:]) if path.startswith("b/") else None
            if current is not None:
                changed.setdefault(current, set())
        elif line.startswith("@@") and current is not None:
            m = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            start, count = int(m.group(1)), int(m.group(2) or 1)
            changed[current].update(range(start, start + count))
    # untracked files count as fully changed
    for path in _git(
        root, "ls-files", "--others", "--exclude-standard", "--full-name", "--", "."
    ).splitlines():
        r = rel(path)
        if r and r.endswith(".py"):
            n = len(
                (top / path).read_text(encoding="utf-8", errors="replace").splitlines()
            )
            changed[r] = set(range(1, n + 1))
    return {"changed": changed, "renamed": renamed}
