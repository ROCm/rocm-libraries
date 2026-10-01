#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Reject newly added absolute imports of the TensileLite package namespaces.

Inside ``projects/hipblaslt/tensilelite/Tensile`` and
``projects/hipblaslt/tensilelite/tensilelite`` every import of ``Tensile`` or
``tensilelite`` (``import X``, ``import X.y``, ``from X import ...``,
``from X.y import ...``) must be package-relative.

Test code (``<package root>/Tests``) is out of scope and never checked.

Only lines that the change adds or modifies are checked, so existing absolute
imports never fail a commit, and a new absolute import in a file that already
has many is the only thing reported. The diff is chosen like this:

* ``PRE_COMMIT_FROM_REF`` / ``PRE_COMMIT_TO_REF`` set (``pre-commit run
  --from-ref/--to-ref``, which is what CI and the pre-push stage do):
  ``FROM...TO``, with file contents read from ``TO``.
* otherwise: ``HEAD`` against the working tree, which is the staged change when
  running as a commit hook.

A multi-line import statement counts as added when its first line (the line
holding ``import`` / ``from``) is added or modified. Appending a name to an
existing parenthesised import therefore does not report that import.

Imports are found with ``ast``, so import-looking text inside strings and
comments is ignored.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

PACKAGE_ROOTS = (
    "projects/hipblaslt/tensilelite/Tensile",
    "projects/hipblaslt/tensilelite/tensilelite",
)
BANNED_TOP_LEVEL_NAMES = frozenset({"Tensile", "tensilelite"})
EXCLUDED_SUBDIRECTORIES = ("Tests",)

# TODO: remove together with the Tensile -> tensilelite compatibility bridge.
# The alias package has to name the canonical package absolutely to install
# the alias.
TEMPORARY_EXEMPTIONS = frozenset(
    {
        (
            "projects/hipblaslt/tensilelite/Tensile/__init__.py",
            "tensilelite._namespace_bridge",
        ),
    }
)

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _in_scope(path: str) -> bool:
    if not path.endswith(".py"):
        return False
    for root in PACKAGE_ROOTS:
        if not path.startswith(root + "/"):
            continue
        relative = path[len(root) + 1 :]
        return relative.split("/", 1)[0] not in EXCLUDED_SUBDIRECTORIES
    return False


def parse_added_lines(diff_text: str) -> dict[str, set[int]]:
    """Map each changed path to the new-side line numbers a ``-U0`` diff adds."""
    added: dict[str, set[int]] = {}
    current: set[int] | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            if target == "/dev/null":
                current = None
            else:
                path = target[2:] if target.startswith("b/") else target
                current = added.setdefault(path, set())
            continue
        match = HUNK_RE.match(line)
        if match and current is not None:
            start = int(match.group(1))
            count = 1 if match.group(2) is None else int(match.group(2))
            current.update(range(start, start + count))
    return added


def banned_imports(source: str, path: str) -> list[tuple[int, int, str]]:
    """Return ``(line, column, module)`` of every banned absolute import."""
    found = []
    for node in ast.walk(ast.parse(source, filename=path)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            continue
        for module in modules:
            if module.split(".")[0] not in BANNED_TOP_LEVEL_NAMES:
                continue
            if (path, module) in TEMPORARY_EXEMPTIONS:
                continue
            found.append((node.lineno, node.col_offset, module))
    return sorted(set(found))


def violations_in(
    path: str, source: str, added_lines: set[int]
) -> list[tuple[int, int, str]]:
    return [v for v in banned_imports(source, path) if v[0] in added_lines]


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", "-c", "core.quotepath=off", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def diff_added_lines(
    from_ref: str | None, to_ref: str | None
) -> dict[str, set[int]]:
    # The pathspec is the whole package roots, not just the files pre-commit
    # passes in: rename detection needs the deleted old path in the diff, and
    # it keeps the diff (and blob fetches in a partial clone) small.
    diff_args = ["diff", "-U0", "--no-color", "--no-ext-diff", "--find-renames"]
    if from_ref and to_ref:
        diff_args.append(f"{from_ref}...{to_ref}")
    else:
        diff_args.append("HEAD")
    return parse_added_lines(_git([*diff_args, "--", *PACKAGE_ROOTS]))


def read_source(path: str, to_ref: str | None) -> str:
    if to_ref:
        return _git(["show", f"{to_ref}:{path}"])
    return Path(path).read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("filenames", nargs="*", help="files pre-commit selected")
    args = parser.parse_args(argv)

    candidates = {PurePosixPath(f).as_posix() for f in args.filenames}
    candidates = {f for f in candidates if _in_scope(f)}
    if not candidates:
        return 0

    from_ref = os.environ.get("PRE_COMMIT_FROM_REF")
    to_ref = os.environ.get("PRE_COMMIT_TO_REF")
    if not (from_ref and to_ref):
        from_ref = to_ref = None

    added = diff_added_lines(from_ref, to_ref)

    failures = 0
    for path in sorted(candidates & added.keys()):
        if not added[path]:
            continue
        source = read_source(path, to_ref)
        try:
            found = violations_in(path, source, added[path])
        except SyntaxError as exc:
            print(
                f"{path}:{exc.lineno or 0}: cannot parse with Python "
                f"{sys.version.split()[0]}, so its imports were not checked: {exc.msg}"
            )
            failures += 1
            continue
        lines = source.splitlines()
        for lineno, col, module in found:
            failures += 1
            print(f"{path}:{lineno}:{col + 1}: absolute import of '{module}'")
            print(f"    {lines[lineno - 1].strip()}")

    if failures:
        print(
            "\nImports of 'Tensile' and 'tensilelite' inside these packages must "
            "be relative\n(e.g. 'from . import x', 'from ..Common import y'). "
            "Only lines added by this\nchange are checked; existing absolute "
            "imports are not reported."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
