# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocKE review verifier's team member file, as a review in parts reads it (rkmon_lib/split.py) and check 18 reads
a team's own (rkmon_lib/coherence.py).

- FILE: the file's path in an agentic folder (the package's agentic/, a team's skills folder), with forward slashes
  (paths.VERIFIER_FILE, its home).
- CHECKS: each kind of area -> the leads of the checklist lines a part of that kind quotes from the file: the bold lead
  of each one's bullet (What to Check §7 and §8). A part quotes each bullet whole.
- BULLET: a bullet line with a bold lead, `- **…`, at any indent (its indent, then the text after the `**`).
- `leads(kinds=None)`: the leads of KINDS, each once, in CHECKS' order of each kind's; every kind's without KINDS.
- `bullet(lines, lead)`: the index in LINES of the first bullet whose bold text starts with LEAD, or -1: a lead the
  file lacks, which a part names as missing.
- `missing(text)`: the leads of every kind the file's TEXT lacks.

The standard library and rkmon_lib/paths.py only, on the host's Python (deployment.HOST_PYTHON): rkmon runs check 18
on the host.
"""

from __future__ import annotations

import re

from . import paths

FILE = paths.VERIFIER_FILE
CHECKS = {
    "kernels": (
        "New `/`, `%`, shift or index",
        "New size, offset or count",
        "New spec or request fields",
        "Pointers the kernel declares",
        "New guards, defaults",
    ),
    "library": (
        "New spec or request fields",
        "New guards, defaults",
        "New `/`, `%`, shift or index",
        '"Engine import probe"',
    ),
    "bindings": (
        "Launch paths",
        "Pointers the kernel declares",
        "New guards, defaults",
        "New size, offset or count",
    ),
    "platform_cpp": (
        "New IR op handlers",
        "New size, offset or count",
        "New guards, defaults",
        "New `/`, `%`, shift or index",
    ),
    "runtime": ("Launch paths", "New size, offset or count", "New guards, defaults"),
    "platform_python": (
        "New IR op handlers",
        "New guards, defaults",
        "New spec or request fields",
        '"Engine import probe"',
    ),
    "rust": ("New IR op handlers", "New size, offset or count", "New guards, defaults"),
    "tools": ("New guards, defaults", '"References to moved or deleted files"'),
    "tests": ("New test assertions", '"CTest labels and tiers"'),
    "docs": (
        "Changed docs, comments and PR-body claims",
        '"References to moved or deleted files"',
    ),
    "build": ('"CTest labels and tiers"', '"References to moved or deleted files"'),
    "code": ("New size, offset or count", "New guards, defaults", "Launch paths"),
}
BULLET = re.compile(r"^(\s*)- \*\*(.+)$")


def leads(kinds=None) -> list:
    """The leads of KINDS, each once: see the module's docstring."""
    out = []
    for kind in CHECKS if kinds is None else kinds:
        for lead in CHECKS.get(kind, ()):
            if lead not in out:
                out.append(lead)
    return out


def bullet(lines: list, lead: str) -> int:
    """The index of LEAD's bullet in LINES, or -1: see the module's docstring."""
    for i, line in enumerate(lines):
        m = BULLET.match(line)
        if m and m.group(2).startswith(lead):
            return i
    return -1


def missing(text: str) -> list:
    """The leads of every kind TEXT lacks: see the module's docstring."""
    lines = text.splitlines()
    return [lead for lead in leads() if bullet(lines, lead) < 0]
