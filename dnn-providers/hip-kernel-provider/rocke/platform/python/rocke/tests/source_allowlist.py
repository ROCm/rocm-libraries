# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared half of the tree-wide source checks: the allowlist that holds the
findings a check tolerates until they are removed. (The files a check reads are
those of :func:`rocke.tests.source_tree.source_files`.)

A source check (``rocke.tests.torch_uses``, ``rocke.tests.toolchain_pins``)
reports findings: each has a 1-based ``line``, a ``kind`` (a member of the
check's own ``enum.Enum``) and the ``detail`` of the code it found. Its gate
compares the tree's findings with an allowlist file, one line per file::

    library/tests/example.py: kind_a kind_b

The list only shrinks: a file not on it, or a kind its line does not name, fails
the gate, and so does a line or kind the tree no longer needs (the stale line
goes). A list read with ``require_issue=True`` names, on every line, the issue
that removes it, after the kinds; a list read without it names none::

    library/tests/pinned.py: kind_a  # AICK-1234

``#`` lines and blank lines are comments.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .pending_lanes import ISSUE_PATTERN

__all__ = [
    "AllowlistEntry",
    "Finding",
    "allowlist_breaches",
    "file_kinds",
    "format_allowlist_entry",
    "read_allowlist",
]

_ISSUE_MARK = "  # "


class Finding(Protocol):
    """One finding of a source check: its line, kind and the code it found."""

    @property
    def line(self) -> int: ...

    @property
    def kind(self) -> enum.Enum: ...

    @property
    def detail(self) -> str: ...


@dataclass(frozen=True)
class AllowlistEntry:
    """One allowlist line: the kinds its file may still have, and the issue that
    removes them (None on a list that names none)."""

    kinds: frozenset[enum.Enum]
    issue: str | None = None


def file_kinds(findings: Sequence[Finding]) -> frozenset[enum.Enum]:
    """The distinct kinds of ``findings``."""
    return frozenset(finding.kind for finding in findings)


def format_allowlist_entry(
    path: str, kinds: frozenset[enum.Enum], issue: str | None = None
) -> str:
    """One allowlist line: ``path: kind kind``, kinds sorted, then ``  # issue``
    when there is one."""
    line = f"{path}: {' '.join(sorted(kind.value for kind in kinds))}"
    return line if issue is None else f"{line}{_ISSUE_MARK}{issue}"


def read_allowlist(
    path: Path, kind_type: type[enum.Enum], *, require_issue: bool = False
) -> dict[str, AllowlistEntry]:
    """The allowlist at ``path``, whose kinds are members of ``kind_type``.

    A malformed line, an unknown kind, a file listed twice, or (with
    ``require_issue``) a line without an issue, or with text after the kinds that
    is not one, is a ``ValueError`` naming the line."""
    known = {kind.value: kind for kind in kind_type}
    allowed: dict[str, AllowlistEntry] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{path.name}:{number}"
        entry_text, marked, issue = line.partition(_ISSUE_MARK)
        listed, separator, kinds_text = entry_text.partition(": ")
        kinds = kinds_text.split()
        if not separator or not kinds:
            raise ValueError(
                f"read_allowlist failed - {where}={line!r}, expected 'file: kind ...'"
            )
        unknown = [kind for kind in kinds if kind not in known]
        if unknown:
            raise ValueError(
                f"read_allowlist failed - {where} kind={unknown[0]!r}, expected one "
                f"of {sorted(known)}"
            )
        if marked and not (require_issue and ISSUE_PATTERN.fullmatch(issue)):
            raise ValueError(
                f"read_allowlist failed - {where} text={issue!r}, expected "
                + (
                    "the issue that removes the line: a Jira id (AICK-1234) or "
                    "ROCm/<repo>#<number>"
                    if require_issue
                    else "nothing after the kinds"
                )
            )
        if require_issue and not marked:
            raise ValueError(
                f"read_allowlist failed - {where}={line!r}, expected "
                f"'file: kind ...{_ISSUE_MARK}<issue>', the issue that removes it"
            )
        if listed in allowed:
            raise ValueError(
                f"read_allowlist failed - {where} file={listed!r}, expected each file "
                "listed once"
            )
        allowed[listed] = AllowlistEntry(
            frozenset(known[kind] for kind in kinds), issue if marked else None
        )
    return allowed


def _lines_of(findings: Sequence[Finding], kinds: frozenset[enum.Enum]) -> str:
    return "; ".join(
        f"line {finding.line} {finding.kind.value}: {finding.detail}"
        for finding in findings
        if finding.kind in kinds
    )


def allowlist_breaches(
    found: Mapping[str, Sequence[Finding]],
    allowed: Mapping[str, AllowlistEntry],
    *,
    subject: str,
) -> list[str]:
    """What breaks the allowlist, one line per file, sorted by file; ``subject``
    names a finding in the messages (``"torch use"``).

    A file not on the list has a finding; a listed file has a kind its line does
    not list; a listed file no longer has a kind its line lists, or any finding
    (the line is stale: the list only shrinks, so its lines go when the
    findings go).
    """
    breaches = []
    for path in sorted(set(found) | set(allowed)):
        findings = found.get(path, ())
        kinds = file_kinds(findings)
        entry = allowed.get(path)
        if entry is None:
            breaches.append(f"{path}: new {subject}: {_lines_of(findings, kinds)}")
            continue
        if added := kinds - entry.kinds:
            breaches.append(
                f"{path}: new kind of {subject}: "
                f"{_lines_of(findings, frozenset(added))}"
            )
        if not kinds:
            breaches.append(f"{path}: stale entry, no {subject} left: delete its line")
        elif gone := entry.kinds - kinds:
            kept = format_allowlist_entry(path, entry.kinds - gone, entry.issue)
            breaches.append(
                f"{path}: stale kinds, no longer used: "
                f"{' '.join(sorted(kind.value for kind in gone))}; make the line {kept!r}"
            )
    return breaches
