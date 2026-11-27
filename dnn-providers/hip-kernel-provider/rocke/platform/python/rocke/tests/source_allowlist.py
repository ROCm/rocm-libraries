# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared half of the tree-wide source checks: the allowlist that holds the
findings a check tolerates until they are removed. (The files a check reads are
those of :func:`rocke.tests.source_tree.source_files`.)

A source check (``rocke.tests.torch_uses``, ``rocke.tests.toolchain_pins``)
reports findings: each has a 1-based ``line``, a ``kind`` (a member of the
check's own ``enum.Enum``) and the ``detail`` of the code it found. Its gate
compares the tree's findings with an allowlist file, one line per file, naming
each kind the file has and how many findings of that kind::

    library/tests/example.py: kind_a=3 kind_b=1

The list only shrinks: a file not on it, a kind its line does not name, or more
findings of a kind than its line counts, fails the gate, and so does a line,
kind or count the tree no longer needs (the stale line is shortened or goes).
A list read with ``require_issue=True`` names, on every line, the issue that
removes it, after the counts; a list read without it names none::

    library/tests/pinned.py: kind_a=1  # AICK-1234

``#`` lines and blank lines are comments.
"""

from __future__ import annotations

import collections
import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from .pending_lanes import ISSUE_PATTERN

__all__ = [
    "AllowlistEntry",
    "Finding",
    "allowlist_breaches",
    "file_kinds",
    "format_allowlist_entry",
    "kind_counts",
    "read_allowlist",
]

_ISSUE_MARK = "  # "
_COUNT_MARK = "="


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
    """One allowlist line: how many findings of each kind its file may still
    have, and the issue that removes them (None on a list that names none)."""

    counts: Mapping[enum.Enum, int]
    issue: str | None = None

    def __post_init__(self) -> None:
        if not self.counts:
            raise ValueError(
                "AllowlistEntry failed - counts={}, expected at least one kind"
            )
        for kind, count in self.counts.items():
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(
                    f"AllowlistEntry failed - {kind.value}={type(count).__name__}, "
                    "expected int"
                )
            if count < 1:
                raise ValueError(
                    f"AllowlistEntry failed - {kind.value}={count!r}, expected a "
                    "whole number above 0"
                )
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    @property
    def kinds(self) -> frozenset[enum.Enum]:
        """The kinds the line names."""
        return frozenset(self.counts)


def file_kinds(findings: Sequence[Finding]) -> frozenset[enum.Enum]:
    """The distinct kinds of ``findings``."""
    return frozenset(finding.kind for finding in findings)


def kind_counts(findings: Sequence[Finding]) -> dict[enum.Enum, int]:
    """How many of ``findings`` have each kind; kinds with none are left out."""
    return dict(collections.Counter(finding.kind for finding in findings))


def format_allowlist_entry(
    path: str, counts: Mapping[enum.Enum, int], issue: str | None = None
) -> str:
    """One allowlist line: ``path: kind=count kind=count``, kinds sorted, then
    ``  # issue`` when there is one."""
    named = sorted(
        f"{kind.value}{_COUNT_MARK}{count}" for kind, count in counts.items()
    )
    line = f"{path}: {' '.join(named)}"
    return line if issue is None else f"{line}{_ISSUE_MARK}{issue}"


def _read_counts(
    tokens: Sequence[str], known: Mapping[str, enum.Enum], where: str, line: str
) -> dict[enum.Enum, int]:
    counts: dict[enum.Enum, int] = {}
    for token in tokens:
        kind_text, marked, count_text = token.partition(_COUNT_MARK)
        if not marked:
            raise ValueError(
                f"read_allowlist failed - {where}={line!r}, expected "
                "'file: kind=count ...'"
            )
        if kind_text not in known:
            raise ValueError(
                f"read_allowlist failed - {where} kind={kind_text!r}, expected one "
                f"of {sorted(known)}"
            )
        if not count_text.isdecimal() or int(count_text) < 1:
            raise ValueError(
                f"read_allowlist failed - {where} {kind_text}={count_text!r}, "
                "expected a whole number above 0"
            )
        kind = known[kind_text]
        if kind in counts:
            raise ValueError(
                f"read_allowlist failed - {where} kind={kind_text!r}, expected each "
                "kind named once"
            )
        counts[kind] = int(count_text)
    return counts


def read_allowlist(
    path: Path, kind_type: type[enum.Enum], *, require_issue: bool = False
) -> dict[str, AllowlistEntry]:
    """The allowlist at ``path``, whose kinds are members of ``kind_type``.

    A malformed line, an unknown kind, a count that is not a whole number above
    0, a kind named twice, a file listed twice, or (with ``require_issue``) a
    line without an issue, or with text after the counts that is not one, is a
    ``ValueError`` naming the line."""
    known = {kind.value: kind for kind in kind_type}
    allowed: dict[str, AllowlistEntry] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{path.name}:{number}"
        entry_text, marked, issue = line.partition(_ISSUE_MARK)
        listed, separator, counts_text = entry_text.partition(": ")
        tokens = counts_text.split()
        if not separator or not tokens:
            raise ValueError(
                f"read_allowlist failed - {where}={line!r}, expected "
                "'file: kind=count ...'"
            )
        counts = _read_counts(tokens, known, where, line)
        if marked and not (require_issue and ISSUE_PATTERN.fullmatch(issue)):
            raise ValueError(
                f"read_allowlist failed - {where} text={issue!r}, expected "
                + (
                    "the issue that removes the line: a Jira id (AICK-1234) or "
                    "ROCm/<repo>#<number>"
                    if require_issue
                    else "nothing after the counts"
                )
            )
        if require_issue and not marked:
            raise ValueError(
                f"read_allowlist failed - {where}={line!r}, expected "
                f"'file: kind=count ...{_ISSUE_MARK}<issue>', the issue that "
                "removes it"
            )
        if listed in allowed:
            raise ValueError(
                f"read_allowlist failed - {where} file={listed!r}, expected each file "
                "listed once"
            )
        allowed[listed] = AllowlistEntry(counts, issue if marked else None)
    return allowed


def _lines_of(findings: Sequence[Finding], kinds: frozenset[enum.Enum]) -> str:
    return "; ".join(
        f"line {finding.line} {finding.kind.value}: {finding.detail}"
        for finding in findings
        if finding.kind in kinds
    )


def _count_changes(
    changed: Sequence[enum.Enum],
    listed: Mapping[enum.Enum, int],
    found: Mapping[enum.Enum, int],
) -> str:
    return ", ".join(
        f"{kind.value} {listed[kind]} -> {found.get(kind, 0)}"
        for kind in sorted(changed, key=lambda kind: kind.value)
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
    not list, or more findings of a kind than its line counts; a listed file has
    fewer findings of a kind than its line counts, none of a kind its line
    lists, or none at all (the line is stale: the list only shrinks, so its
    counts drop and its lines go when the findings go).
    """
    breaches = []
    for path in sorted(set(found) | set(allowed)):
        findings = found.get(path, ())
        counts = kind_counts(findings)
        entry = allowed.get(path)
        if entry is None:
            breaches.append(
                f"{path}: new {subject}: {_lines_of(findings, frozenset(counts))}"
            )
            continue
        if added := frozenset(counts) - entry.kinds:
            breaches.append(
                f"{path}: new kind of {subject}: {_lines_of(findings, added)}"
            )
        if grown := [
            kind
            for kind in entry.kinds & frozenset(counts)
            if counts[kind] > entry.counts[kind]
        ]:
            breaches.append(
                f"{path}: more {subject} than listed: "
                f"{_count_changes(grown, entry.counts, counts)}: "
                f"{_lines_of(findings, frozenset(grown))}"
            )
        if not counts:
            breaches.append(f"{path}: stale entry, no {subject} left: delete its line")
        elif shrunk := [
            kind for kind in entry.kinds if counts.get(kind, 0) < entry.counts[kind]
        ]:
            kept = {
                kind: min(counts[kind], entry.counts[kind])
                for kind in entry.kinds
                if kind in counts
            }
            remedy = (
                f"make the line {format_allowlist_entry(path, kept, entry.issue)!r}"
                if kept
                else "delete its line"
            )
            breaches.append(
                f"{path}: stale counts, fewer than listed: "
                f"{_count_changes(shrunk, entry.counts, counts)}; {remedy}"
            )
    return breaches
