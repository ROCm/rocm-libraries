# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.source_allowlist: an allowlist line names its kinds and, where the
list requires it, the issue that removes it; the comparison with the tree names
every breach and keeps the issue in the line it asks for."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import pytest
from rocke.tests.source_allowlist import (
    AllowlistEntry,
    allowlist_breaches,
    format_allowlist_entry,
    read_allowlist,
)


class Kind(enum.Enum):
    FIRST = "first"
    SECOND = "second"


@dataclass(frozen=True)
class Found:
    line: int
    kind: Kind
    detail: str


def _allowlist(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "allowlist.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_line_reads_its_kinds_and_its_issue(tmp_path):
    path = _allowlist(
        tmp_path,
        "# a comment\n\nlisted.py: second first  # AICK-1234\n"
        "other.py: first  # ROCm/rocm-libraries#12\n",
    )
    assert read_allowlist(path, Kind, require_issue=True) == {
        "listed.py": AllowlistEntry(frozenset({Kind.FIRST, Kind.SECOND}), "AICK-1234"),
        "other.py": AllowlistEntry(frozenset({Kind.FIRST}), "ROCm/rocm-libraries#12"),
    }


def test_a_list_without_issues_reads_none(tmp_path):
    path = _allowlist(tmp_path, "listed.py: first\n")
    assert read_allowlist(path, Kind) == {
        "listed.py": AllowlistEntry(frozenset({Kind.FIRST}))
    }


@pytest.mark.parametrize(
    "text, require_issue, message",
    [
        (
            "listed.py: first\n",
            True,
            r"allowlist.txt:1='listed.py: first', expected 'file: kind ...  # <issue>'",
        ),
        (
            "listed.py: first  # none filed yet\n",
            True,
            r"allowlist.txt:1 text='none filed yet', expected the issue that removes",
        ),
        (
            "listed.py: first  # AICK-1234\n",
            False,
            r"allowlist.txt:1 text='AICK-1234', expected nothing after the kinds",
        ),
        (
            "listed.py: third  # AICK-1234\n",
            True,
            r"allowlist.txt:1 kind='third', expected one of \['first', 'second'\]",
        ),
        (
            "listed.py\n",
            False,
            r"allowlist.txt:1='listed.py', expected 'file: kind \.\.\.'",
        ),
        (
            "listed.py: first\nlisted.py: second\n",
            False,
            r"allowlist.txt:2 file='listed.py', expected each file listed once",
        ),
    ],
)
def test_a_malformed_line_is_refused_with_its_place(
    tmp_path, text, require_issue, message
):
    path = _allowlist(tmp_path, text)
    with pytest.raises(ValueError, match=rf"read_allowlist failed - {message}"):
        read_allowlist(path, Kind, require_issue=require_issue)


def test_a_line_formats_with_its_issue():
    kinds = frozenset({Kind.SECOND, Kind.FIRST})
    assert format_allowlist_entry("a.py", kinds) == "a.py: first second"
    assert (
        format_allowlist_entry("a.py", kinds, "AICK-1")
        == "a.py: first second  # AICK-1"
    )


def test_every_breach_is_named_and_a_shortened_line_keeps_its_issue():
    found = {
        "new.py": [Found(3, Kind.FIRST, "pin")],
        "grown.py": [Found(1, Kind.FIRST, "a"), Found(2, Kind.SECOND, "b")],
        "shrunk.py": [Found(4, Kind.FIRST, "c")],
    }
    allowed = {
        "grown.py": AllowlistEntry(frozenset({Kind.FIRST}), "AICK-1"),
        "shrunk.py": AllowlistEntry(frozenset({Kind.FIRST, Kind.SECOND}), "AICK-2"),
        "gone.py": AllowlistEntry(frozenset({Kind.FIRST}), "AICK-3"),
    }
    assert allowlist_breaches(found, allowed, subject="pin") == [
        "gone.py: stale entry, no pin left: delete its line",
        "grown.py: new kind of pin: line 2 second: b",
        "new.py: new pin: line 3 first: pin",
        (
            "shrunk.py: stale kinds, no longer used: second; make the line "
            "'shrunk.py: first  # AICK-2'"
        ),
    ]
