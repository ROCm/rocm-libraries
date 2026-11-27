# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.source_allowlist: an allowlist line counts its findings of each
kind and, where the list requires it, names the issue that removes it; the
comparison with the tree names every breach and keeps the issue in the line it
asks for."""

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


def test_a_line_reads_its_counts_and_its_issue(tmp_path):
    path = _allowlist(
        tmp_path,
        "# a comment\n\nlisted.py: second=2 first=1  # AICK-1234\n"
        "other.py: first=3  # ROCm/rocm-libraries#12\n",
    )
    assert read_allowlist(path, Kind, require_issue=True) == {
        "listed.py": AllowlistEntry({Kind.FIRST: 1, Kind.SECOND: 2}, "AICK-1234"),
        "other.py": AllowlistEntry({Kind.FIRST: 3}, "ROCm/rocm-libraries#12"),
    }


def test_a_list_without_issues_reads_none(tmp_path):
    path = _allowlist(tmp_path, "listed.py: first=1\n")
    assert read_allowlist(path, Kind) == {"listed.py": AllowlistEntry({Kind.FIRST: 1})}


@pytest.mark.parametrize(
    "text, require_issue, message",
    [
        (
            "listed.py: first=1\n",
            True,
            (
                r"allowlist.txt:1='listed.py: first=1', expected "
                r"'file: kind=count \.\.\.  # <issue>'"
            ),
        ),
        (
            "listed.py: first=1  # none filed yet\n",
            True,
            r"allowlist.txt:1 text='none filed yet', expected the issue that removes",
        ),
        (
            "listed.py: first=1  # AICK-1234\n",
            False,
            r"allowlist.txt:1 text='AICK-1234', expected nothing after the counts",
        ),
        (
            "listed.py: third=1  # AICK-1234\n",
            True,
            r"allowlist.txt:1 kind='third', expected one of \['first', 'second'\]",
        ),
        (
            "listed.py\n",
            False,
            r"allowlist.txt:1='listed.py', expected 'file: kind=count \.\.\.'",
        ),
        (
            "listed.py: first\n",
            False,
            r"allowlist.txt:1='listed.py: first', expected 'file: kind=count \.\.\.'",
        ),
        (
            "listed.py: first=0\n",
            False,
            r"allowlist.txt:1 first='0', expected a whole number above 0",
        ),
        (
            "listed.py: first=two\n",
            False,
            r"allowlist.txt:1 first='two', expected a whole number above 0",
        ),
        (
            "listed.py: first=1 first=2\n",
            False,
            r"allowlist.txt:1 kind='first', expected each kind named once",
        ),
        (
            "listed.py: first=1\nlisted.py: second=1\n",
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
    counts = {Kind.SECOND: 2, Kind.FIRST: 1}
    assert format_allowlist_entry("a.py", counts) == "a.py: first=1 second=2"
    assert (
        format_allowlist_entry("a.py", counts, "AICK-1")
        == "a.py: first=1 second=2  # AICK-1"
    )


@pytest.mark.parametrize("count", [0, -1])
def test_an_entry_refuses_a_count_below_1(count):
    with pytest.raises(
        ValueError,
        match=rf"AllowlistEntry failed - first={count}, expected a whole number above 0",
    ):
        AllowlistEntry({Kind.FIRST: count})


@pytest.mark.parametrize("count", [True, 1.0, "1"])
def test_an_entry_refuses_a_count_that_is_not_an_int(count):
    with pytest.raises(
        TypeError,
        match=rf"AllowlistEntry failed - first={type(count).__name__}, expected int",
    ):
        AllowlistEntry({Kind.FIRST: count})


def test_an_entry_refuses_no_kinds():
    with pytest.raises(ValueError, match=r"expected at least one kind"):
        AllowlistEntry({})


def test_every_breach_is_named_and_a_shortened_line_keeps_its_issue():
    found = {
        "new.py": [Found(3, Kind.FIRST, "pin")],
        "new_kind.py": [Found(1, Kind.FIRST, "a"), Found(2, Kind.SECOND, "b")],
        "more.py": [Found(5, Kind.FIRST, "d"), Found(6, Kind.FIRST, "e")],
        "fewer.py": [Found(7, Kind.FIRST, "f"), Found(8, Kind.SECOND, "g")],
        "shrunk.py": [Found(4, Kind.FIRST, "c")],
    }
    allowed = {
        "new_kind.py": AllowlistEntry({Kind.FIRST: 1}, "AICK-1"),
        "more.py": AllowlistEntry({Kind.FIRST: 1}, "AICK-4"),
        "fewer.py": AllowlistEntry({Kind.FIRST: 3, Kind.SECOND: 1}, "AICK-5"),
        "shrunk.py": AllowlistEntry({Kind.FIRST: 1, Kind.SECOND: 2}, "AICK-2"),
        "gone.py": AllowlistEntry({Kind.FIRST: 1}, "AICK-3"),
    }
    assert allowlist_breaches(found, allowed, subject="pin") == [
        (
            "fewer.py: stale counts, fewer than listed: first 3 -> 1; make the line "
            "'fewer.py: first=1 second=1  # AICK-5'"
        ),
        "gone.py: stale entry, no pin left: delete its line",
        "more.py: more pin than listed: first 1 -> 2: line 5 first: d; line 6 first: e",
        "new.py: new pin: line 3 first: pin",
        "new_kind.py: new kind of pin: line 2 second: b",
        (
            "shrunk.py: stale counts, fewer than listed: second 2 -> 0; make the "
            "line 'shrunk.py: first=1  # AICK-2'"
        ),
    ]


def test_a_line_whose_listed_kinds_are_all_gone_is_deleted_beside_the_new_kind():
    found = {"swapped.py": [Found(1, Kind.SECOND, "b")]}
    allowed = {"swapped.py": AllowlistEntry({Kind.FIRST: 1})}
    assert allowlist_breaches(found, allowed, subject="pin") == [
        "swapped.py: new kind of pin: line 1 second: b",
        "swapped.py: stale counts, fewer than listed: first 1 -> 0; delete its line",
    ]
