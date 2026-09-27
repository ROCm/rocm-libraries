# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the Windows dependent-DLL directory resolution (_candidate_dll_dirs).

These exercise the pure ordering/dedup/dirname logic directly, with synthetic
inputs -- no real directories and no os.add_dll_directory (which exists only on
Windows), so they run on any platform. Like the sibling helper tests (see
test_staleness.py), importing rocisa still loads the built _rocisa extension.
"""

import os

from rocisa import _candidate_dll_dirs

_J = os.path.join


def test_dep_dirs_then_ext_dir_in_order():
    dirs = _candidate_dll_dirs(
        [_J("a", "origami.dll"), _J("b", "amdhip64_7.dll")],
        "extdir",
    )
    assert dirs == ["a", "b", "extdir"]


def test_dedup_preserves_first_occurrence_order():
    # Two deps in the same directory, and ext_dir equal to a dep dir.
    dirs = _candidate_dll_dirs(
        [_J("lib", "one.dll"), _J("lib", "two.dll")],
        "lib",
    )
    assert dirs == ["lib"]


def test_empty_deps_yields_only_ext_dir():
    assert _candidate_dll_dirs([], "ext") == ["ext"]


def test_falsy_dep_entries_are_skipped():
    dirs = _candidate_dll_dirs([""], "ext")
    assert dirs == ["ext"]
