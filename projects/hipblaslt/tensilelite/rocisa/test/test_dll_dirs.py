################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Tests for the Windows dependent-DLL directory resolution (_candidate_dll_dirs).

These exercise the pure ordering/dedup/dirname logic directly, with synthetic
inputs, so they run on any platform without a compiled extension, real
directories, or os.add_dll_directory (which exists only on Windows).
"""

import os

from rocisa import _candidate_dll_dirs

_J = os.path.join


def test_dep_dirs_then_sdk_then_ext_dir_in_order():
    dirs = _candidate_dll_dirs(
        [_J("a", "origami.dll"), _J("b", "amdhip64_7.dll")],
        {"ROCM_PATH": "rocm"},
        "extdir",
    )
    assert dirs == ["a", "b", _J("rocm", "bin"), "extdir"]


def test_dedup_preserves_first_occurrence_order():
    # Two deps in the same directory, and ext_dir equal to a dep dir.
    dirs = _candidate_dll_dirs(
        [_J("lib", "one.dll"), _J("lib", "two.dll")],
        {},
        "lib",
    )
    assert dirs == ["lib"]


def test_hip_path_precedes_rocm_path():
    dirs = _candidate_dll_dirs([], {"HIP_PATH": "hip", "ROCM_PATH": "rocm"}, "ext")
    assert dirs == [_J("hip", "bin"), _J("rocm", "bin"), "ext"]


def test_empty_deps_and_env_yields_only_ext_dir():
    assert _candidate_dll_dirs([], {}, "ext") == ["ext"]


def test_falsy_dep_entries_are_skipped():
    dirs = _candidate_dll_dirs([""], {}, "ext")
    assert dirs == ["ext"]
