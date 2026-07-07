################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``TensileLogic.ValidWorkGroupMappingXCC``.

``_validateWorkGroupMappingXCC`` checks that ``WorkGroupMappingXCC`` for a
solution under a ``*_<N>cu`` directory is ``-1`` or a power-of-two dividing the
directory's CU count. It keeps **module-global** per-file failure counts
(``_xcc_failures_by_file``) so it prints one full message + one "...more" per
file rather than per solution.

Determinism: every test calls ``reset_reported_failures()`` first so results
never depend on test order, and snapshots capture both the return *and* the
resulting global state (``{returned, reported_failures}``) — the goal's
"snapshot the state, not just the return". Filepaths are synthetic POSIX
relative paths so the state keys are stable.
"""

from pathlib import Path

import pytest

import Tensile.TensileLogic.ValidWorkGroupMappingXCC as xcc_mod
from Tensile.TensileLogic.ValidWorkGroupMappingXCC import (
    _cu_count_from_path,
    _validateWorkGroupMappingXCC,
    reset_reported_failures,
)

pytestmark = pytest.mark.unit

_CU_DIR = Path("logic/aquavanjaram/gfx942_38cu/Equality/logic.yaml")
_CU64_DIR = Path("logic/aquavanjaram/gfx942_64cu/Equality/logic.yaml")
_PLAIN_DIR = Path("logic/aquavanjaram/gfx942/Equality/logic.yaml")


def _state():
    return dict(sorted(xcc_mod._xcc_failures_by_file.items()))


def _run(solution, filepath):
    """Reset global state, validate once, return {returned, reported_failures}."""
    reset_reported_failures()
    ret = _validateWorkGroupMappingXCC(solution, filepath)
    return {"returned": ret, "reported_failures": _state()}


# --- early-accept branches --------------------------------------------------

def test_non_cu_variant_dir_skips(snapshot):
    # No *_Ncu component -> cu_count == 0 -> skip the check (accept).
    assert _run({"WorkGroupMappingXCC": 4, "SolutionIndex": 0}, _PLAIN_DIR) == snapshot


def test_xcc_minus_one_accepts(snapshot):
    # WorkGroupMappingXCC == -1 is always allowed.
    assert _run({"WorkGroupMappingXCC": -1, "SolutionIndex": 0}, _CU_DIR) == snapshot


def test_xcc_missing_defaults_to_minus_one(snapshot):
    # Absent key defaults to -1 -> accept.
    assert _run({"SolutionIndex": 0}, _CU_DIR) == snapshot


def test_xcc_valid_power_of_two_divisor(snapshot):
    # 64 % 4 == 0 and 4 is a power of two -> accept.
    assert _run({"WorkGroupMappingXCC": 4, "SolutionIndex": 0}, _CU64_DIR) == snapshot


# --- reject branches --------------------------------------------------------

def test_xcc_zero_rejected(snapshot):
    # 0 is neither -1 nor positive.
    assert _run({"WorkGroupMappingXCC": 0, "SolutionIndex": 1}, _CU_DIR) == snapshot


def test_xcc_negative_other_than_minus_one_rejected(snapshot):
    # -2 != -1 and <= 0 -> "must be -1 or positive".
    assert _run({"WorkGroupMappingXCC": -2, "SolutionIndex": 1}, _CU_DIR) == snapshot


def test_xcc_not_power_of_two_rejected(snapshot):
    # 3 > 0 but not a power of two.
    assert _run({"WorkGroupMappingXCC": 3, "SolutionIndex": 1}, _CU_DIR) == snapshot


def test_xcc_power_of_two_not_dividing_cu_rejected(snapshot):
    # 4 is a power of two but 38 % 4 != 0.
    assert _run({"WorkGroupMappingXCC": 4, "SolutionIndex": 1}, _CU_DIR) == snapshot


# --- exception path ---------------------------------------------------------

def test_non_dict_solution_hits_exception_branch(snapshot):
    # solution.get(...) on None raises -> caught -> False (no state change,
    # since the failure occurs before _report_xcc_failure).
    assert _run(None, _CU_DIR) == snapshot


# --- per-file dedup accounting (module-global state) -------------------------

def test_report_dedup_counts_across_solutions(snapshot):
    # Three rejecting solutions in the SAME file, WITHOUT reset between: the
    # global count goes 1 (full message), 2 ("...more"), 3 (silent). Snapshot
    # the accumulated state to pin the one-message-per-file behaviour.
    reset_reported_failures()
    returns = []
    for idx in range(3):
        returns.append(
            _validateWorkGroupMappingXCC(
                {"WorkGroupMappingXCC": 0, "SolutionIndex": idx}, _CU_DIR
            )
        )
    assert {"returns": returns, "reported_failures": _state()} == snapshot


def test_reset_clears_state(snapshot):
    # After accumulating a failure, reset_reported_failures() empties the map.
    reset_reported_failures()
    _validateWorkGroupMappingXCC({"WorkGroupMappingXCC": 0, "SolutionIndex": 0}, _CU_DIR)
    before = _state()
    reset_reported_failures()
    after = _state()
    assert {"before_reset": before, "after_reset": after} == snapshot


# --- _cu_count_from_path ----------------------------------------------------

@pytest.mark.parametrize("name,parts", [
    ("plain_38cu", "logic/gfx942_38cu/Equality/x.yaml"),
    ("uppercase_64CU", "logic/gfx942_64CU/Equality/x.yaml"),
    ("no_cu_suffix", "logic/gfx942/Equality/x.yaml"),
    ("not_at_end", "logic/gfx942_38cu_extra/Equality/x.yaml"),
    ("first_match_wins", "outer_20cu/inner_38cu/x.yaml"),
])
def test_cu_count_from_path(name, parts, snapshot):
    assert _cu_count_from_path(Path(parts)) == snapshot(name=name)
