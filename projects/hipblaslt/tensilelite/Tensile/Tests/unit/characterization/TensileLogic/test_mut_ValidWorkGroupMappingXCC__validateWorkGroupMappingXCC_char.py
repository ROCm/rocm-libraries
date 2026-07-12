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

"""Mutation-targeted characterization tests for
``_validateWorkGroupMappingXCC`` in
``Tensile.TensileLogic.ValidWorkGroupMappingXCC``.

These are add-only tests that pin the *current* (clean-source) behavior of
specific branches so that the corresponding mutmut survivors fail when applied.
Each test resets the module-global per-file failure map first so results never
depend on test order. Tests that pin the printed diagnostic message capture
stdout via ``capsys`` (the message — including the offending value and the
solution's ``SolutionIndex`` — is the only observable for several mutants).
"""

from pathlib import Path

import pytest

from Tensile.TensileLogic.ValidWorkGroupMappingXCC import (
    _validateWorkGroupMappingXCC,
    reset_reported_failures,
)

pytestmark = pytest.mark.unit

_CU38_DIR = Path("logic/aquavanjaram/gfx942_38cu/Equality/logic.yaml")
_CU64_DIR = Path("logic/aquavanjaram/gfx942_64cu/Equality/logic.yaml")
_CU1_DIR = Path("logic/aquavanjaram/gfx942_1cu/Equality/logic.yaml")
_PLAIN_DIR = Path("logic/aquavanjaram/gfx942/Equality/logic.yaml")


# --- cu_count guard boundary (mutants 3, 4) ---------------------------------

def test_non_cu_dir_skips_invalid_xcc():
    # mutant_3: `cu_count <= 0` -> `cu_count < 0`.
    # A non-CU dir yields cu_count == 0. Original skips the whole check and
    # ACCEPTS even an otherwise-invalid xcc. The mutant would not skip and would
    # then reject xcc=3 (not a power of two). Pin the accept.
    reset_reported_failures()
    assert _validateWorkGroupMappingXCC(
        {"WorkGroupMappingXCC": 3, "SolutionIndex": 0}, _PLAIN_DIR
    ) is True


def test_one_cu_dir_validates_invalid_xcc():
    # mutant_4: `cu_count <= 0` -> `cu_count <= 1`.
    # A `*_1cu` dir yields cu_count == 1. Original does NOT skip, so an invalid
    # xcc=3 is rejected (False). The mutant would skip (cu_count <= 1) and
    # accept. Pin the rejection.
    reset_reported_failures()
    assert _validateWorkGroupMappingXCC(
        {"WorkGroupMappingXCC": 3, "SolutionIndex": 0}, _CU1_DIR
    ) is False


# --- xcc==1 accept boundary (mutants 21, 31) --------------------------------

def test_xcc_one_is_accepted_on_cu_dir():
    # mutant_21: `xcc <= 0` -> `xcc <= 1`  (would reject xcc=1 as non-positive)
    # mutant_31: `xcc & (xcc-1)` -> `xcc & (xcc-2)` (1 & -1 == 1 != 0, would
    #            reject xcc=1 as "not a power of two")
    # Original: xcc=1 is positive, is a power of two (1 & 0 == 0), and divides
    # any CU count -> ACCEPT. Pin the accept; this distinguishes both mutants.
    reset_reported_failures()
    assert _validateWorkGroupMappingXCC(
        {"WorkGroupMappingXCC": 1, "SolutionIndex": 0}, _CU38_DIR
    ) is True


# --- diagnostic message content for the "not positive" branch (23, 24) ------

def test_nonpositive_message_includes_index_and_detail(capsys):
    # mutant_23: solution -> None  (drops the real SolutionIndex from message)
    # mutant_24: detail   -> None  (drops the human-readable reason)
    reset_reported_failures()
    ret = _validateWorkGroupMappingXCC(
        {"WorkGroupMappingXCC": 0, "SolutionIndex": 7}, _CU38_DIR
    )
    out = capsys.readouterr().out
    assert ret is False
    assert "WorkGroupMappingXCC must be -1 or positive (WorkGroupMappingXCC=0)" in out
    assert "index: 7" in out


# --- diagnostic message content for the "not power of two" branch (35, 36) ---

def test_not_power_of_two_message_includes_index_and_detail(capsys):
    # mutant_35: solution -> None  (drops SolutionIndex)
    # mutant_36: detail   -> None  (drops reason)
    reset_reported_failures()
    ret = _validateWorkGroupMappingXCC(
        {"WorkGroupMappingXCC": 3, "SolutionIndex": 9}, _CU38_DIR
    )
    out = capsys.readouterr().out
    assert ret is False
    assert "WorkGroupMappingXCC must be -1 or a power of two (WorkGroupMappingXCC=3)" in out
    assert "index: 9" in out


# --- diagnostic message content for the "does not divide" branch (45, 46) ----

def test_does_not_divide_message_includes_index_and_detail(capsys):
    # mutant_45: solution -> None  (drops SolutionIndex)
    # mutant_46: detail   -> None  (drops reason)
    # 4 is a power of two but 38 % 4 != 0.
    reset_reported_failures()
    ret = _validateWorkGroupMappingXCC(
        {"WorkGroupMappingXCC": 4, "SolutionIndex": 5}, _CU38_DIR
    )
    out = capsys.readouterr().out
    assert ret is False
    assert "WorkGroupMappingXCC=4 must divide CU count 38" in out
    assert "index: 5" in out


# --- exception-path diagnostic message (mutant 52) --------------------------

def test_exception_path_prints_error_message(capsys):
    # mutant_52: `print(f"Error: ... {e} ...")` -> `print(None)`.
    # Passing a non-dict solution makes `solution.get(...)` raise; the except
    # block prints a descriptive Error line and returns False. Pin that the
    # printed line is the descriptive message, not the literal "None".
    reset_reported_failures()
    ret = _validateWorkGroupMappingXCC(None, _CU38_DIR)
    out = capsys.readouterr().out
    assert ret is False
    assert "Error: ValidWorkGroupMappingXCC failed:" in out
    assert out.strip() != "None"
