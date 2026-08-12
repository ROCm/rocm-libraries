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

"""Characterization tests for ``Validators.MatrixInstruction``.

Two public functions are pinned:

* ``matrixInstructionToMIParameters`` — converts a 9-item MI into the derived
  MI-parameter dict. Pure (modulo asmCaps booleans read from ``isaInfoMap``);
  we snapshot the returned dict across representative ISAs/dtypes/flags. This
  file covers that conversion (Part 1). ``validateMIParameters`` is pinned in
  ``test_matrixinstruction_validate_char.py`` (Part 2).

Determinism note: the only environment-dependent inputs are the per-ISA
``asmCaps`` booleans (HasMFMA/HasWMMA), derived once from the live assembler
via ``makeIsaInfoMap``. Those booleans are stable for a given ISA + toolchain,
so the snapshots are reproducible in the dev container. See resistance.md.
"""

import pytest

from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.DataType import DataType
from Tensile.Common.Types import IsaVersion
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.SolutionStructs.Validators.MatrixInstruction import (
    matrixInstructionToMIParameters,
)

pytestmark = pytest.mark.unit

# Real ISA capability map (asmCaps queried from the assembler once per session).
_cxx = validateToolchain("amdclang++")
isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, _cxx)

# Representative ISAs (see capability survey in resistance.md):
GFX90A = IsaVersion(9, 0, 10)   # HasMFMA
GFX942 = IsaVersion(9, 4, 2)    # HasMFMA + HasSMFMA
GFX950 = IsaVersion(9, 5, 0)    # HasMFMA + HasSMFMA, isgfx950 path
GFX1100 = IsaVersion(11, 0, 0)  # HasWMMA (navi: not hasMFMA, isa[0]==11)
GFX1200 = IsaVersion(12, 0, 0)  # HasWMMA + HasSWMMAC

# A generic, internally-arbitrary 9-item MI. matrixInstructionToMIParameters
# only requires length 9; values need not be a "valid" instruction.
MI9 = [16, 16, 16, 1, 1, 2, 2, 2, 2]


def _convert(isa, pt, mi=None, wf=64, wg=(16, 16, 1)):
    return matrixInstructionToMIParameters(
        mi if mi is not None else list(MI9),
        isa, wf, pt, list(wg) if wg is not None else wg, isaInfoMap,
    )


# --- length guard -----------------------------------------------------------

def test_wrong_length_raises_value_error():
    with pytest.raises(ValueError):
        _convert(GFX90A, {"DataType": "h"}, mi=[16, 16, 16, 1])


# --- core conversion across ISAs / dtypes -----------------------------------

def test_mfma_gfx90a_half(snapshot):
    assert _convert(GFX90A, {"DataType": "h"}) == snapshot


def test_mfma_gfx942_half(snapshot):
    assert _convert(GFX942, {"DataType": "h"}) == snapshot


def test_mfma_bf16_branch_gfx90a(snapshot):
    # bf16 dtype exercises the MFMA_BF16_1K computation branch.
    assert _convert(GFX90A, {"DataType": "b"}) == snapshot


def test_navi_wmma_gfx1100(snapshot):
    # not hasMFMA & hasWMMA & isa[0]==11 -> MIInputPerThread = mi[2] branch.
    assert _convert(GFX1100, {"DataType": "h"}) == snapshot


def test_swmmac_isa_gfx1200(snapshot):
    assert _convert(GFX1200, {"DataType": "h"}) == snapshot


# --- WorkGroup present vs absent (custom-kernel path) ------------------------

def test_no_workgroup_skips_workgroup_field(snapshot):
    # workGroup falsy -> result has no "WorkGroup" key (branch 87->92).
    assert _convert(GFX90A, {"DataType": "h"}, wg=None) == snapshot


# --- F32 XDL math op enable path --------------------------------------------

def test_f32_xdl_math_op_enabled(snapshot):
    # DataType single (f32) + F32XdlMathOp not single (bf16) -> enableF32xdl.
    pt = {"DataType": DataType("s"), "F32XdlMathOp": DataType("b")}
    assert _convert(GFX90A, pt) == snapshot


# --- sparse paths -----------------------------------------------------------

def test_sparse_a_gfx942(snapshot):
    # Sparse==1 -> sparseA True (MIInputPerThreadA halved); metadata //8.
    assert _convert(GFX942, {"DataType": "h", "Sparse": 1}) == snapshot


def test_sparse_b_gfx942(snapshot):
    # Sparse==2 -> sparseB True (MIInputPerThreadB halved).
    assert _convert(GFX942, {"DataType": "h", "Sparse": 2}) == snapshot


# --- MX scale block paths ---------------------------------------------------

def test_mx_blocks_non_gfx950(snapshot):
    # MXBlockA/B set on a non-gfx950 ISA -> duplicateFactor = 32 // MatrixInst.
    pt = {"DataType": "h", "MXBlockA": 32, "MXBlockB": 32}
    assert _convert(GFX942, pt) == snapshot


def test_mx_blocks_gfx950_workaround(snapshot):
    # gfx950 -> isgfx950 True -> duplicateFactor forced to 1.
    pt = {"DataType": "h", "MXBlockA": 32, "MXBlockB": 32}
    assert _convert(GFX950, pt) == snapshot


def test_sparse_and_mx_combo_gfx950(snapshot):
    pt = {"DataType": "h", "Sparse": 1, "MXBlockA": 32, "MXBlockB": 32}
    assert _convert(GFX950, pt) == snapshot
