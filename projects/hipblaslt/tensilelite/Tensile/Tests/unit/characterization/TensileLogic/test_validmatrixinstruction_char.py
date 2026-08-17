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

"""Characterization tests for
``TensileLogic.ValidMatrixInstruction._validateMatrixInstruction``.

A thin CLI wrapper around ``Validators.MatrixInstruction.validateMIParameters``:
run the validator, assert ``solution["Valid"]``, return ``True``; on
``AssertionError`` print a tagged message and return ``False``.

Two paths are pinned (snapshot ``{returned, valid}``):

* **accept** — a self-consistent solution (built by running
  ``matrixInstructionToMIParameters`` on a known-good 9-item MI, the same
  trick the ``Validators`` suite uses) with ``Valid`` pre-set; the validator
  leaves ``Valid`` untouched on success, the wrapper's assert passes ->
  ``True``.
* **reject** — an invalid MFMA 4-item MI. The wrapper calls the validator
  with the default ``printSolutionRejectionReason=True``, so ``reject()``
  would *raise* (not just return) if a real ``SolutionIndex`` were present
  ("rejection of a LibraryLogic is not expected"); we set
  ``SolutionIndex=-1`` so ``reject`` instead sets ``Valid=False`` and the
  wrapper's ``assert solution["Valid"]`` produces the caught ``AssertionError``
  -> ``False``. (The ``"reject: ..."`` stdout is incidental and not
  snapshotted.) See ``../resistance.md``.
"""

import copy
from pathlib import Path

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.Types import IsaVersion
from Tensile.SolutionStructs.Validators.MatrixInstruction import (
    matrixInstructionToMIParameters,
)
from Tensile.TensileLogic.ValidMatrixInstruction import _validateMatrixInstruction

pytestmark = pytest.mark.unit

GFX90A = IsaVersion(9, 0, 10)
_FILE = Path("logic/asm/aquavanjaram/mi.yaml")

# Known-good CDNA problem type (mirrors test_MatrixInstructionConversion.py and
# the Validators characterization suite).
_HALF_PT = {
    "OperationType": "GEMM", "DataTypeA": "f8n", "DataTypeB": "h",
    "UseScaleAB": "Scalar", "DataType": "h", "DestDataType": "s",
    "ComputeDataType": "s", "HighPrecisionAccumulate": True,
    "TransposeA": False, "TransposeB": False, "UseBias": 1, "Activation": True,
    "UseScaleAlphaVec": 1, "UseBeta": True, "Batched": True,
    "GroupedGemm": True, "SupportUserArgs": True,
}


def _result(solution, isaInfoMap):
    ret = _validateMatrixInstruction(solution, isaInfoMap, _FILE)
    return {"returned": ret, "valid": solution.get("Valid", "unset")}


def test_accept_consistent_solution(snapshot, isa_info_map):
    pt = dict(_HALF_PT)
    derived = matrixInstructionToMIParameters(
        [32, 32, 8, 1, 1, 31, 16, 4, 2], GFX90A, 48, pt, [16, 16, 1], isa_info_map
    )
    sol = copy.deepcopy(defaultSolution)
    sol["ProblemType"] = pt
    sol.update(derived)
    sol["Valid"] = True
    assert _result(sol, isa_info_map) == snapshot


def test_reject_invalid_mfma(snapshot, isa_info_map):
    sol = {
        "MatrixInstruction": [16, 16, 4, 1], "EnableMatrixInstruction": True,
        "ISA": (9, 0, 10), "ProblemType": {"DataType": "h"},
        "MatrixInstM": 16, "MatrixInstN": 16, "MatrixInstK": 4, "MatrixInstB": 1,
        "MatrixInstBM": 1, "MIWaveTile": [1, 1], "MIWaveGroup": [1, 1],
        "MIBlock": [16, 16, 4, 1, 1, 1], "WavefrontSize": 64,
        "SolutionIndex": -1,
    }
    assert _result(sol, isa_info_map) == snapshot
