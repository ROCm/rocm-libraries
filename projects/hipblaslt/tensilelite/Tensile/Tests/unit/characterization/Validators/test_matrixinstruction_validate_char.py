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
``Validators.MatrixInstruction.validateMIParameters`` (Part 2).

This function is assertion-heavy and validates a fully-derived Solution. Two
construction strategies are used:

* **Consistent solutions** (happy / navi / bf16-1k paths): built by running
  ``matrixInstructionToMIParameters`` on a known-valid 9-item MI and merging
  into ``defaultSolution`` (deep-copied so the shared global is never
  mutated). This makes every internal consistency assertion pass.
* **Minimal solutions** (reject paths, early returns): hand-built dicts with
  just enough keys to reach the branch under test. Reject paths run with
  ``printSolutionRejectionReason=False`` so ``reject`` silently sets
  ``state["Valid"]=False`` and returns — deterministic, no stdout, no raise.

Snapshot shape: ``{"returned": <bool>, "valid": <Valid or "unset">}``.
"""

import copy
import pytest

from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.Types import IsaVersion
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.SolutionStructs.Validators.MatrixInstruction import (
    matrixInstructionToMIParameters,
    validateMIParameters,
)

pytestmark = pytest.mark.unit

_cxx = validateToolchain("amdclang++")
isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, _cxx)

GFX90A = IsaVersion(9, 0, 10)
GFX942 = IsaVersion(9, 4, 2)
GFX1100 = IsaVersion(11, 0, 0)
GFX1200 = IsaVersion(12, 0, 0)


def _result(sol, printReason=False):
    ret = validateMIParameters(sol, isaInfoMap, printReason)
    return {"returned": ret, "valid": sol.get("Valid", "unset")}


def _consistent(isa, pt, mi9, wf, wg=(16, 16, 1)):
    """Build a self-consistent solution via the conversion function."""
    derived = matrixInstructionToMIParameters(mi9, isa, wf, pt, list(wg), isaInfoMap)
    sol = copy.deepcopy(defaultSolution)
    sol["ProblemType"] = pt
    sol.update(derived)
    return sol


# --- happy paths (consistent solutions) -------------------------------------

# Known-good CDNA config (mirrors test_MatrixInstructionConversion.py).
_HALF_PT = {
    "OperationType": "GEMM", "DataTypeA": "f8n", "DataTypeB": "h",
    "UseScaleAB": "Scalar", "DataType": "h", "DestDataType": "s",
    "ComputeDataType": "s", "HighPrecisionAccumulate": True,
    "TransposeA": False, "TransposeB": False, "UseBias": 1, "Activation": True,
    "UseScaleAlphaVec": 1, "UseBeta": True, "Batched": True,
    "GroupedGemm": True, "SupportUserArgs": True,
}


def test_validate_happy_cdna_gfx90a(snapshot):
    sol = _consistent(GFX90A, dict(_HALF_PT), [32, 32, 8, 1, 1, 31, 16, 4, 2], wf=48)
    assert _result(sol) == snapshot


def test_validate_happy_navi_wmma_gfx1100(snapshot):
    # Navi WMMA: valid WMMA mi4, exercises the isa[0]==11 / navi-range
    # MIInputPerThread asserts (L321-327).
    sol = _consistent(GFX1100, {"DataType": "h"}, [16, 16, 16, 1, 1, 2, 2, 2, 2], wf=32)
    assert _result(sol) == snapshot


def test_validate_bf16_1k_branch_gfx90a(snapshot):
    # bf16 mi4 in validMFMA["B1k"] but not in ["BB"] -> MFMA_BF16_1K assert path.
    sol = _consistent(GFX90A, {"DataType": "b"}, [32, 32, 4, 2, 1, 2, 2, 2, 2], wf=64)
    assert _result(sol) == snapshot


def test_validate_happy_sparse_smfma_gfx942(snapshot):
    # Sparse + valid SMFMA mi4 -> passes the sparse validity check and falls
    # through to the post-validity asserts (branch 297->303).
    sol = _consistent(GFX942, {"DataType": "h", "Sparse": 1},
                      [16, 16, 32, 1, 1, 2, 2, 2, 2], wf=64)
    assert _result(sol) == snapshot


def test_validate_happy_wmma_gfx1200(snapshot):
    # gfx1200: hasWMMA, not hasMFMA, isa[0]==12 -> the (isa[0]==10 or 11) check
    # is False (branch 322->326), then the navi-range check is also False.
    sol = _consistent(GFX1200, {"DataType": "h"},
                      [16, 16, 16, 1, 1, 2, 2, 2, 2], wf=32)
    assert _result(sol) == snapshot


def test_validate_no_metadata_key(snapshot):
    # A valid solution without "MIInputPerThreadMetadata" -> branch 331->334
    # (the optional-key check is skipped) and the function returns True.
    sol = _consistent(GFX90A, dict(_HALF_PT), [32, 32, 8, 1, 1, 31, 16, 4, 2], wf=48)
    sol.pop("MIInputPerThreadMetadata", None)
    assert _result(sol) == snapshot


# --- early returns ----------------------------------------------------------

def test_validate_empty_mi_returns_true(snapshot):
    sol = {
        "MatrixInstruction": [], "EnableMatrixInstruction": False,
        "ISA": (9, 0, 10), "ProblemType": {"DataType": "h"},
    }
    assert _result(sol) == snapshot


def test_validate_mi_disabled_returns_false(snapshot):
    sol = {
        "MatrixInstruction": [16, 16, 16, 1], "EnableMatrixInstruction": False,
        "ISA": (9, 0, 10), "ProblemType": {"DataType": "h"},
        "MatrixInstM": 16, "MatrixInstN": 16, "MatrixInstK": 16, "MatrixInstB": 1,
        "MatrixInstBM": 1, "MIWaveTile": [1, 1], "MIWaveGroup": [1, 1],
    }
    assert _result(sol) == snapshot


# --- reject paths (minimal dicts) -------------------------------------------

def _reject_dict(isa, mi4, sparse=False):
    pt = {"DataType": "h"}
    if sparse:
        pt["Sparse"] = 1
    return {
        "MatrixInstruction": list(mi4), "EnableMatrixInstruction": True,
        "ISA": tuple(isa), "ProblemType": pt,
        "MatrixInstM": mi4[0], "MatrixInstN": mi4[1],
        "MatrixInstK": mi4[2], "MatrixInstB": mi4[3],
        "MatrixInstBM": 1, "MIWaveTile": [1, 1], "MIWaveGroup": [1, 1],
        "MIBlock": [mi4[0], mi4[1], mi4[2], mi4[3], 1, 1], "WavefrontSize": 64,
    }


def test_validate_reject_mfma_invalid(snapshot):
    # gfx90a MFMA: mi4 in validMatrixInstructions but not in validMFMA["HH"].
    assert _result(_reject_dict(GFX90A, [16, 16, 4, 1])) == snapshot


def test_validate_dtype_key_fallback_mixed_mac(snapshot):
    # MacDataTypeA=h, MacDataTypeB=b -> key "HB" is not in the MFMA/SMFMA
    # tables, exercising the dtype-key fallback block (the macA==macB check is
    # False, the cb+ca "BH" check is also False). mi4 is table-valid so the
    # flow reaches the dtype-key usage, then rejects on the unknown key.
    sol = _reject_dict(GFX90A, [32, 32, 8, 1])
    sol["ProblemType"]["MacDataTypeA"] = "h"
    sol["ProblemType"]["MacDataTypeB"] = "b"
    assert _result(sol) == snapshot


def test_validate_reject_wmma_invalid(snapshot):
    # gfx1100 WMMA: mi4 not in validWMMA.
    assert _result(_reject_dict(GFX1100, [32, 32, 8, 1])) == snapshot


def test_validate_reject_smfma_invalid(snapshot):
    # gfx942 sparse SMFMA: mi4 not in validSMFMA["HH"].
    assert _result(_reject_dict(GFX942, [16, 16, 4, 1], sparse=True)) == snapshot


def test_validate_reject_swmmac_invalid(snapshot):
    # gfx1200 sparse SWMMAC: mi4 not in validSWMMAC.
    assert _result(_reject_dict(GFX1200, [16, 16, 16, 1], sparse=True)) == snapshot
