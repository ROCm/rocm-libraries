# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Local-read width check for mixed MXFP4 x MXFP8 subtile kernels (gfx950).

A solution is rejected when an operand's local-read width is below the matrix
instruction's input size per thread, except for 8-bit float operands with
MFMA K 64 or 128. The exemption must follow each operand's own type: with
MXFP4 A and MXFP8 B, DataType is A's type, and testing it for B rejected every
mixed kernel that reads B through LDS without DirectToLds.
"""

import copy

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


@pytest.fixture(scope="module")
def _gp_gfx950(gfx950_iim):
    """Assign process-global parameters for gfx950; restore after module."""
    from tensilelite.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from tensilelite.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx950_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


# MT128x32, MIWT 2x2, MIWG 4x1.
_MI = [16, 16, 128, 1, 1, 2, 2, 4, 1]


def _derive(gfx950_iim, assembler, capsys, typeA, typeB):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx950")
    problem_type = {
        "OperationType": "GEMM",
        "DataType": typeA,
        "DataTypeA": typeA,
        "DataTypeB": typeB,
        "MacDataTypeA": typeA,
        "MacDataTypeB": typeB,
        "DestDataType": "b",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
    }
    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": _MI,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 64,
        "DepthU": 256,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 3,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "InnerUnroll": 1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "DirectToLds": 0,
        "StreamK": 0,
        "PrefetchAcrossPersistent": 0,
        "UseSubtileImpl": True,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 4,
    }
    params.update(matrixInstructionToMIParameters(
        _MI, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx950_iim
    ))
    sol = Solution(params, False, True, False, assembler, gfx950_iim)
    return sol, capsys.readouterr().out


@pytest.mark.parametrize("typeA, typeB", [("F4", "F8"), ("F8", "F4")])
def test_mixed_mx_without_direct_to_lds_is_accepted(_gp_gfx950, gfx950_iim, assembler,
                                                     capsys, typeA, typeB):
    sol, out = _derive(gfx950_iim, assembler, capsys, typeA, typeB)
    assert "is less than MIInput" not in out
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
