# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 fp4/fp6 AssertSummationElementMultiple overrides raise, never lower.

The fp4 and fp6 tail loops require K to be a multiple of 8 and 16 elements. The
override used to assign those values outright, which silently discarded a larger
user assertion: an fp4 kernel with AssertSummationElementMultiple == DepthU
could never be NoTailLoop. Each test derives a real gfx1250 Solution in-process.
"""

import copy

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

# Sibling tests mutate the process-global defaultSolution in place; snapshot it at import.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from tensilelite.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from tensilelite.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


def _derive(gfx1250_iim, assembler, capsys, dataType, **overrides):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = [16, 16, 128, 1, 1, 2, 2, 2, 2]
    problemType = {
        "OperationType": "GEMM",
        "DataType": dataType,
        "DestDataType": "F8",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "DataTypeMXSA": "E8",
        "DataTypeMXSB": "E8",
    }
    params = {
        "ProblemType": problemType,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 32,
        "DepthU": 256,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 4,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "TDMInst": 3,
        "MXLoadInst": "TDM",
        "MXScaleFormat": "InMemorySwizzle",
        "LDSTrInst": False,
        "SourceSwap": True,
        "WorkGroupMapping": 1,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problemType, params["WorkGroup"], gfx1250_iim))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


@pytest.mark.parametrize("dataType, floor", [("F4", 8), ("B6", 16)])
def test_default_assertion_is_raised_to_the_tail_loop_floor(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, dataType, floor
):
    sol, out = _derive(gfx1250_iim, assembler, capsys, dataType, AssertSummationElementMultiple=1)
    assert sol.get("Valid") is True, out
    # The MX block (32) raises the floor further.
    assert sol["AssertSummationElementMultiple"] == max(floor, 32)
    assert sol["NoTailLoop"] is False


@pytest.mark.parametrize("dataType", ["F4", "B6"])
@pytest.mark.parametrize("depthU", [256, 512])
def test_assertion_of_depthu_is_kept_and_removes_the_tail_loop(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, dataType, depthU
):
    sol, out = _derive(gfx1250_iim, assembler, capsys, dataType,
                       DepthU=depthU, AssertSummationElementMultiple=depthU)
    assert sol.get("Valid") is True, out
    assert sol["AssertSummationElementMultiple"] == depthU
    assert sol["NoTailLoop"] is True
