# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx942 DirectToVgpr swizzle path is not subject to the swizzled-TDM rejects.

The swizzled-TDM rules (SolutionStructs/Solution.py) guard the gfx1250 path only. A gfx942 LibraryLogic
carries swizzle-A kernels that are valid there, e.g. VectorWidthA=2 on a BF16 DirectToVgprA kernel, and
rejecting one is an error ("Any rejection of a LibraryLogic is not expected").
"""

import copy

import pytest

from test_halfplr_streamk_rejects import _PRISTINE_DEFAULT_SOLUTION

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def _gp_gfx942(gfx942_iim):
    from tensilelite.Common.GlobalParameters import defaultSolution, globalParameters, assignGlobalParameters
    from tensilelite.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx942_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


def _derive(iim, assembler, **fork):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Solution import Solution
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

    isa = gfxToIsa("gfx942")
    problem_type = {
        "OperationType": "GEMM", "DataType": "B", "DestDataType": "B", "ComputeDataType": "s",
        "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False, "UseBeta": True,
        "Batched": True, "SupportUserArgs": True, "UseBias": 0, "SwizzleTensorA": True,
    }
    mi = [16, 16, 16, 1, 1, 2, 1, 4, 1]          # MT128x16, one 4-wave group along M
    params = {
        "ProblemType": problem_type, "ISA": isa, "MatrixInstruction": mi, "WavefrontSize": 64,
        "WorkGroup": [64, 4, 1], "DepthU": 64, "TDMInst": 0, "DirectToVgprA": 1, "DirectToVgprB": 0,
        "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 4, "VectorWidthA": 2, "VectorWidthB": 1,
        "LdsPadA": 0, "LdsBlockSizePerPadA": 0, "LdsPadB": 16, "LdsBlockSizePerPadB": 128,
        "LocalReadVectorWidth": 8, "NumLoadsCoalescedA": 2, "NumLoadsCoalescedB": 1,
        "ScheduleIterAlg": 3, "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1, "SourceSwap": True,
        "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1, "StreamK": 0, "StoreVectorWidth": 2,
        "KernelLanguage": "Assembly", "1LDSBuffer": 0, "ExpandPointerSwap": False,
    }
    params.update(fork)
    params.update(matrixInstructionToMIParameters(mi, isa, params["WavefrontSize"], problem_type,
                                                  params["WorkGroup"], iim))
    return Solution(params, False, True, False, assembler, iim)


@pytest.mark.parametrize("vwa,svw", [(1, 1), (2, 2)])
def test_gfx942_dtv_swizzle_a_vector_width_is_valid(_gp_gfx942, gfx942_iim, assembler, capsys, vwa, svw):
    """BF16 swizzle-A on the DirectToVgpr path (gfx942), VectorWidthA 1 and 2."""
    sol = _derive(gfx942_iim, assembler, VectorWidthA=vwa, StoreVectorWidth=svw)
    out = capsys.readouterr().out
    assert sol["Valid"], "rejected: %s" % out.strip()
