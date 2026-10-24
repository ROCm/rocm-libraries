# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution-validation guard: 1LDSBuffer with ScheduleIterAlg=1 needs PrefetchGlobalRead=1.

1LDSBuffer=1 + SIA1 + ScheduleLocalWrite=0 + PrefetchGlobalRead=2 builds but
produces wrong results on gfx1151 (bf16 TN MT64x96x64 and MT128x112x32,
norm error ~2.4 at every K); the same kernels with PGR=1, or with SIA3 and PGR=2,
are correct. The base solution is a gfx1151 bf16 TN MT64x96x64 WMMA kernel.
"""

import copy

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


@pytest.fixture(scope="module")
def gfx1151_iim():
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1151")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1151")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1151(gfx1151_iim):
    from tensilelite.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from tensilelite.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx1151_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


def _derive(gfx1151_iim, assembler, capsys, **overrides):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1151")
    mi = [16, 16, 16, 1, 1, 2, 3, 2, 2]  # MT64x96, MIWaveGroup [2,2]
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "B",
        "DestDataType": "B",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
    }
    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [32, 4, 1],
        "WavefrontSize": 32,
        "DepthU": 64,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 1,
        "ScheduleLocalWrite": 0,
        "1LDSBuffer": 1,
        "GlobalSplitU": 1,
        "TransposeLDS": 1,
        "LdsPadA": 8,
        "LdsPadB": 8,
        "LdsBlockSizePerPadA": 128,
        "LdsBlockSizePerPadB": 128,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "StoreVectorWidth": 1,
        "GlobalReadVectorWidthA": 8,
        "GlobalReadVectorWidthB": 8,
        "LocalReadVectorWidth": 16,
        "SourceSwap": True,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1151_iim))
    sol = Solution(params, False, True, False, assembler, gfx1151_iim)
    return sol, capsys.readouterr().out


def test_1ldsbuffer_sia1_pgr2_is_rejected(_gp_gfx1151, gfx1151_iim, assembler, capsys):
    sol, out = _derive(gfx1151_iim, assembler, capsys)
    assert sol.get("Valid") is False
    assert "1LDSBuffer with SIA1 requires PrefetchGlobalRead=1" in out


def test_1ldsbuffer_sia1_pgr1_is_accepted(_gp_gfx1151, gfx1151_iim, assembler, capsys):
    sol, out = _derive(gfx1151_iim, assembler, capsys, PrefetchGlobalRead=1)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"


def test_1ldsbuffer_sia3_pgr2_is_accepted(_gp_gfx1151, gfx1151_iim, assembler, capsys):
    sol, out = _derive(gfx1151_iim, assembler, capsys, ScheduleIterAlg=3, ScheduleLocalWrite=1,
                       PrefetchLocalRead=0)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
