# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution validation for MX scales loaded through TDM by a single wave on gfx1250.

With TDMInst=3 and NumWaves == 1 (not UseSubtileImpl), each of A, B, MXSA and MXSB
owns a TDM descriptor set, but the single-wave setup only programs A, B and sparse
metadata. The MXSA/MXSB loads then read uninitialized descriptors: measured on
gfx1250 every element is wrong, and the four sets also overflow MaxSgpr so a
normal build drops the kernel at codegen. Solution must reject the combination
and leave the multi-wave MX TDM and single-wave non-MX TDM solutions alone.

The base is the MXF8 block-32 TN solution from
``Tests/common/gemm/gfx12/mxf8_gfx1250.yaml``; the harness mirrors
``test_halfplr_streamk_rejects.py``.
"""

import copy

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))

_REJECT_CLAUSE = "TDM with MX scales requires NumWaves > 1"

# [M, N, K, B, MIBlockM, MIWaveTile0, MIWaveTile1, MIWaveGroup0, MIWaveGroup1]
_ONE_WAVE_MI = [16, 16, 128, 1, 1, 4, 4, 1, 1]
_FOUR_WAVE_MI = [16, 16, 128, 1, 1, 2, 2, 2, 2]

_MX_PROBLEM = {
    "MXBlockA": 32,
    "MXBlockB": 32,
    "DataTypeMXSA": "E8",
    "DataTypeMXSB": "E8",
}

_PERSISTENT_DP = {
    "TileProcessingStrategy": "DataParallel",
    "WorkAssignment": "StaticGrid",
    "GlobalSplitU": 0,
    "ScheduleIterAlg": 0,
}


@pytest.fixture(scope="module")
def gfx1250_iim():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    """Assign process-global parameters for gfx1250; restore after module."""
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

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


def _derive(gfx1250_iim, assembler, capsys, mi, mx=True, **overrides):
    """Construct a Solution with reject printing on; return (sol, stdout)."""
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "F8",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "SupportUserArgs": True,
    }
    if mx:
        problem_type.update(_MX_PROBLEM)
    problem_type.update(overrides.pop("ProblemType", {}))

    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 32,
        "DepthU": 256,
        "KernelLanguage": "Assembly",
        "TDMInst": 3,
        "LDSTrInst": False,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 4,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "InnerUnroll": 1,
        "TransposeLDS": -1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "LdsPadMetadata": 0,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
        "UseSubtileImpl": False,
        "HalfPLR": 0,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1250_iim))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


@pytest.mark.parametrize("persistent", [{}, _PERSISTENT_DP], ids=["gemm", "dp"])
def test_multi_wave_mx_tdm_is_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys, persistent):
    sol, out = _derive(gfx1250_iim, assembler, capsys, _FOUR_WAVE_MI, **persistent)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("NumWaves") == 4
    assert sol.get("enableTDMA") and sol.get("enableTDMB")


@pytest.mark.parametrize("persistent", [{}, _PERSISTENT_DP], ids=["gemm", "dp"])
def test_single_wave_mx_tdm_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys, persistent):
    sol, out = _derive(gfx1250_iim, assembler, capsys, _ONE_WAVE_MI, **persistent)
    assert sol.get("Valid") is False
    assert _REJECT_CLAUSE in out


def test_single_wave_mx_tdm_promoted_from_mx_load_inst_is_rejected(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    # MXLoadInst=TDM promotes TDMInst 0 to 3 before the TDM checks run.
    sol, out = _derive(gfx1250_iim, assembler, capsys, _ONE_WAVE_MI,
                       TDMInst=0, MXLoadInst="TDM")
    assert sol.get("Valid") is False
    assert _REJECT_CLAUSE in out


def test_single_wave_tdm_without_mx_is_not_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, _ONE_WAVE_MI, mx=False)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("NumWaves") == 1
    assert _REJECT_CLAUSE not in out
