# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""UseSubtileImpl on wave32 (gfx1250) is implemented only for bf16/fp16 A/B.

The only wave32 subtile geometry is AB_B16_W32.  The 8/6/4-bit geometries and
the MX scale layout model the gfx950 wave64 MFMA operands, so on gfx1250 an fp8
subtile kernel emits WMMA operands of the wrong width (it fails to assemble) and
an MX fp4 one computes a wrong D.  Solution derivation must reject them with a
clear reason, while the 16-bit candidates stay valid.

The harness follows test_halfplr_streamk_rejects.py: real gfx1250 capability
maps and assembler feed Solution.__init__, and the reject reason is read from
stdout.
"""

import copy

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))

REJECT_REASON = "UseSubtileImpl=1 with WavefrontSize 32 is implemented only for bf16/fp16"


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


def _derive(gfx1250_iim, assembler, capsys, problemType, miK, depthU, grvw):
    """Derive the TDM subtile TN candidate of sk_bf16gemm_tdm_subtile.yaml for a dtype."""
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = [16, 16, miK, 1, 1, 3, 3, 2, 2]
    pt = {
        "OperationType": "GEMM",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        **problemType,
    }
    params = {
        "ProblemType": pt,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 32,
        "DepthU": depthU,
        "KernelLanguage": "Assembly",
        "UseSubtileImpl": True,
        "TDMInst": 3,
        "LDSTrInst": True,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 3,
        "TileProcessingStrategy": "DataParallel",
        "WorkAssignment": "StaticGrid",
        "StaggerU": 0,
        "GlobalSplitU": 0,
        "GlobalReadVectorWidthA": grvw,
        "GlobalReadVectorWidthB": grvw,
        "TransposeLDS": -1,
        "ClusterLocalRead": 0,
        "1LDSBuffer": 0,
        "SourceSwap": True,
        "ExpandPointerSwap": False,
        "PrefetchAcrossPersistent": 0,
        "StoreRemapVectorWidth": 0,
        "WorkGroupMapping": 1,
    }
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], pt, params["WorkGroup"], gfx1250_iim))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


@pytest.mark.parametrize("dtype", ["b", "h"])
def test_16bit_subtile_is_accepted_on_wave32(_gp_gfx1250, gfx1250_iim, assembler, capsys, dtype):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       {"DataType": dtype, "DestDataType": dtype}, miK=32, depthU=128, grvw=8)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["_ABTilePairA"] == "AB_B16_W32"
    assert sol["_ABTilePairB"] == "AB_B16_W32"


@pytest.mark.parametrize("problemType,miK,depthU,grvw", [
    pytest.param({"DataType": "F8", "DestDataType": "s"}, 64, 128, 16, id="fp8-mik64"),
    pytest.param({"DataType": "F8", "DestDataType": "s"}, 128, 128, 16, id="fp8-mik128"),
    pytest.param({"DataType": "B8", "DestDataType": "s"}, 64, 128, 16, id="bf8-mik64"),
    pytest.param({"DataType": "F4", "DestDataType": "s", "MXBlockA": 32, "MXBlockB": 32},
                 128, 256, 32, id="mxfp4-mik128"),
])
def test_non_16bit_subtile_is_rejected_on_wave32(_gp_gfx1250, gfx1250_iim, assembler, capsys,
                                                problemType, miK, depthU, grvw):
    sol, out = _derive(gfx1250_iim, assembler, capsys, problemType, miK, depthU, grvw)
    assert sol.get("Valid") is False
    assert REJECT_REASON in out, out
