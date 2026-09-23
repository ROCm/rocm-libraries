################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Step-1 geometries for gfx1250 TN 32x16x128 MXF4 subtile.

Pins AB/scale/C tile selection and TileInfo grids for the smoke YAML
(MT 128x64, DU 256, WG 2x2). Derivation of that YAML is checked when
amdclang++ can target gfx1250.
"""

import copy

import pytest

from Tensile.Common.DataType import DataType
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Components.Subtile.Kernel import (
    AB_B4_W32_M32,
    AB_B4_W32_N16,
    CD_F32_W32,
    CD_F32_W32_M32,
    MXSA_B4_W32_M32,
    MXSB_B4_W32_N16,
    TileInfo,
    selectABGeometry,
    selectDGeometry,
    selectMXScaleGeometry,
)
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))

# Smoke YAML: MI [32, 16, 128, 1, 1, 2, 2, 2, 2] -> MT 128x64, WG 2x2.
_MT0, _MT1, _DU, _WG = 128, 64, 256, [2, 2]


def _fp4_kernel():
    dtype = DataType("f4")
    return {
        "ProblemType": {"DataTypeA": dtype, "DataTypeB": dtype, "MXBlockA": 32, "MXBlockB": 32},
        "WavefrontSize": 32,
        "MatrixInstM": 32,
        "MatrixInstN": 16,
        "MatrixInstK": 128,
        "MacroTileA": _MT0,
        "MacroTileB": _MT1,
        "MacroTile0": _MT0,
        "MacroTile1": _MT1,
        "DepthU": _DU,
        "_DepthUA": _DU,
        "_DepthUB": _DU,
        "_DepthUMXSA": _DU // 32,
        "_DepthUMXSB": _DU // 32,
        "MIWaveGroup": list(_WG),
        "_ABTilePairA": "AB_B4_W32_M32",
        "_ABTilePairB": "AB_B4_W32_N16",
    }


def test_select_ab_geometry_32x16_fp4():
    kernel = _fp4_kernel()
    assert selectABGeometry(kernel, "A") is AB_B4_W32_M32
    assert selectABGeometry(kernel, "B") is AB_B4_W32_N16
    assert AB_B4_W32_M32.gr.mmaLayout.vgprs == 16
    assert AB_B4_W32_N16.gr.mmaLayout.vgprs == 8
    assert AB_B4_W32_M32.mmaTileShape == (32, 128)
    assert AB_B4_W32_N16.mmaTileShape == (16, 128)


def test_select_scale_and_d_geometry_32x16_fp4():
    kernel = _fp4_kernel()
    assert selectMXScaleGeometry(kernel, "MXSA") is MXSA_B4_W32_M32
    assert selectMXScaleGeometry(kernel, "MXSB") is MXSB_B4_W32_N16
    assert MXSA_B4_W32_M32.gr.mmaTileRegCount == 1.0
    assert MXSB_B4_W32_N16.gr.mmaTileRegCount == 0.5
    assert selectDGeometry(kernel) is CD_F32_W32_M32
    assert CD_F32_W32_M32.mmaTileShape == (32, 16)
    assert CD_F32_W32_M32.mmaTileRegCount == 16


def test_select_d_geometry_wave32_16x16_unchanged():
    kernel = _fp4_kernel()
    kernel["MatrixInstM"] = 16
    assert selectDGeometry(kernel) is CD_F32_W32


def test_tileinfo_grids_match_smoke_yaml():
    kernel = _fp4_kernel()
    tiA = TileInfo(AB_B4_W32_M32, "A", None, kernel)
    tiB = TileInfo(AB_B4_W32_N16, "B", None, kernel)
    tiSA = TileInfo(MXSA_B4_W32_M32, "MXSA", None, kernel)
    tiSB = TileInfo(MXSB_B4_W32_N16, "MXSB", None, kernel)
    tiD = TileInfo(CD_F32_W32_M32, "D", None, kernel)

    assert tiA.globalMMATileGrid == [4, 2]
    assert tiA.localMMATileGrid == [2, 2]
    assert tiB.globalMMATileGrid == [4, 2]
    assert tiB.localMMATileGrid == [2, 2]
    assert tiSA.globalMMATileGrid == [4, 2]
    assert tiSA.localMMATileGrid == [2, 2]
    assert tiSB.globalMMATileGrid == [4, 2]
    assert tiSB.localMMATileGrid == [2, 2]
    assert tiD.globalMMATileGrid == [4, 4]
    assert tiD.localMMATileGrid == [2, 2]
    assert tiA.mmaTileRegCount == 16
    assert tiB.mmaTileRegCount == 8
    assert tiD.mmaTileRegCount == 16


def test_tileinfo_mx_tdm_packed_bytes():
    kernel = _fp4_kernel()
    kernel["enableTDMA"] = True
    kernel["enableTDMB"] = True
    tiSA = TileInfo(MXSA_B4_W32_M32, "MXSA", None, kernel)
    tiSB = TileInfo(MXSB_B4_W32_N16, "MXSB", None, kernel)
    assert tiSA.ldsRowPadBytes == 0
    assert tiSB.ldsRowPadBytes == 0
    assert tiSA.depthUBytes == (_DU // 32) * _MT0  # 8 * 128
    assert tiSB.depthUBytes == (_DU // 32) * _MT1  # 8 * 64


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


def _make_smoke_params(gfx1250_iim, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = [32, 16, 128, 1, 1, 2, 2, 2, 2]
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "F4",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "DataTypeMXSA": "e8",
        "DataTypeMXSB": "e8",
    }
    problem_type.update(overrides.pop("ProblemType", {}))
    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WavefrontSize": 32,
        "DepthU": 256,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "ScheduleIterAlg": 3,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "TransposeLDS": -1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsPadMXSA": -1,
        "LdsPadMXSB": -1,
        "1LDSBuffer": 0,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "SourceSwap": False,
        "TDMInst": 3,
        "StreamK": 3,
        "UseSubtileImpl": True,
        "UseSgprForGRO": 0,
        "ForceDisableShadowInit": True,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "PreloadKernArgs": False,
    }
    params.update(overrides)
    params.update(
        matrixInstructionToMIParameters(
            mi, isa, params["WavefrontSize"], problem_type, None, gfx1250_iim
        )
    )
    return params


def test_smoke_yaml_solution_is_valid(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """subtile_mxf4_gfx1250.yaml knobs must derive Valid after the 32x16 guard lift."""
    sol = Solution(_make_smoke_params(gfx1250_iim), False, True, False, assembler, gfx1250_iim)
    out = capsys.readouterr().out
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("_ABTilePairA") == "AB_B4_W32_M32"
    assert sol.get("_ABTilePairB") == "AB_B4_W32_N16"
    assert sol.get("MatrixInstM") == 32
    assert sol.get("MatrixInstN") == 16
    assert sol.get("MacroTile0") == 128
    assert sol.get("MacroTile1") == 64


def test_wave32_fp4_16x16_subtile_is_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = [16, 16, 128, 1, 1, 2, 2, 2, 2]
    params = _make_smoke_params(gfx1250_iim)
    params["MatrixInstruction"] = mi
    params.update(
        matrixInstructionToMIParameters(
            mi, isa, params["WavefrontSize"], params["ProblemType"], None, gfx1250_iim
        )
    )
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    out = capsys.readouterr().out
    assert sol.get("Valid") is False
    assert "32x16" in out
