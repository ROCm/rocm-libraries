# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Vector widths must be powers of two.

F6 and BF6 pack 21 elements per dword (4 // 0.1875), so the old automatic
VectorWidthA/B search started at 21 and could pick 5, 10 or 21. Those kernels
either failed to generate, because the epilogue cannot encode loads and stores
of that width, or generated and computed wrong results. Explicit widths of 3
and 6, which the config validator accepts, also computed wrong results.

The Solution tests reuse the in-process gfx1250 harness from
``test_halfplr_streamk_rejects``.
"""

import copy

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Solution import Solution, _defaultVectorWidth

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


# --------------------------------------------------------------------------- #
# _defaultVectorWidth
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type_name", ["S", "D", "Z", "H", "B", "F8", "F4", "F6", "B6"])
@pytest.mark.parametrize("mi_wave_tile", range(1, 65))
def test_auto_vector_width_only_changes_non_power_of_two_choices(type_name, mi_wave_tile):
    """The search always picks a power of two, and picks the same width as the
    old search whenever the old one picked a power of two. Only F6 and BF6
    differ, where the old search picked 5, 10 or 21."""
    reg = DataType(type_name).numRegisters()

    def search(start):
        vw = start
        while mi_wave_tile % vw:
            vw //= 2
        return vw

    old = search(int(4 // reg))
    new = search(_defaultVectorWidth(reg))
    assert new & (new - 1) == 0
    if old & (old - 1) == 0:
        assert new == old


# --------------------------------------------------------------------------- #
# Solution derivation on gfx1250
# --------------------------------------------------------------------------- #


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


def _derive(gfx1250_iim, assembler, capsys, mi=None, problem_type_overrides=None, **overrides):
    """F6 TN MX GEMM; default MT160x192 (MIWaveTile [5, 6]) from the issue configs."""
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    if mi is None:
        mi = [16, 16, 128, 1, 1, 5, 6, 2, 2]
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "F6",
        "DestDataType": "B",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
    }
    problem_type.update(problem_type_overrides or {})
    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [32, 4, 1],
        "WavefrontSize": 32,
        "DepthU": 256,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 4,
        "StaggerU": 0,
        "GlobalSplitU": 0,
        "InnerUnroll": 1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "SourceSwap": True,
        "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "TDMInst": 3,
        "LDSTrInst": True,
        "TileProcessingStrategy": "DataParallel",
        "WorkGroupMapping": 8,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1250_iim
    ))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


# Strategy "None" needs GlobalSplitU >= 1; the persistent strategies use 0.
@pytest.mark.parametrize("strategy, gsu", [("None", 1), ("DataParallel", 0), ("StreamK", 0)])
def test_f6_odd_wave_tile_gets_power_of_two_store_width(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, strategy, gsu
):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       TileProcessingStrategy=strategy, GlobalSplitU=gsu)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    # Previously VectorWidthA = StoreVectorWidth = 5.
    assert sol["VectorWidthA"] == 1
    assert sol["StoreVectorWidth"] == 1


# 3 and 6 are accepted by the config validator; 5 is the width the old
# automatic search picked. On gfx1250 these kernels either computed wrong
# results or, with SourceSwap and an odd VectorWidthA, failed to generate.
@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("tc, width", [("A", 3), ("A", 5), ("A", 6), ("B", 3), ("B", 6)])
def test_non_power_of_two_vector_width_is_rejected(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, tc, width, source_swap
):
    wave_tile = {3: 6, 5: 10, 6: 6}[width]  # a MIWaveTile the width divides
    mi = [16, 16, 128, 1, 1, wave_tile, 6, 2, 2] if tc == "A" else [16, 16, 128, 1, 1, 6, wave_tile, 2, 2]
    sol, out = _derive(gfx1250_iim, assembler, capsys, mi=mi, SourceSwap=source_swap,
                       **{"VectorWidth%s" % tc: width})
    assert sol.get("Valid") is False
    assert "VectorWidth%s %u is not a power of two" % (tc, width) in out


# The divisibility checks that follow also reject an odd StoreVectorWidth, but
# WaveTileContiguousA skips them, and with SourceSwap off its native store width
# is 8, so 3 passes its own limit. VectorWidthA=6 is exempt in this mode.
def test_non_power_of_two_store_vector_width_is_rejected(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    sol, out = _derive(gfx1250_iim, assembler, capsys, mi=[16, 16, 128, 1, 1, 12, 4, 2, 2],
                       problem_type_overrides={"DataType": "F4", "DestDataType": "S"},
                       VectorWidthA=6, VectorWidthB=2, StoreVectorWidth=3, SourceSwap=False,
                       WaveTileContiguousA=True, HalfPLR=1, LDSSegmentBRotate=1)
    assert sol.get("Valid") is False
    assert "VectorWidthA 6 is not a power of two" not in out, out
    assert "StoreVectorWidth 3 is not a power of two" in out, out
