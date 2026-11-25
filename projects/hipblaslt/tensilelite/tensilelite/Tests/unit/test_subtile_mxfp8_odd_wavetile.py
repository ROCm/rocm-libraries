# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Odd MIWaveTile for the mxfp8 subtile path (gfx950).

MX scales are host-pre-swizzled in 32-row groups, two 16-row MMA tiles per
group, and the scale local read packs 2 tiles in M x 2 in K per VGPR
(MXScaleLRGeometry.subtileShape = (2, 2)).  An even MIWaveTile keeps every wave
and workgroup on a group boundary, so the tile pair sharing a dword is fixed at
compile time.  With an odd MIWaveTile a wave or workgroup can start in the
second half of a group, so mxfp8 reads one byte-rotated dword per tile through
two address registers set up for the runtime start parity
(TileInfo.perTileScale).

These tests pin the contract:
  * odd MIWaveTile mxfp8 solutions VALIDATE on the dual-period path (DU128,
    PGR2): 16-wide, 48/80-wide single-wave, multi-wave, and both dims odd.
  * they are rejected off that path (DU256), as is 16x16 (not dual-period) and
    a tile whose data global read would start a wave mid-load.
  * a small even tile like [2,2] still does not take dual-period at DU128.
  * the emitted assembly has the per-tile structure (one scale read per tile,
    two address registers, the start-parity setup).
"""

import copy
import shutil

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


@pytest.fixture(scope="module")
def gfx950_iim():
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx950")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx950")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx950(gfx950_iim):
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


def _make_params(gfx950_iim, mi, depth_u):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx950")
    problem_type = {
        "OperationType": "GEMM", "DataType": "F8", "DestDataType": "b",
        "ComputeDataType": "s", "HighPrecisionAccumulate": True,
        "TransposeA": True, "TransposeB": False, "UseBeta": True,
        "Batched": True, "StridedBatched": True, "MXBlockA": 32, "MXBlockB": 32,
    }
    params = {
        "ProblemType": problem_type, "ISA": isa, "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1], "WavefrontSize": 64, "DepthU": depth_u,
        "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": 0,
        "ScheduleIterAlg": 3, "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1,
        "LdsPadA": -1, "LdsPadB": -1, "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0, "VectorWidthA": -1, "VectorWidthB": -1, "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1, "LocalReadVectorWidth": -1,
        "SourceSwap": False, "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer", "DirectToLds": 1, "StreamK": 0,
        "PrefetchAcrossPersistent": 0, "UseSubtileImpl": True, "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False, "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False, "WorkGroupMapping": 4,
    }
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx950_iim))
    return params


def _solve(gfx950_iim, assembler, mi, depth_u=128):
    params = _make_params(gfx950_iim, mi, depth_u)
    return Solution(params, False, True, False, assembler, gfx950_iim)


# MIWaveTile [M, N] encoded at mi[5], mi[6]; MIWaveGroup at mi[7], mi[8].
# MacroTile = 16 * MIWaveTile * MIWaveGroup.
_SKINNY_M    = [16, 16, 128, 1, 1, 1, 4, 1, 4]  # MT16x256  16-wide M
_SKINNY_N    = [16, 16, 128, 1, 1, 4, 1, 4, 1]  # MT256x16  16-wide N
_ODD_M3      = [16, 16, 128, 1, 1, 3, 4, 1, 1]  # MT48x64   one wave, 3 tiles
_ODD_M5_1W   = [16, 16, 128, 1, 1, 5, 4, 1, 2]  # MT80x128  one wave in M, 5 tiles
_ODD_M5_2W   = [16, 16, 128, 1, 1, 5, 4, 2, 2]  # MT160x128 two waves in M, 5 tiles each
_ODD_N5_2W   = [16, 16, 128, 1, 1, 4, 5, 2, 2]  # MT128x160 two waves in N
_BOTH_ODD    = [16, 16, 128, 1, 1, 3, 3, 1, 1]  # MT48x48   both odd, one wave
_BOTH_ODD_2W = [16, 16, 128, 1, 1, 5, 5, 2, 2]  # MT160x160 both odd, 2x2 waves
_SKINNY_MWG  = [16, 16, 128, 1, 1, 1, 4, 2, 2]  # MT32x128  1 tile per wave, two waves in M
_EVEN_SMALL  = [16, 16, 128, 1, 1, 2, 2, 1, 1]  # MT32x32   (even, not partitionable)
_EVEN        = [16, 16, 128, 1, 1, 2, 4, 1, 4]  # MT32x256  (even control)
_SKINNY_SQ   = [16, 16, 128, 1, 1, 1, 1, 1, 1]  # MT16x16   (cannot partition -> not dual-period)
# A's global read covers two strips per load (8 waves) and M has two waves of 5
# strips: wave 1 would start on the second strip of a shared load.
_SHARED_GR   = [16, 16, 128, 1, 1, 5, 4, 2, 4]  # MT160x256

_ODD_SUPPORTED = [_SKINNY_M, _SKINNY_N, _ODD_M3, _ODD_M5_1W, _ODD_M5_2W, _ODD_N5_2W,
                  _BOTH_ODD, _BOTH_ODD_2W, _SKINNY_MWG]


# ---------------------------------------------------------------------------
# Gate: odd MIWaveTile mxfp8 validates and takes the dual-period path.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mi", _ODD_SUPPORTED)
def test_odd_miwavetile_mxfp8_validates(_gp_gfx950, gfx950_iim, assembler, mi):
    sol = _solve(gfx950_iim, assembler, mi, depth_u=128)
    assert sol.get("Valid") is True, f"odd MIWaveTile mxfp8 rejected: mi={mi}"
    assert sol.get("_DualPeriodMXFP8") is True, f"expected dual-period: mi={mi}"


# ---------------------------------------------------------------------------
# A small even non-partitionable tile falls off dual-period to the DU256
# non-dual-period solution (and must not take dual-period at DU128).
# ---------------------------------------------------------------------------
def test_even_non_partitionable_avoids_dual_period(_gp_gfx950, gfx950_iim, assembler):
    sol128 = _solve(gfx950_iim, assembler, _EVEN_SMALL, depth_u=128)
    assert sol128.get("_DualPeriodMXFP8") is False
    sol256 = _solve(gfx950_iim, assembler, _EVEN_SMALL, depth_u=256)
    assert sol256.get("Valid") is True and sol256.get("_DualPeriodMXFP8") is False


# ---------------------------------------------------------------------------
# Unsupported configs must be REJECTED (not emitted as a wrong kernel):
#  - any odd MIWaveTile off the dual-period path (DU256): the legacy scale emit
#    pairs tiles at compile time,
#  - 16x16: cannot partition, so it is not dual-period at DU128 either,
#  - a wave that would start on the second strip of a shared data global read
#    (wrong results on GPU, also with unit scales).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mi", _ODD_SUPPORTED + [_SKINNY_SQ, _SHARED_GR])
def test_odd_rejected_off_dual_period(_gp_gfx950, gfx950_iim, assembler, mi):
    sol = _solve(gfx950_iim, assembler, mi, depth_u=256)
    assert sol.get("Valid") is False, f"odd MIWaveTile must be rejected at DU256: mi={mi}"


@pytest.mark.parametrize("mi", [_SKINNY_SQ, _SHARED_GR])
def test_unsupported_odd_rejected(_gp_gfx950, gfx950_iim, assembler, mi):
    sol = _solve(gfx950_iim, assembler, mi, depth_u=128)
    assert sol.get("Valid") is False, f"unsupported config must be rejected: mi={mi}"


# ---------------------------------------------------------------------------
# Codegen: the emitted assembly has the per-tile scale structure.
# ---------------------------------------------------------------------------
def _emit(gfx950_iim, assembler, mi):
    from tensilelite.Toolchain.Validators import validateToolchain
    cxx = validateToolchain("amdclang++")
    if not cxx or not shutil.which(cxx):
        pytest.skip("gfx950 assembly toolchain (amdclang++) unavailable")

    from tensilelite.create_library.run import (
        generateKernelObjectsFromSolutions, processKernelSource)
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.Common.Types import DebugConfig
    import rocisa

    sol = _solve(gfx950_iim, assembler, mi, depth_u=128)
    assert sol.get("Valid") and sol.get("_DualPeriodMXFP8")
    for kernel in generateKernelObjectsFromSolutions([sol]):
        ri = rocisa.rocIsa.getInstance()
        ri.init(tuple(kernel["ISA"]), shutil.which(cxx))
        ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
        kernel["BaseName"] = "odd_mxfp8_probe"
        try:
            kernel.duplicate = False
        except Exception:
            pass
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
        assert res.err == 0, f"odd mxfp8 codegen failed (err={res.err}) mi={mi}"
        return res.src.decode(errors="replace") if isinstance(res.src, (bytes, bytearray)) else res.src
    pytest.fail("no kernel object generated")


def _scale_reads(src, tc):
    """[(what, index, address register)] of the main-loop-style scale ds_reads."""
    import re
    pat = re.compile(r"ds_read_b32 v\d+, (v\d+) offset:\d+\s*// scale%s\[(tile|group)(\d+)," % tc)
    return sorted({(m.group(2), int(m.group(3)), m.group(1)) for m in pat.finditer(src)})


@pytest.mark.parametrize("mi", _ODD_SUPPORTED)
def test_odd_mxfp8_emits(_gp_gfx950, gfx950_iim, assembler, mi):
    _emit(gfx950_iim, assembler, mi)


def test_per_tile_scale_structure_multi_wave(_gp_gfx950, gfx950_iim, assembler):
    # MT160x128: 5 A tiles per wave, two waves in M -> per-wave start parity.
    src = _emit(gfx950_iim, assembler, _ODD_M5_2W)
    reads = _scale_reads(src, "MXSA")
    assert [(w, i) for w, i, _ in reads] == [("tile", i) for i in range(5)], reads
    regs = {i: r for _, i, r in reads}
    assert regs[0] == regs[2] == regs[4] and regs[1] == regs[3] and regs[0] != regs[1], regs
    assert "wave starts in the 2nd half of a swizzle group" in src
    assert "sgprScaleByteOffMXSA" not in src
    # the even N dim keeps the pair-per-register path
    assert [w for w, _, _ in _scale_reads(src, "MXSB")] == ["group", "group"]


def test_per_tile_scale_structure_single_wave(_gp_gfx950, gfx950_iim, assembler):
    # MT80x128: 5 A tiles in one wave -> start parity is the workgroup parity.
    src = _emit(gfx950_iim, assembler, _ODD_M5_1W)
    reads = _scale_reads(src, "MXSA")
    assert [(w, i) for w, i, _ in reads] == [("tile", i) for i in range(5)], reads
    assert len({r for _, _, r in reads}) == 2
    assert "workgroup starts in the 2nd half of a swizzle group" in src
    assert "wave starts in the 2nd half of a swizzle group" not in src


def test_even_tile_keeps_pair_scale_reads(_gp_gfx950, gfx950_iim, assembler):
    src = _emit(gfx950_iim, assembler, _EVEN)
    assert {w for w, _, _ in _scale_reads(src, "MXSA") + _scale_reads(src, "MXSB")} == {"group"}
    assert "swizzle group?" not in src and "ScaleByteOff" not in src
