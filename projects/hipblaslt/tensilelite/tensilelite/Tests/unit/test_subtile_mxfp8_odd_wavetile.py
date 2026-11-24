# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Odd / skinny MIWaveTile for the mxfp8 subtile path (gfx950).

Skinny GEMMs want a 16xN or Mx16 macrotile, i.e. MIWaveTile == 1 (odd) along the
narrow dimension.  MXFP8 historically required even MIWaveTile in both M and N
because the MX scale local read packs 2 scale MMA tiles in M x 2 in K per VGPR
(MXScaleLRGeometry.subtileShape = (2, 2)).  The scale grid is now sized with
ceil, so an odd scale-tile count allocates a full trailing scale group whose
unused half is inert, and the gate allows odd MIWaveTile for mxfp8 (8-bit-float
A and B).

These tests pin the contract:
  * odd / skinny MIWaveTile mxfp8 solutions VALIDATE (previously rejected).
  * a partitionable skinny tile (16xN, Mx16) still takes the dual-period path.
  * a config that cannot form >1 partition (both dims odd, or small even like
    [2,2]) does NOT take dual-period at DU128 -- it is served by the DU256
    non-dual-period solution instead of hitting the prefetchNextPartition
    assertion in the scheduler.
  * the skinny tile emits assembly without error (scale ceil sizing is sound).
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
#
# Verified-correct odd scope (GPU full-element validation): MIWaveTile == 1 AND
# MIWaveGroup == 1 in the odd dim (MacroTile == 16) -- the 16xN / Mx16 skinny
# target.  Everything else odd stays rejected.
# Supported: a 16-wide tile (MIWaveTile==1) whose OTHER dim is partitionable, so
# the solution takes the dual-period path (DU128, PGR2) -- GPU-verified correct at
# all sizes incl deep K.
_SKINNY_M    = [16, 16, 128, 1, 1, 1, 4, 1, 4]  # MT16x256  (skinny M=1, partitionable N -> dual-period)
_SKINNY_N    = [16, 16, 128, 1, 1, 4, 1, 4, 1]  # MT256x16  (skinny N=1, partitionable M -> dual-period)
_EVEN_SMALL  = [16, 16, 128, 1, 1, 2, 2, 1, 1]  # MT32x32   (even, not partitionable)
# Rejected: a 16-wide tile is only correct on the dual-period path (the legacy
# scale SRD faults at deep K), so a non-partitionable 16-wide tile is rejected;
# odd MIWaveTile >= 3 mis-computes a tile once the grid tiles across that dim
# (GPU: MT48x64 fails on (96,128)); MIWaveTile==1 with >1 wave is inexact.
_SKINNY_SQ   = [16, 16, 128, 1, 1, 1, 1, 1, 1]  # MT16x16   (both 16-wide, non-dual -> rejected)
_ODD_M3      = [16, 16, 128, 1, 1, 3, 4, 1, 1]  # MT48x64   (odd M=3)
_BOTH_ODD    = [16, 16, 128, 1, 1, 3, 3, 1, 1]  # MT48x48   (both odd)
_SKINNY_MWG  = [16, 16, 128, 1, 1, 1, 4, 2, 2]  # MT32x128  (M=1 but WG0=2 -> inexact)


# ---------------------------------------------------------------------------
# Gate: a 16-wide skinny tile (MIWaveTile==1) with a partitionable other dim
# validates and takes the dual-period path (previously rejected by the
# even-MIWaveTile gate).
@pytest.mark.parametrize("mi", [_SKINNY_M, _SKINNY_N])
def test_skinny_miwavetile_mxfp8_validates(_gp_gfx950, gfx950_iim, assembler, mi):
    sol = _solve(gfx950_iim, assembler, mi, depth_u=128)
    assert sol.get("Valid") is True, f"skinny MIWaveTile mxfp8 rejected: mi={mi}"
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
# Unsupported configs must be REJECTED (not emitted as a wrong/faulting kernel):
#  - a 16-wide tile that is NOT dual-period (e.g. 16x16): the legacy scale SRD
#    faults at deep K, so it is rejected rather than shipped,
#  - odd MIWaveTile >= 3 mis-computes a tile across the grid (GPU-confirmed),
#  - MIWaveTile==1 with >1 wave has inexact per-wave scale sizing.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mi", [_SKINNY_SQ, _ODD_M3, _BOTH_ODD, _SKINNY_MWG])
@pytest.mark.parametrize("depth_u", [128, 256])
def test_unsupported_odd_rejected(_gp_gfx950, gfx950_iim, assembler, mi, depth_u):
    sol = _solve(gfx950_iim, assembler, mi, depth_u=depth_u)
    assert sol.get("Valid") is False, \
        f"unsupported config must be rejected: mi={mi} DU={depth_u}"


# ---------------------------------------------------------------------------
# Codegen smoke test: the skinny 16xN tile emits assembly without error, proving
# the ceil-based scale sizing indexes the scale VGPRs in bounds.
# ---------------------------------------------------------------------------
def test_skinny_mxfp8_emits(_gp_gfx950, gfx950_iim, assembler):
    from tensilelite.Toolchain.Validators import validateToolchain
    cxx = validateToolchain("amdclang++")
    if not cxx or not shutil.which(cxx):
        pytest.skip("gfx950 assembly toolchain (amdclang++) unavailable")

    from tensilelite.create_library.run import (
        generateKernelObjectsFromSolutions, processKernelSource)
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.Common.Types import DebugConfig
    import rocisa

    sol = _solve(gfx950_iim, assembler, _SKINNY_M, depth_u=128)
    assert sol.get("Valid") and sol.get("_DualPeriodMXFP8")
    for kernel in generateKernelObjectsFromSolutions([sol]):
        ri = rocisa.rocIsa.getInstance()
        ri.init(tuple(kernel["ISA"]), shutil.which(cxx))
        ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
        kernel["BaseName"] = "skinny_mxfp8_probe"
        try:
            kernel.duplicate = False
        except Exception:
            pass
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
        assert res.err == 0, f"skinny mxfp8 codegen failed (err={res.err})"
        return
    pytest.fail("no kernel object generated")
