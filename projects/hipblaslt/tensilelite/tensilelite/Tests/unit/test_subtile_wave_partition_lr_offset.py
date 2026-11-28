# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wave-partition local-read offset for subtile kernels with several waves on both axes.

Every wave reads its A strip and its B strip from its own slot in LDS, so the
local-read address needs a per-wave offset along each axis that has more than
one wave.  The offset used to be skipped whenever one global read covers two or
more tile strips (loadRatioGR >= 2.0), which is only right when the tensor's own
axis has a single wave.  In a 4x2 wave group B has loadRatioGR == 2.0 (four
M-waves cooperate on a load) and two N strips: without the offset all eight
waves read the first N strip and half of the output is wrong.  A 2x4 group has
the same problem on A.  The dual-period path always took the offset, so these
tests use the single-period path (DepthU=256, PrefetchGlobalRead=1).
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


def _emit(gfx950_iim, assembler, mi):
    """Assembly of the single-period MXFP8 subtile kernel for MatrixInstruction `mi`."""
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters
    from tensilelite.Toolchain.Validators import validateToolchain
    from tensilelite.create_library.run import generateKernelObjectsFromSolutions, processKernelSource
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.Common.Types import DebugConfig
    import rocisa

    cxx = validateToolchain("amdclang++")
    if not cxx or not shutil.which(cxx):
        pytest.skip("gfx950 assembly toolchain (amdclang++) unavailable")
    isa = gfxToIsa("gfx950")
    problem_type = {
        "OperationType": "GEMM", "DataType": "F8", "DestDataType": "b",
        "ComputeDataType": "s", "HighPrecisionAccumulate": True,
        "TransposeA": True, "TransposeB": False, "UseBeta": True,
        "Batched": True, "StridedBatched": True, "MXBlockA": 32, "MXBlockB": 32,
    }
    params = {
        "ProblemType": problem_type, "ISA": isa, "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1], "WavefrontSize": 64, "DepthU": 256,
        "KernelLanguage": "Assembly", "PrefetchGlobalRead": 1, "PrefetchLocalRead": 1,
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
    sol = Solution(params, False, True, False, assembler, gfx950_iim)
    assert sol.get("Valid"), f"solution rejected: mi={mi}"
    assert not sol.get("_DualPeriodMXFP8"), "expected the single-period path"
    for kernel in generateKernelObjectsFromSolutions([sol]):
        ri = rocisa.rocIsa.getInstance()
        ri.init(tuple(kernel["ISA"]), shutil.which(cxx))
        ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
        kernel["BaseName"] = "wave_partition_probe"
        try:
            kernel.duplicate = False
        except Exception:
            pass
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(), False, kernel)
        assert res.err == 0, f"codegen failed (err={res.err}) mi={mi}"
        return res.src.decode(errors="replace") if isinstance(res.src, (bytes, bytearray)) else res.src
    pytest.fail("no kernel object generated")


_A_OFFSET = "A: wave partition LR offset"
_B_OFFSET = "B: wave partition LR offset"

# MIWaveTile [M, N] at mi[5], mi[6]; MIWaveGroup at mi[7], mi[8].
_WG_4x2 = [16, 16, 128, 1, 1, 4, 4, 4, 2]  # MT256x128, 8 waves
_WG_2x4 = [16, 16, 128, 1, 1, 4, 4, 2, 4]  # MT128x256, 8 waves
_WG_2x2 = [16, 16, 128, 1, 1, 4, 4, 2, 2]  # MT128x128, 4 waves
_WG_4x1 = [16, 16, 128, 1, 1, 4, 4, 4, 1]  # MT256x64, 4 waves on M only
_WG_1x4 = [16, 16, 128, 1, 1, 4, 4, 1, 4]  # MT64x256, 4 waves on N only


@pytest.mark.parametrize("mi", [_WG_4x2, _WG_2x4, _WG_2x2])
def test_both_axes_get_the_offset_when_both_have_several_waves(_gp_gfx950, gfx950_iim, assembler, mi):
    src = _emit(gfx950_iim, assembler, mi)
    assert _A_OFFSET in src, f"A wave-partition LR offset missing: mi={mi}"
    assert _B_OFFSET in src, f"B wave-partition LR offset missing: mi={mi}"


def test_single_wave_axis_has_no_offset(_gp_gfx950, gfx950_iim, assembler):
    # 4x1: the four M-waves share the one N strip, so B needs no offset; A does.
    src = _emit(gfx950_iim, assembler, _WG_4x1)
    assert _A_OFFSET in src and _B_OFFSET not in src
    # 1x4: mirrored.
    src = _emit(gfx950_iim, assembler, _WG_1x4)
    assert _B_OFFSET in src and _A_OFFSET not in src
