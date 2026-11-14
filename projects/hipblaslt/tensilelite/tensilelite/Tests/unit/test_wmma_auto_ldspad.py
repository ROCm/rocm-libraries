# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Auto LdsPad (-1) for K-major LDS on WMMA V1/V2 pads by one ds_load_b128.

A local read wider than 16 bytes is issued as several b128 reads, and each
one meets the banks on its own. Padding a row by one b128 (16 bytes) shifts
each row by a 4-bank window; padding by 32 bytes (the old result for fp16
LRVW=16 on gfx11, or for the MI16 doubling of a b128 read) makes the row
stride 4*even banks, a 2-way conflict. MFMA keeps its doubling.

Each case derives a full Solution on real capability maps (needs amdclang++).
"""

import copy

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


@pytest.fixture(scope="module")
def toolchain():
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

    try:
        cxx = validateToolchain("amdclang++")
        bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    except (ValueError, FileNotFoundError) as e:
        pytest.skip(f"amdclang++ is unavailable: {e}")
    return cxx, makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture
def derive(toolchain):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Common.GlobalParameters import assignGlobalParameters, globalParameters
    from tensilelite.Common.ValidParameters import validParameters
    from tensilelite.SolutionStructs.Solution import Solution
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    cxx, assembler = toolchain
    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))

    def _derive(arch, dataType, transA, transB, lrvw, mi, wavefrontSize):
        isa = gfxToIsa(arch)
        iim = makeIsaInfoMap([isa], cxx)
        if not iim[isa].asmCaps["SupportedISA"]:
            pytest.skip(f"amdclang++ in this environment does not support {arch}")
        defaultSolution.clear()
        defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
        assignGlobalParameters({}, iim)

        isInt8 = dataType == "I8"
        problemType = {
            "OperationType": "GEMM",
            "DataType": dataType,
            "DestDataType": "I" if isInt8 else dataType,
            "ComputeDataType": "I" if isInt8 else "s",
            "HighPrecisionAccumulate": True,
            "TransposeA": transA,
            "TransposeB": transB,
            "UseBeta": True,
            "Batched": True,
        }
        params = {
            "ProblemType": problemType,
            "ISA": isa,
            "MatrixInstruction": mi,
            "WavefrontSize": wavefrontSize,
            "DepthU": 32,
            "KernelLanguage": "Assembly",
            "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 1,
            "ScheduleIterAlg": 3,
            "TransposeLDS": 1,
            "LdsPadA": -1,
            "LdsPadB": -1,
            "LdsBlockSizePerPadA": -1,
            "LdsBlockSizePerPadB": -1,
            "LocalReadVectorWidth": lrvw,
            "VectorWidthA": -1,
            "VectorWidthB": -1,
            "GlobalReadVectorWidthA": -1,
            "GlobalReadVectorWidthB": -1,
            "SourceSwap": True,
            "1LDSBuffer": 0,
            "StreamK": 0,
            "GlobalSplitU": 1,
            "StaggerU": 0,
            "WorkGroupMapping": 1,
        }
        params.update(
            matrixInstructionToMIParameters(mi, isa, wavefrontSize, problemType, None, iim))
        return Solution(params, False, True, False, assembler, iim)

    yield _derive

    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


_MI16_64x64 = [16, 16, 16, 1, 1, 2, 2, 2, 2]


@pytest.mark.parametrize(
    "arch, dataType, transA, transB, lrvw, padA, padB",
    [
        # gfx11 (WMMA V1) needs LRVW >= MIInputPerThread = 16 for fp16: two b128
        # reads per lane. One b128 is 8 halves.
        ("gfx1151", "H", True, False, 16, 8, 8),
        ("gfx1100", "H", True, False, 16, 8, 8),
        # NN: only B is K-major in LDS; A keeps its own (TLU) formula.
        ("gfx1151", "H", False, False, 16, 32, 8),
        # 8-bit: one b128 is 16 bytes.
        ("gfx1151", "I8", True, False, 16, 16, 16),
        # gfx12 (WMMA V2): a single b128 read is no longer doubled.
        ("gfx1201", "H", True, False, 8, 8, 8),
        ("gfx1201", "H", True, False, 16, 8, 8),
    ],
)
def test_wmma_k_major_auto_pad_is_one_b128(derive, arch, dataType, transA, transB, lrvw, padA, padB):
    sol = derive(arch, dataType, transA, transB, lrvw, _MI16_64x64, 32)
    assert sol.get("Valid") is True
    assert (sol["LdsPadA"], sol["LdsPadB"]) == (padA, padB)


def test_mfma_mi16_b128_doubling_is_unchanged(derive):
    sol = derive("gfx942", "H", True, False, 8, _MI16_64x64, 64)
    assert sol.get("Valid") is True
    assert (sol["LdsPadA"], sol["LdsPadB"]) == (16, 16)
