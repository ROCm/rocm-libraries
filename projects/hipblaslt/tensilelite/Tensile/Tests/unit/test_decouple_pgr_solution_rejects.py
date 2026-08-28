################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
import copy

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


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


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = overrides.pop("MatrixInstruction", [16, 16, 128, 1, 1, 2, 16, 2, 2])
    workGroup = overrides.pop("WorkGroup", [32, 4, 1])
    problemType = {
        "OperationType": "GEMM",
        "MacDataTypeA": "F8",
        "MacDataTypeB": "F4",
        "DataType": "F8",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "DataTypeMXSA": "E8",
        "DataTypeMXSB": "E8",
    }
    problemType.update(overrides.pop("ProblemType", {}))
    params = {
        "ProblemType": problemType,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": workGroup,
        "WavefrontSize": 32,
        "DepthU": 256,
        "MaxLDS": 327680,
        "KernelLanguage": "Assembly",
        "TDMInst": 3,
        "MXScaleFormat": "InMemorySwizzle",
        "LDSTrInst": True,
        "TDMFuse": 0,
        "TDMSplit": False,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": 1,
        "PrefetchGlobalReadB": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 0,
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
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problemType, workGroup, gfx1250_iim))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


@pytest.mark.parametrize("pgrA, pgrB, label",
                         [(1, 2, "PGRA1_PGRB2"), (2, 1, "PGRA2_PGRB1")])
def test_the_divergent_pairs_are_accepted(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, label):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, f"{label} rejected with: {out!r}"


def test_rejects_cluster_with_divergent_pgr(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       ClusterDim=[2, 1], TDMFuse=5)
    assert sol.get("Valid") is False
    assert "ClusterDim != [1, 1] is incompatible with divergent" in out
    assert "cluster barrier drains tensorcnt" in out


def test_cluster_allows_equal_pgr(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       ClusterDim=[2, 1], TDMFuse=5,
                       PrefetchGlobalReadA=2, PrefetchGlobalReadB=2)
    assert sol.get("Valid") is True, f"rejected with: {out!r}"


def test_rejects_sparse(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, ProblemType={"Sparse": 1})
    assert sol.get("Valid") is False
    assert "PrefetchGlobalReadA/B: Sparse is not supported yet" in out


def test_rejects_prefetch_across_persistent(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchAcrossPersistent=1, StreamK=3)
    assert sol.get("Valid") is False
    assert "PrefetchGlobalReadA/B: PrefetchAcrossPersistent is not" in out
    assert "over-fills a tensor at level 0" in out


def test_rejects_one_lds_buffer(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **{"1LDSBuffer": 1})
    assert sol.get("Valid") is False
    assert "1LDSBuffer=1 gives every tensor one shared LDS block" in out
    assert "ask for 1 and 2" in out


@pytest.mark.parametrize(
    "pgrA, pgrB, label",
    [(1, 1, "one and one"), (0, 1, "zero and one"), (1, 0, "one and zero")],
)
def test_rejects_both_tensors_on_one_lds_block(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, label):
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchGlobalRead=1,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is False, label
    assert "put both tensors on a single LDS block" in out
    assert "computes wrong results from K = 2*DepthU" in out


@pytest.mark.parametrize(
    "overrides, clause, label",
    [
        ({"ScheduleIterAlg": 3},
         "only ScheduleIterAlg=0 places the fill where it can be moved", "sia"),
        ({"PrefetchLocalRead": 0},
         "PrefetchLocalRead must be at least 1", "plr below one"),
        ({"DepthU": 512, "PrefetchLocalRead": 4},
         "is not below LoopIters=4", "plr at loop iters"),
        ({"PrefetchGlobalRead": 3, "PrefetchGlobalReadB": 3},
         "more than two LDS blocks for a tensor is not supported", "three blocks"),
        ({"MatrixInstruction": [16, 16, 128, 1, 1, 2, 16, 1, 1],
          "WorkGroup": [32, 1, 1]},
         "wave-separated TDM descriptor (NumWaves > 1)", "one wave"),
    ],
)
def test_divergent_pair_reject_reports_its_reason(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, overrides, clause, label):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **overrides)
    assert sol.get("Valid") is False, label
    assert "need a slot in the unrolled loop to move the single-buffered" in out
    assert clause in out


def test_the_divergent_reject_is_silent_on_an_equal_pair(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, ScheduleIterAlg=3,
                       PrefetchGlobalReadA=2, PrefetchGlobalReadB=2)
    assert sol.get("Valid") is True, f"rejected with: {out!r}"
    assert "need a slot in the unrolled loop" not in out
