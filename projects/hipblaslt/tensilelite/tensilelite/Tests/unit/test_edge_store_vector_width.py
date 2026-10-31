# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Edge-tile store vector width from AssertFree0ElementMultiple (AF0EM).

gfx1250 non-sparse MFMA kernels store edge tiles with the widest
StoreVectorWidth/2^n that divides AF0EM; other archs, sparse, OutputAmaxD
and GSU AtomicDest partial stores keep the old width.
AF0EM -1 resolves to StoreVectorWidth on gfx1250.
"""

import copy
import types

import pytest

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Components.NotLocalFullTileElements import NotLocalFullTileElementsMFMA
from tensilelite.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


def _edge_vw(svw, af0em, isa=(12, 5, 0), sparse=0, lsu=1, amax=False, gsu=1, gsua="MultipleBuffer"):
    """Edge store width = last entry of the halving chain from getElements."""
    kernel = {
        "StoreVectorWidth": svw, "_VectorStore": 1, "GlobalSplitUAlgorithm": gsua, "LocalSplitU": lsu,
        "GlobalSplitU": gsu, "ISA": isa, "ProblemType": {"Sparse": sparse, "OutputAmaxD": amax},
        "AssertFree0ElementMultiple": af0em, "MatrixInstM": 16, "MatrixInstN": 16,
        "MatrixInstBM": 1, "MatrixInstBN": 1, "WavefrontSize": 32, "MIWaveTile": [2, 2],
        "SourceSwap": True, "VectorWidthA": svw, "VectorWidthB": 1, "MIOutputVectorWidth": 8,
    }
    writer = types.SimpleNamespace(maxGwvw=lambda k: 1000)
    widths, _ = NotLocalFullTileElementsMFMA().getElements(writer, kernel)
    return widths[0], widths[-1]


@pytest.mark.parametrize("svw, af0em, edge", [
    (4, 1, 1), (4, 2, 2), (4, 4, 4), (4, 8, 4),   # widest SVW/2^n dividing AF0EM
    (8, 4, 4), (2, 16, 2), (1, 8, 1),
])
def test_gfx1250_edge_vw_follows_af0em(svw, af0em, edge):
    full, got = _edge_vw(svw, af0em)
    assert full == svw
    assert got == edge


@pytest.mark.parametrize("isa, sparse", [((9, 4, 2), 0), ((9, 5, 0), 0), ((12, 5, 0), 1)])
def test_other_archs_and_sparse_keep_vw1(isa, sparse):
    assert _edge_vw(4, 4, isa=isa, sparse=sparse)[1] == 1


@pytest.mark.parametrize("kw, edge", [
    ({"lsu": 2}, 4), ({"amax": True}, 1),
    ({"gsu": 2, "gsua": "SingleBuffer"}, 4),
    ({"gsu": 2, "gsua": "AtomicDest"}, 2),          # packed bf16 atomics: pairs, as before
    ({"gsu": 2, "gsua": "MultipleBuffer"}, 4),
])
def test_gated_paths_keep_old_width(kw, edge):
    assert _edge_vw(4, 4, **kw)[1] == edge


# ---------------------------------------------------------------------------
# AF0EM -1 resolution through Solution derivation (real gfx1250 caps).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx1250_iim():
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Validators import validateToolchain

    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], validateToolchain("amdclang++"))
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

    return makeAssemblyToolchain(validateToolchain("amdclang++"),
                                 validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER), "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from tensilelite.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from tensilelite.Common.ValidParameters import validParameters

    saved = [copy.deepcopy(dict(x)) for x in (globalParameters, validParameters, defaultSolution)]
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    for x, s in zip((globalParameters, validParameters, defaultSolution), saved):
        x.clear()
        x.update(s)


def _solution(gfx1250_iim, assembler, af0em):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

    isa = gfxToIsa("gfx1250")
    mi = [16, 16, 32, 1, 1, 8, 8, 2, 2]     # bf16 TN MT256x256, as 1024_vgpr_gfx1250.yaml
    pt = {"OperationType": "GEMM", "DataType": "b", "DestDataType": "b", "ComputeDataType": "s",
          "HighPrecisionAccumulate": True, "TransposeA": True, "TransposeB": False, "UseBeta": True,
          "Batched": True}
    params = {
        "ProblemType": pt, "ISA": isa, "MatrixInstruction": mi, "WorkGroup": [16, 16, 1],
        "WavefrontSize": 32, "DepthU": 64, "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1, "ScheduleIterAlg": 4, "TDMInst": 3, "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer", "StoreRemapVectorWidth": 0, "StoreVectorWidth": -1,
        "VectorWidthA": -1, "VectorWidthB": -1, "MaxLDS": 327680, "1LDSBuffer": 0,
        "AssertFree0ElementMultiple": af0em,
    }
    params.update(matrixInstructionToMIParameters(mi, isa, 32, pt, params["WorkGroup"], gfx1250_iim))
    return Solution(params, False, True, False, assembler, gfx1250_iim)


def test_af0em_auto_resolves_to_store_vector_width(_gp_gfx1250, gfx1250_iim, assembler):
    sol = _solution(gfx1250_iim, assembler, -1)
    assert sol.get("Valid") is True
    assert sol["StoreVectorWidth"] > 1
    assert sol["AssertFree0ElementMultiple"] == sol["StoreVectorWidth"]


def test_af0em_explicit_value_is_kept(_gp_gfx1250, gfx1250_iim, assembler):
    sol = _solution(gfx1250_iim, assembler, 1)
    assert sol.get("Valid") is True
    assert sol["AssertFree0ElementMultiple"] == 1
