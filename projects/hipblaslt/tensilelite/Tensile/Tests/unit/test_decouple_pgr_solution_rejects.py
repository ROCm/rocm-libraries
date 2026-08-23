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
"""The PrefetchGlobalReadA/B rejects, on a derived solution rather than a helper.

test_decouple_pgr_lds_blocks covers the helpers under these rejects -- the block
count map, the pair predicates, and every clause of
divergentPairUnsupportedReason -- by calling them directly. That is where the
arithmetic belongs, but it leaves the rejects themselves untested: a helper can
keep returning the right reason while the reject that was supposed to consult it
stops being reached, and no kernel name changes when that happens.

So this file asks the other half of the question. Each test derives a real
gfx1250 solution, and pins the clause that identifies which reject fired. Two of
them are worth the trouble on their own:

  - The divergent-pair reject FRAMES divergentPairUnsupportedReason's answer
    rather than restating it, so its five clauses are only reachable through it.
    Testing the helper proves the reason is computed; testing here proves it is
    consulted and reported.
  - decoupledOneBlockBoth guards a kernel that builds, fits, and computes wrong
    results from K = 2*DepthU. There is no assertion downstream to catch it, so
    this reject is the only thing standing between that kernel and a library.
    The threshold is derived from the loop trip count and is sourced, together
    with the FFM sweep that independently lands on it, at the reject itself in
    Solution.depthUIteration. It is not one of the empirical SIA4 thresholds,
    which were withdrawn as intermittent.

Three of the rejects in this block cannot be reached on gfx1250 and are
deliberately not tested, because a test that cannot fail is worse than no test:

  - DirectToLds, ON THIS TARGET ONLY -- the finding is arch-scoped and is not
    a statement about the guard. gfx1250 reports HasDirectToLds=0, so
    isDirectToLdsDoable clears DirectToLdsA/B before this reject reads them;
    measured, requesting both leaves the solution valid with both keys False.
    On a target where HasDirectToLds is 1 the keys survive and this reject is
    live, so it must not be removed on the strength of this note. What cannot
    be written is a gfx1250 test: it would pass with the guard deleted, which
    is the vacuous kind this file is trying not to add.
  - UseSubtileImpl. On the MX shape the feature needs, the subtile pipeline is
    refused earlier by "Unable to load MXSB scales using one load per wave".
  - TDMInst != 3. Refused earlier still, by "TDMA and TDMB must be enabled
    simultaneously", which is asserted by the TDMFuse files.
"""
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
    """The F8F4 TN MT64x512 DepthU256 divergent pair (PGRA=1, PGRB=2).

    Naming: hero=(1,2), mirror=(2,1). Quote the pair, not a nickname.
    TDMFuse=0 so grouping guards cannot shadow the reject under test.
    """
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


# The control comes first: every reject below is vacuous if the pair it starts
# from is itself refused. Both orientations, since the single-buffered tensor
# being A rather than B is a different cadence in the emitter.
@pytest.mark.parametrize("pgrA, pgrB, label",
                         [(1, 2, "PGRA1_PGRB2"), (2, 1, "PGRA2_PGRB1")])
def test_the_divergent_pairs_are_accepted(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, label):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, f"{label} rejected with: {out!r}"


# ---------------------------------------------------------------------------
# The unsupported-combination rejects, one message each.
# ---------------------------------------------------------------------------
def test_rejects_sparse(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Metadata would have to follow the sparse operand's per-tensor count.

    It has no per-tensor key of its own, so there is nothing for it to follow.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, ProblemType={"Sparse": 1})
    assert sol.get("Valid") is False
    assert "PrefetchGlobalReadA/B: Sparse is not supported yet" in out


def test_rejects_prefetch_across_persistent(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """StreamK=3 is carried because without it PrefetchAcrossPersistent is off.

    A bare PrefetchAcrossPersistent=1 is reset to 0 long before this reject, so
    a test that only set the one key would pass on an accepted solution and
    prove nothing. The mechanism is that the next-tile prefetch group re-fills
    every tensor once, which over-fills a tensor sitting at level 0.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchAcrossPersistent=1, StreamK=3)
    assert sol.get("Valid") is False
    assert "PrefetchGlobalReadA/B: PrefetchAcrossPersistent is not" in out
    assert "over-fills a tensor at level 0" in out


def test_rejects_one_lds_buffer(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """One shared block contradicts any pair asking for two.

    The reject has to see the pair to say so, which is why equalPairDegenerates
    ToScalar deliberately leaves an explicit 1LDSBuffer=1 pair unresolved --
    resolving it away would delete the keys and let the solution quietly build
    one block instead of the two it asked for.
    """
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
    """The reject that stands between a wrong-results kernel and a library.

    All three spellings emit byte-identical assembly, since a level of 0 and a
    level of 1 are both one block, so all three have to be refused. This builds
    and fits; nothing downstream asserts. The K = 2*DepthU threshold is in the
    message because it is the mechanism: two full trips through the unrolled
    loop is what it takes for a fill to overwrite a block still being read.
    The reject in Solution.depthUIteration carries that argument and the FFM
    sweep that agrees with it, so nothing here is asking to be taken on trust.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchGlobalRead=1,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is False, label
    assert "put both tensors on a single LDS block" in out
    assert "computes wrong results from K = 2*DepthU" in out


# ---------------------------------------------------------------------------
# The divergent-pair reject, and the five clauses it frames.
# ---------------------------------------------------------------------------
# divergentPairUnsupportedReason is unit-tested directly in
# test_decouple_pgr_lds_blocks. What is checked here is the other half: that the
# reject consults it and reports what it said. Each row picks a different clause,
# so a reject that stopped calling the helper -- or called it and dropped the
# reason -- fails here even though the helper's own tests stay green.
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
    # The frame, which says which reject fired...
    assert "need a slot in the unrolled loop to move the single-buffered" in out
    # ...and the clause, which says why. Neither alone identifies the failure.
    assert clause in out


def test_the_divergent_reject_is_silent_on_an_equal_pair(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """An equal pair has no single-buffered tensor, so it needs no slot.

    Worth pinning alongside the rows above: they would all still pass against a
    guard that refused every decoupled solution outright, which would delete the
    feature rather than protect it.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, ScheduleIterAlg=3,
                       PrefetchGlobalReadA=2, PrefetchGlobalReadB=2)
    assert sol.get("Valid") is True, f"rejected with: {out!r}"
    assert "need a slot in the unrolled loop" not in out
