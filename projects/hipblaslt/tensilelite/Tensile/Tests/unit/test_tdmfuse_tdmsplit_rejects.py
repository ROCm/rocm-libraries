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
"""TDMSplit x TDMFuse, and the per-row TDMFuse rejects, by their own messages.

TDMFuse is a reject-heavy parameter: a nonzero value PINS a descriptor grouping,
so every combination the writer would not actually group that way has to come
back refused rather than quietly building a different kernel under a name that
claims otherwise. Four rows carry twenty-eight such refusals between them and
the characterization goldens assert only {basename, err}, so a guard that stops
firing changes no kernel name and nothing notices.

What the messages are for. Each reject explains a different mechanism, and the
mechanisms are what tell two refusals apart: asserting only that a solution was
refused would pass just as happily if a neighbouring guard fired instead, which
is the regression this file exists to catch. So every test below pins a
substring that no other reject can produce -- in practice the row number plus
the clause naming the mechanism, never a whole sentence, because these texts are
prose and get reworded.

Row 5 keeps its own guards next to its feature in test_tdmfuse_paired, so what
is here for that row is the cross-product cell and the one clause that file does
not reach, the sparse metadata tensor. Rows 2, 4 and 6 are covered here outright:
before this file row 2 had one message pinned, row 4 none, and row 6 one.

The TDMSplit axis is the one worth stating outright, because it is not uniform:

    TDMFuse   0     2       4     5       6
    TDMSplit  ok    reject  ok    reject  reject

Rows 2, 5 and 6 each refuse TDMSplit for a reason of their own -- 2 has retired
the parity pairing its select depends on, 5 has two shared descriptors and no
arithmetic naming the second, 6 has de-aliased the pairing away -- and rows 0
and 4 accept it, because a grouping that keeps one parity-selected shared
descriptor is exactly what TDMSplit's multi-wave increment recomputes. The
asymmetry is the content: a guard copied to row 4 by mistake, or dropped from
row 4's neighbours, is a silent change of the selectable space.

TDMSplit is disabled upstream (PR #10911). Cases that set TDMSplit=True
are xfail(strict=False) so they revive when the blanket reject lifts.
Keep the per-row table; do not drop those xfails.
"""
import copy

import pytest

from Tensile.Common.DecouplePgr import tdmFuseAMx
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

# TDMSplit=True cases are unreachable until PR #10911 lifts; keep xfail.
_TDMSPLIT_DISABLED_UPSTREAM = pytest.mark.xfail(
    reason="TDMSplit is currently disabled upstream (PR #10911)", strict=False)

# Sibling unit tests mutate the process-global defaultSolution in place, which
# makes Solution.__init__'s `for key in defaultSolution` loop order-dependent.
# Snapshot it at collection time, as test_tdmfuse_paired does.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))

# One wave count serves every row: 4 is the only one TDMFuse=2 accepts, and it
# is a power of two above one, which is all 4, 5 and 6 ask for.
_ONE_WAVE_MI = [16, 16, 128, 1, 1, 2, 16, 1, 1]
_ONE_WAVE_WG = [32, 1, 1]

# Dropping MXBlockB takes the B scales away without making the shape invalid;
# MacDataTypeB has to come with it or the F4 operand has no scales to describe.
_NO_MX_ON_B = {"MacDataTypeB": "F8", "DataTypeMXSB": "E8", "MXBlockB": 0}


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
    """The F8F4 TN MT64x512 DepthU256 shape every row here can reach.

    The EQUAL (2,2) per-tensor pair is the default rather than the divergent
    (1,2) the decoupled feature exists for, because TDMFuse=2 refuses a
    divergent pair on its own account -- so a divergent base would mask every
    other row-2 guard behind that one message. `_remove` drops keys outright,
    which is the only way to spell the legacy solution that carries no
    per-tensor key at all.
    """
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    # Popped, not merged: matrixInstructionToMIParameters re-derives both keys
    # from the instruction, so a value left in overrides would be overwritten.
    mi = overrides.pop("MatrixInstruction", [16, 16, 128, 1, 1, 2, 16, 2, 2])
    workGroup = overrides.pop("WorkGroup", [32, 4, 1])
    remove = overrides.pop("_remove", ())
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
        "TDMFuse": 6,
        "TDMSplit": False,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": 2,
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
    for key in remove:
        params.pop(key, None)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problemType, workGroup, gfx1250_iim))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


# ---------------------------------------------------------------------------
# Controls. Every reject below is vacuous if the shape it starts from is itself
# refused, so the accepts come first and cover all four rows.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fuse", [0, 2, 4, 5, 6])
def test_the_base_shape_is_accepted_by_every_row(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse)
    assert sol.get("Valid") is True, f"TDMFuse={fuse} rejected with: {out!r}"


# ---------------------------------------------------------------------------
# TDMSplit x TDMFuse. The axis the pre-development analysis called out.
# ---------------------------------------------------------------------------
# None means the row accepts TDMSplit. Otherwise the clause that names why THIS
# row cannot express it -- the three refusals share a first half, so the clause
# is what distinguishes them and a test pinned to the shared half would pass on
# the wrong guard.
_TDMSPLIT_MECHANISM = {
    0: None,
    2: "retires the parity pairing that select depends on",
    4: None,
    5: "two shared descriptors and no arithmetic that names the second",
    6: None,
}


@pytest.mark.parametrize("fuse", [0, 2, 4, 5, 6])
@_TDMSPLIT_DISABLED_UPSTREAM
def test_tdmsplit_across_every_grouping(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """The cross-product, stated as one table rather than five separate facts.

    Rows 0 and 4 keep one parity-selected shared descriptor, which is what
    TDMSplit's multi-wave increment recomputes, so they accept. Rows 2, 5 and 6
    have each retired that pairing in a different way and refuse.

    Removing any one of the three rejects fails this test at that row: the
    solution comes back Valid and the assert on `is False` fires. Demonstrated,
    not reasoned -- see the report accompanying this file.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, TDMSplit=True)
    mechanism = _TDMSPLIT_MECHANISM[fuse]

    if fuse not in (2, 5, 6):
        assert sol.get("Valid") is True, f"TDMFuse={fuse} rejected with: {out!r}"
        return

    assert sol.get("Valid") is False
    # The row number is in the message, so no other row's guard can satisfy this.
    assert "TDMFuse=%d is not available with TDMSplit" % fuse in out
    if mechanism:
        assert mechanism in out


@_TDMSPLIT_DISABLED_UPSTREAM
def test_row_four_is_the_only_fused_grouping_that_keeps_tdmsplit(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Stated on its own because it is the easiest thing here to get wrong.

    {MXSA,MXSB} + {A,B} leaves the data tensors on the parity pair TDMSplit's
    increment selects on, so this row has nothing to refuse -- unlike its three
    neighbours. A guard copied here out of symmetry would silently delete a
    selectable combination, and no kernel name would change.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=4, TDMSplit=True)
    assert sol.get("Valid") is True, f"rejected with: {out!r}"
    assert "not available with TDMSplit" not in out


@pytest.mark.parametrize(
    "pair, label",
    [
        ({"PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 2}, "equal"),
        ({"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2}, "divergent"),
        ({"_remove": ("PrefetchGlobalReadA", "PrefetchGlobalReadB")}, "legacy"),
    ],
)
@_TDMSPLIT_DISABLED_UPSTREAM
def test_tdmsplit_is_refused_whatever_the_pair_spelling(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pair, label):
    """TDMSplit is a descriptor-grouping objection, so no pair spelling escapes.

    Worth pinning because the neighbouring guards in this row DO key on the
    pair: a TDMSplit clause accidentally written inside a pair-conditional arm
    would keep refusing the case a test happened to use and quietly admit the
    other two.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       TDMFuse=6, TDMSplit=True, **pair)
    assert sol.get("Valid") is False, label
    assert "TDMFuse=6 is not available with TDMSplit" in out


@pytest.mark.xfail(
    reason="premise preempted: upstream's blanket TDMSplit reject (PR #10911) "
           "fires above both row-2 guards whose order this pins",
    strict=False)
def test_tdmsplit_is_refused_before_the_equal_pair_guard(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Two of row 2's guards apply at once here, and the order decides the text.

    A divergent pair with TDMSplit violates both "requires an equal decoupled
    pair" and "not available with TDMSplit". TDMSplit is checked first, so that
    is the message, and a test asserting only that the solution was refused
    could not tell the two apart. Reordering the block would flip this without
    changing any kernel that builds.

    ITS PREMISE IS PREEMPTED, not just its expected text -- which is why this
    one is marked separately from the other nine. Upstream's blanket reject
    sits at function scope above BOTH guards being ordered, so neither is
    consulted and the precedence this test exists to pin is currently
    unobservable rather than false. Note what that does to the second
    assertion: "requires an equal decoupled pair" is indeed absent from the
    output, but for a new reason, so it would keep passing while proving
    nothing. Rewriting the first assertion against the blanket message would
    turn this green at the cost of deleting the ordering fact, so it is parked
    instead: when the blanket reject lifts, both guards become reachable again
    in the same order and this test resumes protecting it unchanged.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=2, TDMSplit=True,
                       PrefetchGlobalReadA=1, PrefetchGlobalReadB=2)
    assert sol.get("Valid") is False
    assert "TDMFuse=2 is not available with TDMSplit" in out
    assert "requires an equal decoupled pair" not in out


# ---------------------------------------------------------------------------
# TDMFuse=2, {A,MXSA,MXSB} + {B}. The row with the most refusals and, before
# this file, one covered message.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mi, wg, numWaves",
    [
        (_ONE_WAVE_MI, _ONE_WAVE_WG, 1),
        ([16, 16, 128, 1, 1, 2, 16, 2, 1], [32, 2, 1], 2),
        ([16, 16, 128, 1, 1, 2, 16, 4, 2], [32, 8, 1], 8),
    ],
)
def test_row_two_requires_exactly_four_waves(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, mi, wg, numWaves):
    """Not "more than one wave" -- exactly four, and for an arithmetic reason.

    The dispatch is 1/1/2: one wave for MXSA, one for MXSB, and A takes
    numWaves - 2, which is a power of two only at 4. So both 2 and 8 are
    refused, which is what separates this row from 4, 5 and 6 -- they hold at
    every power-of-two wave count above one. A guard written as `<= 1` by
    analogy with its neighbours would admit 2 and 8 and emit a grouping the
    writer cannot dispatch.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       TDMFuse=2, MatrixInstruction=mi, WorkGroup=wg)
    assert sol.get("Valid") is False, numWaves
    assert "1/1/2 split is a remainder policy" in out
    assert "got NumWaves=%d" % numWaves in out


def test_row_two_requires_mx_scales_on_both_tensors(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Without B's scales the shared group is just {A}, which is a different row.

    That is the mechanism worth pinning rather than the bare requirement: the
    grouping does not become invalid, it becomes TDMFuse=6 with the scales
    moved, so accepting it here would name one kernel with another's token.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=2,
                       ProblemType=_NO_MX_ON_B)
    assert sol.get("Valid") is False
    assert "TDMFuse=2 names MXSA and MXSB as the two single-wave members" in out
    assert "this is TDMFuse=6 with the scales moved" in out


def test_row_two_accepts_stagger(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """StaggerU is selected by YAML, not rejected by TDMFuse validation.

    Keep this case valid so TDMFuse=2 does not silently regain the removed
    StaggerU=0 hard restriction.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=2, StaggerU=32)
    assert sol.get("Valid") is True
    assert "requires StaggerU=0" not in out


def test_row_two_requires_an_equal_decoupled_pair(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """MXSB rides A's descriptor set but follows B's LDS block count.

    Kept here as well as beside TDMFuse=5, which exists because of it, so that
    this row's guard set is complete in one place: the divergent pair is the
    only one of row 2's refusals that turns on the decoupled feature at all.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=2,
                       PrefetchGlobalReadA=1, PrefetchGlobalReadB=2)
    assert sol.get("Valid") is False
    assert "TDMFuse=2 requires an equal decoupled pair" in out
    assert "its single swap arm cannot express both" in out


def test_row_two_declines_when_the_predicate_disagrees(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """The last guard in the row is a drift detector, and it must stay silent.

    tdmFuseAMx is what the writer actually reads, and the solution-level guards
    are meant to be exactly its preconditions. Reaching the final reject means
    the two have drifted apart, so the assertion is that an ACCEPTED solution
    satisfies the predicate -- not that the message can be produced, which on a
    correct tree it cannot.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=2)
    assert sol.get("Valid") is True, f"rejected with: {out!r}"
    assert tdmFuseAMx(sol._state) is True
    assert "tdmFuseAMx declined the solution" not in out


# ---------------------------------------------------------------------------
# TDMFuse=4, {MXSA,MXSB} + {A,B}. The default pairing, pinned rather than
# derived. No guard of this row was covered before this file.
# ---------------------------------------------------------------------------
def test_row_four_requires_wave_separated_tdm(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """At one wave every tensor already owns its descriptor and nothing is fused.

    Pinning a grouping that is not produced would put a TDMF4 token on a kernel
    identical to the unfused one, so two different names would describe the
    same assembly.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=4,
                       MatrixInstruction=_ONE_WAVE_MI, WorkGroup=_ONE_WAVE_WG)
    assert sol.get("Valid") is False
    assert "TDMFuse=4 requires wave-separated TDM (NumWaves > 1)" in out


def test_row_four_requires_mx_scales_on_both_tensors(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Without both scale tensors the only group left is {A,B}.

    The message names the group that goes missing, which is what distinguishes
    this from row 2's and row 6's MX refusals -- all three would otherwise read
    as the same objection.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=4,
                       ProblemType=_NO_MX_ON_B)
    assert sol.get("Valid") is False
    assert "TDMFuse=4 names the MX scale group {MXSA,MXSB}" in out
    assert "without them the only group is {A,B}" in out


# ---------------------------------------------------------------------------
# The sparse metadata clause. All four rows carry one and none was covered.
# ---------------------------------------------------------------------------
# TDMFuse names data and scale tensors only, so the metadata descriptor
# (tdmMetadataGroup0) is a tensor no value of the parameter accounts for. Rows 2
# and 5 call it "a descriptor", row 4 "a third descriptor", row 6 says only that
# the row does not describe it -- three texts, one mechanism, and the row number
# is what makes each assertion unambiguous.
@pytest.mark.parametrize("fuse", [2, 4, 5, 6])
def test_every_row_refuses_the_sparse_metadata_tensor(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Sparse puts a third tensor on the TDM that no grouping value names.

    Reached on the same MX shape as the rest of the file: Sparse turns
    enableTDMMetadata on, and these clauses sit before the decoupled feature's
    own "Sparse is not supported yet", so the row's objection is what surfaces.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse,
                       ProblemType={"Sparse": 1})
    assert sol.get("Valid") is False
    assert "TDMFuse=%d does not describe the sparse metadata tensor" % fuse in out


# ---------------------------------------------------------------------------
# TDMFuse=6, {A} + {B} + {MXSA,MXSB}.
# ---------------------------------------------------------------------------
def test_row_six_requires_mx_scales_on_both_tensors(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=6,
                       ProblemType=_NO_MX_ON_B)
    assert sol.get("Valid") is False
    assert "TDMFuse=6 names the MX scale group {MXSA,MXSB}" in out


@pytest.mark.parametrize(
    "pair, label",
    [
        ({"PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 2}, "equal"),
        ({"_remove": ("PrefetchGlobalReadA", "PrefetchGlobalReadB")}, "legacy"),
    ],
)
def test_row_six_without_a_divergent_pair_is_admitted_at_sia_four(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pair, label):
    """This pinned the opposite until the refusal's premise was measured away.

    e9504db146a refused row 6 at SIA=4 without a divergent pair because the
    equal pair computed wrong results there: at StinkyTofu OptLevel 3 the
    barrier rebuild left the LDS0 barrier inside the wave-parity region around
    the de-aliased fill, so half the workgroup skipped it. 4d9145fb2dc then
    corrected the rebuild to walk the whole tree, and the refusal outlived the
    defect it was written for -- self-sealingly, because the guard kept this
    arm out of every corpus, so no post-fix corpus contained it to re-measure.

    The premise changed, not the tolerance. Re-measured on the tree this test
    ships with, on b8-3 (MI450 B0) at MT64x512, DepthU 256, PGRA=PGRB=2,
    num-elements-to-validate=-1 and Random MX scales on BOTH tensors: 15 of 15
    reps PASSED at K=384, 512, 1024 and 1152. The negative control ran in the
    same session, on a build differing only in KernelWriter.py with 4d9145fb2dc
    and its dependent follow-up 0e0b4c342fd backed out, and FAILED 15 of 15 at
    K=1024 with 212638 of 262144 elements wrong and at K=1152 with 261736 --
    the same two K values and the same element counts as the original report in
    b8-3:~/_pgrval19/val.log. So the fix is what moves this arm, and the counts
    say so twice. Logs b8-3:~/f6val/{head,ctl}_rep1..15.log.

    What that does not license is a barrier COUNT as the regression check.
    Both builds emit eight barriers and only placement differs; the
    parity-guarded count is what separates them, reading 1 without the rebuild
    against 0 with it.

    An equal pair resolves away to its legacy scalar before this point, so both
    spellings arrive here as the same solution, and both have to be admitted.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       TDMFuse=6, ScheduleIterAlg=4, **pair)
    assert sol.get("Valid") is True, f"{label} rejected with: {out!r}"


def test_row_six_at_sia_four_with_a_divergent_pair_is_admitted(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Row 6 SIA clause admits divergent pairs.

    divergentPairUnsupportedReason reads derived _ScheduleIterAlg (SIA4 -> 0).
    OptLevel 3 rebuilds barriers from the whole tree; they must sit before the
    wave-parity branch around a de-aliased fill. hero=(1,2) mirror=(2,1).
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=6, ScheduleIterAlg=4,
                       PrefetchGlobalReadA=1, PrefetchGlobalReadB=2)
    assert sol.get("Valid") is True, f"rejected with: {out!r}"
    # The guard that used to refuse this ahead of the exemption stays silent.
    assert "only ScheduleIterAlg=0 places the fill where it can be moved" not in out


# The half of the old tripwire that is still load-bearing, and the reason the
# test above is not just a flipped assertion. SIA=4 is admitted because it
# derives to 0, not because the level was relaxed, so the hazard the refusal
# guarded still needs pinning: 1, 2 and 3 are different schedulers and none of
# them can host the relocated fill. 3 is the sharp one -- with the clause taken
# out it emits the relocation wrapping four empty `.header` modules and both
# early-return guards pass spuriously, so the kernel builds, fits and computes
# wrong results from K = 2*DepthU instead of coming back refused. That K is
# derived, not sampled -- two trips through the unrolled loop is what it takes
# for a fill to overwrite a block still being read, and the reject in
# Solution.depthUIteration carries the argument and the FFM sweep that agrees
# with it. It is not one of the empirical SIA4 thresholds, which were
# withdrawn.
#
# The clause differs at 2 because SIA=2 derives 1LDSBuffer=1, and that
# contradiction with a (1,2) pair is caught before the divergent guard is
# consulted. Pinning each message rather than just the refusal is what keeps
# that precedence visible: a widening of the derived-key read fails at 1 and 3
# here, and 2 keeps saying which guard is actually holding it.
@pytest.mark.parametrize(
    "sia, clause",
    [
        (1, "only ScheduleIterAlg=0 places the fill where it can be moved"),
        (2, "1LDSBuffer=1 gives every tensor one shared LDS block"),
        (3, "only ScheduleIterAlg=0 places the fill where it can be moved"),
    ],
)
def test_row_six_with_a_divergent_pair_is_still_refused_below_sia_four(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, sia, clause):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=6,
                       ScheduleIterAlg=sia,
                       PrefetchGlobalReadA=1, PrefetchGlobalReadB=2)
    assert sol.get("Valid") is False, sia
    assert clause in out


def test_row_six_accepts_every_pair_spelling_at_sia_zero(
        _gp_gfx1250, gfx1250_iim, assembler, capsys):
    """The complement of the SIA clause, and a correction to its reading.

    This row is often described as requiring a divergent pair. It does not: the
    equal and legacy spellings are accepted at ScheduleIterAlg=0, and only the
    (no divergent pair, SIA=4) combination is refused. Without this the SIA test
    above would pass just as well against a guard that refused equal pairs
    outright.
    """
    for pair, label in (
            ({"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2}, "divergent"),
            ({"PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 2}, "equal"),
            ({"_remove": ("PrefetchGlobalReadA", "PrefetchGlobalReadB")}, "legacy"),
    ):
        sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=6, **pair)
        assert sol.get("Valid") is True, f"{label} rejected with: {out!r}"


# ---------------------------------------------------------------------------
# Shared envelope. One message, four rows, and it is reachable from none of
# them on an MX shape -- so assert what is actually produced.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fuse", [2, 4, 5, 6])
def test_one_sided_tdm_is_refused_before_the_grouping_is_considered(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Every row here needs MX scales on both tensors, and an MX shape with a
    one-sided TDM is rejected by the scale-transport validator first.

    So TDMFuse's own "needs the TDM on both tensors" clause is unreachable from
    this direction rather than dead -- it is reachable only on a non-MX shape,
    which no row of this file can use. Assert the refusal that is produced, as
    test_tdmfuse_paired does for the same message.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, TDMInst=1)
    assert sol.get("Valid") is False
    assert "TDMA and TDMB must be enabled simultaneously" in out
    assert "describes how TDM transfers share descriptors" not in out


@pytest.mark.parametrize("fuse", [2, 4, 5, 6])
def test_subtile_is_refused_before_the_grouping_is_considered(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """Same precedence for UseSubtileImpl, and for the same reason.

    The subtile pipeline cannot move MXSB with one load per wave at all, which
    is a stronger objection than how descriptors would be grouped. All four
    rows carry their own UseSubtileImpl clause and none of them can be reached
    on a shape that has the MX scales they require.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse,
                       UseSubtileImpl=True)
    assert sol.get("Valid") is False
    assert "Unable to load MXSB scales using one load per wave" in out
    assert "not available with UseSubtileImpl" not in out


