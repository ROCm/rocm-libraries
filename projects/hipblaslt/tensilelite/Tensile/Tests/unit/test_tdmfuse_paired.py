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
"""TDMFuse=5, the crossed {MXSA,A} + {MXSB,B} TDM descriptor grouping.

Three things here are load-bearing and nothing else in-tree states them.

The wave division. A and MXSB take the even waves, MXSA and B the odd ones, so
A and B keep the parity the default pairing gives them and only the scales change
side. tdmWavePartition is the single source of truth for it, and the LDS side,
the global-address side and the dispatch guards all read that one answer -- a
grouping that got it wrong there would have a wave move one tile from memory and
write another tile's LDS slot, which validates or not by luck rather than failing
to build.

The consequence the design of this row did not expect. Cadence-homogeneous sets
are what let one swap arm serve a set, but they are also what stops WAVE PARITY
from deciding a wave's LDS cadence: with {MXSA,A} all on block count A and
{MXSB,B} all on B, each parity ends up carrying one single-buffered tensor and
one double-buffered one. KernelWriter._dcpScheduleSingleBufferedFillLate splits
by parity for every other grouping, so at a divergent PrefetchGlobalReadA/B pair
this row needs the per-set relocation instead. test_cadence_is_not_separated_by
_parity pins the property the two branches turn on; it is the reason the code has
two branches at all.

The guards. Each is a reject, not an assert: an AssertionError in a solution
predicate takes down a whole TensileCreateLibrary run rather than dropping the
one solution that cannot be built.
"""
import copy

import pytest

from Tensile.Common.DecouplePgr import (
    decouplePgrBlocks,
    tdmFuseAMx,
    tdmFusePaired,
    tdmWaveCompIdMode,
    tdmWaveComponents,
    tdmWavePartition,
)
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.SolutionStructs.Naming import getKernelNameMin
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit


# Sibling unit tests mutate the process-global defaultSolution in place, which
# makes Solution.__init__'s `for key in defaultSolution` loop order-dependent.
# Snapshot it at collection time, as test_halfplr_streamk_rejects does.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))

_TENSORS = ("A", "MXSA", "MXSB", "B")


def _ks(fuse=5, pgrA=1, pgrB=2, **overrides):
    """The smallest dict the grouping predicates read."""
    ks = {
        "TDMFuse": fuse,
        "TDMInst": 3,
        "TDMSplit": False,
        "NumWaves": 4,
        "UseSubtileImpl": False,
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
        "enableTDMA": True,
        "enableTDMB": True,
        "LdsOffsetBlkA": 1 << 15,
        "LdsOffsetBlkB": 1 << 16,
    }
    ks.update(overrides)
    return ks


class _Grouping:
    """Just enough of KernelWriterAssembly to answer the grouping questions.

    Every member is the real method, so these tests fail when the writer's
    predicates change rather than agreeing with a second implementation of them.
    """

    isTdmWaveSeparated = KernelWriterAssembly.isTdmWaveSeparated
    tdmDealiasAB = KernelWriterAssembly.tdmDealiasAB
    tdmFuseAMx = KernelWriterAssembly.tdmFuseAMx
    tdmFusePaired = KernelWriterAssembly.tdmFusePaired
    tdmSeparateABDescriptors = KernelWriterAssembly.tdmSeparateABDescriptors
    tdmDescriptorSetOwner = KernelWriterAssembly.tdmDescriptorSetOwner
    _tdmPairedParityOrder = KernelWriterAssembly._tdmPairedParityOrder
    _tdmAliasPartner = KernelWriterAssembly._tdmAliasPartner
    _tdmParityMembers = KernelWriterAssembly._tdmParityMembers
    _tdmDecoupledBlocks = KernelWriterAssembly._tdmDecoupledBlocks
    _tdmDecoupledGroup = KernelWriterAssembly._tdmDecoupledGroup
    _tdmDecoupledAliasPartner = KernelWriterAssembly._tdmDecoupledAliasPartner
    _tdmDecoupledAliasPartnerPaired = KernelWriterAssembly._tdmDecoupledAliasPartnerPaired


W = _Grouping()


# ---------------------------------------------------------------------------
# The predicate.
# ---------------------------------------------------------------------------
def test_paired_is_selected_only_by_five():
    """Nothing derives this grouping, so 0 stays inert and 2/4/6 keep theirs."""
    assert tdmFusePaired(_ks(fuse=5)) is True
    for other in (0, 2, 4, 6):
        assert tdmFusePaired(_ks(fuse=other)) is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"NumWaves": 1},
        {"UseSubtileImpl": True},
        {"TDMSplit": True},
        {"TDMInst": 1},
        {"TDMInst": 2},
        {"ProblemType": {"MXBlockA": 0, "MXBlockB": 32}},
        {"ProblemType": {"MXBlockA": 32, "MXBlockB": 0}},
    ],
)
def test_paired_declines_outside_its_envelope(overrides):
    assert tdmFusePaired(_ks(**overrides)) is False


@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1), (2, 2), (0, 2), (1, 1)])
def test_paired_does_not_key_on_the_block_counts(pgrA, pgrB):
    """Unlike 2 and 6 this row holds at every pair, which is the point of it.

    2 is rejected at a divergent pair because its three-member set carries two
    cadences; 6 requires one because its de-aliased cadence was verified nowhere
    else. Each of this row's sets is one cadence at every pair, so neither
    argument reaches it.
    """
    assert tdmFusePaired(_ks(pgrA=pgrA, pgrB=pgrB)) is True


@pytest.mark.parametrize("numWaves", [2, 4, 8])
def test_paired_holds_at_every_wave_count_parity_can_split(numWaves):
    """No remainder policy here, unlike TDMFuse=2's 1/1/2 dispatch at four waves."""
    assert tdmFusePaired(_ks(NumWaves=numWaves)) is True
    assert tdmFuseAMx(_ks(fuse=2, NumWaves=numWaves)) is (numWaves == 4)


# ---------------------------------------------------------------------------
# The wave division: A and MXSB even, MXSA and B odd.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tc, waves",
    [
        ("A", (0, 2)),
        ("MXSA", (1, 3)),
        ("MXSB", (0, 2)),
        ("B", (1, 3)),
    ],
)
def test_crossed_wave_assignment(tc, waves):
    numComp, got = tdmWavePartition(_ks(), tc)
    assert got == waves
    # Two waves per component, so the component id is WaveIdx >> 1 -- the same
    # shift the default pairing uses, which is what keeps the global-address
    # arithmetic of the data tensors unchanged.
    assert numComp == 2
    assert tdmWaveCompIdMode(_ks(), tc) == "parity"
    assert tdmWaveComponents(_ks(), tc) == (2, 1)


def test_only_the_scales_change_parity_against_the_default():
    """The data tensors' division is byte-identical to TDMFuse=0's.

    That is the whole argument for taking the crossed form over the parallel one:
    A's and B's global addressing does not move, and every wave still carries one
    data tensor -- rather than both data tensors landing on the even waves and
    both of the much smaller scale tensors on the odd.
    """
    paired, default = _ks(fuse=5), _ks(fuse=0)
    assert tdmWavePartition(paired, "A") == tdmWavePartition(default, "A")
    assert tdmWavePartition(paired, "B") == tdmWavePartition(default, "B")
    assert tdmWavePartition(paired, "MXSA") == tdmWavePartition(default, "MXSB")
    assert tdmWavePartition(paired, "MXSB") == tdmWavePartition(default, "MXSA")

    for wave in range(4):
        carried = [tc for tc in _TENSORS if wave in tdmWavePartition(paired, tc)[1]]
        assert len(carried) == 2
        assert sum(1 for tc in carried if "MXS" in tc) == 1


# ---------------------------------------------------------------------------
# The descriptor sets, and what the emitter derives from them.
# ---------------------------------------------------------------------------
def test_each_data_tensor_owns_a_set_and_carries_its_own_scales():
    ks = _ks()
    assert {tc: W.tdmDescriptorSetOwner(ks, tc) for tc in _TENSORS} == {
        "A": "A", "MXSA": "A", "MXSB": "B", "B": "B"}
    # The default pairs across tensors instead: one set for the data, one for
    # the scales.
    assert {tc: W.tdmDescriptorSetOwner(_ks(fuse=0), tc) for tc in _TENSORS} == {
        "A": "A", "MXSA": "MXSA", "MXSB": "MXSA", "B": "A"}


def test_aliases_follow_the_owner():
    ks = _ks()
    assert W._tdmAliasPartner(ks, "A") == "MXSA"
    assert W._tdmAliasPartner(ks, "B") == "MXSB"
    assert W._tdmAliasPartner(_ks(fuse=0), "A") == "B"
    assert W._tdmAliasPartner(_ks(fuse=0), "MXSA") == "MXSB"


def test_a_and_b_land_on_distinct_register_sets():
    """B owns a set here, so its fill is its own instruction and its own swap."""
    assert W.tdmSeparateABDescriptors(_ks()) is True
    assert W.tdmSeparateABDescriptors(_ks(fuse=0)) is False


def test_swap_arms_are_placed_by_parity_not_by_owner():
    """The set B owns holds MXSB on its EVEN waves.

    _tdmSwapLdsOffsetDecoupled compares the live LDS address against the second-
    copy base of the tensor the wave actually holds, so an arm placed by which
    name owns the set would compare B's base on the waves holding MXSB and swap
    to the wrong tile.
    """
    ks = _ks()
    assert W._tdmParityMembers(ks, "A", "MXSA") == ("A", "MXSA")
    assert W._tdmParityMembers(ks, "B", "MXSB") == ("MXSB", "B")
    # Nothing crosses under the default pairing, so both arms keep the pair order.
    assert W._tdmParityMembers(_ks(fuse=0), "A", "B") == ("A", "B")
    assert W._tdmParityMembers(_ks(fuse=0), "MXSA", "MXSB") == ("MXSA", "MXSB")


def test_the_scale_call_programs_the_set_b_rides():
    """The wave-separated helpers are called per (A,B) and per (MXSA,MXSB) pair.

    This row crosses those pairs, so the scale call's even member is MXSB -- and
    with it the descriptor that call initialises, offsets and advances. Getting
    this wrong programs one set twice per wave and leaves the other stale.
    """
    ks = _ks()
    tPA, tPB = {"tensorChar": "A"}, {"tensorChar": "B"}
    tMXA, tMXB = {"tensorChar": "MXSA"}, {"tensorChar": "MXSB"}
    assert W._tdmPairedParityOrder(ks, tPA, tPB) == (tPA, tPB)
    assert W._tdmPairedParityOrder(ks, tMXA, tMXB) == (tMXB, tMXA)
    assert W._tdmPairedParityOrder(_ks(fuse=0), tMXA, tMXB) == (tMXA, tMXB)


def test_each_set_carries_one_lds_block_count():
    """One cadence per set is what lets the parity-aware swap arm express it.

    TDMFuse=2 is rejected at a divergent pair for want of exactly this: its
    {A,MXSA,MXSB} set carries A's count and B's at once.
    """
    hero = _ks(pgrA=1, pgrB=2)
    setBlocks = {}
    for owner in ("A", "B"):
        members = [tc for tc in _TENSORS if W.tdmDescriptorSetOwner(hero, tc) == owner]
        setBlocks[owner] = {W._tdmDecoupledBlocks(hero, tc)[0] for tc in members}
    assert setBlocks == {"A": {1}, "B": {2}}


def test_cadence_is_not_separated_by_parity():
    """The price of one cadence per set, and the reason the late fill changed.

    KernelWriter._dcpScheduleSingleBufferedFillLate duplicates the fill group at
    two slots under complementary wave-parity guards, which is only correct while
    one parity is entirely single-buffered and the other entirely double. That
    holds under the default pairing and does not hold here: each parity carries
    one of each, so this row moves the single-buffered SET's fill instead of
    guarding on parity. Both the parallel and the crossed wave assignment have
    this property -- it follows from the grouping, not from which parity the
    scales are put on.
    """
    def cadencesByParity(ks):
        out = {}
        for parity in (0, 1):
            out[parity] = {W._tdmDecoupledBlocks(ks, tc)[0] for tc in _TENSORS
                           if parity in [w % 2 for w in tdmWavePartition(ks, tc)[1]]}
        return out

    hero = _ks(pgrA=1, pgrB=2)
    assert cadencesByParity(hero) == {0: {1, 2}, 1: {1, 2}}
    # The default pairing is the case the parity split was written for.
    assert cadencesByParity(_ks(fuse=0, pgrA=1, pgrB=2)) == {0: {1}, 1: {2}}


# ---------------------------------------------------------------------------
# Kernel naming.
# ---------------------------------------------------------------------------
def _namingKernel(**overrides):
    kernel = copy.deepcopy(dict(_PRISTINE_DEFAULT_SOLUTION))
    kernel["ProblemType"] = {
        "OperationIdentifier": "Cijk_Alik_Bljk",
        "DataType": 0,
        "DestDataType": 0,
        "ComputeDataType": 0,
        "GroupedGemm": False,
        "UseBeta": False,
        "UseBias": 0,
    }
    kernel["MacroTile0"] = 64
    kernel["MacroTile1"] = 512
    kernel["DepthU"] = 512
    kernel["MatrixInstM"] = 16
    kernel["MatrixInstN"] = 16
    kernel["MatrixInstB"] = 1
    kernel["MatrixInstruction"] = [16, 16, 128, 1]
    kernel["MIWaveTile"] = [2, 16]
    kernel["WorkGroup"] = [32, 4, 1]
    kernel["ISA"] = (12, 5, 0)
    kernel.update(overrides)
    return kernel


def test_paired_is_named():
    """A pinned grouping has to be in the name, or two kernels dedup to one."""
    assert "TDMF5" in getKernelNameMin(_namingKernel(TDMFuse=5), False)


def test_off_leaves_the_name_alone():
    """0 derives nothing, so naming it would rename every pre-existing kernel."""
    name = getKernelNameMin(_namingKernel(TDMFuse=0), False)
    assert "TDMF" not in name
    assert name == getKernelNameMin(_namingKernel(), False)


def test_each_grouping_gets_its_own_name():
    names = {f: getKernelNameMin(_namingKernel(TDMFuse=f), False) for f in (0, 2, 4, 5, 6)}
    assert len(set(names.values())) == 5
    assert "TDMF5" in names[5] and "TDMF5" not in names[2]


# ---------------------------------------------------------------------------
# Solution-level guards, on a real derived gfx1250 solution.
# ---------------------------------------------------------------------------
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
    """The F8F4 TN MT64x512 DepthU256 shape this row is meant for, plus overrides.

    DepthU 256: at 512 the equal pair exceeds MaxLDS for every TDMFuse value.
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
        "TDMFuse": 5,
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
        mi, isa, params["WavefrontSize"], problemType, params["WorkGroup"], gfx1250_iim))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


# The accept cases come first: every reject below is vacuous if the shape it
# starts from is itself rejected.
@pytest.mark.parametrize(
    "pgrA, pgrB, label",
    [(1, 2, "hero"), (2, 1, "mirror"), (2, 2, "equal")],
)
def test_accepted_at_every_pair(_gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB, label):
    """Including the hero (1,2) this row was asked for, where TDMFuse=2 declines.

    The equal pair is here too: it resolves away to legacy PrefetchGlobalRead=2
    before this reject runs, and this row needs no pair at all -- each of its
    sets is one cadence whether or not the two cadences differ.
    Mirror (2,1) is Valid: thick-wait1 is on A (2 LDS blocks), symmetric with
    hero's wait on B.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalRead=max(pgrA, pgrB),
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, f"{label} rejected with: {out!r}"


def test_accepts_mirror_divergent_thick_wait(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """A=2/B=1: same Valid contract as hero; wait belongs on thick A."""
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalRead=2,
                       PrefetchGlobalReadA=2, PrefetchGlobalReadB=1)
    assert sol.get("Valid") is True, f"mirror_f5 rejected with: {out!r}"
    assert "TDMFuse=5 cannot honour its divergent thick-wait" not in out


class _ThickWaitStub:
    """Stand in for KernelWriter: only _dcpApplyThickWait1 + decouplePgrBlocks."""

    class states:
        memTokenLdsDcp = {"A": (0, 1), "B": (0, 1)}

    def _dcpDivergent(self, kernel):
        return True


@pytest.mark.parametrize(
    "pgrA, pgrB, thick, marker",
    [(1, 2, "B", "DcpEarlyFillB"), (2, 1, "A", "DcpEarlyFillA")],
)
def test_thick_wait1_lands_on_the_thick_tensor(pgrA, pgrB, thick, marker):
    """Hero waits on B; mirror waits on A. Both rewrite to tensorcnt(1)."""
    from Tensile.KernelWriter import KernelWriter

    asm = (
        f"{marker}End_0:\n"
        "s_wait_tensorcnt 0\n"
        "s_nop 0\n"
        f"{marker}End_1:\n"
        "s_wait_tensorcnt 0\n"
        "s_nop 0\n"
        "DcpLateFillXEnd_0:\n"
        "s_wait_tensorcnt 0\n"
    )
    kernel = {
        "TDMFuse": 5,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
    }
    out = KernelWriter._dcpApplyThickWait1(_ThickWaitStub(), kernel, asm)
    waits = [ln.strip() for ln in out.splitlines() if ln.startswith("s_wait_tensorcnt")]
    # Two cloned thick-header waits become 1; the late/thin wait stays 0.
    assert waits == ["s_wait_tensorcnt 1", "s_wait_tensorcnt 1", "s_wait_tensorcnt 0"], waits
    assert marker in out


def test_two_is_still_refused_at_the_hero(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """The reason this row was asked for, kept next to the row that answers it."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=2)
    assert sol.get("Valid") is False
    assert "TDMFuse=2 requires an equal decoupled pair" in out


def test_rejects_one_wave(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       MatrixInstruction=[16, 16, 128, 1, 1, 2, 16, 1, 1],
                       WorkGroup=[32, 1, 1])
    assert sol.get("Valid") is False
    assert "TDMFuse=5 splits each of its two descriptor sets by wave parity" in out


def test_rejects_subtile(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Refused, but by an earlier and more fundamental guard.

    The subtile pipeline cannot move MXSB with one load per wave at all, which is
    a stronger objection than how the descriptors would be grouped, so this
    row's own UseSubtileImpl message is unreachable from this direction rather
    than dead -- the same precedence TDMFuse=2's TDMInst message has. The
    predicate half is covered by test_paired_declines_outside_its_envelope.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, UseSubtileImpl=True)
    assert sol.get("Valid") is False
    assert "Unable to load MXSB scales using one load per wave" in out


def test_rejects_without_mx_scales_on_both(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       ProblemType={"MacDataTypeB": "F8", "DataTypeMXSB": "E8",
                                    "MXBlockB": 0})
    assert sol.get("Valid") is False
    assert "TDMFuse=5 names MXSA and MXSB as the odd-wave member" in out


@pytest.mark.xfail(
    reason="TDMSplit is currently disabled upstream (PR #10911)", strict=False)
def test_rejects_tdmsplit(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Row 5's own TDMSplit refusal, which upstream now preempts.

    97e1223a3f9 (PR #10911) rejects TDMSplit at function scope in Solution.py,
    above this row's guard, so the message below is unreachable rather than
    wrong. strict=False so this revives by itself once that temporary reject is
    lifted; the guard it covers is left in place for the same reason.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMSplit=True)
    assert sol.get("Valid") is False
    assert "TDMFuse=5 is not available with TDMSplit" in out


def test_accepts_stagger(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """StaggerU is selected by YAML, not rejected by TDMFuse validation.

    Keep this case valid so TDMFuse=5 does not silently regain the removed
    StaggerU=0 hard restriction.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, StaggerU=32)
    assert sol.get("Valid") is True
    assert "requires StaggerU=0" not in out


def test_rejects_halfplr_at_a_divergent_pair(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """HalfPLR's increment mask rides in the module the relocation moves.

    Leaving it at the top would be wrong for the set that moved; moving it late
    leaves the set that stayed advancing on the final iteration. Equal pairs run
    no relocation and are not covered by this.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, HalfPLR=1)
    assert sol.get("Valid") is False
    assert "TDMFuse=5 requires HalfPLR=0 at a divergent decoupled pair" in out


def test_rejects_without_tdm_on_both_tensors(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Also refused earlier, on this MX shape.

    The shared "TDMFuse=%d describes how TDM transfers share descriptors" clause
    is reachable only on a non-MX shape, and this row requires MX scales on both
    tensors, so on any shape it could apply to the one-sided TDMInst is rejected
    first. Assert the refusal, not a message that cannot be produced here.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMInst=1)
    assert sol.get("Valid") is False
    assert "TDMA and TDMB must be enabled simultaneously" in out


def test_guards_agree_with_the_predicate(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """An accepted solution must be one the writer actually groups this way.

    The last reject in the block declines when the two disagree; reaching it
    means a guard drifted, so assert the accepted state satisfies the predicate
    rather than only that the message exists.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"rejected with: {out!r}"
    assert tdmFusePaired(sol._state) is True
    assert "tdmFusePaired declined the solution" not in out
    assert decouplePgrBlocks(sol._state)[1:] == (1, 2)
