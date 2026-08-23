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
"""Block-count map behind PrefetchGlobalReadA/B ("Decouple PGR", AIHPBLAS-4159).

``ldsBlocksForPgrLevel`` decides the whole LDS budget of a decoupled solution
and nothing else in-tree pins it down, so a silent change to the map is a
silent change to every decoupled kernel's LDS footprint.

The guards below it are here for the same reason. Each was wrong on a pushed
branch, and each was wrong in the way a guard fails quietly: some let a
solution through validation to die on an emitter assertion, one resolved a pair
away before the reject that contradicted it could see it, and one warned about
a kernel that computes wrong results instead of dropping it. None of that shows
up in a build that only asks whether kernels came out.
"""
import pytest

from Tensile.Common.DecouplePgr import (
    decoupledOneBlockBoth,
    decouplePgrBlocks,
    decoupledSingleBuffered,
    divergentPairUnsupportedReason,
    equalPairDegeneratesToScalar,
    ldsBlocksForPgrLevel,
)


# A block count, not a loop level: 0 and 1 are both a single block (1
# prefetches into it, 0 does not), 2 is the ping-pong pair.
@pytest.mark.parametrize(
    "level, blocks",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    ],
)
def test_lds_blocks_for_pgr_level(level, blocks):
    assert ldsBlocksForPgrLevel(level) == blocks


# (decoupled, blocksA, blocksB). An absent key falls back to the scalar, for
# that tensor only.
@pytest.mark.parametrize(
    "pgr, pgrA, pgrB, expected",
    [
        (2, None, None, (False, 2, 2)),
        (0, 0, 0, (True, 1, 1)),
        (1, 1, 1, (True, 1, 1)),
        (2, 2, 2, (True, 2, 2)),
        (2, 1, 2, (True, 1, 2)),
        (2, 2, 1, (True, 2, 1)),
        (2, 0, 2, (True, 1, 2)),
        (2, 2, 0, (True, 2, 1)),
        (2, 1, None, (True, 1, 2)),
        (2, None, 1, (True, 2, 1)),
    ],
)
def test_decouple_pgr_blocks(pgr, pgrA, pgrB, expected):
    ks = {"PrefetchGlobalRead": pgr}
    if pgrA is not None:
        ks["PrefetchGlobalReadA"] = pgrA
    if pgrB is not None:
        ks["PrefetchGlobalReadB"] = pgrB

    assert decouplePgrBlocks(ks) == expected


# Only a one-block tensor sharing a loop with a two-block one needs the
# write-after-read barriers. Equal counts do not: one block each is covered by
# decoupledOneBlockBoth, two blocks each ping-pong.
@pytest.mark.parametrize(
    "pgrA, pgrB, single",
    [
        (1, 2, True),
        (2, 1, True),
        (0, 2, True),
        (0, 0, False),
        (1, 1, False),
        (0, 1, False),
        (2, 2, False),
    ],
)
def test_decoupled_single_buffered(pgrA, pgrB, single):
    ks = {
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
    }

    assert decoupledSingleBuffered(ks) is single


def test_legacy_solution_is_not_decoupled():
    """No per-tensor key means the scalar owns the block count outright."""
    assert decoupledSingleBuffered({"PrefetchGlobalRead": 2}) is False


def _divergentSolution(**overrides):
    """A (1,2) pair with every precondition of the late fill satisfied.

    The loop shape is carried because one precondition is about the number of
    sub-iterations: DepthU 512 over MatrixInstK 128 is LoopIters 4, so the
    PrefetchLocalRead of 1 here sits well inside it.

    _ScheduleIterAlg is derived from ScheduleIterAlg rather than written out,
    because the guard reads the derived key and a test that set only the raw
    one would be pinning a state the deriver never produces.
    """
    ks = {
        "PrefetchGlobalRead": 1,
        "PrefetchGlobalReadA": 1,
        "PrefetchGlobalReadB": 2,
        "ScheduleIterAlg": 0,
        "PrefetchLocalRead": 1,
        "NumWaves": 4,
        "DepthU": 512,
        "LocalSplitU": 1,
        "InnerUnroll": 1,
        "MatrixInstK": 128,
        "EnableMatrixInstruction": True,
        "ClusterLocalRead": 1,
        "ForceUnrollSubIter": False,
    }
    ks.update(overrides)
    # Derived the way Solution.assignProblemIndependentDerivedParameters
    # derives it. An override that names the derived key directly still wins.
    ks.setdefault("_ScheduleIterAlg",
                  0 if ks["ScheduleIterAlg"] == 4 else ks["ScheduleIterAlg"])
    return ks


# Each of these is a precondition of relocating the single-buffered tensor's
# fill, and each has to come back as a reject: a divergent pair that reaches
# KernelWriter._dcpScheduleSingleBufferedFillLate without one dies on a bare
# assertion, which takes down the whole Tensile invocation instead of dropping
# the one solution that is unbuildable.
@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, None),
        ({"PrefetchGlobalReadB": 3}, "more than two LDS blocks"),
        ({"ScheduleIterAlg": 3}, "ScheduleIterAlg=0"),
        ({"PrefetchLocalRead": 0}, "PrefetchLocalRead must be at least 1"),
        ({"PrefetchLocalRead": 4}, "is not below LoopIters=4"),
        ({"NumWaves": 1}, "NumWaves > 1"),
    ],
)
def test_divergent_pair_unsupported_reason(overrides, expected):
    reason = divergentPairUnsupportedReason(_divergentSolution(**overrides))

    if expected is None:
        assert reason is None
    else:
        assert reason is not None
        assert expected in reason


def test_divergent_pair_at_one_wave_is_rejected_not_asserted():
    """NumWaves=1 is the precondition nothing used to check.

    Parity selects which tensor a wave fills, and parity means nothing with one
    wave -- KernelWriterAssembly.isTdmWaveSeparated is false there, which is
    exactly what the emitter asserts. Read this as: the guard exists at all, so
    the failure is a reject on one solution rather than an AssertionError on the
    sweep that contained it.
    """
    assert divergentPairUnsupportedReason(_divergentSolution(NumWaves=1)) is not None
    assert divergentPairUnsupportedReason(_divergentSolution(NumWaves=2)) is None


# The guard is on the derived schedule, not on the level the user asked for.
# ScheduleIterAlg=4 is StinkyTofu at OptLevel 3 over _ScheduleIterAlg=0, so the
# unrolled loop is the SIA0 loop and the slot the relocated fill needs is there;
# reading the raw key dropped these solutions for a scheduler they never run.
# 1, 2 and 3 are genuinely different schedulers and the reject is load-bearing
# for them: with the clause taken out, SIA3 emits the relocation guard with no
# instructions inside it, against six at SIA0.
@pytest.mark.parametrize(
    "scheduleIterAlg, accepted",
    [
        (0, True),
        (1, False),
        (2, False),
        (3, False),
        (4, True),
    ],
)
def test_divergent_pair_follows_the_derived_schedule_iter_alg(scheduleIterAlg, accepted):
    reason = divergentPairUnsupportedReason(
        _divergentSolution(ScheduleIterAlg=scheduleIterAlg)
    )

    if accepted:
        assert reason is None
    else:
        assert reason is not None and "ScheduleIterAlg" in reason


# 1LDSBuffer=1 gives every tensor one shared LDS block, which contradicts any
# pair asking for two. Degenerating an equal pair deletes both keys, so the
# reject that catches the contradiction never sees it and the solution silently
# builds the one block -- byte-identical to the legacy kernel and half the LDS
# that was asked for. Only an explicit 1 holds the pair back; -1 is the auto
# rule, which legacy PrefetchGlobalRead=k gets too.
@pytest.mark.parametrize(
    "pgrA, pgrB, oneLdsBuffer, degenerates",
    [
        (2, 2, 0, True),
        (2, 2, -1, True),
        (2, 2, None, True),
        (2, 2, 1, False),
        (0, 0, 0, True),
        (0, 0, 1, True),
        (1, 1, 0, False),
        (1, 1, 1, False),
        (1, 2, 0, False),
        (1, 2, 1, False),
    ],
)
def test_equal_pair_degenerates_to_scalar(pgrA, pgrB, oneLdsBuffer, degenerates):
    ks = {
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
    }
    if oneLdsBuffer is not None:
        ks["1LDSBuffer"] = oneLdsBuffer

    assert equalPairDegeneratesToScalar(ks) is degenerates


def test_legacy_solution_does_not_degenerate():
    """There is no pair to resolve away, whatever 1LDSBuffer says."""
    assert equalPairDegeneratesToScalar({"PrefetchGlobalRead": 2, "1LDSBuffer": 1}) is False


# PrefetchLocalRead at or above LoopIters is rewritten to 0 by
# Solution.assignDerivedParameters, after this reject has already looked at it,
# so the clause that rejects 0 outright never sees the 0 that gets made. Every
# cell here was measured on the emitter: below LoopIters it builds, at or above
# it asserted. LoopIters is DepthU / LocalSplitU / InnerUnroll / MatrixInstK,
# which for MatrixInstK 128 is 4, 2 and 1 at DepthU 512, 256 and 128.
#
# Rows pin what divergentPairUnsupportedReason returns for this grid.
@pytest.mark.parametrize(
    "depthU, prefetchLocalRead, rejected",
    [
        (512, 1, False),
        (512, 2, False),
        (512, 3, False),
        (512, 4, True),
        (512, 5, True),
        (512, 6, True),
        (512, 7, True),
        (512, 8, True),
        (256, 1, False),
        (256, 2, True),
        (256, 3, True),
        (256, 4, True),
        (256, 5, True),
        (128, 1, True),
    ],
)
def test_prefetch_local_read_below_loop_iters(depthU, prefetchLocalRead, rejected):
    reason = divergentPairUnsupportedReason(
        _divergentSolution(DepthU=depthU, PrefetchLocalRead=prefetchLocalRead)
    )

    if rejected:
        assert reason is not None and "LoopIters" in reason
    else:
        assert reason is None


def test_prefetch_local_read_rewrite_is_rejected_not_asserted():
    """Reaching PrefetchLocalRead=0 by rewrite has to reject like writing it does.

    Both routes end at the same kernel, so they cannot end at different
    outcomes. Writing 0 rejected already; arriving at 0 through the rewrite ran
    on to KernelWriter._dcpScheduleSingleBufferedFillLate and asserted.
    """
    written = divergentPairUnsupportedReason(_divergentSolution(PrefetchLocalRead=0))
    rewritten = divergentPairUnsupportedReason(
        _divergentSolution(DepthU=256, PrefetchLocalRead=2)
    )

    assert written is not None
    assert rewritten is not None


def test_prefetch_local_read_guard_is_only_for_the_rewrite():
    """No rewrite, no reject: the guard covers that rewrite and nothing wider.

    The rewrite is conditional on ClusterLocalRead, so with it off the value
    survives and the emitter is not handed a 0 it did not ask for. Rejecting
    there would drop solutions that were never affected.
    """
    ks = _divergentSolution(DepthU=256, PrefetchLocalRead=2, ClusterLocalRead=0)

    assert divergentPairUnsupportedReason(ks) is None


def test_more_lds_blocks_masks_the_loop_iters_guard():
    """(1,4) at DepthU 128 is caught by block count first, not by this guard.

    Worth pinning because it cuts the other way from most masking: the block
    count clause is what keeps aggressive pairs away from the assertion today,
    so relaxing it without this guard in place would widen the assertion rather
    than narrow it.
    """
    aggressive = _divergentSolution(DepthU=128, PrefetchGlobalReadB=4, PrefetchLocalRead=1)

    assert "more than two LDS blocks" in divergentPairUnsupportedReason(aggressive)


# (0,1)/(1,0)/(1,1) are one LDS block; reject from K=2*DepthU.
@pytest.mark.parametrize(
    "pgrA, pgrB, oneBlockBoth",
    [
        (1, 1, True),
        (0, 1, True),
        (1, 0, True),
        (0, 0, False),
        (1, 2, False),
        (2, 1, False),
        (2, 2, False),
    ],
)
def test_decoupled_one_block_both(pgrA, pgrB, oneBlockBoth):
    ks = {
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
    }

    assert decoupledOneBlockBoth(ks) is oneBlockBoth


def test_legacy_one_block_is_out_of_reach_of_the_pair_guard():
    """The identical legacy kernel is not this guard's to reject.

    PrefetchGlobalRead=1 with 1LDSBuffer=1 emits assembly byte-identical to the
    (1,1) pair and carries the same defect, but it is selected without any
    per-tensor parameter and predates this feature. Rejecting it here would be
    a guard on the pair keys reaching past the pair keys; AIHPBLAS-4159 covers
    fixing the shared emit properly.
    """
    assert decoupledOneBlockBoth({"PrefetchGlobalRead": 1, "1LDSBuffer": 1}) is False
