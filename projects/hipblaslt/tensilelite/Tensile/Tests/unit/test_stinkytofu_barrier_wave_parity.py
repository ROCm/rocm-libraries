# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No barrier may be reachable only under a wave-parity guard.

At ScheduleIterAlg=4 (StinkyTofu OptLevel 3) postMainLoopBarrierCheckAndReset
deletes every barrier in the kernel and rebuilds placement from memory tokens.
The guard that decides which wave fills which tensor -- s_bitcmp1_b32 on bit 0 of
the wave index, then an s_cbranch -- is emitted in a DIFFERENT module from the
tensor_load_to_lds it guards, by
KernelWriter._dcpScheduleSingleBufferedFillLate and
KernelWriterAssembly._emitTdmDealiasedIssue. A rebuild that recursed module by
module could not see that branch, so it put the barrier after it.

The invariant is asserted, not the count: a pass can emit exactly the right
number of barriers and still place one where half the workgroup branches over
it, which assembles cleanly and returns wrong results on hardware. The resulting
failures are intermittent rather than tied to any problem size, so a barrier
count -- or a single green run -- cannot stand in for the invariant.

The last test is the other half of the invariant: a workgroup-uniform branch must
NOT move a barrier, because the unroll loop is itself skipped by one of those and
hoisting out of it would take the barrier out of the loop.
"""
from rocisa.code import Label, Module
from rocisa.container import DSModifiers, MemTokenData, sgpr, vgpr
from rocisa.instruction import (
    DSLoadB64,
    Instruction,
    SAndB32,
    SBarrier,
    SBitcmp1B32,
    SCBranchSCC1,
    SCmpEQU32,
    SCmpLeU32,
    SLShiftRightB32,
    TensorLoadToLds,
    VReadfirstlaneB32,
)

from Tensile.KernelWriter import KernelWriter

LDS_TOKEN = 0


class _Writer:
    """The pass reads the fallback ScheduleIterAlg, the kernel name, and the
    overflowedResources flag it sets to decline a solution. That is little enough
    to run against a hand-built module tree without standing up a whole
    KernelWriter -- but the flag has to be per-instance, or one refusal would
    leak into every later test.
    """

    class _States:
        def __init__(self):
            self.scheduleIterAlg = 0
            self.kernelName = "unit_test_kernel"
            self.overflowedResources = 0

    class _DebugConfig:
        printSolutionRejectionReason = False

    def __init__(self):
        self.states = _Writer._States()
        self.debugConfig = _Writer._DebugConfig()


def _kernel(**overrides):
    # The pass's own gate: OptLevel 3, derived SIA 0, TDM on both tensors, more
    # than one wave. ScheduleIterAlg=4 is what produces the first two.
    kernel = {
        "NumThreads": 128,
        "WavefrontSize": 32,
        "_StinkyTofuOptLevel": 3,
        "_ScheduleIterAlg": 0,
        "enableTDMA": True,
        "enableTDMB": True,
        "PrefetchGlobalRead": 2,
    }
    kernel.update(overrides)
    return kernel


def _runPass(kernel, root):
    writer = _Writer()
    KernelWriter.postMainLoopBarrierCheckAndReset(writer, kernel, root)
    return writer


# checkResources code for "no workgroup-wide position for a rebuilt LDS barrier".
_REJECTED = 9


def _read():
    inst = DSLoadB64(dst=vgpr("ValuA_X0_I0", 2), src=vgpr("LocalReadAddrA"),
                     ds=DSModifiers(offset=0), comment="read the block")
    inst.setMemToken(MemTokenData([LDS_TOKEN]))
    return inst


def _fill():
    inst = TensorLoadToLds(sgpr("tdmAGroup0", 4), sgpr("tdmAGroup1", 8), None, None,
                          "refill the block")
    inst.setMemToken(MemTokenData([LDS_TOKEN]))
    return inst


def _guardedFillTree(guardOpener):
    """The writer's shape: the guard in one module, the conflicting fill in another.

    A read of the block comes first, so the refill is a write-after-read and the
    pass has a transition to place a barrier on at all. `guardOpener` supplies the
    comparison, which is the only difference between a wave-parity guard and a
    workgroup-uniform one.
    """
    root = Module("kernelBody")
    root.add(_read())

    end = Label("DcpEarlyFillAEnd", "")
    guarded = Module("TDM decoupled early fill B")
    guarded.add(guardOpener())
    guarded.add(SCBranchSCC1(labelName=end.getLabelName(),
                             comment="B is single-buffered, its fill moves late"))
    # The separate module is the whole point: it is what the old per-module
    # recursion descended into with the branch above already out of scope.
    fillGroup = Module("globalReadA")
    fillGroup.add(_fill())
    guarded.add(fillGroup)
    guarded.add(end)
    root.add(guarded)
    return root, guarded, fillGroup


def _waveParityCompare():
    return SBitcmp1B32(src0=sgpr("WaveIdx"), src1=0, comment="check wave parity")


def _tripCountCompare():
    return SCmpLeU32(src0=sgpr("LoopCounterL"), src1=1, comment="LoopCounterL < EndCounter")


# The number the allocator happens to hand out for the recomputed index. Which
# number it is does not matter; that it is a number and not a symbol does.
_WAVE_TEMP = 5


def _recomputedWaveParityCompare():
    """The same guard, emitted where sgprWaveIdx is no longer live.

    KernelWriterAssembly releases sgprWaveIdx after the stagger, so the guard
    reads the index back out of vgprSerial into a temporary and compares that
    number. There is no symbol here for the pass to match on, and the thread id
    the read lands has to be divided by the wave length in place first.
    """
    module = Module("recomputed wave index")
    module.add(VReadfirstlaneB32(dst=sgpr(_WAVE_TEMP), src=vgpr("Serial"),
                                 comment="get tId"))
    module.add(SLShiftRightB32(dst=sgpr(_WAVE_TEMP), shiftHex=5,
                               src=sgpr(_WAVE_TEMP), comment="waveId"))
    module.add(SBitcmp1B32(src0=sgpr(_WAVE_TEMP), src1=0, comment="check wave parity"))
    return module


def _reusedTemporaryCompare():
    """The temporary goes back to the allocator and something unrelated takes the
    number, which is guaranteed to happen now that sgprWaveIdx is released early.

    The comparison names a register that once held a wave index and no longer
    does, and the branch it feeds is taken by the whole workgroup.
    """
    module = Module("reused temporary")
    module.add(VReadfirstlaneB32(dst=sgpr(_WAVE_TEMP), src=vgpr("Serial"),
                                 comment="get tId"))
    module.add(SLShiftRightB32(dst=sgpr(_WAVE_TEMP), shiftHex=5,
                               src=sgpr(_WAVE_TEMP), comment="waveId"))
    module.add(SAndB32(dst=sgpr(_WAVE_TEMP), src0=sgpr("GSU"), src1=0x3fff,
                       comment="the number is reused for something else"))
    module.add(SCmpEQU32(src0=sgpr(_WAVE_TEMP), src1=1, comment="GSU == 1"))
    return module


def _flatten(module):
    out = []
    for item in module.items():
        if isinstance(item, Module):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def _isLabelDef(leaf):
    return not isinstance(leaf, Instruction) and hasattr(leaf, "getLabelName")


def _waveParityGuardedBarriers(module):
    """Barriers reachable only when a wave-parity branch falls through.

    Written independently of how the pass tracks this: it walks the finished tree,
    follows SCC from the comparison that names the wave index to the branch that
    reads it, and reports every barrier between that branch and its target.
    """
    leaves = _flatten(module)
    labelIndex = {}
    for i, leaf in enumerate(leaves):
        if _isLabelDef(leaf):
            labelIndex.setdefault(leaf.getLabelName(), i)

    offending = []
    openEnds = []
    sccIsWaveParity = False
    for i, leaf in enumerate(leaves):
        openEnds = [e for e in openEnds if e > i]
        if isinstance(leaf, SBarrier):
            if openEnds:
                offending.append((i, str(leaf).strip()))
            continue
        if isinstance(leaf, (SBitcmp1B32, SCmpLeU32)):
            sccIsWaveParity = "WaveIdx" in str(leaf)
            continue
        target = getattr(leaf, "labelName", None)
        if target is not None and sccIsWaveParity:
            end = labelIndex.get(target)
            if end is not None and end > i:
                openEnds.append(end)
    return offending


def _barriers(module):
    return [leaf for leaf in _flatten(module) if isinstance(leaf, SBarrier)]


def test_no_barrier_is_reachable_only_under_a_wave_parity_guard():
    root, _guarded, _fillGroup = _guardedFillTree(_waveParityCompare)
    _runPass(_kernel(), root)

    offending = _waveParityGuardedBarriers(root)
    assert not offending, (
        "%d barrier(s) sit between a wave-parity s_cbranch and its target, so only "
        "the waves that fall through execute them: %s" % (len(offending), offending))


def test_the_guarded_fill_is_still_synchronised():
    """The invariant must not be met by emitting nothing.

    A write-after-read on the block needs one barrier, and it has to be somewhere
    the whole workgroup reaches -- which here is the module holding the branch,
    ahead of it, not the module holding the fill.
    """
    root, guarded, fillGroup = _guardedFillTree(_waveParityCompare)
    _runPass(_kernel(), root)

    assert len(_barriers(root)) == 1, \
        "expected exactly one barrier for the one write-after-read, got %d" % len(_barriers(root))
    assert not [x for x in fillGroup.items() if isinstance(x, SBarrier)], \
        "the barrier is still inside the guarded fill group"
    items = guarded.items()
    barrierAt = next(i for i, x in enumerate(items) if isinstance(x, SBarrier))
    branchAt = next(i for i, x in enumerate(items) if isinstance(x, SCBranchSCC1))
    assert barrierAt < branchAt, \
        "the barrier must precede the branch that only some waves fall through"


def test_a_workgroup_uniform_branch_does_not_move_the_barrier():
    """A trip-count branch is taken by the whole workgroup, so a barrier inside it
    is already correct. Moving one out matters because the unroll loop is entered
    past exactly such a branch: hoisting there would lift a barrier out of the
    loop and change how many times it runs."""
    root, guarded, fillGroup = _guardedFillTree(_tripCountCompare)
    _runPass(_kernel(), root)

    assert len(_barriers(root)) == 1
    assert [x for x in fillGroup.items() if isinstance(x, SBarrier)], \
        "the barrier moved out of a branch that the whole workgroup takes together"


def test_the_pass_does_not_run_below_optlevel_3():
    """ScheduleIterAlg=0 keeps the barriers the writer placed, untouched."""
    root, _guarded, fillGroup = _guardedFillTree(_waveParityCompare)
    authored = SBarrier(comment="author barrier")
    fillGroup.add(authored)

    _runPass(_kernel(_StinkyTofuOptLevel=0), root)

    assert [x for x in fillGroup.items() if x is authored], \
        "the pass ran at OptLevel 0 and removed a barrier the writer placed"
    assert len(_barriers(root)) == 1


def test_a_wave_index_recomputed_into_a_temporary_still_guards():
    """The by-number half of the detector, which is the half a liveness rule can
    silently switch off: every recomputed index is refined in place right after it
    is read, so a rule that ended the tenure on any write would see no guards at
    all here while the by-symbol tests above kept passing."""
    root, _guarded, fillGroup = _guardedFillTree(_recomputedWaveParityCompare)
    _runPass(_kernel(), root)

    assert len(_barriers(root)) == 1, \
        "expected exactly one barrier for the one write-after-read, got %d" % len(_barriers(root))
    assert not [x for x in fillGroup.items() if isinstance(x, SBarrier)], \
        "the barrier is still inside a fill guarded by a wave index held in a temporary"
    leaves = _flatten(root)
    barrierAt = next(i for i, x in enumerate(leaves) if isinstance(x, SBarrier))
    branchAt = next(i for i, x in enumerate(leaves) if isinstance(x, SCBranchSCC1))
    assert barrierAt < branchAt, \
        "the barrier must precede the branch that only some waves fall through"


def test_a_temporary_reused_after_the_wave_index_is_not_a_wave_index():
    """A register that once held a wave index is not one forever. Reading
    membership of a set that only ever grew reported this uniform branch as a
    wave-parity guard and hoisted a barrier out of it."""
    root, _guarded, fillGroup = _guardedFillTree(_reusedTemporaryCompare)
    _runPass(_kernel(), root)

    assert len(_barriers(root)) == 1
    assert [x for x in fillGroup.items() if isinstance(x, SBarrier)], \
        "a barrier was hoisted out of a branch the whole workgroup takes, because " \
        "the register it compares had held a wave index earlier"


# Branch targets come from the label object: getLabelName() prefixes the name it
# was constructed with.


def _seq(*leaves):
    """One module holding the given leaves, so flattened order is written order."""
    root = Module("kernelBody")
    for leaf in leaves:
        root.add(leaf)
    return root


def _branchTo(label, comment):
    return SCBranchSCC1(labelName=label.getLabelName(), comment=comment)


def test_a_token_already_touched_in_the_divergent_region_is_rejected():
    """The block is read and then refilled inside one divergent region, so
    hoisting the barrier ahead of the branch would move it ahead of the read it
    has to separate."""
    guardEnd = Label("BarrierRejectGuardEnd", "")
    root = _seq(
        _fill(),                                        # block written outside
        _waveParityCompare(),
        _branchTo(guardEnd, "only one parity falls through"),
        _read(),                                        # hoists cleanly
        _fill(),                                        # token already touched inside
        guardEnd,
    )

    writer = _runPass(_kernel(), root)

    assert writer.states.overflowedResources == _REJECTED, (
        "the pass accepted a kernel whose barrier can only be placed where some "
        "waves branch over it")
    assert not _barriers(root), "declined, but a barrier was still inserted"


def test_a_guard_that_opens_outside_the_loop_is_rejected():
    """The guard opens outside the loop the access is in, so hoisting there
    turns a per-iteration barrier into a per-kernel one."""
    guardEnd = Label("BarrierRejectGuardEnd", "")
    loopBegin = Label("LoopBeginL", "")
    root = _seq(
        _fill(),
        _waveParityCompare(),
        _branchTo(guardEnd, "region opens before the loop"),
        loopBegin,                                      # loop boundary
        _read(),                                        # transition inside the loop
        _branchTo(loopBegin, "back-edge"),
        guardEnd,
    )

    # PrefetchGlobalRead=2 leaves the back-edge token model empty, isolating the
    # loop boundary from loop-head token state.
    writer = _runPass(_kernel(PrefetchGlobalRead=2), root)

    assert writer.states.overflowedResources == _REJECTED, (
        "a barrier was accepted whose only placement moves it out of the loop it "
        "has to run in")
    assert not _barriers(root)


def test_a_loop_prologue_barrier_inside_a_divergent_region_is_rejected():
    """A loop-prologue barrier is pinned to the loop label, so when the label
    itself sits inside a divergent region it cannot move."""
    guardEnd = Label("BarrierRejectGuardEnd", "")
    loopBegin = Label("LoopBeginL", "")
    root = _seq(
        _fill(),
        _waveParityCompare(),
        _branchTo(guardEnd, "region opens before the loop label"),
        loopBegin,                                      # prologue barrier belongs here
        _read(),
        _branchTo(loopBegin, "back-edge"),
        guardEnd,
    )

    # PrefetchGlobalRead=1 turns on the back-edge model that produces a
    # loop-prologue barrier.
    writer = _runPass(_kernel(PrefetchGlobalRead=1), root)

    assert writer.states.overflowedResources == _REJECTED
    assert not _barriers(root)


def test_the_successful_hoist_is_still_accepted():
    root, _guarded, _fillGroup = _guardedFillTree(_waveParityCompare)

    writer = _runPass(_kernel(), root)

    assert writer.states.overflowedResources == 0, \
        "the production shape was declined, so the refusal is over-conservative"
    assert len(_barriers(root)) == 1
