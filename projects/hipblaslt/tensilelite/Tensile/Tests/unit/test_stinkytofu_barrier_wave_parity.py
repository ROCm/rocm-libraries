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
"""No barrier may be reachable only under a wave-parity guard.

At ScheduleIterAlg=4 (StinkyTofu OptLevel 3) postMainLoopBarrierCheckAndReset
deletes every barrier in the kernel and rebuilds placement from memory tokens.
The guard that decides which wave fills which tensor -- s_bitcmp1_b32 on bit 0 of
the wave index, then an s_cbranch -- is emitted in a DIFFERENT module from the
tensor_load_to_lds it guards, by
KernelWriter._dcpScheduleSingleBufferedFillLate and
KernelWriterAssembly._emitTdmDealiasedIssue. A rebuild that recursed module by
module could not see that branch, so it put the barrier after it.

The invariant is asserted, not the count. The broken pass emitted exactly the
right NUMBER of barriers -- three of them in a divergent decoupled-PGR kernel
were simply executed by half the workgroup -- so a test on the count would have
passed while gfx1250 silicon returned wrong results. That silicon failure is
intermittent rather than K-gated: it was observed at K=1024 and K=1152 at DepthU
256 (b8-3:~/_pgrval19/val.log), and the same defect class failed and then passed
at the SAME K on one unchanged tree and code object. No K is a safe region and no
single green run settles anything, which is exactly why this test pins the
invariant instead of a pass count.

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
    """All the pass reads off the writer is the fallback ScheduleIterAlg, used for a
    kernel that does not carry the derived one. So it can be run against a
    hand-built module tree without standing up a whole KernelWriter."""

    class states:
        scheduleIterAlg = 0


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
    KernelWriter.postMainLoopBarrierCheckAndReset(_Writer(), kernel, root)


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
