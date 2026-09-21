# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for cluster-barrier signal/wait pairing.

The cluster-scope barrier handshake is split into a ``signal`` half and a
``wait`` half so the cross-CU latency can hide behind intervening WMMAs. The
wait must nonetheless dominate every early-exit branch (``SkipToNLL``,
``SkipToNGLL``, ``SkipTailLoopL`` etc.) that follows the signal; otherwise a
taken branch leaves the barrier signalled but never waited, desyncing the
cluster handshake (FFM assertion / incorrect on hardware).

These tests exercise ``insertClusterBarrier`` directly with a mock writer so
they stay fast and hardware-free.
"""

from unittest.mock import MagicMock

from rocisa.code import Label, Module
from rocisa.container import DSModifiers, MemTokenData, sgpr, vgpr
from rocisa.instruction import DSLoadB64, Instruction, SAndB32, SBarrier, SBitcmp1B32, SCBranchSCC1, SCmpEQU32, SCmpLeU32, SLShiftRightB32, TensorLoadToLds, VReadfirstlaneB32

from Tensile.Components.DecouplePGR import dcpThickGateFromTokenPasses
from Tensile.Components.Subtile.ClusterBarrier import insertClusterBarrier
from Tensile.KernelWriter import KernelWriter


def _mockWriter():
    writer = MagicMock()
    # ClusterBarrier requires the HasClusterBarrier asm capability.
    writer.states.asmCaps.get.return_value = True
    writer.labels.getUniqueNamePrefix.side_effect = lambda prefix: prefix + "_TEST"
    return writer


def _kernel():
    return {"ClusterBarrier": True}


def _isSignal(inst):
    return isinstance(inst, SBarrier) and "cluster_barrier signal" in str(inst)


def _isClusterWait(inst):
    return (isinstance(inst, SBarrier)
            and "cluster_barrier wait" in str(inst)
            and "workgroup" not in str(inst))


def _indices(items):
    sig = next(i for i, x in enumerate(items) if _isSignal(x))
    wait = next(i for i, x in enumerate(items) if _isClusterWait(x))
    return sig, wait


def test_wait_dominates_early_exit_branch():
    """The cluster wait must sit after the signal but before the first exit."""
    module = Module("section")
    module.add(SCmpEQU32(sgpr("LoopCounterL"), 1, "LoopCounter LE 1?"))
    module.add(SCBranchSCC1("label_SkipToNLL", "skip to NLL"))
    module.add(SCmpEQU32(sgpr("LoopCounterL"), 2, "LoopCounter LE 2?"))
    module.add(SCBranchSCC1("label_SkipToNGLL", "skip to NGLL"))

    items = insertClusterBarrier(module, _mockWriter(), _kernel()).flatitems()
    sig, wait = _indices(items)
    firstExit = next(i for i, x in enumerate(items) if isinstance(x, SCBranchSCC1))

    assert sig < wait < firstExit, (
        "cluster wait must follow the signal and precede the first early-exit "
        "branch so no taken branch can skip it"
    )


def test_wait_never_placed_after_exit_branch():
    """Anti-regression: no early-exit branch may precede the cluster wait."""
    module = Module("section")
    module.add(SCmpEQU32(sgpr("LoopCounterL"), 32, "LoopCounter LE 32?"))
    module.add(SCBranchSCC1("label_SkipTailLoopL", "early-exit tail"))

    items = insertClusterBarrier(module, _mockWriter(), _kernel()).flatitems()
    _, wait = _indices(items)

    exitsBeforeWait = [i for i, x in enumerate(items)
                       if isinstance(x, SCBranchSCC1) and i < wait]
    assert not exitsBeforeWait, (
        "an early-exit branch precedes the cluster wait; the barrier can be "
        "signalled but never waited on that path"
    )


# Wave-divergent barrier hazard: no barrier only under a wave-parity guard.


LDS_TOKEN = 0


class _Writer:
    """Pass reads fallback SIA, kernel name, and per-instance overflowedResources."""

    class _States:
        def __init__(self):
            self.scheduleIterAlg = 0
            self.kernelName = "unit_test_kernel"
            self.overflowedResources = 0
            # No fixture builds the three-buffer TDM path; the real method returns {}.
            self.kernel = {}

    class _DebugConfig:
        printSolutionRejectionReason = False

    def __init__(self):
        self.states = _Writer._States()
        self.debugConfig = _Writer._DebugConfig()

    def _ldsTokenBackEdgeMap(self):
        """Delegate to KernelWriter._ldsTokenBackEdgeMap."""
        return KernelWriter._ldsTokenBackEdgeMap(self)


def _waveParityKernel(**overrides):
    # OptLevel 3 + derived SIA 0 + TDM on both tensors + more than one wave.
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


# Divergent pair with separate TDM descriptors => DCP_THICK_GATE_TOKENS.
_TOKEN_GATE_KEYS = {
    "PrefetchGlobalReadA": 2,
    "PrefetchGlobalReadB": 1,
    "TDMFuse": 1,
    "TDMInst": 3,
    "TDMSplit": False,
    "UseSubtileImpl": False,
    "NumWaves": 4,
    "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
}


def _runPass(kernel, root):
    writer = _Writer()
    KernelWriter.postMainLoopBarrierCheckAndReset(writer, kernel, root)
    return writer


# overflowedResources code: no workgroup-wide position for a rebuilt LDS barrier.
_REJECTED = 10


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
    """Writer shape: the guard in one module, the conflicting fill in another."""
    root = Module("kernelBody")
    root.add(_read())

    end = Label("DcpEarlyFillAEnd", "")
    guarded = Module("TDM decoupled early fill B")
    guarded.add(guardOpener())
    guarded.add(SCBranchSCC1(labelName=end.getLabelName(),
                             comment="B is single-buffered, its fill moves late"))
    # Separate module: the fill is not in the same module as the branch.
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


# Allocator temp for the recomputed index: a number, not a symbol.
_WAVE_TEMP = 5


def _recomputedWaveParityCompare():
    """Same guard where sgprWaveIdx is no longer live; the compare is a number, not a symbol."""
    module = Module("recomputed wave index")
    module.add(VReadfirstlaneB32(dst=sgpr(_WAVE_TEMP), src=vgpr("Serial"),
                                 comment="get tId"))
    module.add(SLShiftRightB32(dst=sgpr(_WAVE_TEMP), shiftHex=5,
                               src=sgpr(_WAVE_TEMP), comment="waveId"))
    module.add(SBitcmp1B32(src0=sgpr(_WAVE_TEMP), src1=0, comment="check wave parity"))
    return module


def _reusedTemporaryCompare():
    """The temporary is reused for something else; the branch is workgroup-uniform."""
    module = Module("reused temporary")
    module.add(VReadfirstlaneB32(dst=sgpr(_WAVE_TEMP), src=vgpr("Serial"),
                                 comment="get tId"))
    module.add(SLShiftRightB32(dst=sgpr(_WAVE_TEMP), shiftHex=5,
                               src=sgpr(_WAVE_TEMP), comment="waveId"))
    module.add(SAndB32(dst=sgpr(_WAVE_TEMP), src0=sgpr("GSU"), src1=0x3fff,
                       comment="the number is reused for something else"))
    module.add(SCmpEQU32(src0=sgpr(_WAVE_TEMP), src1=1, comment="GSU == 1"))
    return module


def _packedArgTypeWaveParityCompare():
    """Same guard where sgprWaveIdx has been released; parity rides ArgType bit 8."""
    return SBitcmp1B32(src0=sgpr("ArgType"), src1=8, comment="check wave parity")


def _maskedArgTypeDomainCompare():
    """cmpNamedArgTypeEq shape: the named domain is the low byte, masked into a temp."""
    module = Module("masked ArgType domain")
    module.add(SAndB32(dst=sgpr(_WAVE_TEMP), src0=sgpr("ArgType"), src1=hex(0xFF),
                       comment="mask ArgType domain (bits 8+ = TDM wave id)"))
    module.add(SCmpEQU32(src0=sgpr(_WAVE_TEMP), src1=2, comment="ArgType == 2 ?"))
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
    """Barriers reachable only when a wave-parity branch falls through."""
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
            text = str(leaf)
            sccIsWaveParity = "sgprWaveIdx" in text or "sgprArgType" in text
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
    _runPass(_waveParityKernel(), root)

    offending = _waveParityGuardedBarriers(root)
    assert not offending, (
        "%d barrier(s) sit between a wave-parity s_cbranch and its target, so only "
        "the waves that fall through execute them: %s" % (len(offending), offending))


def test_the_guarded_fill_is_still_synchronised():
    """Must not be met by emitting nothing."""
    root, guarded, fillGroup = _guardedFillTree(_waveParityCompare)
    _runPass(_waveParityKernel(), root)

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
    """A trip-count branch is workgroup-uniform; hoisting would leave the loop."""
    root, guarded, fillGroup = _guardedFillTree(_tripCountCompare)
    _runPass(_waveParityKernel(), root)

    assert len(_barriers(root)) == 1
    assert [x for x in fillGroup.items() if isinstance(x, SBarrier)], \
        "the barrier moved out of a branch that the whole workgroup takes together"


def test_the_pass_does_not_run_below_optlevel_3():
    root, _guarded, fillGroup = _guardedFillTree(_waveParityCompare)
    authored = SBarrier(comment="author barrier")
    fillGroup.add(authored)

    _runPass(_waveParityKernel(_StinkyTofuOptLevel=0), root)

    assert [x for x in fillGroup.items() if x is authored], \
        "the pass ran at OptLevel 0 and removed a barrier the writer placed"
    assert len(_barriers(root)) == 1


def test_the_pass_runs_at_optlevel_0_for_a_decoupled_token_gate():
    """A TOKENS thick gate rebuilds barriers at any opt level."""
    kernel = _waveParityKernel(_StinkyTofuOptLevel=0, **_TOKEN_GATE_KEYS)
    assert dcpThickGateFromTokenPasses(kernel), \
        "expected token-based thick gate"

    root, guarded, fillGroup = _guardedFillTree(_waveParityCompare)
    authored = SBarrier(comment="author barrier")
    fillGroup.add(authored)

    _runPass(kernel, root)

    assert not [x for x in fillGroup.items() if x is authored], \
        "the writer's barrier survived, so the token-driven rebuild did not run"
    assert len(_barriers(root)) == 1
    items = guarded.items()
    barrierAt = next(i for i, x in enumerate(items) if isinstance(x, SBarrier))
    branchAt = next(i for i, x in enumerate(items) if isinstance(x, SCBranchSCC1))
    assert barrierAt < branchAt, \
        "the rebuilt barrier must precede the branch only some waves fall through"


def test_a_wave_index_recomputed_into_a_temporary_still_guards():
    """By-number half of the detector; tenure must survive in-place refine."""
    root, _guarded, fillGroup = _guardedFillTree(_recomputedWaveParityCompare)
    _runPass(_waveParityKernel(), root)

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
    """A register that once held a wave index is not one forever."""
    root, _guarded, fillGroup = _guardedFillTree(_reusedTemporaryCompare)
    _runPass(_waveParityKernel(), root)

    assert len(_barriers(root)) == 1
    assert [x for x in fillGroup.items() if isinstance(x, SBarrier)], \
        "a barrier was hoisted out of a branch the whole workgroup takes, because " \
        "the register it compares had held a wave index earlier"


def test_a_wave_parity_packed_into_argtype_still_guards():
    """Packed half of the detector; WaveIdx is released before the guard runs."""
    root, _guarded, fillGroup = _guardedFillTree(_packedArgTypeWaveParityCompare)
    _runPass(_waveParityKernel(), root)

    assert not _waveParityGuardedBarriers(root), \
        "a barrier sits between a packed-parity s_cbranch and its target, so only the " \
        "waves that fall through execute it"
    assert len(_barriers(root)) == 1, \
        "expected exactly one barrier for the one write-after-read, got %d" % len(_barriers(root))
    assert not [x for x in fillGroup.items() if isinstance(x, SBarrier)], \
        "the barrier is still inside a fill guarded by the parity packed in ArgType bit 8"
    leaves = _flatten(root)
    barrierAt = next(i for i, x in enumerate(leaves) if isinstance(x, SBarrier))
    branchAt = next(i for i, x in enumerate(leaves) if isinstance(x, SCBranchSCC1))
    assert barrierAt < branchAt, \
        "the barrier must precede the branch that only some waves fall through"


def test_a_masked_argtype_domain_test_is_not_a_wave_index():
    """The named domain is the low byte, which the whole workgroup reads the same."""
    root, _guarded, fillGroup = _guardedFillTree(_maskedArgTypeDomainCompare)
    _runPass(_waveParityKernel(), root)

    assert len(_barriers(root)) == 1
    assert [x for x in fillGroup.items() if isinstance(x, SBarrier)], \
        "a barrier was hoisted out of a branch the whole workgroup takes, because the " \
        "domain test it compares names ArgType in its comment"


def _seq(*leaves):
    """One module holding the given leaves, so flattened order is written order."""
    root = Module("kernelBody")
    for leaf in leaves:
        root.add(leaf)
    return root


def _branchTo(label, comment):
    return SCBranchSCC1(labelName=label.getLabelName(), comment=comment)


def test_a_token_already_touched_in_the_divergent_region_is_rejected():
    """Read then refill inside one divergent region needs a workgroup-wide barrier."""
    guardEnd = Label("BarrierRejectGuardEnd", "")
    root = _seq(
        _fill(),                                        # block written outside
        _waveParityCompare(),
        _branchTo(guardEnd, "only one parity falls through"),
        _read(),                                        # hoists cleanly
        _fill(),                                        # token already touched inside
        guardEnd,
    )

    writer = _runPass(_waveParityKernel(), root)

    assert writer.states.overflowedResources == _REJECTED, (
        "the pass accepted a kernel whose barrier can only be placed where some "
        "waves branch over it")
    assert not _barriers(root), "declined, but a barrier was still inserted"


def test_a_guard_that_opens_outside_the_loop_is_rejected():
    """Hoisting outside the loop turns a per-iteration barrier into a per-kernel one."""
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
    writer = _runPass(_waveParityKernel(PrefetchGlobalRead=2), root)

    assert writer.states.overflowedResources == _REJECTED, (
        "a barrier was accepted whose only placement moves it out of the loop it "
        "has to run in")
    assert not _barriers(root)


def test_a_loop_prologue_barrier_inside_a_divergent_region_is_rejected():
    """A loop-prologue barrier is pinned to the loop label."""
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
    writer = _runPass(_waveParityKernel(PrefetchGlobalRead=1), root)

    assert writer.states.overflowedResources == _REJECTED
    assert not _barriers(root)


def test_the_successful_hoist_is_still_accepted():
    root, _guarded, _fillGroup = _guardedFillTree(_waveParityCompare)

    writer = _runPass(_waveParityKernel(), root)

    assert writer.states.overflowedResources == 0, \
        "expected overflowedResources=0"
    assert len(_barriers(root)) == 1
