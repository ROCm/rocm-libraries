################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
################################################################################

# Store-pairs ahead a pair's terminal MFMAs are issued (the MFMA->accvgpr_read
# latency window). Single source of truth shared by the eligibility gate
# (computeSubtilePlsin) and the scheduler weave (LogicalScheduler): the gate admits
# a tile only when numStorePairs > this value, so at least one pair is left in the
# loop to hide the woven ones under. They MUST move together.
PLSIN_WEAVE_LOOKAHEAD = 2


def plsinLargeTile(kernel):
    """Tiles whose macro tile exceeds 256 in either dimension.

    Such tiles already peak at the architectural VGPR ceiling inside the loop, so
    the fused store keeps its terminal MFMAs in-loop (no weave) and lends the
    now-dead input-tile VGPRs to the store pool instead. Single source of truth
    for the LogicalScheduler weave/hoist decisions and the KernelWriterAssembly
    store-init hoist gate.
    """
    return (kernel["MacroTile0"] > 256) or (kernel["MacroTile1"] > 256)


def computeSubtilePlsin(kernel):
    """Derive the PostLoopStoreInNll (PLSIN) eligibility for a subtile kernel.

    PLSIN is an internal, subtile-owned decision: it is NOT a public solution
    parameter and is NOT part of the kernel name. It is a deterministic function
    of the already-derived solution/problem parameters, computed once at
    kernel-writer init time and carried on ``writer.states.postLoopStoreInNll``.

    The gate only ever AUTO-DISABLES; it never enables an ineligible config
    (sub-threshold tiles stay disabled, eligible tiles stay on).

    Returns:
        postLoopStoreInNll: bool
    """
    isa = tuple(kernel["ISA"])

    problemType = kernel["ProblemType"]
    mxBlockA = bool(problemType["MXBlockA"])
    mxBlockB = bool(problemType["MXBlockB"])
    isMxFloat4 = ((mxBlockA and problemType["DataTypeA"].isFloat4()) or
                  (mxBlockB and problemType["DataTypeB"].isFloat4()))
    isgfx950 = isa[:2] == (9, 5)
    destType = problemType["DestDataType"]
    depthU = kernel["DepthU"]
    depthUPow2 = depthU > 0 and (depthU & (depthU - 1)) == 0
    # The fused store is _emit16bitSubtilePairedStore: it only exists for a
    # bf16/half dest with HPA on wave64, and not for the StreamK workspace
    # (MultipleBuffer*) accumulation paths.
    pairedStoreAvailable = (
        (destType.isBFloat16() or destType.isHalf()) and
        problemType["HighPrecisionAccumulate"] and
        kernel["WavefrontSize"] != 32 and
        kernel["_GlobalAccumulation"] not in ("MultipleBufferSingleKernel", "MultipleBuffer")
    )
    # Barrier-free-store precondition: only StoreRemapVectorWidth>0 puts an
    # s_barrier *inside* the ds_bpermute paired store (LDS remap + barriers) and
    # uses an entirely different store mechanism, so it stays excluded.
    barrierFreeStore = (
        kernel["StoreRemapVectorWidth"] == 0
    )
    # Production support is intentionally limited to non-atomic SK3/4/5. Keeping
    # SK0 out of the gate avoids carrying an untested alternate control-flow path.
    streamKSupported = kernel["StreamK"] in (3, 4, 5) and not kernel["StreamKAtomic"]
    # Spill tiles: MIWaveTile product > 64 spills accumulators into arch VGPRs
    # and overflows the occ-1 budget under the fused store. MIWaveTile is only
    # present for EnableMatrixInstruction solutions; guard the lookup.
    miwt = kernel.get("MIWaveTile")
    spillFree = bool(miwt) and len(miwt) == 2 and (miwt[0] * miwt[1] <= 64)
    # Store-footprint fit: large asymmetric tiles (min>=4 and max>=14) overflow
    # the arch-VGPR budget and emit out-of-range v>=256.
    storeFitsVgpr = not (bool(miwt) and len(miwt) == 2 and
                         min(miwt[0], miwt[1]) >= 4 and max(miwt[0], miwt[1]) >= 14)
    # Overlap feasibility: the weave only weaves store-pairs with pair index >=
    # weaveLA. numStorePairs = MIWT0*MIWT1//2; at or below the threshold no pair
    # is woven, so PLSIN would be pure overhead. weaveLA is the shared constant the
    # scheduler weave (Components/Subtile/LogicalScheduler.py) reads too, so the gate
    # and the weave move together.
    weaveLA = PLSIN_WEAVE_LOOKAHEAD
    numStorePairs = (miwt[0] * miwt[1] // 2) if (bool(miwt) and len(miwt) == 2) else 0
    overlapPossible = numStorePairs > weaveLA
    # Two-wide MIWaveTile dimensions leave insufficient SGPR headroom for the
    # MX-scale descriptors once the fused-store state is live (for example,
    # [2,4] needs 103 SGPRs against gfx950's 102-SGPR cap).
    mxBlockScaled = mxBlockA or mxBlockB
    mxBlockScaleSgprFits = not (mxBlockScaled and bool(miwt) and len(miwt) == 2 and
                                min(miwt[0], miwt[1]) <= 2)
    # These epilogues need transient SRDs/state that the in-NLL store does not yet
    # reproduce completely. Fail safe to the normal post-loop store until each is
    # implemented and hardware-verified on SK3/4/5.
    epilogueSupported = (not problemType.get("UseE", False)
                         and not problemType.get("UseGateResidual", False))
    # Hard structural: the fused store literally cannot be emitted. ALWAYS enforced.
    structuralFail = ((not isMxFloat4)
                      or (not kernel["UseSubtileImpl"])
                      or (not isgfx950)
                      or (not kernel["EnableMatrixInstruction"])
                      or (kernel["PrefetchGlobalRead"] < 1)
                      or (not kernel["BufferStore"])
                      or (not pairedStoreAvailable)
                      or (not streamKSupported)
                      or (not problemType["ComputeDataType"].isSingle())
                      or (not depthUPow2)
                      or kernel.get("TailloopInNll", False)
                      or (not epilogueSupported)
                      or (not barrierFreeStore))
    # Register / spill budget: the fused store would overflow the arch-VGPR /
    # SGPR ceiling for this tile. ALWAYS enforced.
    registerFail = ((not spillFree)
                    or (not storeFitsVgpr)
                    or (not mxBlockScaleSgprFits))
    # Pure profitability (weave-overlap threshold): correct and register-safe, just
    # below the pair-count where the weave overlaps anything. The structural NLL
    # requirement (PGR >= 1) is already enforced by structuralFail above.
    profitFail = not overlapPossible

    return not (structuralFail or registerFail or profitFail)
