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

from ...Common import plsinDebugEnv


def computeSubtilePlsin(kernel):
    """Derive the PostLoopStoreInNll (PLSIN) eligibility and store mode for a
    subtile kernel.

    PLSIN is an internal, subtile-owned decision: it is NOT a public solution
    parameter and is NOT part of the kernel name. It is a deterministic function
    of the already-derived solution/problem parameters, computed once at
    kernel-writer init time and carried on ``writer.states`` (postLoopStoreInNll
    and plsinStoreMode).

    The gate only ever AUTO-DISABLES; it never enables an ineligible config.
    With the debug env unset this reproduces the shipped default decision exactly
    (sub-threshold tiles stay disabled, eligible tiles stay on).

    Returns:
        (postLoopStoreInNll: bool, plsinStoreMode: str)
    """
    isa = tuple(kernel["ISA"])
    plsin = True
    # PLSINStoreMode was a solution-selection choice (Weave/Lend); the shipped
    # default is always "Weave" (no config ever set "Lend" -- Lend is selected at
    # schedule time via largeTile / the TENSILE_PLSIN_SMALLTILE_LEND override).
    # Canonicalize to "Weave" here so the decision stays codegen-neutral.
    storeMode = "Weave"

    isFloat4 = kernel["ProblemType"]["DataTypeA"].isFloat4() or \
               kernel["ProblemType"]["DataTypeB"].isFloat4()
    isgfx950 = isa[:2] == (9, 5)
    destType = kernel["ProblemType"]["DestDataType"]
    # TEST-ONLY force: turn PLSIN on for fp4 kernels with an f16 (half) C/D dest
    # even when only the weave-overlap PROFITABILITY heuristic would auto-disable
    # it. The structural and register/spill gates below stay enforced, so this can
    # only ever force NON-SPILL macrotiles on (never past the VGPR/SGPR ceiling).
    # Unset (default "0") reproduces the shipped library byte-for-byte.
    forcePlsinFp4F16 = (isFloat4
                        and destType.isHalf()
                        and plsinDebugEnv("TENSILE_PLSIN_FORCE_FP4_F16", "0") != "0")
    # The fused store is _emit16bitSubtilePairedStore: it only exists for a
    # bf16/half dest with HPA on wave64, and not for the StreamK workspace
    # (MultipleBuffer*) accumulation paths.
    pairedStoreAvailable = (
        (destType.isBFloat16() or destType.isHalf()) and
        kernel["ProblemType"]["HighPrecisionAccumulate"] and
        kernel["WavefrontSize"] != 32 and
        kernel["_GlobalAccumulation"] not in ("MultipleBufferSingleKernel", "MultipleBuffer")
    )
    # Barrier-free-store precondition: only StoreRemapVectorWidth>0 puts an
    # s_barrier *inside* the ds_bpermute paired store (LDS remap + barriers) and
    # uses an entirely different store mechanism, so it stays excluded.
    barrierFreeStore = (
        kernel["StoreRemapVectorWidth"] == 0
    )
    # StreamK support: only the non-atomic reduction (SK3/4/5) is eligible.
    streamKAtomicFree = not (kernel["StreamK"] and kernel["StreamKAtomic"])
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
    # is woven, so PLSIN would be pure overhead. This reads the same
    # TENSILE_PLSIN_DEBUG="TENSILE_WEAVE_LA=..." override and the same production
    # default as the scheduler (Components/Subtile/LogicalScheduler.py), so the
    # gate and the weave move together.
    weaveLA = int(plsinDebugEnv("TENSILE_WEAVE_LA", "2"))
    numStorePairs = (miwt[0] * miwt[1] // 2) if (bool(miwt) and len(miwt) == 2) else 0
    overlapPossible = numStorePairs > weaveLA
    # HOIST-ONLY opt-in: keep PLSIN ON for sub-threshold tiles (numStorePairs <=
    # weaveLA) that pass every structural/register gate, instead of auto-disabling
    # them. Requires at least one real store pair. Default off, so unset
    # reproduces the shipped library byte-for-byte.
    admitHoistOnly = (plsinDebugEnv("TENSILE_PLSIN_HOIST_ONLY", "0") != "0"
                      and numStorePairs >= 1)
    streamKFixupSafe = True
    # MX-block-scaled fp4 extreme skews ([2,16]/[16,2]) overflow the 102-SGPR
    # gfx9 ceiling; auto-disable just those.
    mxBlockScaled = bool(kernel["ProblemType"]["MXBlockA"] or kernel["ProblemType"]["MXBlockB"])
    mxBlockScaleSgprFits = not (mxBlockScaled and bool(miwt) and len(miwt) == 2 and
                                min(miwt[0], miwt[1]) <= 2 and max(miwt[0], miwt[1]) >= 16)
    # Hard structural: the fused store literally cannot be emitted. ALWAYS enforced.
    structuralFail = ((not isFloat4)
                      or (not kernel["UseSubtileImpl"])
                      or (not isgfx950)
                      or (not kernel["EnableMatrixInstruction"])
                      or (kernel["PrefetchGlobalRead"] < 1)
                      or (not kernel["BufferStore"])
                      or (not pairedStoreAvailable)
                      or (not streamKAtomicFree)
                      or (not barrierFreeStore))
    # Register / spill budget: the fused store would overflow the arch-VGPR /
    # 102-SGPR ceiling for this tile. ALWAYS enforced, so the force only ever
    # turns on non-spill macrotiles (never pushes a tile past the ceiling).
    registerFail = ((not spillFree)
                    or (not storeFitsVgpr)
                    or (not mxBlockScaleSgprFits))
    # Pure profitability (weave-overlap threshold): correct and register-safe, just
    # below the pair-count where the weave overlaps anything. The ONLY group the
    # TENSILE_PLSIN_FORCE_FP4_F16 force bypasses.
    profitFail = (((not overlapPossible) and not admitHoistOnly)
                  or (not streamKFixupSafe))
    if structuralFail or registerFail or (profitFail and not forcePlsinFp4F16):
        plsin = False

    # PLSIN fuses the D store into the No-Load Loop, so it needs a structural NLL
    # (PGR >= 1). Defensive re-check.
    if plsin and kernel["PrefetchGlobalRead"] < 1:
        plsin = False

    # PLSINStoreMode only matters when PLSIN is on; canonicalize when disabled.
    if not plsin:
        storeMode = "Weave"

    return plsin, storeMode
