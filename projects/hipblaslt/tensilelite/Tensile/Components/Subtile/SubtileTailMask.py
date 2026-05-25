# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-lane K-tail masking for the subtile path.

Builds and applies the lane-level cndmasks that zero A/B (and
optionally MXSA/MXSB) VGPRs whose K-position falls past
``LoopCounterL = K mod DU`` on the last m-row of the tail loop.

Public surface (all helpers take the live ``KernelWriter`` as the
first arg ``kw``):

- :func:`subtileTailByteShiftApplies` - static predicate gating the
  sub-lane byte-refine path on / off.
- :func:`emitTailKPosCmpSubtile` - coarse per-mmak ``v_cmp_ge_i32``
  used by the legacy / scale-aware path.
- :func:`emitTailSubLaneMaskRefineSubtile` - sub-lane byte-refine
  chain that v_ands directly back onto the A/B tile VGPRs (used by
  the ``SubtileTailMaskPrecompute=False`` reversibility path).
- :func:`emitTailSubLaneMaskInitFused`,
  :func:`emitTailSubLaneMaskChainIntoVgprFused`,
  :func:`emitTailSubLaneMaskChainIntoVgpr`,
  :func:`emitTailSubLaneMaskPrecomputeSubtile`,
  :func:`emitTailSubLaneMaskApplySubtile` - precompute pipeline:
  build per-(operand, mmak, ir) masks into long-lived VGPRs ONCE
  before the per-mmak MFMA loop, then per-mmak just v_ands them
  onto each boundary VGPR.

The helpers reach into ``kw.states.*`` for laneSGPRCount,
``kw.vgprPool`` for per-mask scratch, ``kw.allocTmpSgpr`` for the
SGPR-pair holding the VOPC result, and ``kw._subtileCmpSrc1FitsInline``
to gate the literal-staging path for the fused chain's VOPC literals.
"""

from rocisa.code import Module
from rocisa.container import sgpr, vgpr
from rocisa.instruction import (
    SMovB32,
    VAddU32, VAndB32, VCmpGEI32, VCmpGTI32, VCmpLeI32, VCmpLtI32,
    VCndMaskB32, VMovB32, VSubI32,
)


def emitTailKPosCmpSubtile(kw, kPosBaseVgpr, mmak, miK, maskSgpr):
    """Per-mmak kPosCur add + v_cmp_ge_i32 into maskSgpr (held by caller)."""
    module = Module("tailKPosCmpSubtile mmak=%u" % mmak)
    laneSGPRCount = kw.states.laneSGPRCount
    kPosCur = kw.vgprPool.checkOut(1, "kRegCur")
    module.add(VAddU32(dst=vgpr(kPosCur), src0=mmak * miK, src1=vgpr(kPosBaseVgpr),
                       comment="kPosCur = kPosBase + mmak * miK"))
    module.add(VCmpGEI32(dst=sgpr(maskSgpr, laneSGPRCount),
                         src0=vgpr(kPosCur), src1=sgpr("LoopCounterL"),
                         comment="check K_idx >= K-tail size"))
    kw.vgprPool.checkIn(kPosCur)
    return module


def subtileTailByteShiftApplies(kernel, numMIInUnroll):
    """Gate for the sub-lane K-tail mask refinement.

    Returns True when the helper should run: ASEM < numMIInUnroll
    (otherwise the coarse per-lane cndmask already covers every
    K-element past LoopCounterL), the MX scale path is inactive
    (MX uses its own padded-scale handling), and each operand has
    an integer-yielding `bpe` of at least 1 byte and at most one
    register width (so `elementsPerVgpr = max(1, bpr // bpe) >= 1`
    is well-defined). The helper itself is bpe-parametric so this
    gate can be relaxed in a follow-up to enable fp8/MX-data tails;
    the integer-bpe constraint excludes mxfp4 (numBytes=0.5) for
    now since the byte-mask construction assumes byte-aligned mod
    boundaries.
    """
    asem = kernel["AssertSummationElementMultiple"]
    if asem >= numMIInUnroll:
        return False
    if kernel["ProblemType"].get("MXBlockA", 0) > 0:
        return False
    if kernel["ProblemType"].get("MXBlockB", 0) > 0:
        return False
    bpr = 4
    for tc in ("DataTypeA", "DataTypeB"):
        bpeRaw = kernel["ProblemType"][tc].numBytes()
        if int(bpeRaw) != bpeRaw:
            return False
        bpe = int(bpeRaw)
        if bpe < 1 or bpe > bpr:
            return False
    return True


def emitTailSubLaneMaskRefineSubtile(kw, kernel, kPosBaseVgpr, mmak, miK,
                                     numMIInUnroll, aIndicesByIr, bIndicesByIr):
    """Sub-lane K-tail mask refinement: zero past-LoopCounterL bytes
    within each per-VGPR K-window for the boundary lane group that
    the coarse per-lane cndmask cannot reach.

    For each operand and each ir slot, builds a single per-lane
    32-bit `maskVgpr` by chaining mod=elementsPerVgpr-1 down to
    mod=0 mask-byte selects, then `v_and`s the accumulated mask
    against every boundary VGPR at that ir. The mod=k mask byte
    is `(1 << (k * bpe * 8)) - 1` (keeps the lo `k` elements within
    the VGPR when the K-position at byte offset `k * bpe` is past
    LoopCounterL):

      bf16 (bpe=2, elementsPerVgpr=2): {mod=1: 0xFFFF, mod=0: 0}
      fp8  (bpe=1, elementsPerVgpr=4): {mod=3: 0x00FFFFFF,
                                        mod=2: 0x0000FFFF,
                                        mod=1: 0x000000FF,
                                        mod=0: 0}

    Static skip: when `ASEM*bpe % bpr == 0` (K_remain in bytes is a
    multiple of a register), only the mod=0 step is reachable and
    the mod>0 chain collapses. Otherwise the full mod chain is
    emitted unconditionally -- the mod>0 chain is short (4 instr
    per step for bf16, 12 total for fp8) and a 3-instr scalar
    runtime gate to skip it is not worth the branch overhead in
    the precompute path (this chain runs ONCE per (operand, mmak,
    ir) before the per-mmak MFMA loop, not in the hot path). The
    mod=0 step is always emitted (it's the "this VGPR is entirely
    past LoopCounterL" case the coarse cndmask used to handle and
    that #5 lets the byte refine own).

    A and B operand chains are emitted independently per #3 so a
    mixed-bpe problem (e.g. asymmetric fp8/bf16 in a future PR) gets
    correct per-operand mod chains; in the same-bpe (bf16/bf16)
    common case the cmps do duplicate per ir but the helper stays
    bpe-agnostic. Gated by `subtileTailByteShiftApplies`.
    """
    assert kernel["ProblemType"].get("MXBlockA", 0) == 0, (
        "sub-lane K-tail mask refinement does not handle the MX scale path.")
    assert kernel["ProblemType"].get("MXBlockB", 0) == 0, (
        "sub-lane K-tail mask refinement does not handle the MX scale path.")

    module = Module("tailSubLaneMaskRefineSubtile mmak=%u" % mmak)
    laneSGPRCount = kw.states.laneSGPRCount

    # bpr = bytes per register. `DataType.numBytes()` returns float
    # for sub-32b dtypes (bf16/fp16: 2.0, fp8: 1.0, mxfp4: 0.5) so
    # round-trip through int and assert integrality before deriving
    # mask widths -- a non-integer bpe would propagate floats into
    # rocisa arithmetic below, and the predicate gate already
    # rejects mxfp4 / future sub-byte dtypes upstream.
    bpr = 4
    bpeARaw = kernel["ProblemType"]["DataTypeA"].numBytes()
    bpeBRaw = kernel["ProblemType"]["DataTypeB"].numBytes()
    bpeA = int(bpeARaw)
    bpeB = int(bpeBRaw)
    assert bpeA == bpeARaw and 1 <= bpeA <= bpr, (
        "sub-lane refine: DataTypeA bpe must be integer in [1, bpr]; "
        "got %r" % (bpeARaw,))
    assert bpeB == bpeBRaw and 1 <= bpeB <= bpr, (
        "sub-lane refine: DataTypeB bpe must be integer in [1, bpr]; "
        "got %r" % (bpeBRaw,))
    elementsPerVgprA = max(1, bpr // bpeA)
    elementsPerVgprB = max(1, bpr // bpeB)
    asem = kernel["AssertSummationElementMultiple"]

    irKeys = set(aIndicesByIr.keys()) | set(bIndicesByIr.keys())
    if not irKeys:
        return module
    vgprPerInUnroll = max(irKeys) + 1

    kPosCur = kw.vgprPool.checkOut(1, "kPosCurByteRefine")
    maskVgpr = kw.vgprPool.checkOut(1, "subLaneByteMask")
    seedVgpr = kw.vgprPool.checkOut(1, "subLaneByteSeed")

    def _emitChain(operand, idxs, ir, bpe, elementsPerVgpr):
        """Emit one (ir, operand) mask chain and v_and the accumulated
        mask into each boundary VGPR.
        """
        staticSkipPartial = (asem * bpe) % bpr == 0
        module.add(VMovB32(
            dst=vgpr(maskVgpr), src=hex(0xFFFFFFFF),
            comment="byteRefine[%s ir=%d mmak=%d]: mask seed = full keep"
                    % (operand, ir, mmak)))
        if not staticSkipPartial:
            with kw.allocTmpSgpr(laneSGPRCount,
                                 alignment=laneSGPRCount) as maskInfo:
                maskSgpr = maskInfo.idx
                # mod = elementsPerVgpr-1 down to 1: each step folds the
                # mask down to `(1 << (mod*bpe*8)) - 1` on past-boundary
                # lanes, leaving in-range lanes unchanged. The mod=0 step
                # below is statically reachable so it lives outside this
                # block.
                for mod in range(elementsPerVgpr - 1, 0, -1):
                    maskByte = (1 << (mod * bpe * 8)) - 1
                    # The past-boundary mask byte lives in src1 of cndmask,
                    # which on gfx950 cannot hold a 32-bit literal -- stage
                    # it through a VGPR seed.
                    module.add(VMovB32(
                        dst=vgpr(seedVgpr), src=hex(maskByte),
                        comment="byteRefine[%s ir=%d mod=%d]: keep mask = 0x%X"
                                % (operand, ir, mod, maskByte)))
                    kElemOffset = mmak * miK + ir * elementsPerVgpr + mod
                    module.add(VAddU32(
                        dst=vgpr(kPosCur), src0=kElemOffset, src1=vgpr(kPosBaseVgpr),
                        comment="byteRefine[%s ir=%d mod=%d]: K_pos = kPosBase + %d"
                                % (operand, ir, mod, kElemOffset)))
                    module.add(VCmpGEI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(kPosCur), src1=sgpr("LoopCounterL"),
                        comment="byteRefine[%s ir=%d mod=%d]: K_pos >= LoopCounterL ?"
                                % (operand, ir, mod)))
                    module.add(VCndMaskB32(
                        dst=vgpr(maskVgpr),
                        src0=vgpr(maskVgpr), src1=vgpr(seedVgpr),
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment="byteRefine[%s ir=%d mod=%d]: mask = past ? 0x%X : prev"
                                % (operand, ir, mod, maskByte)))

        # mod=0: lanes whose K-position is at or past LoopCounterL get
        # mask = 0 (entire VGPR zeroed by the v_and below). Always
        # emitted -- subsumes the coarse per-VGPR cndmask for the
        # operand/ir VGPRs (#5).
        kElemOffset0 = mmak * miK + ir * elementsPerVgpr
        module.add(VAddU32(
            dst=vgpr(kPosCur), src0=kElemOffset0, src1=vgpr(kPosBaseVgpr),
            comment="byteRefine[%s ir=%d mod=0]: K_pos = kPosBase + %d"
                    % (operand, ir, kElemOffset0)))
        with kw.allocTmpSgpr(laneSGPRCount,
                             alignment=laneSGPRCount) as maskInfo:
            maskSgpr = maskInfo.idx
            module.add(VCmpGEI32(
                dst=sgpr(maskSgpr, laneSGPRCount),
                src0=vgpr(kPosCur), src1=sgpr("LoopCounterL"),
                comment="byteRefine[%s ir=%d mod=0]: K_pos >= LoopCounterL ?"
                        % (operand, ir)))
            module.add(VCndMaskB32(
                dst=vgpr(maskVgpr),
                src0=vgpr(maskVgpr), src1=0,
                src2=sgpr(maskSgpr, laneSGPRCount),
                comment="byteRefine[%s ir=%d mod=0]: mask = past ? 0 : prev"
                        % (operand, ir)))

        for vIdx in idxs:
            module.add(VAndB32(
                dst=vgpr(vIdx), src0=vgpr(maskVgpr), src1=vgpr(vIdx),
                comment="byteRefine[%s ir=%d]: apply mask to Valu%s[%u]"
                        % (operand, ir, operand, vIdx)))

    for ir in range(vgprPerInUnroll):
        aIdxs = aIndicesByIr.get(ir, [])
        bIdxs = bIndicesByIr.get(ir, [])
        if aIdxs:
            _emitChain("A", aIdxs, ir, bpeA, elementsPerVgprA)
        if bIdxs:
            _emitChain("B", bIdxs, ir, bpeB, elementsPerVgprB)

    kw.vgprPool.checkIn(kPosCur)
    kw.vgprPool.checkIn(maskVgpr)
    kw.vgprPool.checkIn(seedVgpr)
    return module


def emitTailSubLaneMaskInitFused(kw, kPosBaseVgpr, numMIInUnroll,
                                 bpe, vgprPerInUnroll):
    """Emit fused-form per-lane invariants for the K-tail mask
    chain (BF16 byte-refine path: bpe=2, elementsPerVgpr=2). Returns
    the persistent VGPRs the per-(operand, mmak, ir) chain consults
    via `emitTailSubLaneMaskChainIntoVgprFused`:

      * `diffVgpr` (1 VGPR): `LoopCounterL - kPosBase` (signed
        v_sub_i32). Persists across every per-(mmak, ir) chain so
        that chain's only per-call cost is two VOPC cmps + two
        cndmasks (no per-call subtract / add of K_pos).
      * `boundaryMaskVgprs` (`vgprPerInUnroll` VGPRs): per-vgpr-in-
        tile 3-state partial mask derived once from
        `d = LoopCounterL & (numMIInUnroll - 1)`. For BF16 (kStride=2):
          - d <=  i*2     -> boundary[i] = 0             (this vgpr's K-pos already past)
          - d ==  i*2 + 1 -> boundary[i] = 0x0000FFFF   (low BF16 in, high past)
          - d >=  i*2 + 2 -> boundary[i] = -1           (both BF16 in)
        These cover the BOUNDARY lane only; the per-(mmak, ir) chain
        uses sFull / sZero cmps to override to full (-1) / zero (0)
        when the entire lane is in / out of range.

    Stays paired with the long-lived per-(mmak, ir) precompute
    storage in `emitTailSubLaneMaskPrecomputeSubtile` so the
    per-mmak apply step stays a pure `v_and_b32`. The chain init +
    per-(mmak, ir) precompute both run before the GR swait /
    sbarrier drain so the cmp/cndmask traffic co-issues with the
    buffer-load latency; the per-mmak hot path just v_ands the
    precomputed mask onto each tile VGPR.

    Asserted scope: bpe == 2 only (BF16). Other byte-refine bpe
    values (bpe in {1, 4}) keep the legacy
    `emitTailSubLaneMaskChainIntoVgpr` chain via the precompute
    dispatcher.
    """
    laneSGPRCount = kw.states.laneSGPRCount
    module = Module("tailSubLaneMaskInitFused")
    assert bpe == 2 and vgprPerInUnroll >= 1, \
        ("fused mask init currently supports BF16 (bpe=2) only; "
         "got bpe=%r vgprPerInUnroll=%r" % (bpe, vgprPerInUnroll))
    elementsPerVgpr = 2

    diffVgpr = kw.vgprPool.checkOut(1, "subLaneMaskDiffFused")
    module.add(VSubI32(
        dst=vgpr(diffVgpr), src0=sgpr("LoopCounterL"), src1=vgpr(kPosBaseVgpr),
        comment="subLaneMask fused: diff = LoopCounterL - kPosBase "
                "(signed; per-(mmak,ir) chain uses fullLit/zeroLit folded)"))

    # halfKeep = 0x0000FFFF (keep low BF16 K-element, zero high)
    halfMaskVgpr = kw.vgprPool.checkOut(1, "subLaneMaskHalfKeep")
    module.add(VMovB32(
        dst=vgpr(halfMaskVgpr), src="0x0000FFFF",
        comment="subLaneMask fused: halfKeep mask = 0x0000FFFF"))

    # d = LoopCounterL % numMIInUnroll. numMIInUnroll is a power of 2
    # for every gfx950 BF16 MFMA we emit (MI_M * MI_K / WaveSize = 8
    # for MI16x16x32 BF16), so `& (numMIInUnroll - 1)` is the divide.
    vDLaneRem = kw.vgprPool.checkOut(1, "subLaneMaskDLaneRem")
    module.add(VAndB32(
        dst=vgpr(vDLaneRem),
        src0=numMIInUnroll - 1, src1=sgpr("LoopCounterL"),
        comment="subLaneMask fused: d = LoopCounterL %% %u" % numMIInUnroll))

    boundaryMaskVgprs = []
    with kw.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount) as tmpSgprInfo:
        maskSgpr = tmpSgprInfo.idx
        for i in range(vgprPerInUnroll):
            bm = kw.vgprPool.checkOut(1, "subLaneMaskBoundary%u" % i)
            boundaryMaskVgprs.append(bm)
            hiBound = i * elementsPerVgpr + elementsPerVgpr   # d < hi -> halfKeep else full
            loBound = i * elementsPerVgpr + 1                  # d < lo -> 0 else prev
            module.add(VCmpLtI32(
                dst=sgpr(maskSgpr, laneSGPRCount),
                src0=vgpr(vDLaneRem), src1=hiBound,
                comment="subLaneMask boundary[%u]: d < %u ?" % (i, hiBound)))
            module.add(VCndMaskB32(
                dst=vgpr(bm),
                src0=-1, src1=vgpr(halfMaskVgpr),
                src2=sgpr(maskSgpr, laneSGPRCount),
                comment="subLaneMask boundary[%u] = (d<%u) ? halfKeep : full"
                        % (i, hiBound)))
            module.add(VCmpLtI32(
                dst=sgpr(maskSgpr, laneSGPRCount),
                src0=vgpr(vDLaneRem), src1=loBound,
                comment="subLaneMask boundary[%u]: d < %u ?" % (i, loBound)))
            module.add(VCndMaskB32(
                dst=vgpr(bm), src0=vgpr(bm), src1=0,
                src2=sgpr(maskSgpr, laneSGPRCount),
                comment="subLaneMask boundary[%u] = (d<%u) ? 0 : prev"
                        % (i, loBound)))

    kw.vgprPool.checkIn(halfMaskVgpr)
    kw.vgprPool.checkIn(vDLaneRem)
    return module, diffVgpr, boundaryMaskVgprs


def emitTailSubLaneMaskChainIntoVgprFused(kw, diffVgpr, boundaryMaskVgpr,
                                          operand, mmak, ir, miK,
                                          numMIInUnroll, targetMaskVgpr):
    """Fused-form per-(operand, mmak, ir) K-tail mask chain. Two
    VOPC cmps + two cndmasks compute a 3-state mask
    (full / boundary[ir] / zero) keyed off the single `diffVgpr` and
    `boundaryMaskVgpr` precomputed once by
    `emitTailSubLaneMaskInitFused`.

    Per call:
      fullLit = mmak*miK + numMIInUnroll - 1
      zeroLit = mmak*miK
      sFull   = (diff >  fullLit)   -> "ALL of this lane's K is in range"
      sZero   = (diff <= zeroLit)   -> "NONE of this lane's K is in range"
      targetMaskVgpr = sFull ? -1 : boundaryMaskVgpr
                     = sZero ?  0 : prev

    For VOPC inline range [-16, 64], literals past 64 are staged
    through a scratch sgpr (mirrors `_emitSubtileScalarCmpLitOrStaged`
    for the SOPC class but for VOPC src1). For BF16 MI_K=32,
    numMIInUnroll=8: fullLit > 64 starts at mmak=2 (`2*32+7=71`),
    zeroLit > 64 starts at mmak=3 (`3*32=96`). DU<=64 (mmak<=1) is
    entirely inline.
    """
    laneSGPRCount = kw.states.laneSGPRCount
    fullLit = mmak * miK + numMIInUnroll - 1
    zeroLit = mmak * miK

    module = Module("tailSubLaneMaskChainFused %s mmak=%u ir=%u"
                    % (operand, mmak, ir))

    with kw.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount) as tmpSgprInfo:
        maskSgpr = tmpSgprInfo.idx

        def _vopcCmp(cmpCls, literal, comment):
            if kw._subtileCmpSrc1FitsInline(literal):
                module.add(cmpCls(
                    dst=sgpr(maskSgpr, laneSGPRCount),
                    src0=vgpr(diffVgpr), src1=literal, comment=comment))
            else:
                with kw.allocTmpSgpr(1) as litSgprInfo:
                    litSgpr = litSgprInfo.idx
                    module.add(SMovB32(
                        dst=sgpr(litSgpr), src=hex(literal),
                        comment="stage literal %u (non-inline) for vopc src1" % literal))
                    module.add(cmpCls(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(diffVgpr), src1=sgpr(litSgpr), comment=comment))

        _vopcCmp(VCmpGTI32, fullLit,
                 "subLaneMask[%s mmak=%u ir=%u]: sFull = diff > %u (all-in)"
                 % (operand, mmak, ir, fullLit))
        module.add(VCndMaskB32(
            dst=vgpr(targetMaskVgpr),
            src0=vgpr(boundaryMaskVgpr), src1=-1,
            src2=sgpr(maskSgpr, laneSGPRCount),
            comment="subLaneMask[%s mmak=%u ir=%u] = sFull ? full : boundary[%u]"
                    % (operand, mmak, ir, ir)))
        _vopcCmp(VCmpLeI32, zeroLit,
                 "subLaneMask[%s mmak=%u ir=%u]: sZero = diff <= %u (none-in)"
                 % (operand, mmak, ir, zeroLit))
        module.add(VCndMaskB32(
            dst=vgpr(targetMaskVgpr),
            src0=vgpr(targetMaskVgpr), src1=0,
            src2=sgpr(maskSgpr, laneSGPRCount),
            comment="subLaneMask[%s mmak=%u ir=%u] = sZero ? 0 : prev"
                    % (operand, mmak, ir)))

    return module


def emitTailSubLaneMaskChainIntoVgpr(kw, kernel, operand, kPosBaseVgpr,
                                     mmak, ir, miK, bpe, elementsPerVgpr,
                                     targetMaskVgpr, kPosCurVgpr, seedVgpr):
    """Emit ONE per-(operand, mmak, ir) byte-mask chain into
    `targetMaskVgpr`. Mirrors the chain body from
    `emitTailSubLaneMaskRefineSubtile._emitChain` (static skip +
    mod>0 chain + mod=0 step) but writes its result to a
    caller-owned VGPR instead of v_anding it back into A/B tile
    VGPRs. The apply step (`emitTailSubLaneMaskApplySubtile`) does
    the v_and per (operand, ir, vIdx).

    Caller owns the lifecycle of `targetMaskVgpr` (held live across
    the per-mmak loop), `kPosCurVgpr` and `seedVgpr` (transient
    scratch shared across all chains in a precompute call).

    Legacy chain form -- per-(mmak, ir) emits its own kPos
    computation and the bpe-parametric mod chain. Retained for the
    bpe != 2 byte-refine path (fp8 / int8 anyK, not exercised in
    the current gauntlet) and for the
    `SubtileTailMaskFusedForm=False` fallback. The default BF16
    (bpe=2) path now uses
    `emitTailSubLaneMaskChainIntoVgprFused` instead (lower
    instruction count on the MT320x320 side-by-side).
    """
    laneSGPRCount = kw.states.laneSGPRCount
    bpr = 4
    asem = kernel["AssertSummationElementMultiple"]

    module = Module("tailSubLaneMaskChainIntoVgpr %s mmak=%u ir=%u"
                    % (operand, mmak, ir))
    staticSkipPartial = (asem * bpe) % bpr == 0
    module.add(VMovB32(
        dst=vgpr(targetMaskVgpr), src=hex(0xFFFFFFFF),
        comment="byteRefine[%s ir=%d mmak=%d]: mask seed = full keep"
                % (operand, ir, mmak)))

    if not staticSkipPartial:
        with kw.allocTmpSgpr(laneSGPRCount,
                             alignment=laneSGPRCount) as maskInfo:
            maskSgpr = maskInfo.idx
            for mod in range(elementsPerVgpr - 1, 0, -1):
                maskByte = (1 << (mod * bpe * 8)) - 1
                module.add(VMovB32(
                    dst=vgpr(seedVgpr), src=hex(maskByte),
                    comment="byteRefine[%s ir=%d mod=%d]: keep mask = 0x%X"
                            % (operand, ir, mod, maskByte)))
                kElemOffset = mmak * miK + ir * elementsPerVgpr + mod
                module.add(VAddU32(
                    dst=vgpr(kPosCurVgpr), src0=kElemOffset, src1=vgpr(kPosBaseVgpr),
                    comment="byteRefine[%s ir=%d mod=%d]: K_pos = kPosBase + %d"
                            % (operand, ir, mod, kElemOffset)))
                module.add(VCmpGEI32(
                    dst=sgpr(maskSgpr, laneSGPRCount),
                    src0=vgpr(kPosCurVgpr), src1=sgpr("LoopCounterL"),
                    comment="byteRefine[%s ir=%d mod=%d]: K_pos >= LoopCounterL ?"
                            % (operand, ir, mod)))
                module.add(VCndMaskB32(
                    dst=vgpr(targetMaskVgpr),
                    src0=vgpr(targetMaskVgpr), src1=vgpr(seedVgpr),
                    src2=sgpr(maskSgpr, laneSGPRCount),
                    comment="byteRefine[%s ir=%d mod=%d]: mask = past ? 0x%X : prev"
                            % (operand, ir, mod, maskByte)))

    kElemOffset0 = mmak * miK + ir * elementsPerVgpr
    module.add(VAddU32(
        dst=vgpr(kPosCurVgpr), src0=kElemOffset0, src1=vgpr(kPosBaseVgpr),
        comment="byteRefine[%s ir=%d mod=0]: K_pos = kPosBase + %d"
                % (operand, ir, kElemOffset0)))
    with kw.allocTmpSgpr(laneSGPRCount,
                         alignment=laneSGPRCount) as maskInfo:
        maskSgpr = maskInfo.idx
        module.add(VCmpGEI32(
            dst=sgpr(maskSgpr, laneSGPRCount),
            src0=vgpr(kPosCurVgpr), src1=sgpr("LoopCounterL"),
            comment="byteRefine[%s ir=%d mod=0]: K_pos >= LoopCounterL ?"
                    % (operand, ir)))
        module.add(VCndMaskB32(
            dst=vgpr(targetMaskVgpr),
            src0=vgpr(targetMaskVgpr), src1=0,
            src2=sgpr(maskSgpr, laneSGPRCount),
            comment="byteRefine[%s ir=%d mod=0]: mask = past ? 0 : prev"
                    % (operand, ir)))
    return module


def emitTailSubLaneMaskPrecomputeSubtile(kw, kernel, kPosBaseVgpr,
                                         numMmaks, miK, numMIInUnroll):
    """Precompute every per-(operand, mmak, ir) K-tail byte mask into
    long-lived scratch VGPRs *before* the per-mmak MFMA loop runs.
    The hot per-mmak path then becomes a pure
    `v_and_b32 vIdx, maskVgpr, vIdx` apply step (see
    `emitTailSubLaneMaskApplySubtile`), hoisting the cmp + cndmask
    chain out of the loop.

    Storage layout / dedup:
      - Returns `maskVgprMap[(operand, mmak, ir)] -> vgpr_index`.
      - When `bpeA == bpeB` (and therefore the chain produces
        identical masks for both operands at the same (mmak, ir)),
        the A and B keys map to the SAME vgpr (halves the per-(mmak, ir)
        VGPR cost for the common bf16/bf16 case).
      - Persistent VGPR count for the common case:
        `numMmaks * (numMIInUnroll // elementsPerVgpr)` per-(mmak, ir)
        precomputed masks PLUS, on the fused path, the init
        invariants: `1 + (numMIInUnroll // elementsPerVgpr)` shared
        across all (mmak, ir). For BF16 ASEM<8 with numMIInUnroll=8,
        elementsPerVgpr=2 (4 vgprs per mmak): 2 mmaks (DU=64) -> 8
        precomputed + 5 init = 13 vgprs. Legacy chain holds 8
        precomputed + 2 transient (freed after precompute) = 8
        persistent.

    Chain dispatch:
      - `kernel.get("SubtileTailMaskFusedForm", True)` && bpeA ==
        bpeB == 2 (BF16): use the fused init+chain form (single
        `diff` + per-i `boundary[ir]` precomputed once, then 2 cmps +
        2 cndmasks per (mmak, ir)).
      - Otherwise: legacy per-(operand, mmak, ir) chain via
        `emitTailSubLaneMaskChainIntoVgpr` (bpe-parametric mod
        chain with static skip + runtime gate). Covers fp8/int8
        byte-refine paths and asymmetric bpe configs (not exercised
        in the current gauntlet) plus the explicit fallback when
        the fused form is disabled.

    Returns:
      `(module, maskVgprMap, allocatedMaskVgprs)`:
        - `module` holds the emitted chain instructions (init +
          per-(mmak, ir) chain).
        - `maskVgprMap` is the (operand, mmak, ir) -> vgpr lookup
          the apply step consults.
        - `allocatedMaskVgprs` is the list the caller must
          `vgprPool.checkIn` AFTER the per-mmak loop completes
          (includes both init VGPRs and per-(mmak, ir) precomputed
          VGPRs so cleanup matches alloc).
    """
    assert kernel["ProblemType"].get("MXBlockA", 0) == 0, (
        "sub-lane K-tail mask precompute does not handle the MX scale path.")
    assert kernel["ProblemType"].get("MXBlockB", 0) == 0, (
        "sub-lane K-tail mask precompute does not handle the MX scale path.")

    bpr = 4
    bpeARaw = kernel["ProblemType"]["DataTypeA"].numBytes()
    bpeBRaw = kernel["ProblemType"]["DataTypeB"].numBytes()
    bpeA = int(bpeARaw)
    bpeB = int(bpeBRaw)
    assert bpeA == bpeARaw and 1 <= bpeA <= bpr, (
        "sub-lane precompute: DataTypeA bpe must be integer in [1, bpr]; "
        "got %r" % (bpeARaw,))
    assert bpeB == bpeBRaw and 1 <= bpeB <= bpr, (
        "sub-lane precompute: DataTypeB bpe must be integer in [1, bpr]; "
        "got %r" % (bpeBRaw,))
    elementsPerVgprA = max(1, bpr // bpeA)
    elementsPerVgprB = max(1, bpr // bpeB)
    vgprPerInUnrollA = max(1, numMIInUnroll // elementsPerVgprA)
    vgprPerInUnrollB = max(1, numMIInUnroll // elementsPerVgprB)
    shareAB = (bpeA == bpeB and elementsPerVgprA == elementsPerVgprB)

    # Fused form gating: opt-in for the BF16/BF16 symmetric path
    # only (the chain shares one boundary[ir] vgpr across operands,
    # so asymmetric bpe doesn't fit). `SubtileTailMaskFusedForm`
    # defaults to True; setting it False reverts to the legacy
    # per-(operand, mmak, ir) bpe-parametric chain (kept for fp8 /
    # int8 byte-refine paths and the reversible escape hatch).
    useFusedForm = (kernel.get("SubtileTailMaskFusedForm", True)
                    and shareAB and bpeA == 2)

    module = Module("tailSubLaneMaskPrecomputeSubtile")
    maskVgprMap = {}
    allocatedMaskVgprs = []

    if useFusedForm:
        # Init invariants once (diff + boundaryMask[ir]). The init
        # VGPRs persist across every per-(mmak, ir) chain emit AND the
        # per-mmak apply loop (the chain re-reads boundaryMask[ir];
        # the apply itself ignores them, but freeing here would race
        # the next mmak's chain emit).
        initModule, diffVgpr, boundaryMaskVgprs = \
            emitTailSubLaneMaskInitFused(
                kw, kPosBaseVgpr, numMIInUnroll, bpeA, vgprPerInUnrollA)
        module.add(initModule)
        allocatedMaskVgprs.append(diffVgpr)
        allocatedMaskVgprs.extend(boundaryMaskVgprs)

        for mmak in range(numMmaks):
            for ir in range(vgprPerInUnrollA):
                vMaskA = kw.vgprPool.checkOut(
                    1, "subLaneMask_A_mmak%d_ir%d" % (mmak, ir))
                allocatedMaskVgprs.append(vMaskA)
                maskVgprMap[("A", mmak, ir)] = vMaskA
                module.add(emitTailSubLaneMaskChainIntoVgprFused(
                    kw, diffVgpr, boundaryMaskVgprs[ir],
                    "A", mmak, ir, miK, numMIInUnroll, vMaskA))
            # bpeA == bpeB == 2 guaranteed by useFusedForm; B operand
            # shares the per-(mmak, ir) precomputed mask.
            for ir in range(vgprPerInUnrollA):
                maskVgprMap[("B", mmak, ir)] = maskVgprMap[("A", mmak, ir)]
        return module, maskVgprMap, allocatedMaskVgprs

    # Legacy bpe-parametric chain. Kept for fp8 / int8 byte-refine
    # configs (not in current gauntlet) and the
    # SubtileTailMaskFusedForm=False reversibility path.
    kPosCur = kw.vgprPool.checkOut(1, "kPosCurPrecompute")
    seedVgpr = kw.vgprPool.checkOut(1, "subLaneByteSeedPrecompute")

    for mmak in range(numMmaks):
        for ir in range(vgprPerInUnrollA):
            vMaskA = kw.vgprPool.checkOut(
                1, "subLaneMask_A_mmak%d_ir%d" % (mmak, ir))
            allocatedMaskVgprs.append(vMaskA)
            maskVgprMap[("A", mmak, ir)] = vMaskA
            module.add(emitTailSubLaneMaskChainIntoVgpr(
                kw, kernel, "A", kPosBaseVgpr, mmak, ir, miK,
                bpeA, elementsPerVgprA, vMaskA, kPosCur, seedVgpr))
        if shareAB:
            for ir in range(vgprPerInUnrollA):
                maskVgprMap[("B", mmak, ir)] = maskVgprMap[("A", mmak, ir)]
        else:
            for ir in range(vgprPerInUnrollB):
                vMaskB = kw.vgprPool.checkOut(
                    1, "subLaneMask_B_mmak%d_ir%d" % (mmak, ir))
                allocatedMaskVgprs.append(vMaskB)
                maskVgprMap[("B", mmak, ir)] = vMaskB
                module.add(emitTailSubLaneMaskChainIntoVgpr(
                    kw, kernel, "B", kPosBaseVgpr, mmak, ir, miK,
                    bpeB, elementsPerVgprB, vMaskB, kPosCur, seedVgpr))

    kw.vgprPool.checkIn(kPosCur)
    kw.vgprPool.checkIn(seedVgpr)
    return module, maskVgprMap, allocatedMaskVgprs


def emitTailSubLaneMaskApplySubtile(kw, mmak, maskVgprMap,
                                    aIndicesByIr, bIndicesByIr):
    """Per-mmak v_and-only apply step for precomputed K-tail masks.
    Walks `aIndicesByIr` / `bIndicesByIr` (which still reflect the
    per-mmak A/B vgprTile slice) and emits one
    `v_and_b32 vIdx, maskVgprMap[(operand, mmak, ir)], vIdx`
    per boundary VGPR. No cmps, no cndmasks: those were emitted once
    in `emitTailSubLaneMaskPrecomputeSubtile` before the loop.
    """
    module = Module("tailSubLaneMaskApplySubtile mmak=%u" % mmak)
    for ir, idxs in sorted(aIndicesByIr.items()):
        vMask = maskVgprMap.get(("A", mmak, ir))
        assert vMask is not None, (
            "missing precomputed A mask for mmak=%d ir=%d "
            "(maskVgprMap keys=%r)" % (mmak, ir, sorted(maskVgprMap.keys())))
        for vIdx in idxs:
            module.add(VAndB32(
                dst=vgpr(vIdx), src0=vgpr(vMask), src1=vgpr(vIdx),
                comment="byteRefine[A ir=%d mmak=%d]: apply precomputed mask "
                        "to ValuA[%u]" % (ir, mmak, vIdx)))
    for ir, idxs in sorted(bIndicesByIr.items()):
        vMask = maskVgprMap.get(("B", mmak, ir))
        assert vMask is not None, (
            "missing precomputed B mask for mmak=%d ir=%d "
            "(maskVgprMap keys=%r)" % (mmak, ir, sorted(maskVgprMap.keys())))
        for vIdx in idxs:
            module.add(VAndB32(
                dst=vgpr(vIdx), src0=vgpr(vMask), src1=vgpr(vIdx),
                comment="byteRefine[B ir=%d mmak=%d]: apply precomputed mask "
                        "to ValuB[%u]" % (ir, mmak, vIdx)))
    return module
