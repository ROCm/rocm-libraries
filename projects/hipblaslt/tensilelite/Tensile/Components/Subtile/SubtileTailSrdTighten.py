# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SRD NumRecords tightening at K-tail entry for the subtile path.

Each helper clips ``Srd<tc>+2`` (the buffer NumRecords field) so that
the re-issued tail GR cannot OOR-read past the operand's allocated
bytes on the last m-row of the tile. Three flavours, all called
unconditionally from `_emitTailLoopScaffoldSubtile` after openLoop:

- :func:`emitTailSrdTightenSubtile`        — bf16/fp16/int8 A/B
                                             (bpr=4 align).
- :func:`emitTailSrdTightenSubtileMX`      — MX scale operands
                                             (K=256 pad granularity).
- :func:`emitTailSrdTightenSubtileMXData`  — MX data operands
                                             (K=256 pad, per-operand
                                             bpe shift).

Each helper is gated to a static no-op when its preconditions don't
hold (non-MX vs MX, swizzled-A/B, asymmetric bpe, DepthU <= MX_PAD_K,
…) and returns an empty Module in that case so the call sequence at
the scaffold stays uniform.

The kernel-writer object (``kw`` first arg) is the live KernelWriter
instance; helpers reach into ``kw.states.*`` for tile metadata and
into ``kw.allocTmpSgpr`` for scratch allocation.
"""

import math

from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import (
    SLShiftLeftB32, SLShiftRightB32, SMovB32,
    SAddU32, SAndB32, SSubU32,
)


# Host MX-scale K-padding granularity guaranteed by
# `DataInitialization.cpp::rearrangePaddedMXScaleLayout` (pads K-blocks
# to a multiple of 8 mxBlocks = 256 K-elements). The value is specific
# to the MX scale's host pre-pad and is not a generic "swizzle K pad" --
# the subtile-swizzled A/B path's K-direction swizzle size is 32 (per
# the `swizzleSize1 = 32` in `KernelWriterAssembly.computeLoadSrd()`),
# not 256, so keeping the name MX-scoped avoids a misleading rename.
# (Nakajee's PR review flagged this rename as deferrable; keeping it
# accurate over generic.)
MX_PAD_K = 256


def _swizzleSize0ForMN(kernel, tc):
    """Return the M/N-direction swizzle block size used by the
    ``computeLoadSrd()`` byte-stride math for operand ``tc``.

    Mirrors the same branch as
    ``KernelWriterAssembly.computeLoadSrd()`` (look for
    ``isMxSwizzledScaleLayout`` / ``isSwizzledSubtile``):

      * MX scale (``tc in {MXSA, MXSB}``) with a swizzled
        ``MXScaleFormat`` (``HostPreSwizzle`` / ``InMemorySwizzle``)
        -> 32 (M/N rows packed per 256-K swizzle block; the per-K-
        element byte stride along K is ``swizzleSize0 / mxBlock``,
        so for ``mxBlock == 32`` it collapses to 1 byte/K-element).
      * Subtile A/B with ``SwizzleTensor{A,B}`` (PreShuffled DTV
        path) and ``UseSubtileImpl=1`` -> 16 (MXFP4 swizzle's M/N
        block size; the per-K-element byte stride is
        ``swizzleSize0 / 1 = 16`` bytes).
      * Anything else (plain bf16/fp16/int8 A/B, non-swizzled MX
        scale, non-subtile) -> 1. The K-direction byte stride for
        this path is then ``1 * bpe = bpe`` bytes/K-element, which
        is what :func:`emitTailSrdTightenSubtile` uses directly.

    With this helper, the SRD-tighten delta math is the same shape
    for all three paths:

        delta_bytes = delta_K * (swizzleSize0 / blockGran) * bpe_K

    where ``blockGran`` is ``mxBlock`` for MX scale and ``1``
    otherwise, and ``bpe_K`` is the per-K-element data bpe (only the
    non-swizzled bf16/fp16/int8 path uses ``bpe_K > 1``; swizzled MX
    paths absorb the K-direction bpe into ``swizzleBlockSize``).
    Keeping the helper here (rather than re-importing the predicate
    from :mod:`KernelWriterAssembly`) avoids a circular import; the
    branches must stay in sync with ``computeLoadSrd()``.
    """
    useSubtile = bool(kernel.get("UseSubtileImpl"))
    mxScaleFormat = kernel.get("MXScaleFormat", "NoSwizzle")
    isMxSwizzledScale = ("MXS" in tc) and \
        mxScaleFormat in ("InMemorySwizzle", "HostPreSwizzle")
    if isMxSwizzledScale:
        return 32
    isPreShuffledAB = tc in ("A", "B") and \
        kernel["ProblemType"].get("SwizzleTensor%s" % tc, False)
    if isPreShuffledAB and useSubtile:
        return 16
    return 1


def emitTailSrdTightenSubtile(kw, kernel):
    """Tighten ``Srd<tc>+2`` for bf16/fp16/int8 A/B at tail entry so
    the K-direction reads of the re-issued tail GR cannot wander past
    A/B's actual end-of-array on the last m-row (OOR clip).

    Formula (align to bpr=4)::

        alignedBytes = roundUp(LoopCounterL * bpe, 4)    # bpr=4
        delta_bytes  = DepthU * bpe - alignedBytes
        Srd<tc>+2   -= delta_bytes

    ``roundUp(..., bpr=4)`` is the tightest clip that still keeps the
    wide DTL load valid for the trailing odd-K element on the last
    m-row: per-thread ``buffer_load_<...>`` is a BufferLoad-aligned
    hardware multiple of bpr, so the smallest granularity past
    K_remain that the load *needs* to cover is one bpr-worth of bytes
    (=2 bf16 / 4 int8 elements). Earlier m-rows are over-protected
    (their k_lane >= K_remain reads still pass the tightened limit
    but get zeroed in VGPR by the per-MFMA lane mask + sub-lane
    refine, so correctness is unchanged).

    Align-UP (rather than literal align-DOWN ``remainK & 0xfffffffe``)
    is required because the gfx950 assembler rejects the narrow
    trailing-element load (``buffer_load_d16_b16 ... lds``) the
    align-DOWN strategy depended on; that helper was previously
    deleted in ``7df7d24``. Without it, align-DOWN would drop the
    trailing odd-K element from the wide load. Our wide DTL +
    per-lane refine path handles the trailing element when it is
    INCLUDED in the wide load (which requires align-UP). bpr=4 is
    the finest align-UP granularity, costing at most one DWORD
    (4 B = 2 bf16) of slack relative to a per-byte clip. Tensilelite
    always K-pads to MIK boundary, so DWORD-level past-K reads stay
    within-page.

    ``delta_bytes >= 0`` is provably non-negative under align-UP to
    bpr=4 (alignedBytes <= roundUp((DepthU-1)*bpe, 4) <= DepthU*bpe
    for bpe in {1, 2}), so no runtime clamp / skip-label is needed
    -- when delta=0 the two ``s_sub_u32`` lines become harmless
    no-ops.

    Gating (the helper is only called from
    ``_emitTailLoopScaffoldSubtile``, which already runs only for
    ``UseSubtileImpl=True`` kernels with ``NoTailLoop=False``, so this
    helper does not re-check those two; it gates the remaining
    out-of-scope subtile variants):

      - non-MX (``MXBlock{A,B} == 0``) -- MX scales have their own
        host re-scatter padding plus a separate swizzleBlock-aware
        MXSA/MXSB tightener (:func:`emitTailSrdTightenSubtileMX`)
        when DepthU > 256;
      - non-swizzled A/B (``SwizzleTensor{A,B}=False``) -- subtile
        mxfp4 swizzled A/B (DTV path) live on a different tail-loop
        emitter entirely;
      - symmetric per-tensor bpe (A and B same dtype), bpe in {1, 2}
        (bf16 / fp16 / int8 anyk paths are the immediate consumer);
      - symmetric per-tensor ``loadWidthGR`` (per-thread B128 / B64 /
        B32 etc. shape matches across A/B in the gated kernels).

    No SRD restore is needed: the tail body is the last GR site for
    A/B before the kernel epilogue (epilogue uses SrdC / SrdD).

    Swizzle-size cross-reference (see :func:`_swizzleSize0ForMN`):
    this helper is the ``swizzleSize0 == 1`` case of the generalized
    SRD-tighten formula ``delta_bytes = delta_K * swizzleSize0 *
    bpe_K`` -- non-MX, non-swizzled A/B reduces to ``delta_bytes =
    delta_K * 1 * bpe = delta_K * bpe`` (modulo the bpr=4 round-UP
    on K_remain*bpe). The swizzled A/B path (DTV / PreShuffled) has
    ``swizzleSize0 == 16`` and is gated out below; see the MX
    tighteners for the ``swizzleSize0 in {16, 32}`` paths.
    """
    module = Module("tailSrdTightenSubtile")
    if kernel["ProblemType"].get("MXBlockA", 0) > 0 \
       or kernel["ProblemType"].get("MXBlockB", 0) > 0:
        return module
    if kernel["ProblemType"].get("SwizzleTensorA", False) \
       or kernel["ProblemType"].get("SwizzleTensorB", False):
        return module

    tiA = kw.states.a.tileInfo
    tiB = kw.states.b.tileInfo
    bpeA = int(tiA.bpe)
    bpeB = int(tiB.bpe)
    if bpeA != bpeB or bpeA not in (1, 2):
        return module
    bpe = bpeA
    loadBytesA = int(tiA.loadWidthGR)
    loadBytesB = int(tiB.loadWidthGR)
    if loadBytesA != loadBytesB or loadBytesA <= 0:
        return module
    depthU = int(kernel["DepthU"])
    depthUBytes = depthU * bpe
    # bpr=4 alignment requires depthUBytes >= bpr so the natural
    # case `K_remain == 0` (which the tail loop doesn't enter, but
    # is the algebraic boundary) gives delta <= depthUBytes.
    bpr = 4
    if depthUBytes < bpr:
        return module
    alignMaskInv = (~(bpr - 1)) & 0xffffffff

    module.addComment2(
        "Tighten Srd<tc>+2 to K_remain*bpe rounded up to bpr=4 "
        "(OOR clip on last m-row)")
    with kw.allocTmpSgpr(2) as tmpInfo:
        scaledKRem = tmpInfo.idx
        delta = tmpInfo.idx + 1
        if bpe == 2:
            module.add(SLShiftLeftB32(
                dst=sgpr(scaledKRem), src=sgpr("LoopCounterL"), shiftHex=hex(1),
                comment="K_remain * bpe (bpe=2)"))
        else:
            module.add(SMovB32(
                dst=sgpr(scaledKRem), src=sgpr("LoopCounterL"),
                comment="K_remain * bpe (bpe=1)"))
        module.add(SAddU32(
            dst=sgpr(scaledKRem), src0=sgpr(scaledKRem), src1=(bpr - 1),
            comment="+ (bpr-1) for roundUp"))
        module.add(SAndB32(
            dst=sgpr(scaledKRem), src0=sgpr(scaledKRem), src1=hex(alignMaskInv),
            comment="alignedBytes = roundUp(K_remain*bpe, %u)" % bpr))
        # delta provably >= 0 under align-UP to bpr=4; when delta=0
        # the SSubs are harmless no-ops, so no runtime skip needed.
        module.add(SSubU32(
            dst=sgpr(delta), src0=depthUBytes, src1=sgpr(scaledKRem),
            comment="delta = DepthU*bpe - alignedBytes (>= 0)"))
        module.add(SSubU32(
            dst=sgpr("SrdA+2"), src0=sgpr("SrdA+2"), src1=sgpr(delta),
            comment="Srd A+2 -= delta (clip K past K_remain on last m-row)"))
        module.add(SSubU32(
            dst=sgpr("SrdB+2"), src0=sgpr("SrdB+2"), src1=sgpr(delta),
            comment="Srd B+2 -= delta (clip K past K_remain on last m-row)"))
    return module


def emitTailSrdTightenSubtileMX(kw, kernel):
    """Tighten ``SrdMXS{A,B}+2`` for MX scale buffers when
    ``DepthU > MX_PAD_K`` (=256). Companion to the bf16 tightener::

        remainK_MX = roundUp(K_remain, 256)                # K elements
        delta_K    = DepthU - remainK_MX                   # K elements, >= 0
        SrdMXS{A,B}+2 -= delta_K * bytesPerKElement_MX     # bytes

    where:

      - 256 is the host MX-scale K-padding granularity guaranteed by
        ``DataInitialization.cpp::rearrangePaddedMXScaleLayout``;
      - ``bytesPerKElement_MX = swizzleSize0 / mxBlock``. This is the
        explicit form of ``computeLoadSrd()``'s K-direction stride
        for the MX swizzled scale layout: that helper computes
        ``swizzleBlockSize = swizzleSize0 * swizzleSize1 / mxBlock``
        (with ``swizzleSize0=32``, ``swizzleSize1=256``) and adds
        ``swizzleBlockSize * (DepthU / swizzleSize1)`` bytes to
        ``Srd<tc>+2``, so the per-K-element K-stride is
        ``swizzleBlockSize / swizzleSize1 = swizzleSize0 / mxBlock``.
        For the gauntlet configs (mxfp4/mxfp8, ``mxBlock == 32``)
        this collapses to 1 byte/K-element, matching the previously
        implicit ``lrSubtileSize * lrGlobalSubtileGrid[1] / DepthU``
        derivation that this helper used to use. For ``mxBlock == 16``
        the explicit formula yields 2 bytes/K-element. See
        :func:`_swizzleSize0ForMN` for the cross-reference to
        ``computeLoadSrd()``.

    Static gate -- emit nothing when ``DepthU <= MX_PAD_K=256``: for
    any K_remain in [0, DepthU-1] with DepthU <= 256,
    ``remainK_MX = roundUp(K_remain, 256) >= DepthU``, so
    ``delta_K <= 0`` always. The host padding alone already covers
    any read past K_remain on the last m-row.

    For DepthU=512 (the only DepthU>256 MX config in our gauntlet),
    K_remain in (256, 511] -> remainK_MX=512=DepthU -> delta=0
    (no-op); K_remain in [1, 256] -> remainK_MX=256 -> delta=256*1=256
    bytes (real shrink). For larger DepthU multiples of 256 the same
    pattern extends.

    Gated to:

      - at least one MX side present (``MXBlockA > 0`` OR
        ``MXBlockB > 0``);
      - ``DepthU > 256`` (otherwise static no-op);
      - integer ``bytesPerKElement_MX`` (non-integer would require a
        runtime SMul which no current MX config needs).

    No SRD restore is needed (same reason as the bf16 path: tail GR
    is the last MXSA/MXSB read site before epilogue).
    """
    module = Module("tailSrdTightenSubtileMX")
    hasMxA = kernel["ProblemType"].get("MXBlockA", 0) > 0
    hasMxB = kernel["ProblemType"].get("MXBlockB", 0) > 0
    if not (hasMxA or hasMxB):
        return module
    depthU = int(kernel["DepthU"])
    if depthU <= MX_PAD_K:
        return module
    # Derive bytesPerKElement_MX explicitly as `swizzleSize0 / mxBlock`,
    # matching `computeLoadSrd()`'s K-direction stride. This replaces
    # the previously implicit `lrSubtileSize * lrGlobalSubtileGrid[1]
    # / DepthU` derivation, which gave the same numerical value for
    # the gauntlet configs but obscured the SRD-byte intent (review
    # feedback). Cross-check against the legacy derivation is kept
    # below in debug-only form so a divergence (e.g. a future subtile
    # state change) is caught loudly rather than silently mis-clipping.
    mxOperands = []
    if hasMxA:
        mxOperands.append(('MXSA', kw.states.mxsa.tileInfo))
    if hasMxB:
        mxOperands.append(('MXSB', kw.states.mxsb.tileInfo))
    bytesPerKList = []
    for tc, ti in mxOperands:
        swizzleSize0 = _swizzleSize0ForMN(kernel, tc)
        tcab = "A" if tc == "MXSA" else "B"
        mxBlock = int(kernel["ProblemType"].get("MXBlock%s" % tcab, 0))
        if mxBlock <= 0 or swizzleSize0 <= 0:
            return module
        if swizzleSize0 % mxBlock != 0:
            # Non-integer bytesPerKElement (e.g. swizzleSize0=32 with
            # mxBlock > 32). No current MX config hits this; bail
            # rather than emit a runtime divide.
            return module
        bytesPerKElement = swizzleSize0 // mxBlock
        # Cross-check: the legacy derivation must agree for any
        # config where `lrSubtileSize * lrGlobalSubtileGrid[1]` is
        # populated (it is in all subtile MX paths). If subtile state
        # is absent, skip the assertion; if present, mismatch raises.
        legacyBytesPerDU = int(ti.lrSubtileSize * ti.lrGlobalSubtileGrid[1])
        if legacyBytesPerDU > 0 and legacyBytesPerDU % depthU == 0:
            legacyBytesPerK = legacyBytesPerDU // depthU
            assert legacyBytesPerK == bytesPerKElement, (
                "MX SRD-tighten K-stride mismatch for %s: explicit "
                "swizzleSize0/mxBlock = %d/%d = %d but legacy "
                "lrSubtileSize*lrGlobalSubtileGrid[1]/DepthU = %d/%d "
                "= %d. The SRD-advance and SRD-tighten byte conventions "
                "must agree; investigate before relaxing this check."
                % (tc, swizzleSize0, mxBlock, bytesPerKElement,
                   legacyBytesPerDU, depthU, legacyBytesPerK))
        bytesPerKList.append(bytesPerKElement)
    if any(b != bytesPerKList[0] for b in bytesPerKList):
        return module
    bytesPerKElement = bytesPerKList[0]

    module.addComment2(
        "Tighten SrdMXS<tc>+2 for K_remain (MX K-pad=%u, DepthU=%u; "
        "OOR clip MX scale follow-up)" % (MX_PAD_K, depthU))
    with kw.allocTmpSgpr(2) as tmpInfo:
        remKMx = tmpInfo.idx
        delta = tmpInfo.idx + 1
        # remKMx = roundUp(LoopCounterL, MX_PAD_K)
        #        = (LoopCounterL + MX_PAD_K - 1) & ~(MX_PAD_K - 1)
        padMaskInv = (~(MX_PAD_K - 1)) & 0xffffffff
        module.add(SAddU32(
            dst=sgpr(remKMx), src0=sgpr("LoopCounterL"), src1=(MX_PAD_K - 1),
            comment="K_remain + (MX_pad_K - 1) for roundUp"))
        module.add(SAndB32(
            dst=sgpr(remKMx), src0=sgpr(remKMx), src1=hex(padMaskInv),
            comment="remainK_MX = roundUp(K_remain, %u)" % MX_PAD_K))
        # delta_K provably >= 0: remainK_MX <= roundUp(DepthU-1, 256)
        # <= DepthU for any DepthU that is a multiple of 256 > 256.
        module.add(SSubU32(
            dst=sgpr(delta), src0=depthU, src1=sgpr(remKMx),
            comment="delta_K = DepthU - remainK_MX (>= 0)"))
        if bytesPerKElement != 1:
            module.add(SLShiftLeftB32(
                dst=sgpr(delta), src=sgpr(delta),
                shiftHex=hex(int(math.log2(bytesPerKElement))),
                comment="delta_bytes = delta_K * bytesPerKElement_MX (=%u)"
                        % bytesPerKElement))
        for tc, _ in mxOperands:
            module.add(SSubU32(
                dst=sgpr("Srd%s+2" % tc), src0=sgpr("Srd%s+2" % tc), src1=sgpr(delta),
                comment="Srd%s+2 -= delta (clip K past remainK_MX)" % tc))
    return module


def emitTailSrdTightenSubtileMXData(kw, kernel):
    """Tighten ``Srd{A,B}+2`` for the MX **data** tensors when
    ``DepthU > MX_PAD_K`` (=256). Companion to
    :func:`emitTailSrdTightenSubtileMX` which clips
    ``SrdMXS{A,B}+2``. The data side needs the same K=256-padded clip
    as the scale side -- otherwise the natural DepthU-shaped data
    over-read on the last m-row can fault past the data tensor's
    allocated bytes (the per-lane mask + 0-scale absorb keeps the
    MFMA result correct under garbage data, but garbage reads can
    still page fault past the data buffer).

    Formula::

        remainK_MX = roundUp(K_remain, 256)                # K elements
        delta_K    = DepthU - remainK_MX                   # K elements, >= 0
        Srd{A,B}+2 -= delta_K * bpe_data                   # bytes

    Static gate -- emit nothing when ``DepthU <= MX_PAD_K=256``.

    For MX BF/FP4 / FP8 the per-K-element data bpe is 0.5 / 1
    respectively; both can be expressed as ``bpe = 1 / (1 << shr)``
    with ``shr in {0, 1}`` (fp8: shr=0; fp4: shr=1). The delta-bytes
    SShiftRight is therefore a fixed compile-time shift.

    Gated to:

      - at least one MX side present;
      - non-swizzled A/B on the MX sides we tighten (swizzle adds a
        per-block stride that the simple ``delta_K * bpe`` formula
        does not model; the swizzled-MX clip is deferred);
      - ``DepthU > 256``;
      - per-MX-operand bpe in {0.5, 1}.

    Swizzle-size cross-reference (see :func:`_swizzleSize0ForMN`):
    this helper handles the ``swizzleSize0 == 1`` case for the MX
    DATA tensor side (non-swizzled). The companion path for
    PreShuffled MX data A/B (``swizzleSize0 == 16``) would use
    ``delta_bytes = delta_K * swizzleSize0`` (the bpe is already
    absorbed into ``swizzleBlockSize``); that path is gated out
    below and deferred -- no current gauntlet config exercises it,
    and emitting the swizzled-MX-data clip without runtime coverage
    would risk over- vs under-tightening regressions.
    """
    module = Module("tailSrdTightenSubtileMXData")
    hasMxA = kernel["ProblemType"].get("MXBlockA", 0) > 0
    hasMxB = kernel["ProblemType"].get("MXBlockB", 0) > 0
    if not (hasMxA or hasMxB):
        return module
    depthU = int(kernel["DepthU"])
    if depthU <= MX_PAD_K:
        return module

    def _bpeShiftRight(bpe):
        # `bpe = 1.0 / (1 << shr)` for shr in {0, 1}. bpe>1 (or non
        # power-of-two inverse) bails -- MX data is never bpe>1
        # today, and no shr fits non-powers-of-two cleanly.
        if bpe >= 1.0 and abs(bpe - int(bpe)) < 1e-9:
            return 0 if int(bpe) == 1 else None
        inv = 1.0 / bpe
        if abs(inv - int(round(inv))) > 1e-9:
            return None
        inv = int(round(inv))
        if inv <= 0 or (inv & (inv - 1)) != 0:
            return None
        return inv.bit_length() - 1

    mxOperands = []
    if hasMxA:
        if kernel["ProblemType"].get("SwizzleTensorA", False):
            return module
        shrA = _bpeShiftRight(float(kw.states.a.tileInfo.bpe))
        if shrA is None:
            return module
        mxOperands.append(('A', shrA))
    if hasMxB:
        if kernel["ProblemType"].get("SwizzleTensorB", False):
            return module
        shrB = _bpeShiftRight(float(kw.states.b.tileInfo.bpe))
        if shrB is None:
            return module
        mxOperands.append(('B', shrB))
    if not mxOperands:
        return module

    module.addComment2(
        "Tighten Srd<tc>+2 for MX data K_remain (MX K-pad=%u, DepthU=%u; "
        "OOR clip MX data follow-up)" % (MX_PAD_K, depthU))
    with kw.allocTmpSgpr(2) as tmpInfo:
        remKMx = tmpInfo.idx
        delta = tmpInfo.idx + 1
        padMaskInv = (~(MX_PAD_K - 1)) & 0xffffffff
        module.add(SAddU32(
            dst=sgpr(remKMx), src0=sgpr("LoopCounterL"), src1=(MX_PAD_K - 1),
            comment="K_remain + (MX_pad_K - 1) for roundUp"))
        module.add(SAndB32(
            dst=sgpr(remKMx), src0=sgpr(remKMx), src1=hex(padMaskInv),
            comment="remainK_MX = roundUp(K_remain, %u)" % MX_PAD_K))
        module.add(SSubU32(
            dst=sgpr(delta), src0=depthU, src1=sgpr(remKMx),
            comment="delta_K = DepthU - remainK_MX (>= 0)"))
        # Uniform-shr fast path (all current mxfp4/mxfp4 and
        # mxfp8/mxfp8 configs): one shift, then SSub both SRDs from
        # the same delta_bytes. Mixed shr falls back to per-operand
        # SSubs from per-operand shifted deltas.
        uniformShr = mxOperands[0][1] if all(
            s == mxOperands[0][1] for _, s in mxOperands) else None
        if uniformShr is not None:
            if uniformShr > 0:
                module.add(SLShiftRightB32(
                    dst=sgpr(delta), src=sgpr(delta), shiftHex=hex(uniformShr),
                    comment="delta_bytes = delta_K * bpe_data (bpe=1/%u)"
                            % (1 << uniformShr)))
            for tc, _ in mxOperands:
                module.add(SSubU32(
                    dst=sgpr("Srd%s+2" % tc), src0=sgpr("Srd%s+2" % tc),
                    src1=sgpr(delta),
                    comment="Srd%s+2 -= delta (clip MX data past remainK_MX)" % tc))
        else:
            # Mixed bpe across A and B is not currently realized in
            # any gauntlet config; per-operand path kept for
            # completeness.
            for tc, shr in mxOperands:
                if shr > 0:
                    module.add(SLShiftRightB32(
                        dst=sgpr(remKMx), src=sgpr(delta), shiftHex=hex(shr),
                        comment="Srd%s delta_bytes = delta_K * bpe_data (bpe=1/%u)"
                                % (tc, 1 << shr)))
                    srcDelta = remKMx
                else:
                    srcDelta = delta
                module.add(SSubU32(
                    dst=sgpr("Srd%s+2" % tc), src0=sgpr("Srd%s+2" % tc),
                    src1=sgpr(srcDelta),
                    comment="Srd%s+2 -= delta (clip MX data past remainK_MX)" % tc))
    return module
