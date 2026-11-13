# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""ScalarLoadB: the B operand of f32 source (MAC) kernels comes from SGPRs.

When SubGroup0 is a multiple of the wavefront size, every lane of a wave has the same tile-1
index, so the B elements a wave multiplies with are wave-uniform. Each wave then reads them with
s_load straight into SGPRs and the MACs take B as a scalar operand. B needs no LDS, no VGPRs and
no VGPR read port, which is what lets v_dual_fmac_f32 run with two VGPR reads per pair.

B is consumed as a stream of slices in summation order. A slice is `chunk` summation iterations
of all ThreadTile1 columns. The SGPR ring holds ScalarLoadB iterations (`depth // chunk` slices);
a slot is refilled with the slice `depth` iterations ahead at the iteration that first uses the
slice after it, i.e. right after its last use.
"""

from math import log2

from rocisa.code import Module, Label, RegSet, ValueSet
from rocisa.register import RegisterPool
from rocisa.container import sgpr, vgpr, VCC
from rocisa.instruction import FlatLoadB32, SAddU32, SAddCU32, SCBranchSCC1, SCmpLeU32, SLoadB32, \
    SLoadB64, SLoadB128, SLoadB256, SLoadB512, SLShiftLeftB32, SMinU32, SMovB32, SMulI32, SMulHIU32, SSubU32, \
    SWaitCnt, VAddCCOU32, VAddCOU32, VAddU32, VAndB32, VLShiftRightB32, VMovB32, VMulLOU32, VReadfirstlaneB32

_SLOAD = {1: SLoadB32, 2: SLoadB64, 4: SLoadB128, 8: SLoadB256, 16: SLoadB512}


def scalarLoadBChunk(state):
    """Summation iterations fetched together by ScalarLoadB (0 if disabled).

    The loads of one chunk are issued back to back, so their cache misses overlap, and the ring
    holds at least two chunks, so a chunk is fetched while the previous one is consumed.
    """
    depth = state["ScalarLoadB"]
    if not depth:
        return 0
    return min(4, max(1, depth // 2))


def _loadWidth(kernel) -> int:
    """Elements per scalar load when B is contiguous along the tile index."""
    return min(kernel["VectorWidthB"], 4)


def _geometry(kernel):
    depth = kernel["ScalarLoadB"]
    chunk = scalarLoadBChunk(kernel)
    return depth, chunk, depth // chunk, kernel["ThreadTile1"], kernel["VectorWidthB"]


def _column(kernel, c: int) -> int:
    """Column of ValuC tile-1 index c relative to the first column of the thread."""
    vw = kernel["VectorWidthB"]
    return (c // vw) * kernel["SubGroup1"] * vw + c % vw


def _groups(kernel):
    """SGPR groups of the ring: (number of groups, registers per group). One scalar load fills one group."""
    depth, chunk, slots, tt1, _ = _geometry(kernel)
    if kernel["ProblemType"]["TLUB"]:
        lw = _loadWidth(kernel)
        return depth * (tt1 // lw), lw      # per summation iteration: lw adjacent columns
    return tt1 * slots, chunk               # per column: chunk summation iterations


def _group(kernel, p: int, c: int) -> str:
    """Name of the SGPR group holding ring position p (summation iteration) of tile-1 index c."""
    _, chunk, slots, tt1, _ = _geometry(kernel)
    if kernel["ProblemType"]["TLUB"]:
        lw = _loadWidth(kernel)
        return "ScalarB%u" % (p * (tt1 // lw) + c // lw)
    return "ScalarB%u" % (c * slots + p // chunk)


def operand(kernel, u: int, c: int):
    """SGPR holding B of summation iteration u (within the unrolled block) and tile-1 index c."""
    depth, chunk, _, _, _ = _geometry(kernel)
    p = u % depth
    if kernel["ProblemType"]["TLUB"]:
        return sgpr("%s+%u" % (_group(kernel, p, c), c % _loadWidth(kernel)))
    return sgpr("%s+%u" % (_group(kernel, p, c), p % chunk))


def _loadSlice(writer, kernel, slot: int, comment: str) -> Module:
    """Load the next slice of the stream into ring slot `slot` and advance the stream pointer."""
    depth, chunk, _, tt1, vwb = _geometry(kernel)
    bpe = 4
    module = Module("ScalarLoadB slice")
    ptr = sgpr("ScalarPtrB", 2)
    if kernel["ProblemType"]["TLUB"]:
        # Per summation iteration: the columns come in runs of VectorWidthB adjacent elements.
        lw = _loadWidth(kernel)
        for p in range(slot * chunk, (slot + 1) * chunk):
            for c in range(0, tt1, lw):
                module.add(_SLOAD[lw](dst=sgpr(_group(kernel, p, c), lw), base=ptr,
                                      soffset=hex(_column(kernel, c) * bpe), comment=comment))
            module.add(SAddU32(dst=sgpr("ScalarPtrB+0"), src0=sgpr("ScalarPtrB+0"), src1=sgpr("ScalarStrideB"),
                               comment="next summation iteration of B"))
            module.add(SAddCU32(dst=sgpr("ScalarPtrB+1"), src0=sgpr("ScalarPtrB+1"), src1=0))
    else:
        # Per column: `chunk` summation iterations.
        for c in range(tt1):
            soffset = sgpr("ScalarColOffB%u" % c) if c else hex(0)
            module.add(_SLOAD[chunk](dst=sgpr(_group(kernel, slot * chunk, c), chunk), base=ptr, soffset=soffset,
                                     comment=comment))
        module.add(SAddU32(dst=sgpr("ScalarPtrB+0"), src0=sgpr("ScalarPtrB+0"), src1=chunk * bpe, comment="next B slice"))
        module.add(SAddCU32(dst=sgpr("ScalarPtrB+1"), src0=sgpr("ScalarPtrB+1"), src1=0))
    return module


def init(writer, kernel, tP) -> Module:
    """Stream pointer of this wave and the first slices. Requires SrdB of the new tile.

    The SGPRs are defined here, not with the other loop SGPRs: the B address computation of the
    buffer-load path is dead once the stream pointer exists, and the ring reuses its registers.
    All of them are released with the other loop SGPRs at the end of the summation.
    """
    depth, chunk, slots, tt1, vwb = _geometry(kernel)
    bpe = 4
    tlu = kernel["ProblemType"]["TLUB"]
    module = Module("ScalarLoadB init")
    module.addComment0("ScalarLoadB: per-wave B stream pointer")
    tileStride = writer.strideRef("B", tP["tileIdx"])   # stride between columns
    sumStride = writer.strideRef("B", kernel["ProblemType"]["IndicesSummation"][writer.states.unrollIdx])
    module.add(writer.defineSgpr("ScalarPtrB", 2, 2))
    tmpV = writer.vgprPool.checkOut(1, tag="ScalarLoadB_tid1")
    with writer.allocTmpSgpr(2, tag="ScalarLoadB_init") as tmpInfo:
        tmp = tmpInfo.idx
        colStride = sgpr(tmp + 1)   # bytes between columns (B contiguous along the summation index)
        # First column of the wave: tid1 * VectorWidthB (tid1 = Serial / SubGroup0, wave-uniform).
        module.add(VLShiftRightB32(dst=vgpr(tmpV), shiftHex=hex(int(log2(kernel["SubGroup0"]))),
                                   src=vgpr("Serial"), comment="tid1"))
        module.add(VReadfirstlaneB32(dst=sgpr(tmp), src=vgpr(tmpV), comment="tid1 is the same for every lane"))
        prepad = writer.states.srdShiftLeft["B"] * bpe  # SrdB base sits this far below the tile
        if tlu:
            module.add(SMulI32(dst=sgpr(tmp), src0=sgpr(tmp), src1=vwb * bpe, comment="first column (bytes)"))
            module.add(SAddU32(dst=sgpr(tmp), src0=sgpr(tmp), src1=prepad, comment="undo SRD pre-pad"))
            module.add(SAddU32(dst=sgpr("ScalarPtrB+0"), src0=sgpr("SrdB+0"), src1=sgpr(tmp)))
            module.add(SAddCU32(dst=sgpr("ScalarPtrB+1"), src0=sgpr("SrdB+1"), src1=0))
            module.add(SLShiftLeftB32(dst=colStride, src=sumStride, shiftHex=hex(2),
                                      comment="bytes between summation iterations"))
        else:
            module.add(SLShiftLeftB32(dst=colStride, src=tileStride, shiftHex=hex(2), comment="column stride (bytes)"))
            if vwb > 1:
                module.add(SMulI32(dst=sgpr(tmp), src0=sgpr(tmp), src1=vwb, comment="first column"))
            module.add(SMulHIU32(dst=sgpr("ScalarPtrB+1"), src0=sgpr(tmp), src1=colStride))
            module.add(SMulI32(dst=sgpr(tmp), src0=sgpr(tmp), src1=colStride, comment="first column (bytes)"))
            module.add(SAddU32(dst=sgpr(tmp), src0=sgpr(tmp), src1=prepad, comment="undo SRD pre-pad"))
            module.add(SAddCU32(dst=sgpr("ScalarPtrB+1"), src0=sgpr("ScalarPtrB+1"), src1=0))
            module.add(SAddU32(dst=sgpr("ScalarPtrB+0"), src0=sgpr("SrdB+0"), src1=sgpr(tmp)))
            module.add(SAddCU32(dst=sgpr("ScalarPtrB+1"), src0=sgpr("ScalarPtrB+1"), src1=sgpr("SrdB+1")))
        module.add(prefetchInit(writer, kernel, colStride))
        for name in _DEAD_AFTER_SETUP:
            if _inUse(writer, name):
                module.add(_release(writer, name))
        # The ring takes the aligned blocks first; the single registers then fill the gaps.
        numGroups, size = _groups(kernel)
        for g in range(numGroups):
            module.add(writer.defineSgpr("ScalarB%u" % g, size, min(size, 4)))
        if tlu:
            module.add(writer.defineSgpr("ScalarStrideB", 1))
            module.add(SMovB32(dst=sgpr("ScalarStrideB"), src=colStride))
        else:
            for c in range(1, tt1):
                module.add(writer.defineSgpr("ScalarColOffB%u" % c, 1))
                module.add(SMulI32(dst=sgpr("ScalarColOffB%u" % c), src0=colStride, src1=_column(kernel, c),
                                   comment="byte offset of column %u" % c))
    writer.vgprPool.checkIn(tmpV)
    for slot in range(slots - 1):
        module.add(_loadSlice(writer, kernel, slot, "B slice %u" % slot))
    return module


def prefetchAfterGlobalRead(kernel) -> bool:
    """True if the cache-warming load is issued right after the block's global reads.

    A local write that waits for its global reads must not also wait for the warming load, whose
    data is only needed blocks later: the load has to be the newest vector load at that wait. That
    is known when the reads all sit at the top of the block (unscheduled PrefetchGlobalRead=1,
    or PrefetchGlobalRead=2 with two G2L sets). Otherwise PrefetchGlobalRead=2 issues each read
    after its local write, and the load goes to the top of the block.
    """
    if not kernel.get("ScalarLoadBPrefetch"):
        return False
    return bool(kernel.get("PGR2DoubleG2L")) or (
        kernel["PrefetchGlobalRead"] == 1 and not kernel["ScheduleGlobalRead"] and not kernel["ScheduleLocalWrite"])


def prefetchInit(writer, kernel, stride) -> Module:
    """ScalarLoadBPrefetch: per-lane address of the cache-warming load of block 0.

    The scalar loads of a wave hit DRAM one after the other. One vector load per unrolled block,
    with the lanes spread over the columns of the wave, pulls the lines the scalar loads will need
    into the cache ahead of time. Its result is discarded. The load runs D blocks ahead, clamped
    to the last block. Call before the first slices are loaded. `stride` holds the bytes between
    columns (or between summation iterations if B is contiguous along the tile index).
    """
    dist = kernel["ScalarLoadBPrefetch"]
    module = Module("ScalarLoadB prefetch init")
    if not dist:
        return module
    _, _, _, tt1, vwb = _geometry(kernel)
    tlu = kernel["ProblemType"]["TLUB"]
    addr = writer.states.startVgprScalarLoadBPrefetch
    lo, hi, tmpV = vgpr(addr), vgpr(addr + 1), vgpr(addr + 2)
    groups = tt1 // vwb if tlu else tt1          # distinct lines: column groups (TLU) or columns
    module.addComment0("ScalarLoadB: address of the cache-warming load")
    with writer.allocTmpSgpr(1, tag="ScalarLoadB_prefetch") as tmpInfo:
        tmp = sgpr(tmpInfo.idx)
        # lane -> tile-1 index it warms -> column -> byte offset
        module.add(VAndB32(dst=lo, src0=vgpr("Serial"), src1=groups - 1, comment="lane %% %u" % groups))
        if tlu:
            unit = kernel["SubGroup1"] * vwb * 4        # bytes between column groups
            module.add(VMulLOU32(dst=lo, src0=lo, src1=unit, comment="column group (bytes)"))
        else:
            if vwb > 1:
                module.add(VLShiftRightB32(dst=tmpV, shiftHex=hex(int(log2(vwb))), src=lo))
                module.add(VMulLOU32(dst=tmpV, src0=tmpV, src1=kernel["SubGroup1"] * vwb - vwb))
                module.add(VAddU32(dst=lo, src0=lo, src1=tmpV, comment="column"))
            module.add(VMulLOU32(dst=lo, src0=lo, src1=stride, comment="column (bytes)"))
        loopCounter = writer.loopCounter(kernel, writer.states.unrollIdx)
        module.add(SSubU32(dst=tmp, src0=loopCounter, src1=1, comment="last block"))
        module.add(SMinU32(dst=tmp, src0=tmp, src1=dist, comment="%u blocks ahead" % dist))
        module.add(SMulI32(dst=tmp, src0=tmp, src1=kernel["LoopIters"] * (1 if tlu else 4)))
        if tlu:
            module.add(SMulI32(dst=tmp, src0=tmp, src1=stride))
        module.add(VAddU32(dst=lo, src0=lo, src1=tmp))
        module.add(VMovB32(dst=hi, src=sgpr("ScalarPtrB+1")))
        module.add(VAddCOU32(dst=lo, dst1=VCC(), src0=sgpr("ScalarPtrB+0"), src1=lo))
        module.add(VAddCCOU32(dst=hi, dst1=VCC(), src0=hi, src1=0, src2=VCC()))
    return module


def prefetch(writer, kernel) -> Module:
    """Cache-warming load of one unrolled block (main loop only) and the step to the next block.

    Every block issues exactly one load, so waits can count it. The address stops at the last block.
    """
    dist = kernel["ScalarLoadBPrefetch"]
    module = Module("ScalarLoadB prefetch")
    if not dist:
        return module
    addr = writer.states.startVgprScalarLoadBPrefetch
    lo, hi, tmpV = vgpr(addr), vgpr(addr + 1), vgpr(addr + 2)
    skip = Label(writer.labels.getNameInc("ScalarLoadBPrefetchEnd"), "")
    loopCounter = writer.loopCounter(kernel, writer.states.unrollIdx)
    module.add(FlatLoadB32(dst=tmpV, vaddr=vgpr(addr, 2), comment="warm the cache for scalar B (result unused)"))
    module.add(SCmpLeU32(src0=loopCounter, src1=dist + 1, comment="is the next target past the end?"))
    module.add(SCBranchSCC1(labelName=skip.getLabelName()))
    if kernel["ProblemType"]["TLUB"]:
        with writer.allocTmpSgpr(1, tag="ScalarLoadB_prefetch_step") as tmpInfo:
            module.add(SMulI32(dst=sgpr(tmpInfo.idx), src0=sgpr("ScalarStrideB"), src1=kernel["LoopIters"]))
            module.add(VAddCOU32(dst=lo, dst1=VCC(), src0=sgpr(tmpInfo.idx), src1=lo, comment="next block"))
    else:
        module.add(VAddCOU32(dst=lo, dst1=VCC(), src0=kernel["LoopIters"] * 4, src1=lo, comment="next block"))
    module.add(VAddCCOU32(dst=hi, dst1=VCC(), src0=hi, src1=0, src2=VCC()))
    module.add(skip)
    return module


# SGPRs that are only read again by the store code. Their values wait in VGPRs during the
# summation so that the ring can use their registers. Any use in between fails to assemble
# (the symbol is undefined), so a missed case cannot silently read a B value.
_PARKED = ("Alpha", "Beta", "AddressD", "AddressC", "StridesD", "StridesC")
# Not read after the tile setup; released for the ring.
_DEAD_AFTER_SETUP = ("SrdB", "ShadowLimitB", "GlobalReadIncsB", "AddressA", "AddressB", "StridesA",
                     "StridesB", "WGM", "OrigLoopCounter", "StaggerU")


def _inUse(writer, name: str) -> int:
    """Number of in-use SGPRs carrying this name."""
    if name not in writer.sgprs:
        return 0
    return sum(1 for r in writer.sgprPool.getPool()
               if r.tag == name and r.status == RegisterPool.Status.InUse)


def _undefAliases(writer, name: str) -> Module:
    """Aliases keep the register they were defined with: undefine them with the SGPR."""
    module = Module("undef aliases of %s" % name)
    for alias, _ in writer.states.sgprAliases.get(name, []):
        module.add(ValueSet(name="sgpr" + alias, value="UNDEF", format=-1))
    return module


def _release(writer, name: str) -> Module:
    """Give up an SGPR for good, together with its aliases."""
    module = Module("release %s" % name)
    module.add(_undefAliases(writer, name))
    module.add(writer.undefineSgpr(name))
    return module


def numParkVgprs(kernel) -> int:
    pt = kernel["ProblemType"]
    scalar = pt["ComputeDataType"].numRegisters()
    return 2 * scalar + 2 * 2 + 2 * (pt["NumIndicesC"] - 1)


def park(writer, kernel) -> Module:
    """Move the store-only SGPRs to VGPRs. Emit on a path every wave takes."""
    module = Module("ScalarLoadB park")
    module.addComment0("ScalarLoadB: keep store arguments in VGPRs during the summation")
    v = writer.states.startVgprScalarLoadBPark
    parked = []
    for name in _PARKED:
        size = _inUse(writer, name)
        if not size:
            continue
        for i in range(size):
            module.add(VMovB32(dst=vgpr(v), src=sgpr("%s+%u" % (name, i)), comment="park %s" % name))
            v += 1
        parked.append((name, size))
        module.add(_undefAliases(writer, name))
        module.add(writer.setSgprToFreeState(name))
    assert v - writer.states.startVgprScalarLoadBPark <= numParkVgprs(kernel)
    writer.states.scalarLoadBParked = parked
    return module


def unpark(writer, kernel) -> Module:
    """Bring the parked SGPRs back once the ring is released."""
    module = Module("ScalarLoadB unpark")
    v = writer.states.startVgprScalarLoadBPark
    for name, size in writer.states.scalarLoadBParked:
        # the same registers as before: aliases such as StrideD1J were resolved against them
        module.add(writer.setSgprToInUseState(name))
        for alias, offset in writer.states.sgprAliases.get(name, []):
            module.add(RegSet("s", "sgpr" + alias, "sgpr" + name, offset))
        for i in range(size):
            module.add(VReadfirstlaneB32(dst=sgpr("%s+%u" % (name, i)), src=vgpr(v), comment="unpark %s" % name))
            v += 1
    writer.states.scalarLoadBParked = []
    return module


def iterLoads(writer, kernel, u: int, lastBlock: bool) -> Module:
    """Loads issued before the MACs of summation iteration u of an unrolled block."""
    depth, chunk, slots, _, _ = _geometry(kernel)
    module = Module("ScalarLoadB iter %u" % u)
    if u % chunk != 0:
        return module
    slice_ = u // chunk + slots - 1                  # relative to the first slice of this block
    if lastBlock and slice_ >= kernel["LoopIters"] // chunk:
        return module                                # past the end of the summation range
    module.add(_loadSlice(writer, kernel, slice_ % slots, "B slice +%u" % slice_))
    return module


def iterWait(writer, kernel, u: int) -> Module:
    """The slice first used by iteration u was loaded earlier: it has to have landed."""
    _, chunk, _, _, _ = _geometry(kernel)
    module = Module("ScalarLoadB wait %u" % u)
    if u % chunk == 0:
        module.add(SWaitCnt(dscnt=0, comment="wait for scalar B (SMEM returns out of order)"))
    return module
