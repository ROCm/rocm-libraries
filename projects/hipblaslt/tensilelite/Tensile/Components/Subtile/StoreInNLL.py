# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Correctness guards and code generation for folding D stores into NLL."""

from dataclasses import dataclass
from collections import deque
import re
from typing import Any, Iterable, List, Optional

from rocisa.code import Module
from rocisa.container import MUBUFModifiers, accvgpr, sgpr, vgpr
from rocisa.instruction import (
    BufferStoreB128,
    VAccvgprReadB32,
    VCvtPkF32toBF16,
    VAddU32,
    VAddLShiftLeftU32,
    VLShiftLeftB32,
    VMulF32,
    VMulLOU32,
)

_MFMA_TILE_RE = re.compile(r"MFMA C\[(\d+),(\d+)\]")


def subtileStoreInNLLRejectionReason(state: dict) -> Optional[str]:
    """Return why ``SubtileStoreInNLL`` is inadmissible, or ``None``."""
    if not state.get("SubtileStoreInNLL", 0):
        return None

    pt = state.get("ProblemType", {})
    dest_type = pt.get("DestDataType")
    dest_is_bf16 = (dest_type == 7
                    or getattr(dest_type, "isBFloat16", lambda: False)())
    checks = (
        (tuple(state.get("ISA", ())) == (9, 5, 0), "SubtileStoreInNLL requires gfx950"),
        (state.get("UseSubtileImpl", False), "SubtileStoreInNLL requires UseSubtileImpl=1"),
        (state.get("SourceSwap", False), "SubtileStoreInNLL requires SourceSwap=1"),
        (not state.get("CompactLoopStore", False),
         "SubtileStoreInNLL is incompatible with CompactLoopStore"),
        (state.get("StreamK", 0) == 0, "SubtileStoreInNLL requires StreamK=0"),
        (state.get("GlobalSplitU", 1) == 1, "SubtileStoreInNLL requires GlobalSplitU=1"),
        (state.get("MIWaveTile") == [8, 8],
         "SubtileStoreInNLL requires MIWaveTile=[8,8]"),
        (not state.get("MIArchVgpr", False),
         "SubtileStoreInNLL requires accumulator registers"),
        (pt.get("HighPrecisionAccumulate", False),
         "SubtileStoreInNLL requires HighPrecisionAccumulate=1"),
        (dest_is_bf16,
         "SubtileStoreInNLL requires BF16 destination"),
        (not pt.get("UseE", False), "SubtileStoreInNLL does not support E output"),
        (not pt.get("UseScaleD", False) and not pt.get("UseScaleCD", False),
         "SubtileStoreInNLL does not support scaleD"),
        (not pt.get("OutputAmaxD", False), "SubtileStoreInNLL does not support amaxD"),
        (bool(state.get("NonTemporalD", 0) & 0x4),
         "SubtileStoreInNLL requires NonTemporalD bit 0x4"),
        (state.get("AssertFree0ElementMultiple", 1) % state.get("MacroTile0", 1) == 0,
         "SubtileStoreInNLL requires full MacroTile-aligned M"),
        (state.get("AssertFree1ElementMultiple", 1) % state.get("MacroTile1", 1) == 0,
         "SubtileStoreInNLL requires full MacroTile-aligned N"),
    )
    for accepted, reason in checks:
        if not accepted:
            return reason
    return None


@dataclass(frozen=True)
class NLLStoreEvent:
    """One abstract NLL issue used by :func:`planNLLStoreInterleave`."""

    kind: str
    accumulator: Optional[Any] = None
    payload: Any = None


def planNLLStoreInterleave(
    mfmaAccumulators: Iterable[Any],
    stores: Iterable[NLLStoreEvent],
    minMfmaGap: int = 4,
) -> List[NLLStoreEvent]:
    """Place each store after its accumulator's final producer.

    A store is delayed by ``minMfmaGap`` later MFMA issues when the NLL has that
    much remaining work.  Near the end, it is drained immediately after the
    final available MFMA, so every store remains inside NLL.  The routine is
    intentionally independent of rocISA instruction classes and can schedule
    either direct-AGPR stores or a conversion-plus-store payload.
    """
    if minMfmaGap < 0:
        raise ValueError("minMfmaGap must be non-negative")

    mfmas = list(mfmaAccumulators)
    storeList = list(stores)
    lastProducer = {acc: idx for idx, acc in enumerate(mfmas)}
    buckets = [[] for _ in range(len(mfmas) + 1)]

    for store in storeList:
        if store.kind != "store":
            raise ValueError("stores must contain only kind='store' events")
        if store.accumulator not in lastProducer:
            raise ValueError(f"no NLL producer for accumulator {store.accumulator!r}")
        producer = lastProducer[store.accumulator]
        issueAfter = min(producer + minMfmaGap, len(mfmas) - 1)
        buckets[issueAfter + 1].append(store)

    result: List[NLLStoreEvent] = []
    for idx, accumulator in enumerate(mfmas):
        result.append(NLLStoreEvent("mfma", accumulator))
        result.extend(buckets[idx + 1])
    return result


def nonTemporalDFlags(nonTemporalD: int) -> dict:
    """Return the MUBUF cache flags, preserving the D ``nt`` bit."""
    return {
        "glc": bool(nonTemporalD & 0x1),
        "slc": bool(nonTemporalD & 0x2),
        "nt": bool(nonTemporalD & 0x4),
    }


def emitSubtileStoreInNLLAddressInit(writer, kernel) -> Module:
    """Allocate the folded-store address/scratch VGPRs and form D's lane address."""
    module = Module("SubtileStoreInNLLAddressInit")
    if hasattr(writer.states, "subtileStoreInNLLVaddr"):
        return module

    vaddr = writer.vgprPool.checkOut(1, "subtileStoreInNLLVaddr")
    data = writer.vgprPool.checkOutAligned(8, 4, "subtileStoreInNLLData")
    stride = writer.vgprPool.checkOut(1, "subtileStoreInNLLStride")
    writer.states.subtileStoreInNLLVaddr = vaddr
    writer.states.subtileStoreInNLLData = data
    writer.states.subtileStoreInNLLStride = stride
    strideD1 = "StrideD%s" % writer.states.indexChars[kernel["PackedC1IndicesX"][0]]
    module.add(VMulLOU32(
        dst=vgpr(vaddr),
        src0=vgpr(writer.vgprs.coord1),
        src1=sgpr(strideD1),
        comment="folded-store global D column offset",
    ))
    module.add(VAddLShiftLeftU32(
        dst=vgpr(vaddr),
        src0=vgpr(vaddr),
        src1=vgpr(writer.vgprs.coord0),
        shiftHex=1,
        comment="folded-store D address = (column*strideD + row) * sizeof(bf16)",
    ))
    module.add(VLShiftLeftB32(
        dst=vgpr(stride),
        shiftHex=hex(1),
        src=sgpr(strideD1),
        comment="folded-store BF16 D column stride in bytes"))
    return module


def cleanupSubtileStoreInNLL(writer) -> None:
    """Release folded-store-only VGPRs after post-loop cleanup."""
    if hasattr(writer.states, "subtileStoreInNLLVaddr"):
        writer.vgprPool.checkIn(writer.states.subtileStoreInNLLVaddr)
        del writer.states.subtileStoreInNLLVaddr
    if hasattr(writer.states, "subtileStoreInNLLData"):
        writer.vgprPool.checkIn(writer.states.subtileStoreInNLLData)
        del writer.states.subtileStoreInNLLData
    if hasattr(writer.states, "subtileStoreInNLLStride"):
        writer.vgprPool.checkIn(writer.states.subtileStoreInNLLStride)
        del writer.states.subtileStoreInNLLStride


def _store_unit(writer, kernel, dtileInfo, column: int) -> list:
    """Return one eight-row wide BF16 store for a SourceSwap output column."""
    tiles_m = dtileInfo.localMMATileGrid[0]
    tile_n = column % dtileInfo.localMMATileGrid[1]
    component = column // dtileInfo.localMMATileGrid[1]
    regs = []
    for tile_m in range(tiles_m):
        tile = dtileInfo.vgprTiles[tile_m + tile_n * tiles_m]
        if tile.regList.pool == writer.vgprPool:
            raise RuntimeError("SubtileStoreInNLL does not support accumulator spill to VGPR")
        regs.append(tile.regList.indices[component])
    if len(regs) != 8:
        raise RuntimeError(f"SubtileStoreInNLL expected 8 accumulators, got {len(regs)}")

    flags = nonTemporalDFlags(kernel["NonTemporalD"])
    mubuf = lambda offset: MUBUFModifiers(
        offen=True, offset12=offset, glc=flags["glc"], slc=flags["slc"], nt=flags["nt"])
    vaddr = writer.states.subtileStoreInNLLVaddr
    data = writer.states.subtileStoreInNLLData
    insts = []

    for row, reg in enumerate(regs):
        insts.append(VAccvgprReadB32(
            dst=vgpr(data + row), src=accvgpr(reg),
            comment=f"fold C column {column} row {row}: acc -> vgpr"))
        insts.append(VMulF32(
            dst=vgpr(data + row), src0=sgpr("Alpha"), src1=vgpr(data + row),
            comment="apply alpha before folded BF16 store"))
    for pair in range(4):
        insts.append(VCvtPkF32toBF16(
            dst=vgpr(data + pair),
            src0=vgpr(data + 2 * pair),
            src1=vgpr(data + 2 * pair + 1),
            comment=f"pack folded BF16 rows {2 * pair}:{2 * pair + 2}"))
    insts.append(BufferStoreB128(
        src=vgpr(data, 4),
        vaddr=vgpr(vaddr),
        saddr=sgpr("SrdD", 4),
        soffset=0,
        mubuf=mubuf(0),
        comment=f"folded wide non-temporal D store column {column}"))

    if column != 31:
        insts.append(VAddU32(
            dst=vgpr(vaddr), src0=vgpr(vaddr),
            src1=vgpr(writer.states.subtileStoreInNLLStride),
            comment="advance folded-store address by one BF16 D column"))
    return insts


def interleaveSubtileStoreInNLL(
    module: Module,
    writer,
    kernel,
    dtileInfo,
    minMfmaGap: int = 4,
    issueRate: int = 6,
) -> Module:
    """Pace completed output-column stores through a scheduled NLL MFMA stream."""
    if not kernel.get("SubtileStoreInNLL", 0):
        return module

    items = list(module.flatitems())
    mfmas = []
    for item_index, item in enumerate(items):
        match = _MFMA_TILE_RE.search(str(item))
        if match:
            mfmas.append((item_index, int(match.group(1)), int(match.group(2))))
    if not mfmas:
        raise RuntimeError("SubtileStoreInNLL found no NLL MFMA producers")

    last_producer = {}
    for seq, (_, tile_m, tile_n) in enumerate(mfmas):
        last_producer[(tile_m, tile_n)] = seq
    last_seq = len(mfmas) - 1

    units = []
    for column in range(32):
        tile_n = column % dtileInfo.localMMATileGrid[1]
        producer = max(last_producer[(tile_m, tile_n)]
                       for tile_m in range(dtileInfo.localMMATileGrid[0]))
        release = min(producer + minMfmaGap, last_seq)
        units.append((release, deque(
            _store_unit(writer, kernel, dtileInfo, column))))

    out = Module(module.name)
    pending = deque()
    next_unit = 0
    mfma_seq = -1
    for item in items:
        out.add(item)
        if not _MFMA_TILE_RE.search(str(item)):
            continue
        mfma_seq += 1
        while next_unit < len(units) and units[next_unit][0] <= mfma_seq:
            pending.append(units[next_unit][1])
            next_unit += 1

        budget = issueRate
        while pending and budget:
            unit = pending[0]
            out.add(unit.popleft())
            budget -= 1
            if not unit:
                pending.popleft()

        if mfma_seq == last_seq:
            while next_unit < len(units):
                pending.append(units[next_unit][1])
                next_unit += 1
            while pending:
                unit = pending.popleft()
                while unit:
                    out.add(unit.popleft())
    return out
