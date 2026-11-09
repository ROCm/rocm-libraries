# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction scheduler for subtile-based mainloop.

Interleaves non-MFMA instructions between MFMAs using a slot-based placer
with pluggable scheduling rules.
"""

import copy
import math
from typing import List, Tuple, Optional
from rocisa.code import Module, TextBlock
from rocisa.instruction import SWaitCnt, SBarrier, MFMAInstruction, MXMFMAInstruction, \
    LocalReadInstruction, LocalWriteInstruction, GlobalReadInstruction, GlobalWriteInstruction, \
    CommonInstruction, SMemLoadInstruction, VXorB32


class _SlotPlacer:
    """Generic slot placement engine for interleaving instructions between MFMAs.

    The legacy schedule has 2 placement slots between adjacent MFMAs.
    MXFP8 partition prefetch uses five, one instruction each.
    Rules are injected via callbacks:
      - validators: (placer, pos, inst) -> bool — reject invalid slots
      - preferences: (placer, pos, inst) -> bool — like validators, but
        dropped when no slot satisfies them
      - adjusters:  (placer, limit, inst) -> limit — shift search start
      - onPlace:    (placer, pos, inst) -> None — update rule state after placement
    """

    def __init__(self, intervals: int, numModules: int,
                 pathOrders: List[List[int]],
                 validators=None, adjusters=None, onPlace=None,
                 slotsPerInterval: int = 2,
                 maxPerSlot: int = 2,
                 dataDsReadCap: Optional[Tuple[int, int]] = None,
                 maxInstructionsPerInterval: Optional[int] = None,
                 preferences=None):
        self.slotsPerInterval = slotsPerInterval
        self.maxPerSlot = maxPerSlot
        self.dataDsReadCap = dataDsReadCap
        self.maxInstructionsPerInterval = maxInstructionsPerInterval
        self.totalSlots = intervals * slotsPerInterval
        self._n = numModules
        self._prevInPath: List[int] = [-1] * numModules
        self._nextInPath: List[int] = [-1] * numModules
        for order in pathOrders:
            for a, b in zip(order, order[1:]):
                self._prevInPath[b] = a
                self._nextInPath[a] = b
        self._validators = validators or []
        self._preferences = preferences or []
        self._adjusters = adjusters or []
        self._onPlace = onPlace
        self.denseToFit = False

        self._placed: List[List[Tuple[int, object]]] = [[] for _ in range(self.totalSlots)]
        self._firstPos: List[Optional[int]] = [None] * numModules
        self._lastPos: List[Optional[int]] = [None] * numModules
        self.leftovers: List[Tuple[int, object]] = []

    # ── Placement ──

    def _canPlace(self, pos: int, inst) -> bool:
        if pos < 0 or pos >= self.totalSlots or len(self._placed[pos]) >= self.maxPerSlot:
            return False
        return all(v(self, pos, inst) for v in self._validators)

    def _prefers(self, pos: int, inst) -> bool:
        return all(p(self, pos, inst) for p in self._preferences)

    def _intervalInstructionCount(self, pos: int) -> int:
        begin = (pos // self.slotsPerInterval) * self.slotsPerInterval
        end = min(self.totalSlots, begin + self.slotsPerInterval)
        return sum(
            not isinstance(item[1], TextBlock)
            for slot in range(begin, end)
            for item in self._placed[slot])

    def _intervalHasDtlLoad(self, pos: int) -> bool:
        begin = (pos // self.slotsPerInterval) * self.slotsPerInterval
        end = min(self.totalSlots, begin + self.slotsPerInterval)
        return any(
            _isDtlLoad(item[1])
            for slot in range(begin, end)
            for item in self._placed[slot]
        )

    def adjustLimit(self, limit: int, inst) -> int:
        for adj in self._adjusters:
            limit = adj(self, limit, inst)
        return limit

    def bounds(self, mid: int) -> Tuple[int, int]:
        lo = 0
        pred = self._prevInPath[mid]
        if 0 <= pred < self._n and self._lastPos[pred] is not None:
            lo = self._lastPos[pred] + 1
        hi = self.totalSlots - 1
        succ = self._nextInPath[mid]
        if 0 <= succ < self._n and self._firstPos[succ] is not None:
            hi = self._firstPos[succ] - 1
        return lo, hi

    def findSlot(self, mid: int, inst, limit: int, reverse: bool = False) -> Optional[int]:
        lo, hi = self.bounds(mid)
        if reverse:
            hi = min(hi, limit)
        else:
            lo = max(lo, limit)
        if hi < lo:
            return None
        order = range(hi, lo - 1, -1) if reverse else range(lo, hi + 1)
        if self._preferences:
            for pos in order:
                if self._canPlace(pos, inst) and self._prefers(pos, inst):
                    return pos
        for pos in order:
            if self._canPlace(pos, inst):
                return pos
        return None

    def _forceSlot(self, mid: int, limit: int, reverse: bool) -> int:
        """Find the closest valid slot respecting dependencies, allowing >2 items per slot."""
        lo, hi = self.bounds(mid)
        if reverse:
            hi = min(hi, limit)
            lo = max(lo, 0)
            if hi < lo:
                hi = lo
            return hi
        else:
            lo = max(lo, limit)
            hi = min(hi, self.totalSlots - 1)
            if lo > hi:
                lo = hi
            return lo

    def place(self, pos: int, item: Tuple[int, object], reverse: bool = False):
        mid = item[0]
        if reverse:
            self._placed[pos].insert(0, item)
        else:
            self._placed[pos].append(item)
        if self._firstPos[mid] is None or pos < self._firstPos[mid]:
            self._firstPos[mid] = pos
        if self._lastPos[mid] is None or pos > self._lastPos[mid]:
            self._lastPos[mid] = pos
        if self._onPlace:
            self._onPlace(self, pos, item[1])

    def _placeAtomicDtlPair(self, setupItem, loadItem, limit: int) -> Optional[int]:
        """Keep a final m0 setup and its direct-to-LDS load in one MFMA interval."""
        setupMid, setupInst = setupItem
        loadMid, loadInst = loadItem
        setupLo, setupHi = self.bounds(setupMid)
        loadLo, loadHi = self.bounds(loadMid)
        step = self.slotsPerInterval
        first = max(limit, setupLo)
        for preferred in ((True, False) if self._preferences else (False,)):
            for setupPos in range(first, setupHi + 1):
                intervalEnd = min(
                    self.totalSlots - 1,
                    (setupPos // step + 1) * step - 1)
                if not self._canPlace(setupPos, setupInst) or (
                        preferred and not self._prefers(setupPos, setupInst)):
                    continue
                for loadPos in range(max(setupPos, loadLo), min(intervalEnd, loadHi) + 1):
                    if loadPos == setupPos and (
                            self.maxPerSlot <= 1 or self._placed[setupPos]):
                        continue
                    if not self._canPlace(loadPos, loadInst) or (
                            preferred and not self._prefers(loadPos, loadInst)):
                        continue
                    self.place(setupPos, setupItem)
                    self.place(loadPos, loadItem)
                    return loadPos
        # The interval rule can reject sharing a slot pair. Still keep the
        # request together: the load must not cross an MFMA away from its m0.
        firstInterval = max(0, first // step)
        lastInterval = (self.totalSlots - 1) // step
        for interval in range(firstInterval, lastInterval + 1):
            intervalBegin = interval * step
            intervalEnd = min(self.totalSlots - 1, intervalBegin + step - 1)
            setupPos = max(first, intervalBegin)
            loadPos = max(setupPos, loadLo)
            if self.maxPerSlot <= 1 and loadPos <= setupPos:
                loadPos = setupPos + 1
            if (setupPos <= min(setupHi, intervalEnd)
                    and loadPos <= min(loadHi, intervalEnd)):
                self.place(setupPos, setupItem)
                self.place(loadPos, loadItem)
                return loadPos
        return None

    def _forceAtomicDtlPair(self, setupItem, loadItem, limit: int) -> int:
        """Keep one m0/load pair together after interval capacity is exhausted.

        The load stays in the same MFMA interval as its m0. limit is
        already past every earlier m0, so this cannot move the setup ahead
        of a previous direct-to-LDS request.
        """
        setupMid, _setupInst = setupItem
        loadMid, _loadInst = loadItem
        setupLo, setupHi = self.bounds(setupMid)
        _loadLo, loadHi = self.bounds(loadMid)
        setupPos = min(max(limit, setupLo), setupHi, self.totalSlots - 1)
        intervalEnd = min(
            self.totalSlots - 1,
            (setupPos // self.slotsPerInterval + 1)
            * self.slotsPerInterval - 1,
        )
        loadPos = min(intervalEnd, max(setupPos, loadHi))
        if self.maxPerSlot <= 1 and loadPos <= setupPos:
            loadPos = min(intervalEnd, setupPos + 1)
        if loadPos <= setupPos or loadPos >= self.totalSlots:
            self.place(setupPos, setupItem)
            self.place(setupPos, loadItem)
            return setupPos
        self.place(setupPos, setupItem)
        self.place(loadPos, loadItem)
        return loadPos

    def placePath(self, pathInsts: List[Tuple[int, object]], reverse: bool = False,
                  multiDU: bool = False, startLimit: Optional[int] = None,
                  overflowToTail: bool = False, atomicDtl: bool = False,
                  mfmaBetweenWaitAndBarrier: bool = False):
        """Place a sequence of (moduleId, instruction) items into slots.

        Walks pathInsts in order, applying adjusters (forward only) and
        finding valid slots. When no empty slot is found, force-places at
        the closest valid position respecting dependencies (allowing >2
        items per slot).

        When multiDU is True, wait_gr immediately followed by barrier is
        placed as an atomic pair in the same slot so other paths cannot
        insert m0 setup between them.

        The extra arguments are the dual-period partition-zero phase placer.
        Default calls keep the original walk.
        """
        phase = (startLimit is not None or overflowToTail or atomicDtl
                 or mfmaBetweenWaitAndBarrier)
        limit = (self.totalSlots - 1) if reverse else 0
        if phase and not reverse and not multiDU:
            limit = 0 if startLimit is None else startLimit
            idx = 0
            while idx < len(pathInsts):
                item = pathInsts[idx]
                mid, inst = item
                if (mfmaBetweenWaitAndBarrier
                        and idx + 1 < len(pathInsts)
                        and _isWaitGr(inst)
                        and _isBarrier(pathInsts[idx + 1][1])):
                    limit = self.adjustLimit(limit, inst)
                    waitPos = self.findSlot(mid, inst, limit)
                    if waitPos is None:
                        waitPos = self._forceSlot(mid, limit, False)
                    nextInterval = (
                        waitPos // self.slotsPerInterval + 1
                    ) * self.slotsPerInterval
                    if nextInterval < self.totalSlots:
                        self.place(waitPos, item)
                        barrierItem = pathInsts[idx + 1]
                        barrierPos = self.findSlot(
                            barrierItem[0], barrierItem[1], nextInterval)
                        if barrierPos is None:
                            barrierPos = self._forceSlot(
                                barrierItem[0], nextInterval, False)
                        self.place(barrierPos, barrierItem)
                        limit = barrierPos + 1
                        idx += 2
                        continue
                if (atomicDtl and idx + 1 < len(pathInsts)
                        and _isM0Update(inst)
                        and _isDtlLoad(pathInsts[idx + 1][1])):
                    pairEnd = self._placeAtomicDtlPair(
                        item, pathInsts[idx + 1], limit)
                    if pairEnd is None:
                        pairEnd = self._forceAtomicDtlPair(
                            item, pathInsts[idx + 1], limit)
                    if pairEnd is not None:
                        limit = pairEnd + 1
                        idx += 2
                        continue
                limit = self.adjustLimit(limit, inst)
                pos = self.findSlot(mid, inst, limit, reverse=False)
                if pos is None:
                    if overflowToTail:
                        self.leftovers.extend(pathInsts[idx:])
                        return
                    pos = self._forceSlot(mid, limit, False)
                self.place(pos, item, reverse=False)
                limit = pos + 1
                idx += 1
            return
        if not multiDU:
            for item in pathInsts:
                mid, inst = item
                if not reverse:
                    limit = self.adjustLimit(limit, inst)
                pos = self.findSlot(mid, inst, limit, reverse=reverse)
                if pos is None:
                    pos = self._forceSlot(mid, limit, reverse)
                self.place(pos, item, reverse=reverse)
                limit = (pos - 1) if reverse else (pos + 1)
            return

        idx = 0
        while idx < len(pathInsts):
            item = pathInsts[idx]
            mid, inst = item
            if (not reverse and idx + 1 < len(pathInsts)
                    and _isWaitGr(inst) and _isBarrier(pathInsts[idx + 1][1])):
                if not reverse:
                    limit = self.adjustLimit(limit, inst)
                pos = self.findSlot(mid, inst, limit, reverse=reverse)
                if pos is None:
                    pos = self._forceSlot(mid, limit, reverse)
                self.place(pos, item, reverse=reverse)
                sync_item = pathInsts[idx + 1]
                if self._canPlace(pos, sync_item[1]):
                    self.place(pos, sync_item, reverse=reverse)
                else:
                    sync_pos = self.findSlot(sync_item[0], sync_item[1],
                                             pos + 1, reverse=reverse)
                    if sync_pos is None:
                        sync_pos = self._forceSlot(sync_item[0], pos + 1, reverse)
                    self.place(sync_pos, sync_item, reverse=reverse)
                    pos = sync_pos
                limit = (pos - 1) if reverse else (pos + 1)
                idx += 2
                continue
            if not reverse:
                limit = self.adjustLimit(limit, inst)
            pos = self.findSlot(mid, inst, limit, reverse=reverse)
            if pos is None:
                pos = self._forceSlot(mid, limit, reverse)
            self.place(pos, item, reverse=reverse)
            limit = (pos - 1) if reverse else (pos + 1)
            idx += 1

    # ── Assembly ──

    def assemble(self, mfmas) -> Module:
        intervals = len(mfmas) - 1
        result = Module()
        result.add(mfmas[0])
        width = self.slotsPerInterval
        for i in range(intervals):
            for slot in range(width * i, width * i + width):
                for item in self._placed[slot]:
                    result.add(item[1])
            result.add(mfmas[i + 1])
        for _, inst in self.leftovers:
            result.add(inst)
        return result


# ── Scheduling rules ──

# Hardcoded gap to hide ds_read latency. TODO: compute this more accurately.
# gfx1250 needs a larger gap (8); other archs use 4.
_MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT = 4
_MIN_MFMA_GAP_DS_READ_TO_WAIT_GFX1250 = 8
# gfx950 MXFP8 MFMA is 32 cycles. The issue takes 8 and LD_PRELOAD
# takes 4. The remaining 20 cycles are five follower slots. Every
# ds_read counts toward two per interval, or three when denseToFit
# says a two-read schedule cannot fit.
_MXFP8_FOLLOWER_SLOTS = 5
_MXFP8_DATA_DS_READS = 2
_MXFP8_DATA_DS_READ_OVERFLOW = 3
# A half-disjoint cell holds at most this many consecutive MFMA gaps
# with a direct-to-LDS load. SALU per gap stays under the follower cap.
_MXFP8_MAX_DTL_RUN = 5

# gfx950 scaled MFMAs retain their scale-source VGPR reads for nine issue
# positions.  A direct write of that VGPR has to stay this far beyond the
# last reader.  The loop edge between the last C1 consumer and a C0 shadow
# is three loop-control instructions plus the C0 entry wait.  The write
# itself issues after the target MFMA, so the earliest shadow is one less
# than the remaining retain distance.
_SCALE_SOURCE_WAR_MFMA_GAP = 9
_SOURCE_WAR_LOOP_EDGE = 3 + 1

def _physicalInstructionCount(items):
    """Count emitted instructions while ignoring comments/text markers."""
    return sum(not isinstance(inst, TextBlock) for inst in items)


def _registerIndex(register):
    """Return a concrete VGPR index, or None for a symbolic/non-VGPR operand."""
    if register is None or getattr(register, "regType", None) != "v":
        return None
    index = getattr(register, "regIdx", None)
    return index if isinstance(index, int) else None


def _sourceWriteShadow(lastUse, totalMfmas):
    """Earliest C0 MFMA shadow that may overwrite a VGPR last read in C1."""
    later = totalMfmas - 1 - lastUse
    return max(
        0,
        _SCALE_SOURCE_WAR_MFMA_GAP - later - _SOURCE_WAR_LOOP_EDGE - 1)


def scale_prefetch_fits(pool_size, count):
    """True when ``count`` more VGPRs stay in the current 8-register block.

    The accumulator block starts at that rounded boundary. A checkout that
    stays under it does not change occupancy. A checkout that crosses it
    moves the block and can drop a wave.
    """
    if count <= 0:
        return False

    def rounded(size):
        return int(math.ceil(size / 8.0)) * 8

    return rounded(pool_size + count) == rounded(pool_size)


def selectScaleShadowPlan(
        c0Instructions, c1Instructions, firstPartitionMfmas,
        expectedScaleReads):
    """Select the canonical scales needed before C0 partition zero can issue.

    The count is whatever this macrotile consumes inside partition zero.  It
    is not a fixed two-ScaleA plus two-ScaleB split.  Those values are seeded
    in the preloop and refreshed directly in the prior C1 tail.  Every other
    scale read stays in C0 and overlaps compute that already has its operands.
    """
    c0Reads = {}
    tensorByRegister = {}
    for inst in c0Instructions:
        if not isinstance(inst, LocalReadInstruction):
            continue
        destination = _registerIndex(getattr(inst, "dst", None))
        comment = getattr(inst, "comment", "")
        tensor = next(
            (name for name in ("SA", "SB") if comment.startswith(name + " ")),
            None)
        if destination is None or tensor is None:
            continue
        c0Reads.setdefault(destination, copy.deepcopy(inst))
        tensorByRegister[destination] = tensor

    c0FirstUse = {}
    c1LastUse = {}
    c0Rank = 0
    for inst in c0Instructions:
        if not isinstance(inst, MXMFMAInstruction):
            continue
        for register in (inst.mxsa, inst.mxsb):
            index = _registerIndex(register)
            if index is not None:
                c0FirstUse.setdefault(index, c0Rank)
        c0Rank += 1

    c1Rank = 0
    for inst in c1Instructions:
        if not isinstance(inst, MXMFMAInstruction):
            continue
        for register in (inst.mxsa, inst.mxsb):
            index = _registerIndex(register)
            if index is not None:
                c1LastUse[index] = c1Rank
        c1Rank += 1

    selected = [
        register for register in c0Reads
        if c0FirstUse.get(register, firstPartitionMfmas) < firstPartitionMfmas
        and register in c1LastUse
    ]
    selected.sort(
        key=lambda register: (
            0 if tensorByRegister[register] == "SA" else 1,
            c0FirstUse[register],
            register))
    expectedSA, expectedSB = expectedScaleReads
    if len(selected) != expectedSA + expectedSB:
        return []
    if [tensorByRegister[register] for register in selected].count("SA") != expectedSA:
        return []
    if [tensorByRegister[register] for register in selected].count("SB") != expectedSB:
        return []

    entries = [
        {
            "canonicalVgpr": register,
            "tensor": tensorByRegister[register],
            "readTemplate": c0Reads[register],
            "firstC0Mfma": c0FirstUse[register],
            "lastC1Mfma": c1LastUse[register],
        }
        for register in c0Reads
        if register in c0FirstUse and register in c1LastUse
    ]
    entries.sort(
        key=lambda entry: (
            entry["firstC0Mfma"],
            0 if entry["tensor"] == "SA" else 1,
            entry["canonicalVgpr"]))
    selectedSet = set(selected)
    writeFloor = {}
    for register, lastUse in c1LastUse.items():
        shadow = _sourceWriteShadow(lastUse, c1Rank)
        if shadow > 0:
            writeFloor[register] = shadow
    return {
        "entry": [
            entry for entry in entries
            if entry["canonicalVgpr"] in selectedSet
        ],
        "deferred": [
            entry for entry in entries
            if entry["canonicalVgpr"] not in selectedSet
        ],
        "totalMfmas": c1Rank,
        "writeFloor": writeFloor,
    }


def selectLateScalePrefetch(deferred, extraByCanonical, writeFloor):
    """Point the next-epoch copy of a late scale at its staging VGPR.

    ``deferred`` entries are the scale groups whose first consumer is not
    in partition zero. Their canonical VGPR is still a source of the last
    C1 MFMAs, so the next epoch is read into ``extraByCanonical`` instead.
    The MFMA keeps naming the canonical register; a later copy moves the
    staged dword across once that source is free.
    """
    plan = []
    for entry in deferred:
        canonical = entry["canonicalVgpr"]
        extra = extraByCanonical.get(canonical)
        if extra is None:
            continue
        minimum = writeFloor.get(canonical, 0)
        # The copy has to sit after the source is free and before the
        # first consumer. A scale with no such shadow keeps its C0 read.
        if entry["firstC0Mfma"] <= minimum:
            continue
        read = copy.deepcopy(entry["readTemplate"])
        from rocisa.container import vgpr
        read.dst = vgpr(extra)
        plan.append({
            "canonicalVgpr": canonical,
            "extraVgpr": extra,
            "tensor": entry["tensor"],
            "readTemplate": read,
            "canonicalRead": copy.deepcopy(entry["readTemplate"]),
            "firstC0Mfma": entry["firstC0Mfma"],
            "lastC1Mfma": entry["lastC1Mfma"],
            "minShadow": minimum,
        })
    return plan


def lateScaleReadsPrecedeSecondHalf(module, registers, firstHalfMfmas):
    """Whether every scale read of ``registers`` issues before the barrier
    that opens the second half.

    That barrier is the first one once ``firstHalfMfmas`` MFMAs have
    issued. The second half's scale loads overwrite the buffer these reads
    come from.
    """
    pending = set(registers)
    issued = 0
    for inst in module.flatitems():
        if _isMfma(inst):
            issued += 1
        elif _isBarrier(inst) and issued >= firstHalfMfmas:
            break
        elif _isDsRead(inst):
            comment = getattr(inst, "comment", "") or ""
            if comment.startswith(("SA ", "SB ")):
                pending.discard(_registerIndex(getattr(inst, "dst", None)))
    return not pending


def dropEntryScaleReads(emitted3d, entryRegisters):
    """Remove preloop-seeded scale reads before C0 placement.

    Those VGPRs are already live from the preloop seed and are refreshed in
    the C1 tail.  Leaving the reread in the LR payload occupies an early
    shadow a data read can use, and the placer would overwrite a value C0
    still consumes.  A later copy of a scale VGPR that is already being
    read in this C0 body is the same dword and is dropped too.
    """
    if not entryRegisters:
        return
    seen = set()
    for partition in emitted3d:
        for cell in partition:
            for emitted in cell:
                instructions = emitted.instructions
                metadata = emitted.instructionMetadata
                aligned = len(metadata) == len(instructions)
                kept = []
                keptMeta = []
                for index, inst in enumerate(instructions):
                    destination = _registerIndex(getattr(inst, "dst", None))
                    comment = getattr(inst, "comment", "") or ""
                    scaleRead = (
                        isinstance(inst, LocalReadInstruction)
                        and destination is not None
                        and (
                            comment.startswith("SA ")
                            or comment.startswith("SB "))
                    )
                    if scaleRead and (
                            destination in entryRegisters
                            or destination in seen):
                        continue
                    if scaleRead:
                        seen.add(destination)
                    kept.append(inst)
                    if aligned:
                        keptMeta.append(metadata[index])
                if len(kept) != len(instructions):
                    emitted.instructions = kept
                    if aligned:
                        emitted.instructionMetadata = keptMeta


def appendC1ScaleShadowReads(
        schedule, plan, finalPartitionMfmas,
        warGap=_SCALE_SOURCE_WAR_MFMA_GAP,
        ignoreSourceWar=False):
    """Refresh C0-entry scales in dependency-safe C1 P3 MFMA shadows.

    Prefer an empty final-partition suffix.  A scale ds_read then counts
    toward the same two-read cap as a data read, and uses three only when
    a two-read schedule cannot fit.  An empty shadow may hold as many
    scale reads as it has follower slots.  Every direct write issues at
    least ``warGap`` positions after that scale's final C1 consumer.

    Returns ``None`` when the final partition has no barrier to hang the
    reads on.  Otherwise returns ``(unplaced, c1Mfmas)``.  Unplaced
    entries are past the last C1 shadow or do not fit in it; the caller
    spills those into the next C0.
    """
    if not plan:
        return None

    prefix = []
    mfmas = []
    followers = []
    for inst in schedule.flatitems():
        if isinstance(inst, (MFMAInstruction, MXMFMAInstruction)):
            mfmas.append(inst)
            followers.append([])
        elif not mfmas:
            prefix.append(inst)
        else:
            followers[-1].append(inst)

    if len(mfmas) < finalPartitionMfmas or finalPartitionMfmas <= 0:
        return None
    finalStart = len(mfmas) - finalPartitionMfmas
    barrierShadows = [
        index for index in range(finalStart, len(followers))
        if any(isinstance(inst, SBarrier) for inst in followers[index])
    ]
    if not barrierShadows:
        return None
    afterBarrier = barrierShadows[-1]

    def canAppend(index, requireEmpty, reserved, maxLocalReads):
        group = followers[index]
        if requireEmpty and _physicalInstructionCount(group):
            return False
        if (_physicalInstructionCount(group) + reserved[index]
                >= _MXFP8_FOLLOWER_SLOTS):
            return False
        localReads = sum(
            isinstance(inst, LocalReadInstruction) for inst in group)
        return localReads + reserved[index] < maxLocalReads

    orderedPlan = sorted(
        plan,
        key=lambda entry: (
            entry["lastC1Mfma"],
            0 if entry["tensor"] == "SA" else 1,
            entry["canonicalVgpr"]))

    occupied = [
        index for index in range(finalStart, len(followers))
        if _physicalInstructionCount(followers[index])
    ]
    emptyTailStart = occupied[-1] + 1 if occupied else finalStart

    def makePlacement(start, requireEmpty, maxLocalReads):
        placements = []
        reserved = [0] * len(followers)
        searchFrom = max(start, finalStart, afterBarrier)
        for entry in orderedPlan:
            # A staging VGPR is not a source of the in-flight MFMAs, so the
            # read only has to follow the LDS barrier.
            sourceEarliest = (
                searchFrom if ignoreSourceWar
                else entry["lastC1Mfma"] + warGap - 1)
            earliest = max(searchFrom, sourceEarliest)
            target = next(
                (
                    index for index in range(earliest, len(followers))
                    if canAppend(
                        index, requireEmpty, reserved, maxLocalReads)
                ),
                None)
            if target is None:
                return None
            placements.append((target, entry))
            reserved[target] += 1
            searchFrom = target
        return placements

    placements = makePlacement(
        emptyTailStart, requireEmpty=True,
        maxLocalReads=_MXFP8_DATA_DS_READS)
    if placements is None:
        placements = makePlacement(
            finalStart, requireEmpty=False,
            maxLocalReads=_MXFP8_DATA_DS_READS)
    if placements is None:
        placements = makePlacement(
            finalStart, requireEmpty=False,
            maxLocalReads=_MXFP8_DATA_DS_READ_OVERFLOW)
    def commit(selected):
        for target, entry in selected:
            read = copy.deepcopy(entry["readTemplate"])
            read.comment = (
                f"{getattr(read, 'comment', '')}; "
                "refresh next C0 entry scale in prior C1 shadow; "
                f"{entry['tensor']} canonical=v{entry['canonicalVgpr']} "
                f"last-use=M{entry['lastC1Mfma']}"
            )
            followers[target].append(read)
        items = list(prefix)
        for mfma, group in zip(mfmas, followers):
            items.append(mfma)
            items.extend(group)
        schedule.setItems(items)

    if placements is not None:
        commit(placements)
        return [], len(mfmas)

    # The strict caps could not take every refresh.  Keep the ones that
    # still land in C1.  An empty shadow is scale-only, so it can use
    # every follower slot; a shadow that already has data reads stays
    # on the overflow cap.
    placements = []
    leftovers = []
    reserved = [0] * len(followers)
    searchFrom = max(finalStart, afterBarrier)

    def readCap(index):
        if _physicalInstructionCount(followers[index]) == 0:
            return _MXFP8_FOLLOWER_SLOTS
        return _MXFP8_DATA_DS_READ_OVERFLOW

    for entry in orderedPlan:
        sourceEarliest = (
            searchFrom if ignoreSourceWar
            else entry["lastC1Mfma"] + warGap - 1)
        earliest = max(searchFrom, sourceEarliest)
        target = next(
            (
                index for index in range(earliest, len(followers))
                if canAppend(index, False, reserved, readCap(index))
            ),
            None)
        if target is None:
            leftovers.append(entry)
            continue
        placements.append((target, entry))
        reserved[target] += 1
        searchFrom = target
    commit(placements)
    return leftovers, len(mfmas)


def spillC0ScaleRefreshes(schedule, entries, c1Mfmas):
    """Place C1-tail overflows in the next C0, before their first use.

    A prologue write sits before the C0 entry wait, so that wait retires
    it.  Its issue distance is the remaining C1 MFMAs, the three
    loop-control instructions, and the write.  A write that is one
    short of that gap can sit after the entry wait and before mfma 0;
    the wait then counts, and the read gets its own drain.  A later C0
    shadow uses the same distance as a C1 shadow once the loop edge is
    included, and
    it has to stay before the first consumer and before the scale-pointer
    swap.  A shadow write that the entry wait does not cover gets an
    lgkmcnt drain immediately before its consumer.
    """
    if not entries:
        return True
    items = list(schedule.flatitems())
    mfmaPos = [
        index for index, inst in enumerate(items)
        if isinstance(inst, (MFMAInstruction, MXMFMAInstruction))
    ]
    if not mfmaPos:
        return False
    def drainsLocalReads(inst):
        # ds_read retires on the LDS counter.  The C0 entry wait is a
        # dscnt drain; older paths spell the same drain as kmcnt.
        return isinstance(inst, SWaitCnt) and (
            getattr(inst, "waitAll", False)
            or getattr(inst, "dscnt", -1) == 0
            or getattr(inst, "kmcnt", -1) == 0
        )

    def instText(inst):
        return " ".join(
            part for part in (
                getattr(inst, "comment", "") or "",
                getattr(inst, "text", "") or "",
                getattr(inst, "name", "") or "",
            )
            if part)

    prefix = items[:mfmaPos[0]]
    waitAt = next(
        (
            index for index in range(len(prefix) - 1, -1, -1)
            if drainsLocalReads(prefix[index])
        ),
        None)
    if waitAt is None:
        return False
    shadows = []
    for rank, pos in enumerate(mfmaPos):
        end = mfmaPos[rank + 1] if rank + 1 < len(mfmaPos) else len(items)
        shadows.append(items[pos + 1:end])

    def swapFlat(tensor):
        needle = f"swap MX{tensor} LR"
        index = next(
            (
                found for found, inst in enumerate(items)
                if needle in instText(inst)
            ),
            None)
        if index is None:
            return None
        # The banner can be scheduled just after the xor it describes.
        # The xor is the pointer change.
        cursor = index - 1
        while cursor >= 0 and type(items[cursor]).__name__ == "TextBlock":
            cursor -= 1
        if cursor >= 0 and type(items[cursor]).__name__ == "VXorB32":
            return cursor
        return index

    def region(flatIndex):
        if flatIndex < mfmaPos[0]:
            return ("prefix", flatIndex)
        for rank, pos in enumerate(mfmaPos):
            end = mfmaPos[rank + 1] if rank + 1 < len(mfmaPos) else len(items)
            if flatIndex == pos:
                return ("mfma", rank)
            if pos < flatIndex < end:
                return ("shadow", rank)
        return None

    def beforeSwap(tensor, where):
        flat = swapFlat(tensor)
        if flat is None:
            return True
        loc = region(flat)
        if where[0] == "prologue":
            return loc[0] != "prefix" or loc[1] > waitAt
        if where[0] == "postwait":
            # The read is inserted at the end of the prefix, after the
            # entry wait.  A swap still in the prefix has already
            # retired, or sits in that same suffix.
            return loc[0] != "prefix"
        rank = where[1]
        if loc[0] == "prefix":
            return False
        return rank < loc[1]

    def later(entry):
        return c1Mfmas - 1 - entry["lastC1Mfma"]

    def makeRead(entry, where):
        read = copy.deepcopy(entry["readTemplate"])
        read.comment = (
            f"{getattr(read, 'comment', '')}; "
            f"refresh next C0 entry scale in {where}; "
            f"{entry['tensor']} canonical=v{entry['canonicalVgpr']} "
            f"last-use=M{entry['lastC1Mfma']}"
        )
        return read

    prologue = []
    postwait = []
    shadowAdds = [[] for _ in shadows]
    for entry in sorted(
            entries,
            key=lambda item: (item["firstC0Mfma"], item["lastC1Mfma"])):
        if later(entry) + 3 + 1 >= 9 and beforeSwap(
                entry["tensor"], ("prologue",)):
            prologue.append(makeRead(entry, "next C0 prologue"))
            continue
        # One short of the prologue gap.  The entry wait sits before
        # this write, so it counts, and the write is still before mfma 0.
        if (entry["firstC0Mfma"] == 0
                and later(entry) + _SOURCE_WAR_LOOP_EDGE + 1 >= 9
                and beforeSwap(entry["tensor"], ("postwait",))):
            postwait.append(makeRead(entry, "next C0 before mfma 0"))
            continue
        placed = False
        minimum = max(
            0, 9 - 1 - _SOURCE_WAR_LOOP_EDGE - later(entry))
        for target in range(minimum, entry["firstC0Mfma"]):
            if target >= len(shadows):
                break
            if not beforeSwap(entry["tensor"], ("shadow", target)):
                break
            occupied = (
                _physicalInstructionCount(shadows[target])
                + len(shadowAdds[target]))
            if occupied >= _MXFP8_FOLLOWER_SLOTS:
                continue
            shadowAdds[target].append(
                makeRead(entry, f"C0 shadow M{target}"))
            placed = True
            break
        if not placed:
            return False

    if prologue:
        prefix = prefix[:waitAt] + prologue + prefix[waitAt:]
    if postwait:
        prefix = prefix + postwait
    rebuilt = list(prefix)
    for rank, pos in enumerate(mfmaPos):
        rebuilt.append(items[pos])
        rebuilt.extend(shadows[rank])
        rebuilt.extend(shadowAdds[rank])

    def canonical(inst):
        comment = getattr(inst, "comment", "") or ""
        marker = "canonical=v"
        if marker not in comment or "refresh next C0 entry scale in " not in comment:
            return None
        if "prior C1 shadow" in comment:
            return None
        text = comment.split(marker, 1)[1].split(" ", 1)[0]
        return int(text) if text.isdigit() else None

    def usesRegister(inst, vgpr):
        if not isinstance(inst, MXMFMAInstruction):
            return False
        return any(
            _registerIndex(register) == vgpr
            for register in (inst.mxsa, inst.mxsb))

    waited = []
    for index, inst in enumerate(rebuilt):
        vgpr = canonical(inst)
        if vgpr is None:
            continue
        consumer = next(
            (
                later for later in range(index + 1, len(rebuilt))
                if usesRegister(rebuilt[later], vgpr)
            ),
            None)
        if consumer is None:
            return False
        if any(
                drainsLocalReads(rebuilt[between])
                for between in range(index + 1, consumer)):
            continue
        waited.append(consumer)
    for consumer in sorted(set(waited), reverse=True):
        rebuilt.insert(
            consumer,
            SWaitCnt(
                dscnt=0,
                comment="Wait spilled C0 entry scale refresh"))
    schedule.setItems(rebuilt)
    return True


def placeLateScalePrefetchSpills(schedule, leftovers):
    """Issue a staging read that missed the C1 tail after the C0 entry wait.

    The prologue wait has to stay clear of these reads. A read placed
    before it would hold the first MFMA on the scale latency this prefetch
    exists to move. The read lands in the earliest shadow that still has
    a follower slot, before the scale's first consumer.
    """
    if not leftovers:
        return []
    items = list(schedule.flatitems())
    mfmaPos = [
        index for index, inst in enumerate(items)
        if isinstance(inst, (MFMAInstruction, MXMFMAInstruction))
    ]
    if not mfmaPos:
        return list(leftovers)
    shadows = []
    for rank, pos in enumerate(mfmaPos):
        end = mfmaPos[rank + 1] if rank + 1 < len(mfmaPos) else len(items)
        shadows.append(list(items[pos + 1:end]))
    adds = [[] for _ in shadows]
    unplaced = []
    for entry in sorted(leftovers, key=lambda item: item["firstC0Mfma"]):
        placed = False
        for target in range(0, entry["firstC0Mfma"]):
            if target >= len(shadows):
                break
            occupied = (
                _physicalInstructionCount(shadows[target])
                + len(adds[target]))
            if occupied == 0 or occupied >= _MXFP8_FOLLOWER_SLOTS:
                continue
            read = copy.deepcopy(entry["readTemplate"])
            read.comment = (
                f"{getattr(read, 'comment', '')}; "
                "prefetch next scale epoch into staging VGPR; "
                f"{entry['tensor']} staged=v{entry['extraVgpr']} "
                f"canonical=v{entry['canonicalVgpr']}"
            )
            adds[target].append(read)
            placed = True
            break
        if not placed:
            unplaced.append(entry)
    if not any(adds):
        return unplaced
    rebuilt = list(items[:mfmaPos[0]])
    for rank, pos in enumerate(mfmaPos):
        rebuilt.append(items[pos])
        rebuilt.extend(shadows[rank])
        rebuilt.extend(adds[rank])
    schedule.setItems(rebuilt)
    return unplaced


def insertPrefetchedScaleCopies(schedule, moves):
    """Copy a staged scale into the VGPR its MFMAs already name.

    The copy waits until ``minShadow``, which is the source write-after-read
    floor of the canonical register, and until after a staging read that
    landed in this C0. A read issued in the previous C1 is retired by the
    C0 entry wait. A read issued in this C0 gets its own LDS wait beside
    the copy.
    """
    if not moves:
        return []
    from rocisa.container import vgpr
    from rocisa.instruction import VMovB32

    items = list(schedule.flatitems())
    mfmaPos = [
        index for index, inst in enumerate(items)
        if isinstance(inst, (MFMAInstruction, MXMFMAInstruction))
    ]
    if not mfmaPos:
        return list(moves)

    def drains(inst):
        return isinstance(inst, SWaitCnt) and (
            getattr(inst, "waitAll", False)
            or getattr(inst, "dscnt", -1) == 0
            or getattr(inst, "kmcnt", -1) == 0
        )

    shadows = []
    for rank, pos in enumerate(mfmaPos):
        end = mfmaPos[rank + 1] if rank + 1 < len(mfmaPos) else len(items)
        shadows.append(list(items[pos + 1:end]))
    adds = [[] for _ in shadows]
    unplaced = []
    prefix = list(items[:mfmaPos[0]])

    def readShadow(extra):
        for index, inst in enumerate(prefix):
            if (_registerIndex(getattr(inst, "dst", None)) == extra
                    and isinstance(inst, LocalReadInstruction)):
                return ("prefix", index)
        for rank, group in enumerate(shadows):
            if any(
                    _registerIndex(getattr(inst, "dst", None)) == extra
                    and isinstance(inst, LocalReadInstruction)
                    for inst in group):
                return ("shadow", rank)
        return None

    for move in sorted(moves, key=lambda item: item["firstC0Mfma"]):
        extra = move["extraVgpr"]
        located = readShadow(extra)
        minimum = move.get("minShadow", 0)
        needsWait = False
        if located is not None and located[0] == "shadow":
            minimum = max(minimum, located[1] + 1)
            needsWait = True
        elif located is not None:
            waited = any(drains(inst) for inst in prefix[located[1] + 1:])
            needsWait = not waited
        placed = False
        for target in range(minimum, move["firstC0Mfma"]):
            if target >= len(shadows):
                break
            occupied = (
                _physicalInstructionCount(shadows[target]) + len(adds[target]))
            if occupied == 0:
                continue
            if needsWait:
                adds[target].append(SWaitCnt(
                    dscnt=0,
                    comment=(
                        "Wait prefetched scale before copying "
                        f"v{extra}")))
            adds[target].append(VMovB32(
                dst=vgpr(move["canonicalVgpr"]),
                src=vgpr(extra),
                comment=(
                    f"copy prefetched {move['tensor']} scale "
                    f"v{extra} into v{move['canonicalVgpr']}")))
            placed = True
            break
        if not placed:
            unplaced.append(move)
            continue
    if not any(adds):
        return unplaced
    rebuilt = list(prefix)
    for rank, pos in enumerate(mfmaPos):
        rebuilt.append(items[pos])
        rebuilt.extend(shadows[rank])
        rebuilt.extend(adds[rank])
    schedule.setItems(rebuilt)
    return unplaced


def restoreCanonicalScaleReads(schedule, entries):
    """Put a scale's original C0 read back ahead of the entry wait.

    The staging copy had no legal shadow, so the prefix read that was
    removed for the prefetch has to come back. The entry wait retires it
    before the first MFMA, which is the schedule this scale had before
    staging.
    """
    if not entries:
        return
    items = list(schedule.flatitems())
    mfmaPos = next(
        (index for index, inst in enumerate(items)
         if isinstance(inst, (MFMAInstruction, MXMFMAInstruction))),
        None)
    if mfmaPos is None:
        prefix, tail = items, []
    else:
        prefix, tail = items[:mfmaPos], items[mfmaPos:]
    waitAt = None
    for index in range(len(prefix) - 1, -1, -1):
        inst = prefix[index]
        if isinstance(inst, SWaitCnt) and (
                getattr(inst, "waitAll", False)
                or getattr(inst, "dscnt", -1) == 0
                or getattr(inst, "kmcnt", -1) == 0):
            waitAt = index
            break
    reads = []
    for entry in entries:
        read = copy.deepcopy(entry["canonicalRead"])
        read.comment = (
            f"{getattr(read, 'comment', '')}; "
            "keep C0 scale read; no shadow for the staging copy; "
            f"{entry['tensor']} canonical=v{entry['canonicalVgpr']}"
        )
        reads.append(read)
    if waitAt is None:
        prefix = reads + prefix
    else:
        prefix = prefix[:waitAt] + reads + prefix[waitAt:]
    schedule.setItems(prefix + tail)


_isDsRead = lambda x: isinstance(x, LocalReadInstruction)
_isMfma = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))
# Instruction objects cannot carry a Python attribute, so the prefix pin
# records ids here. emitMainAndExitLoops clears the set around each kernel.
_p0OverwriteReadIds = set()


def clearPartitionZeroOverwriteMarks():
    """Drop overwrite-read marks left by the previous kernel."""
    _p0OverwriteReadIds.clear()
_isBufferLoad = lambda x: isinstance(x, GlobalReadInstruction)
_isBufferStore = lambda x: isinstance(x, GlobalWriteInstruction)
_isWaitCnt = lambda x: isinstance(x, SWaitCnt)
_isM0Update = lambda x: isinstance(x, CommonInstruction) and hasattr(x, 'dst') and hasattr(x.dst, 'regType') and x.dst.regType == 'm'
# Typed op detection (see SWaitCntEx.isWaitGr / SBarrier) rather than matching
# substrings in the instruction comment, which is brittle to comment edits.
_isWaitGr = lambda x: _isWaitCnt(x) and getattr(x, "isWaitGr", False)
_isBarrier = lambda x: isinstance(x, SBarrier)
_isDtlLoad = _isBufferLoad
_isVALU = lambda x: isinstance(x, CommonInstruction) \
    and not isinstance(x, (GlobalReadInstruction, LocalReadInstruction)) \
    and hasattr(x, 'dst') and hasattr(x.dst, 'regType') \
    and x.dst.regType == 'v'


class _SchedulingRules:
    """Scheduling rules for slot placement: validators, adjusters, and placement hooks.

    Owns all rule state (ds_read/waitcnt tracking, buffer-load spreading).
    Bound methods are passed as callbacks to _SlotPlacer.
    """

    def __init__(
            self, totalSlots: int,
            minGapDsReadToWait: int = _MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT,
            allowReadsAcrossWait: bool = False):
        # Cross-path state
        self.lastDsReadPos = -1
        self.earliestWaitCntPos = totalSlots
        self.waitGrBarrierPos: Optional[int] = None
        self.maxDsReadAfterWaitGr = -1
        self.minGap = minGapDsReadToWait
        self.allowReadsAcrossWait = allowReadsAcrossWait
        self.p0OverwritePrerequisiteIds = set()
        self.youngerP0OverwriteIds = set()
        self.partialP0OverwriteWaitIds = set()
        self.earliestWriteShadow = {}
        self.mfmaBase = 0
        self.dtlDsReadLimit: Optional[int] = None
        # Per-path state
        self._resetPath()

    def _resetPath(self):
        self.firstBufLoadPos: Optional[int] = None
        self.bufLoadIdx = 0
        self.bufLoadMaxSlot = 0
        self.numBufLoads = 0

    # ── Validators: (placer, pos, inst) -> bool ──

    def oneDsReadPerInterval(self, placer, pos, inst):
        """Cap ds_reads in one MFMA interval.

        The legacy pair keeps the other slot free of ds_reads. The MXFP8
        budget is two, or three when denseToFit is set. SA and SB count.
        """
        if not _isDsRead(inst):
            return True
        explicit = placer.dataDsReadCap
        if explicit is None:
            peer = pos ^ 1
            return not (0 <= peer < placer.totalSlots
                        and any(_isDsRead(item[1]) for item in placer._placed[peer]))
        ideal, overflow = explicit
        first = (pos // placer.slotsPerInterval) * placer.slotsPerInterval
        end = min(first + placer.slotsPerInterval, placer.totalSlots)
        reads = sum(
            _isDsRead(item[1])
            for slot in range(first, end)
            for item in placer._placed[slot])
        limit = overflow if placer.denseToFit else ideal
        return reads < limit

    def oneDtlLoadPerInterval(self, placer, pos, inst):
        """At most one direct-to-LDS load in an MFMA interval."""
        return not _isDtlLoad(inst) or not placer._intervalHasDtlLoad(pos)

    def maxDtlRun(self, length):
        """Validator: at most length consecutive intervals with a
        direct-to-LDS load."""
        def validate(placer, pos, inst):
            if not _isDtlLoad(inst):
                return True
            step = placer.slotsPerInterval
            interval = pos // step
            last = (placer.totalSlots - 1) // step
            run = 1
            before = interval - 1
            while before >= 0 and placer._intervalHasDtlLoad(before * step):
                run += 1
                before -= 1
            after = interval + 1
            while after <= last and placer._intervalHasDtlLoad(after * step):
                run += 1
                after += 1
            return run <= length
        return validate

    def noDsReadOrValuInFirstSlot(self, placer, pos, inst):
        """Reserve each interval's first slot from ds_reads and VALU.

        SALU, m0, waits, and the barrier may use it.
        """
        if pos % placer.slotsPerInterval != 0:
            return True
        return not (_isDsRead(inst) or _isVALU(inst))

    def minGapDsReadBeforeWait(self, placer, pos, inst):
        """Reject ds_read too close to an already-placed waitcnt ahead."""
        if not _isDsRead(inst):
            return True
        # MXFP8 partition prefetch may issue a read on the far side of a wait.
        # The default gap still applies to every other schedule.
        if self.allowReadsAcrossWait and pos >= self.earliestWaitCntPos:
            return True
        gap = self.minGap * placer.slotsPerInterval
        return self.earliestWaitCntPos - pos >= gap

    def minGapDsReadToWait(self, placer, pos, inst):
        """Reject waitcnt too close to the last placed ds_read."""
        if not _isWaitCnt(inst) or self.lastDsReadPos < 0:
            return True
        if self.allowReadsAcrossWait and pos <= self.lastDsReadPos:
            return True
        gap = self.minGap * placer.slotsPerInterval
        return pos - self.lastDsReadPos >= gap

    def noM0WithBufferLoad(self, placer, pos, inst):
        """Avoid placing M0 updates and buffer_loads in the same MFMA interval."""
        if not _isM0Update(inst) and not _isBufferLoad(inst):
            return True
        begin = (pos // placer.slotsPerInterval) * placer.slotsPerInterval
        end = min(placer.totalSlots, begin + placer.slotsPerInterval)
        if _isM0Update(inst):
            return not any(
                _isBufferLoad(item[1])
                for slot in range(begin, end)
                for item in placer._placed[slot])
        return not any(
            _isM0Update(item[1])
            for slot in range(begin, end)
            for item in placer._placed[slot])

    def valuNeedsLeader(self, placer, pos, inst):
        """Prefer a slot with another instruction ahead of a VALU.

        A VALU first after an MFMA waits out the MFMA's issue, and an
        empty first slot does not help. Paths placed early see empty
        intervals, so this is a preference; keepValuBehindLeader fixes
        what it could not once the body is complete.
        """
        if not _isVALU(inst):
            return True
        begin = (pos // placer.slotsPerInterval) * placer.slotsPerInterval
        return any(
            not _isVALU(item[1]) and not isinstance(item[1], TextBlock)
            for slot in range(begin, pos)
            for item in placer._placed[slot])

    def dtlAvoidsDsRead(self, placer, pos, inst):
        """Prefer an MFMA interval with at most dtlDsReadLimit ds_reads
        for m0 and a direct-to-LDS load. No limit when it is None."""
        if self.dtlDsReadLimit is None or not (
                _isDtlLoad(inst) or _isM0Update(inst)):
            return True
        begin = (pos // placer.slotsPerInterval) * placer.slotsPerInterval
        end = min(placer.totalSlots, begin + placer.slotsPerInterval)
        return sum(
            _isDsRead(item[1])
            for slot in range(begin, end)
            for item in placer._placed[slot]) <= self.dtlDsReadLimit

    def noValuWithDtl(self, placer, pos, inst):
        """Keep VALU and direct-to-LDS requests in different MFMA intervals."""
        begin = (pos // placer.slotsPerInterval) * placer.slotsPerInterval
        end = min(placer.totalSlots, begin + placer.slotsPerInterval)
        placed = [item[1] for slot in range(begin, end) for item in placer._placed[slot]]
        if _isVALU(inst):
            return not any(_isDtlLoad(x) or _isM0Update(x) for x in placed)
        if _isDtlLoad(inst) or _isM0Update(inst):
            return not any(_isVALU(x) for x in placed)
        return True

    def grAfterWaitGrDsReads(self, placer, pos, inst):
        """Keep GR (m0 / buffer_load) out of the window after wait_gr+barrier until ds_reads finish."""
        if not (_isBufferLoad(inst) or _isM0Update(inst)):
            return True
        if self.waitGrBarrierPos is None or self.maxDsReadAfterWaitGr < 0:
            return True
        if pos <= self.waitGrBarrierPos:
            return True
        return pos > self.maxDsReadAfterWaitGr

    # ── Adjusters: (placer, limit, inst) -> limit ──

    def deferGrAfterWaitGrDsReads(self, placer, limit, inst):
        """Once past wait_gr, start searching for GR only after post-barrier ds_reads."""
        if not (_isBufferLoad(inst) or _isM0Update(inst)):
            return limit
        if self.waitGrBarrierPos is None or self.maxDsReadAfterWaitGr < 0:
            return limit
        if limit <= self.waitGrBarrierPos:
            return limit
        return max(limit, self.maxDsReadAfterWaitGr + 1)

    def spreadBufferLoads(self, placer, limit, inst):
        """Spread buffer_load instructions evenly across available range."""
        if not _isBufferLoad(inst) or self.bufLoadMaxSlot <= 0:
            return limit
        if self.firstBufLoadPos is not None:
            stride = max(1, (self.bufLoadMaxSlot - self.firstBufLoadPos)
                         // self.numBufLoads)
            limit = max(limit, self.firstBufLoadPos
                        + self.bufLoadIdx * stride)
        self.bufLoadIdx += 1
        return limit

    def deferLiveSourceWrite(self, placer, limit, inst):
        """Hold a ds_read until its destination's prior reader has retired.

        earliestWriteShadow maps a destination VGPR to the earliest C0 MFMA
        shadow that may overwrite it. mfmaBase converts that global shadow
        into this cell. A register absent from the map is unconstrained.
        """
        if not _isDsRead(inst) or not self.earliestWriteShadow:
            return limit
        dest = _registerIndex(getattr(inst, "dst", None))
        floor = self.earliestWriteShadow.get(dest)
        if not floor:
            return limit
        local = floor - self.mfmaBase
        if local <= 0:
            return limit
        return max(limit, local * placer.slotsPerInterval)

    # ── Placement hook: (placer, pos, inst) -> None ──

    def trackPlacement(self, placer, pos, inst):
        """Update rule state after a successful placement."""
        if _isDsRead(inst):
            self.lastDsReadPos = max(self.lastDsReadPos, pos)
        if _isWaitCnt(inst):
            self.earliestWaitCntPos = min(self.earliestWaitCntPos, pos)
        if _isBufferLoad(inst) and self.firstBufLoadPos is None:
            self.firstBufLoadPos = pos

    # ── Per-path setup ──

    def resetPath(self):
        self._resetPath()

    def setupBufLoadSpreading(self, placer, pathInsts, order):
        """Compute buffer-load spreading bounds for a forward path.

        Reserves tail slots for non-buffer-load instructions in modules that
        follow the last GR module (e.g. GR_INC SRD updates, LDS buffer swaps).
        """
        self.numBufLoads = sum(1 for _, inst in pathInsts if _isBufferLoad(inst))
        if self.numBufLoads > 1:
            _, rawMax = placer.bounds(pathInsts[-1][0])
            grModuleIds = {mid for mid, inst in pathInsts if _isBufferLoad(inst)}
            lastGrIdx = max(order.index(m) for m in grModuleIds if m in order)
            tailModuleIds = set(order[lastGrIdx + 1:])
            numTailInsts = sum(1 for mid, _ in pathInsts if mid in tailModuleIds)
            # this is an approximation as we don't know exactly how many slots will be use by modules after the GR yet (in this codepath)
            self.bufLoadMaxSlot = max(0, rawMax - numTailInsts)

    def setPostWaitGrDsReadFence(self, placer) -> None:
        """After the wait_gr path is placed, record ds_read coverage for GR deferral."""
        wg_pos = None
        for pos, slot in enumerate(placer._placed):
            for _, inst in slot:
                if _isWaitGr(inst):
                    wg_pos = pos if wg_pos is None else min(wg_pos, pos)
        if wg_pos is None:
            self.waitGrBarrierPos = None
            self.maxDsReadAfterWaitGr = -1
            return
        max_ds = -1
        for pos, slot in enumerate(placer._placed):
            if pos <= wg_pos:
                continue
            for _, inst in slot:
                if _isDsRead(inst):
                    max_ds = max(max_ds, pos)
        self.waitGrBarrierPos = wg_pos
        self.maxDsReadAfterWaitGr = max_ds


def _classifyPaths(pathOrders, emittedModules):
    """Classify paths by wait_gr presence, sorted: wait_gr first, then by index."""
    paths = []
    for order in pathOrders:
        hasWaitGR = any(emittedModules[i].opType == "wait_gr" for i in order)
        paths.append((order, hasWaitGR))
    paths.sort(key=lambda p: (0 if p[1] else 1, p[0][0] if p[0] else 10**9))
    return paths


def _flattenPath(order, emittedModules, reverse=False):
    """Flatten a path of module indices into (moduleId, instruction) pairs."""
    pathInsts = [(mid, inst) for mid in order for inst in emittedModules[mid].instructions]
    if reverse:
        pathInsts.reverse()
    return pathInsts


def _instructionMetadata(emittedModules, mid, inst):
    """Return ``(rank, span)`` for one emitted instruction, or None."""
    instructions = emittedModules[mid].instructions
    instruction_index = next(
        (index for index, candidate in enumerate(instructions) if candidate is inst),
        -1)
    metadata = emittedModules[mid].instructionMetadata
    if instruction_index < 0 or instruction_index >= len(metadata):
        return None
    return metadata[instruction_index]


def _isWrapPrefetchPath(order, emittedModules):
    """A wait path whose local reads feed the next macro-tile."""
    return any(
        emittedModules[mid].opType == "lr"
        and getattr(emittedModules[mid].source, "mtIteration", 0) > 0
        for mid in order
    )


def _findPartitionZeroPrefetchOrder(preMfmaOrders, emittedModules):
    """Find partition 0's prefetch path from placement metadata.

    The invariant is an A/B local read in partition 0 whose consumer is the
    next partition of this macro-tile. Wait and barrier ops are not part of
    the signature: the uniform epoch keeps those at the frontier.
    """
    for order in preMfmaOrders:
        readModules = [
            emittedModules[mid] for mid in order
            if emittedModules[mid].opType == "lr"
        ]
        if not readModules:
            continue
        partitions = {
            getattr(em.source, "partition", None) for em in readModules
        }
        if partitions != {0}:
            continue
        hasNextPartitionData = any(
            getattr(em.source, "tensor", None) in ("A", "B")
            and getattr(em.source, "targetPartition", 0) == 1
            and getattr(em.source, "mtIteration", 0) == 0
            for em in readModules
        )
        readCount = sum(
            _isDsRead(inst)
            for em in readModules
            for inst in em.instructions
        )
        if hasNextPartitionData and readCount:
            return order
    return None


def _localReadStaysAheadOfWait(em, retainAllLocalReads=False) -> bool:
    """A local read the partition-zero wait has to cover.

    Scale reads and the current cell's A/B reads carry the issuing
    partition as their target. A hoisted read overlaps LDS the barrier
    is about to release. When one tile axis is a single partition, every
    local read on this path shares that LDS with the other waves, so the
    whole path stays ahead of the wait.
    """
    if em.opType != "lr":
        return False
    if retainAllLocalReads:
        return True
    source = em.source
    if getattr(source, "mustPrecedeP0Overwrite", False):
        return True
    return getattr(source, "targetPartition", None) == getattr(source, "partition", None)


def _splitPartitionZeroPrefetch(preMfmaOrders, emittedModules,
                                retainAllLocalReads=False):
    """Split partition 0's prefetch into a pre-MFMA prefix and its reads.

    Reads this cell consumes stay in the prefix, ahead of the local-read
    wait, as do hoisted reads of LDS the following barrier will release.
    A hoisted read is marked so a later pass can move it into the preloop
    or the previous K iteration once its VGPRs are free. A single-partition
    axis keeps every local read on the path here too: the other waves
    overwrite that LDS after the barrier, and those reads are not marked.
    The returned sync list is the barrier the phase scheduler may move to
    after the first MFMA, once the epoch wait owns the ordering.
    """
    order = _findPartitionZeroPrefetchOrder(preMfmaOrders, emittedModules)
    if order is None:
        return None

    prefixInsts = []
    syncInsts = []
    payloadOrder = []
    for mid in order:
        em = emittedModules[mid]
        if _localReadStaysAheadOfWait(em, retainAllLocalReads):
            protect = (not retainAllLocalReads
                       and getattr(em.source, "mustPrecedeP0Overwrite", False))
            for inst in em.instructions:
                if protect and _isDsRead(inst):
                    _p0OverwriteReadIds.add(id(inst))
                prefixInsts.append((mid, inst))
            continue
        if em.opType in ("wait_gr", "sync", "wait_lr"):
            for inst in em.instructions:
                if em.opType == "wait_lr" and _isWaitCnt(inst):
                    inst.dscnt = 0
                    inst.comment = "Wait for current-cell LR before p0 MFMA"
                item = (mid, inst)
                prefixInsts.append(item)
                if em.opType == "sync":
                    syncInsts.append(item)
            continue
        payloadOrder.append(mid)
    return (order, prefixInsts, payloadOrder,
            _flattenPath(payloadOrder, emittedModules), syncInsts)


def _vgprIndices(register):
    """Concrete VGPR indices one operand covers, or empty."""
    span = _registerSpan(register)
    if span is None or span[0] != "v" or not isinstance(span[1], int):
        return range(0)
    return range(span[1], span[1] + span[2])


def _instructionVgprs(inst):
    """VGPR indices an instruction reads or writes."""
    found = []
    for name in ("a", "b", "mxsa", "mxsb", "acc", "acc2", "dst"):
        found.append(getattr(inst, name, None))
    srcs = getattr(inst, "srcs", None)
    if srcs:
        found.extend(srcs)
    if isinstance(inst, LocalReadInstruction):
        found.extend(inst.getParams())
    indices = set()
    for register in found:
        indices.update(_vgprIndices(register))
    return indices


def _dsReadDstAddr(inst):
    """Destination and address operands of a ds_read."""
    params = list(inst.getParams())
    dst = params[0] if params else None
    addr = params[1] if len(params) > 1 else None
    return dst, addr


def _appendComment(inst, note):
    comment = inst.comment or ""
    if note not in comment:
        inst.comment = f"{comment}; {note}" if comment else note


def _takePrefixOverwriteReads(schedule):
    """Remove marked ds_reads that sit ahead of the first MFMA.

    Returns those reads and the prefix index they occupied, so a read
    that cannot move can go back in front of the entry wait.
    """
    items = list(schedule.flatitems())
    firstMfma = next((index for index, inst in enumerate(items) if _isMfma(inst)), None)
    if firstMfma is None:
        return [], 0
    taken = []
    kept = []
    insertAt = None
    for index, inst in enumerate(items):
        if (index < firstMfma
                and id(inst) in _p0OverwriteReadIds
                and _isDsRead(inst)):
            if insertAt is None:
                insertAt = len(kept)
            taken.append(inst)
            _p0OverwriteReadIds.discard(id(inst))
            continue
        kept.append(inst)
    if taken:
        schedule.setItems(kept)
    return taken, (insertAt if insertAt is not None else 0)


def _restorePrefixReads(schedule, reads, insertAt):
    items = list(schedule.flatitems())
    at = min(insertAt, len(items))
    items[at:at] = list(reads)
    schedule.setItems(items)


def _removeInstructions(schedule, reads):
    targets = {id(inst) for inst in reads}
    schedule.setItems(
        [inst for inst in schedule.flatitems() if id(inst) not in targets])


def _placeOverwriteReadsInPrior(prior, reads, gap):
    """Issue reads after their dests die and the LDS pointer has flipped.

    The source gap is the same MFMA distance a ds_read needs before a
    wait. Each read lands in the earliest later MFMA shadow that still
    has room under the three-read overflow, so the next copy's entry
    wait is not the instruction that issues it. Returns False without
    changing the prior when no such shadow exists.
    """
    items = list(prior.flatitems())
    dest = set()
    addrs = set()
    for read in reads:
        dst, addr = _dsReadDstAddr(read)
        dest.update(_vgprIndices(dst))
        addrs.update(_vgprIndices(addr))
    if not dest or not addrs:
        return False

    lastTouch = None
    xorAt = None
    xorCovered = set()
    mfmas = []
    for index, inst in enumerate(items):
        if dest & _instructionVgprs(inst):
            lastTouch = index
        if isinstance(inst, VXorB32):
            written = set(_vgprIndices(inst.dst)) & addrs
            if written:
                xorCovered |= written
                if addrs <= xorCovered:
                    xorAt = index
        if _isMfma(inst):
            mfmas.append(index)
    if lastTouch is None or xorAt is None:
        return False
    after = [index for index in mfmas if index > lastTouch]
    if len(after) < gap:
        return False

    earliest = max(after[gap - 1], xorAt)
    remaining = list(reads)
    assignments = {}
    previous = earliest
    for mfmaIndex in mfmas:
        if mfmaIndex <= earliest or not remaining:
            continue
        used = sum(_isDsRead(inst) for inst in items[previous + 1:mfmaIndex])
        room = _MXFP8_DATA_DS_READ_OVERFLOW - used
        if room > 0:
            assignments[mfmaIndex] = remaining[:room]
            remaining = remaining[room:]
        previous = mfmaIndex
    if remaining:
        return False

    for read in reads:
        _appendComment(read, "issued in the previous K iteration")
    for mfmaIndex in sorted(assignments, reverse=True):
        items[mfmaIndex:mfmaIndex] = assignments[mfmaIndex]
    prior.setItems(items)
    return True


def _seedOverwriteReadsInPreloop(preloop, reads, comments):
    """Copy the first C0's reads to just after the preloop barrier.

    The first trip has no previous K iteration. The address registers
    still name the same buffer the C0 entry would have read. Later
    preloop ds_reads cover the issue. Returns False without changing
    the preloop when the barrier is missing or the dests are live.
    """
    if preloop is None:
        return False
    items = list(preloop.flatitems())
    barrier = None
    for index, inst in enumerate(items):
        if isinstance(inst, SBarrier):
            barrier = index
    if barrier is None:
        return False
    dest = set()
    for read in reads:
        dest.update(_vgprIndices(_dsReadDstAddr(read)[0]))
    if any(dest & _instructionVgprs(inst) for inst in items[barrier + 1:]):
        return False
    seeded = []
    for read, comment in zip(reads, comments):
        clone = copy.deepcopy(read)
        note = "issued in the preloop"
        clone.comment = f"{comment}; {note}" if comment else note
        seeded.append(clone)
    items[barrier + 1:barrier + 1] = seeded
    preloop.setItems(items)
    return True


def relocatePartitionZeroOverwriteReads(copies, preloop=None, gap=None):
    """Move marked prefix ds_reads to the previous K iteration.

    ``copies`` is the mainloop bodies in issue order. Copy 0 also needs
    a preloop seed, because nothing has run the previous iteration yet.
    A read stays in its prefix when the dest VGPRs are still live, the
    LDS-pointer xor has not happened, or no later shadow can take it.
    Returns the dest VGPR indices that did move, so an exit path that
    issues the same read can wait for the one already in flight.
    """
    if gap is None:
        gap = _MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT
    extracted = [_takePrefixOverwriteReads(copy) for copy in copies]
    moved = set()
    for index, (reads, insertAt) in enumerate(extracted):
        if not reads:
            continue
        saved = [read.comment for read in reads]
        if len(copies) == 1:
            prior = copies[0]
        else:
            prior = copies[(index - 1) % len(copies)]
        placed = _placeOverwriteReadsInPrior(prior, reads, gap)
        # The first copy has no previous iteration. Seeding the preloop
        # is required together with the previous-copy slot; either one
        # alone would issue the read twice or skip the first trip.
        seeded = True
        if placed and index == 0:
            seeded = _seedOverwriteReadsInPreloop(preloop, reads, saved)
        if placed and seeded:
            for read in reads:
                moved.update(_vgprIndices(_dsReadDstAddr(read)[0]))
            continue
        if placed:
            _removeInstructions(prior, reads)
            for read, comment in zip(reads, saved):
                read.comment = comment
        _restorePrefixReads(copies[index], reads, insertAt)
    return moved


def guardExitOverwriteReread(schedule, dests):
    """Wait before an exit path re-reads VGPRs a prior iteration just wrote.

    The hot loop retires that read at the next entry wait. NGLL and NLL
    issue the same ds_read on the way out, and two ds_reads in flight to
    one dest are a WAW. A wait only on that path leaves the loop alone.
    """
    if not dests:
        return schedule
    dests = set(dests)
    items = list(schedule.flatitems())
    firstMfma = next((index for index, inst in enumerate(items) if _isMfma(inst)), len(items))
    for index, inst in enumerate(items[:firstMfma]):
        if not _isDsRead(inst):
            continue
        if not dests & set(_vgprIndices(_dsReadDstAddr(inst)[0])):
            continue
        already = any(
            _isWaitCnt(items[earlier]) and (
                items[earlier].kmcnt == 0 or getattr(items[earlier], "waitAll", False))
            for earlier in range(index))
        if already:
            return schedule
        items.insert(index, SWaitCnt(
            kmcnt=0,
            comment="retire the prior-iteration overwrite read before re-reading it"))
        schedule.setItems(items)
        return schedule
    return schedule


def _orderLocalReadsByFirstConsumer(pathInsts, emittedModules):
    """Sort complete local-read groups by their next MFMA consumer.

    A shorter group wins a tie, so one scale read issues before a
    two-instruction data read of the same MFMA. Pointer updates stay after
    the reads. Return None when a read has no rank, leaving that schedule
    in its original order.
    """
    reads = []
    nonReads = []
    for original_index, item in enumerate(pathInsts):
        mid, inst = item
        if not _isDsRead(inst):
            nonReads.append(item)
            continue
        metadata = _instructionMetadata(emittedModules, mid, inst)
        if metadata is None:
            return None
        rank, span = metadata
        rank = rank if rank is not None else float("inf")
        reads.append((rank, span, original_index, item))
    if not reads:
        return None
    reads.sort(key=lambda entry: entry[:3])
    return [item for _, _, _, item in reads] + nonReads


def _splitAfterFirstConsumerCohort(pathInsts, emittedModules):
    """Keep the earliest consumer's reads together and return the remainder.

    Address setup before the first read stays with that cohort. Reads that
    share a rank stay in one group, so the split does not cut an atom.
    """
    first_rank = None
    split = None
    saw_first_cohort = False
    for index, (mid, inst) in enumerate(pathInsts):
        if not _isDsRead(inst):
            continue
        metadata = _instructionMetadata(emittedModules, mid, inst)
        if metadata is None:
            return [], pathInsts
        rank, _span = metadata
        if rank is None:
            if saw_first_cohort:
                split = index
                break
            continue
        if first_rank is None:
            first_rank = rank
            saw_first_cohort = True
        elif rank != first_rank:
            split = index
            break
    if not saw_first_cohort:
        return [], pathInsts
    if split is None:
        split = len(pathInsts)
    return pathInsts[:split], pathInsts[split:]


def _orderGlobalReadBundlesByFirstConsumer(grPaths, emittedModules):
    """Order whole global-read bundles by their next MFMA consumer.

    A bundle is the address setup, m0, and load that share one metadata
    span. An SRD update or other tail stays after every bundle. Incomplete
    metadata returns the original order.
    """
    original = [item for path in grPaths for item in path]
    bundles = []
    tails = []
    original_index = 0
    saw_rank = False
    for path in grPaths:
        module_groups = []
        for item in path:
            if not module_groups or module_groups[-1][0] != item[0]:
                module_groups.append((item[0], [item]))
            else:
                module_groups[-1][1].append(item)
        in_tail = False
        path_tail = []
        for mid, module_items in module_groups:
            em = emittedModules[mid]
            if em.opType != "gr":
                in_tail = True
                path_tail.extend(module_items)
                continue
            if in_tail:
                return original
            metadata = em.instructionMetadata
            if len(metadata) != len(module_items):
                return original
            offset = 0
            while offset < len(module_items):
                rank, span = metadata[offset]
                if not isinstance(span, int) or span <= 0:
                    return original
                end = offset + span
                if end > len(module_items):
                    return original
                if any(metadata[index] != (rank, span) for index in range(offset, end)):
                    return original
                saw_rank = saw_rank or rank is not None
                bundles.append((
                    rank if rank is not None else float("inf"),
                    original_index,
                    module_items[offset:end],
                ))
                original_index += 1
                offset = end
        tails.extend(path_tail)
    if not saw_rank:
        return original
    bundles.sort(key=lambda entry: entry[:2])
    return [
        item
        for _rank, _index, bundle in bundles
        for item in bundle
    ] + tails


def _nextPlacementSlot(placer):
    for pos in range(placer.totalSlots - 1, -1, -1):
        if placer._placed[pos]:
            return pos + 1
    return 0


def _isMx8Schedule(mfmas) -> bool:
    return bool(mfmas) and all(
        isinstance(inst, MXMFMAInstruction) and inst.getIssueLatency() == 8
        for inst in mfmas
    )


def _registerSpan(register):
    """Return (kind, start, count) for one register operand.

    A negative regIdx is a named scalar such as sgprSrdA. Those names do
    not share an integer range, so identity is the name.
    """
    if register is None or isinstance(register, (int, float)):
        return None
    kind = getattr(register, "regType", None)
    if kind not in ("s", "v", "m"):
        return None
    count = getattr(register, "regNum", 1) or 1
    count = int(count) if isinstance(count, int) else 1
    index = getattr(register, "regIdx", None)
    if isinstance(index, int) and index >= 0:
        return (kind, index, count)
    name = getattr(register, "regName", None)
    if not name:
        return None
    return (kind, name, count)


def _spansOverlap(left, right) -> bool:
    if left is None or right is None or left[0] != right[0]:
        return False
    if isinstance(left[1], int) and isinstance(right[1], int):
        return left[1] < right[1] + right[2] and right[1] < left[1] + left[2]
    if isinstance(left[1], str) and isinstance(right[1], str):
        if left[1] == right[1]:
            return True
        # sgprSrdA+1 is a lane of the sgprSrdA block the load reads.
        return (
            right[1].startswith(left[1] + "+")
            or left[1].startswith(right[1] + "+")
        )
    return False


def _instructionRegisterSpans(inst):
    """Yield destination and source spans for a placed instruction."""
    dst = _registerSpan(getattr(inst, "dst", None))
    if dst is not None:
        yield "dst", dst
    sources = getattr(inst, "srcs", None)
    if sources:
        for src in sources:
            span = _registerSpan(src)
            if span is not None:
                yield "src", span
        return
    getParams = getattr(inst, "getParams", None)
    if getParams is None:
        return
    for value in getParams():
        span = _registerSpan(value)
        if span is not None:
            yield "src", span


def _vmemSourceSpans(setup, load):
    spans = []
    for inst in (setup, load):
        for role, span in _instructionRegisterSpans(inst):
            if role == "src":
                spans.append(span)
    return spans


def _instructionDestSpans(inst):
    return [
        span for role, span in _instructionRegisterSpans(inst)
        if role == "dst"
    ]


def _readsSpans(inst, spans) -> bool:
    if not spans:
        return False
    for role, span in _instructionRegisterSpans(inst):
        if role == "src" and any(
                _spansOverlap(span, wanted) for wanted in spans):
            return True
    return False


def _vmemSlideBlocker(inst, sourceSpans) -> bool:
    """Whether moving a VMEM bundle past inst would break the bundle."""
    if isinstance(inst, TextBlock):
        return False
    if _isBarrier(inst) or _isM0Update(inst) or _isBufferLoad(inst):
        return True
    if _isWaitCnt(inst) and getattr(inst, "vlcnt", -1) >= 0:
        return True
    for role, span in _instructionRegisterSpans(inst):
        if role == "dst" and any(
                _spansOverlap(span, source) for source in sourceSpans):
            return True
    return False


def _liftPlaced(placer, sequence):
    """Remove each captured item from the slot it currently occupies."""
    for pos, item in sequence:
        slot = placer._placed[pos]
        for index, candidate in enumerate(slot):
            if candidate is item:
                del slot[index]
                break


def _insertBefore(placer, cursor, items) -> int:
    """Insert items immediately before an instruction already on the schedule."""
    for pos, slot in enumerate(placer._placed):
        for index, candidate in enumerate(slot):
            if candidate is cursor:
                for offset, item in enumerate(items):
                    slot.insert(index + offset, item)
                return pos
    raise RuntimeError("right-align cursor is not on the schedule")


def _firstScheduleBlocker(placer, start: int, sourceSpans) -> int:
    """Return the first slot at or after start that this instruction must not pass."""
    for pos in range(max(0, start), placer.totalSlots):
        if any(
                _vmemSlideBlocker(inst, sourceSpans)
                for _mid, inst in placer._placed[pos]):
            return pos
    return placer.totalSlots


_GR_ADDRESS_NAMES = frozenset((
    "SrdA", "SrdB", "SrdMXSA", "SrdMXSB",
    "LocalWriteBaseAddrA", "LocalWriteBaseAddrB",
    "LocalWriteBaseAddrMXSA", "LocalWriteBaseAddrMXSB",
))


def _scalarDestName(inst) -> str:
    """Return the base name of a scalar destination, without a lane offset."""
    dst = getattr(inst, "dst", None)
    if dst is None or getattr(dst, "regType", None) != "s":
        return ""
    holder = getattr(dst, "holderName", None)
    if isinstance(holder, str) and holder:
        return holder.split("+", 1)[0]
    regName = getattr(dst, "regName", None)
    name = getattr(regName, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(regName, str) and regName:
        return regName.split("+", 1)[0]
    return ""


def _isGrAddressUpdate(inst) -> bool:
    """Whether inst writes a global-read base or a direct-to-LDS base.

    The SRD advance (and the scale num_records shrink) decides which
    global address a later buffer_load uses. The LocalWriteBaseAddr xor
    decides which LDS half a later m0 names. The scale local-read pointer
    swap writes a VGPR and stays out of this list.
    """
    return _scalarDestName(inst) in _GR_ADDRESS_NAMES


def _bundleEarliestSlot(setupPos: int, width: int) -> int:
    """Keep an m0 from moving into an earlier slot of its own interval."""
    earliest = (setupPos // width) * width
    if earliest < setupPos:
        earliest = (setupPos // width + 1) * width
    return earliest


def _blocksEarlierVmem(inst, sources, dests) -> bool:
    """Whether a VMEM-stream instruction must stay after inst.

    LDS readers of the buffer a load overwrites all sit before its
    overwrite barrier, so the barrier is the only LDS fence needed.
    """
    if isinstance(inst, TextBlock):
        return False
    if _isBarrier(inst) or _isM0Update(inst) or _isBufferLoad(inst):
        return True
    if _isWaitCnt(inst) and getattr(inst, "vlcnt", -1) >= 0:
        return True
    for role, span in _instructionRegisterSpans(inst):
        if role == "dst" and any(
                _spansOverlap(span, wanted) for wanted in sources + dests):
            return True
        if role == "src" and any(
                _spansOverlap(span, wanted) for wanted in dests):
            return True
    return False


def _earliestVmemSlot(placer, pos, insts) -> int:
    """Return the first slot a VMEM-stream instruction may move back to."""
    sources = []
    dests = []
    for inst in insts:
        for role, span in _instructionRegisterSpans(inst):
            (sources if role == "src" else dests).append(span)
    for slot in range(min(pos, placer.totalSlots) - 1, -1, -1):
        if any(_blocksEarlierVmem(inst, sources, dests)
               for _mid, inst in placer._placed[slot]):
            return slot + 1
    return 0


_SCC_NEUTRAL_PREFIXES = (
    "SMov", "SWait", "SBarrier", "SNop", "SSetPrio", "SCBranch", "SBranch",
    "SLoad", "SStore", "SEndpgm", "SSleep", "SSet", "SGet")
_SCC_READERS = ("SAddCU32", "SSubBU32", "SCSelect", "SCBranchSCC")


def _writesScc(inst) -> bool:
    name = type(inst).__name__
    return (name.startswith("S") and not isinstance(inst, TextBlock)
            and not name.startswith(_SCC_NEUTRAL_PREFIXES))


def _readsScc(inst) -> bool:
    return type(inst).__name__.startswith(_SCC_READERS)


def _streamCarriesIntact(placer, streamIds) -> bool:
    """Whether every SCC read in the stream still sees its stream producer."""
    lastWriter = None
    for slot in placer._placed:
        for _mid, inst in slot:
            if _readsScc(inst) and id(inst) in streamIds:
                if lastWriter is None or id(lastWriter) not in streamIds:
                    return False
            if _writesScc(inst):
                lastWriter = inst
    return True


def _rightAlignM0AndVmem(placer, endSlot=None):
    """Place the direct-to-LDS stream from the back of that list.

    The stream ends before endSlot when given, otherwise at the last slot.
    Returns False when some instruction found no legal slot and was stacked
    onto the one behind it.

    The stream is every m0, every direct-to-LDS load, and every SALU that
    writes a global-read SRD or a LocalWriteBaseAddr. Local reads and waits
    stay where the forward pass put them. The list is filled from its last
    instruction toward its first, and each instruction lands strictly before
    the one already placed behind it, so an m0 cannot pass another m0 and a
    load cannot pass the address update that belongs on its other side. A
    load and the m0 directly before it occupy slot 0 and slot 1 of one
    interval. The m0 is SALU, so it may use slot 0. When slot 0 is empty
    and a later slot in that shadow is empty, every instruction below the
    hole slides down by one slot, in order, and the bundle takes the two
    slots that opens at the front. Any other instruction in the list takes
    one slot under the
    same interval cap. A placement stays at or after the forward pass
    unless every such slot is already taken, in which case the instruction
    is inserted immediately before the one behind it. Nothing crosses a
    barrier, a VMEM wait, or a write of a register it reads.
    """
    width = placer.slotsPerInterval
    if width < 2 or placer.maxPerSlot != 1 or placer.totalSlots < 2:
        return True
    sequence = []
    for pos in range(placer.totalSlots):
        for item in placer._placed[pos]:
            if isinstance(item[1], TextBlock):
                continue
            if (_isM0Update(item[1]) or _isDtlLoad(item[1])
                    or _isGrAddressUpdate(item[1])):
                sequence.append((pos, item))
    if not sequence:
        return True
    _liftPlaced(placer, sequence)
    stacked = False

    def floor(pos, insts):
        # Toward a memory boundary the stream may also move ahead of the
        # forward pass, down to the nearest instruction it must not cross.
        if endSlot is None:
            return pos
        return _earliestVmemSlot(placer, pos, insts)

    def place_single(pos, item, limit):
        inst = item[1]
        sources = _vmemSourceSpans(inst, inst)
        hardLimit = min(limit, _firstScheduleBlocker(placer, pos + 1, sources))
        lowest = floor(pos, [inst])
        for slot in range(hardLimit - 1, lowest - 1, -1):
            if placer._canPlace(slot, inst):
                placer.place(slot, item)
                return slot, item
        return None

    def place_bundle(setupPos, setupItem, loadPos, loadItem, limit):
        setup = setupItem[1]
        load = loadItem[1]
        sources = _vmemSourceSpans(setup, load)
        hardLimit = min(
            limit, _firstScheduleBlocker(placer, setupPos + 1, sources))
        earliest = _bundleEarliestSlot(floor(setupPos, [setup, load]), width)
        latest = (hardLimit - 2) // width if hardLimit >= 2 else -1

        def take(slot, item):
            held = placer._placed[slot]
            for index, candidate in enumerate(held):
                if candidate is item:
                    del held[index]
                    return
            raise RuntimeError("pushed instruction is not in its slot")

        def undo(records):
            for origin, dest_slot, moved_item in reversed(records):
                take(dest_slot, moved_item)
                placer._placed[origin].append(moved_item)

        def physical(slot):
            return [
                item for item in placer._placed[slot]
                if not isinstance(item[1], TextBlock)
            ]

        def slide_open(base, end):
            """Slide this shadow down by one slot into its last hole.

            Slot 0 has to be empty. Every instruction below the hole moves
            to the next slot, from the hole backward, so order is kept and
            the slide opens slot 1 as well as slot 0.
            """
            if physical(base):
                return None
            hole = None
            for pos in range(end - 1, base, -1):
                if not physical(pos):
                    hole = pos
                    break
            if hole is None:
                return None
            movers = []
            for pos in range(base + 1, hole):
                held = physical(pos)
                if not held:
                    continue
                if len(held) != 1:
                    return None
                inst = held[0][1]
                if (_isBarrier(inst) or _isWaitCnt(inst)
                        or _isM0Update(inst) or _isBufferLoad(inst)):
                    return None
                movers.append((pos, held[0]))
            moved = []
            for pos, item in reversed(movers):
                inst = item[1]
                take(pos, item)
                dest = pos + 1
                if not placer._canPlace(dest, inst):
                    placer._placed[pos].append(item)
                    undo(moved)
                    return None
                placer._placed[dest].append(item)
                moved.append((pos, dest, item))
            if physical(base) or physical(base + 1):
                undo(moved)
                return None
            return moved

        for interval in range(latest, earliest // width - 1, -1):
            # The m0 is SALU and may use the interval's first slot.
            # The load takes the slot directly after it.
            setupSlot = interval * width
            loadSlot = setupSlot + 1
            end = min(setupSlot + width, hardLimit)
            if setupSlot < earliest or loadSlot >= end:
                continue
            moved = slide_open(setupSlot, end)
            if moved is None:
                continue
            fits = (
                (placer.maxInstructionsPerInterval is None
                 or placer._intervalInstructionCount(setupSlot) + 2
                 <= placer.maxInstructionsPerInterval)
                and placer._canPlace(setupSlot, setup)
                and placer._canPlace(loadSlot, load))
            if fits:
                placer.place(setupSlot, setupItem)
                placer.place(loadSlot, loadItem)
                return setupSlot, setupItem
            undo(moved)
        if (setupPos < hardLimit and loadPos < hardLimit
                and setupPos < loadPos
                and placer._canPlace(setupPos, setup)
                and placer._canPlace(loadPos, load)):
            placer.place(setupPos, setupItem)
            placer.place(loadPos, loadItem)
            return setupPos, setupItem
        return None

    def attach(items):
        nonlocal stacked
        stacked = True
        if cursor is None:
            slot = max(0, limit - 1)
            for item in items:
                placer.place(slot, item)
            return slot, items[0]
        return _insertBefore(placer, cursor, items), items[0]

    cursor = None
    limit = placer.totalSlots
    if endSlot is not None:
        limit = max(1, min(limit, endSlot))
    index = len(sequence) - 1
    while index >= 0:
        pos, item = sequence[index]
        if (index > 0
                and _isDtlLoad(item[1])
                and _isM0Update(sequence[index - 1][1][1])):
            setupPos, setupItem = sequence[index - 1]
            placed = place_bundle(setupPos, setupItem, pos, item, limit)
            if placed is None:
                placed = attach([setupItem, item])
            limit, cursor = placed
            index -= 2
            continue
        placed = place_single(pos, item, limit)
        if placed is None:
            placed = attach([item])
        limit, cursor = placed
        index -= 1
    if stacked:
        return False
    if endSlot is not None:
        return _streamCarriesIntact(
            placer, {id(item[1]) for _pos, item in sequence})
    return True


def _orderReadsAroundLiveSourceWrites(pathInsts, earliestWriteShadow, mfmaBase):
    """Issue unconstrained ds_reads before one whose VGPR is still live.

    The placer walks a path with one cursor. A personal overwrite floor
    raises that cursor, and every later instruction inherits it. Reads that
    do not write the live VGPR stay in their existing order and keep the
    earlier slots. The constrained read then waits at its own floor, and
    pointer updates stay behind the reads they belong to.
    """
    if not pathInsts or not earliestWriteShadow:
        return pathInsts

    def blocked(inst):
        if not _isDsRead(inst):
            return False
        dest = _registerIndex(getattr(inst, "dst", None))
        floor = earliestWriteShadow.get(dest) or 0
        return floor - mfmaBase > 0

    if not any(blocked(inst) for _, inst in pathInsts):
        return pathInsts

    prefix = []
    unconstrained = []
    deferred = []
    tail = []
    seenRead = False
    for item in pathInsts:
        inst = item[1]
        if not _isDsRead(inst):
            if seenRead:
                tail.append(item)
            else:
                prefix.append(item)
            continue
        seenRead = True
        if blocked(inst):
            deferred.append(item)
        else:
            unconstrained.append(item)
    return prefix + unconstrained + deferred + tail


_DATA_TENSORS = ("A", "B")


def _commentTensor(inst):
    """Operand tensor named by a local-read or direct-to-LDS comment."""
    comment = (getattr(inst, "comment", "") or "").lstrip()
    for marker in ("TrSubtile", "Subtile"):
        if comment.startswith(marker):
            rest = comment[len(marker):]
            for tensor in ("MXSA", "MXSB", "SA", "SB", "A", "B"):
                if rest.startswith(tensor):
                    return tensor
            return None
    token = comment.split(" ", 1)[0]
    if token in ("MXSA", "MXSB", "SA", "SB", "A", "B"):
        return token
    return None


def _isScalarAlu(inst) -> bool:
    """Scalar ALU that does not touch LDS, VMEM, or a wait."""
    if (isinstance(inst, TextBlock) or _isMfma(inst) or _isVALU(inst)
            or _isDsRead(inst) or _isBufferLoad(inst) or _isBarrier(inst)
            or _isM0Update(inst) or _isWaitCnt(inst)):
        return False
    if isinstance(inst, (SMemLoadInstruction, LocalWriteInstruction)):
        return False
    return isinstance(inst, CommonInstruction)


def _precededByVmemWait(items, index: int) -> bool:
    """Whether this barrier publishes LDS a vmcnt just made visible.

    The overwrite handshake is preceded by an lgkmcnt or a local read.
    The publish barrier sits on the vmcnt that retires the stores the
    following ds_reads consume, with only math between the two.
    """
    for cursor in range(index - 1, -1, -1):
        inst = items[cursor]
        if (isinstance(inst, TextBlock) or _isMfma(inst)
                or _isVALU(inst) or _isScalarAlu(inst)):
            continue
        return _isWaitCnt(inst) and inst.vlcnt >= 0
    return False


def _canCrossBeforeStore(inst, storeTensor) -> bool:
    """Whether inst may issue before the barrier that guards a store."""
    if isinstance(inst, TextBlock) or _isMfma(inst) or _isVALU(inst) or _isScalarAlu(inst):
        return True
    if _isDsRead(inst):
        readTensor = _commentTensor(inst)
        return (readTensor is not None and storeTensor is not None
                and readTensor != storeTensor)
    return False


def _storeInsertSlot(items, barrierIndex: int):
    """Slot of the m0 that feeds the first A/B store after this barrier.

    Returns None when the barrier publishes a completed store, or when a
    same-tensor local read, another barrier, or a vmcnt sits in front of
    that store.
    """
    loadIndex = None
    storeTensor = None
    for cursor in range(barrierIndex + 1, len(items)):
        inst = items[cursor]
        if isinstance(inst, TextBlock):
            continue
        if _isBufferLoad(inst) and _commentTensor(inst) in _DATA_TENSORS:
            loadIndex = cursor
            storeTensor = _commentTensor(inst)
            break
        if _isBarrier(inst) or (_isWaitCnt(inst) and inst.vlcnt >= 0):
            return None
    if loadIndex is None:
        return None
    insertAt = loadIndex
    if loadIndex > 0 and _isM0Update(items[loadIndex - 1]):
        insertAt = loadIndex - 1
    for cursor in range(barrierIndex + 1, insertAt):
        if not _canCrossBeforeStore(items[cursor], storeTensor):
            return None
    return insertAt


def _isCodeModule(inst) -> bool:
    return hasattr(inst, "setItems") and hasattr(inst, "flatitems")


def _slideOverwriteBarriersToVmem(module) -> None:
    """Move an overwrite barrier to the store it guards.

    The barrier stalls a wave until the others arrive. Math and local
    reads of the other tensor do not depend on that handshake, so they
    stay in front and cover the skew. The direct-to-LDS store issues on
    the first cycle after every wave is in. A barrier that publishes
    LDS for the reads after a vmcnt stays where it is.
    """
    items = list(module.items())
    index = 0
    changed = False
    while index < len(items):
        inst = items[index]
        if _isCodeModule(inst):
            _slideOverwriteBarriersToVmem(inst)
            index += 1
            continue
        if not _isBarrier(inst) or _precededByVmemWait(items, index):
            index += 1
            continue
        dest = _storeInsertSlot(items, index)
        if dest is None or dest <= index + 1:
            index += 1
            continue
        barrier = items.pop(index)
        items.insert(dest - 1, barrier)
        changed = True
        index = dest
    if changed:
        module.setItems(items)


def _isFullLgkmDrain(inst) -> bool:
    """Whether inst waits until no ds_read or scalar load is outstanding."""
    if not isinstance(inst, SWaitCnt):
        return False
    if getattr(inst, "waitAll", False):
        return True
    if inst.dscnt < 0 and inst.kmcnt < 0:
        return False
    lgkm = ((0 if inst.dscnt < 0 else inst.dscnt)
            + (0 if inst.kmcnt < 0 else inst.kmcnt))
    return lgkm == 0


def _incrementsLgkm(inst) -> bool:
    return isinstance(inst, (LocalReadInstruction, LocalWriteInstruction, SMemLoadInstruction))


def dropDominatedLgkmWaits(module):
    """Remove an lgkmcnt(0) that follows another with no new LDS or scalar load.

    A count of 0 drains every outstanding ds_read and s_load. The next
    full drain is redundant until one of those issues again. A wait that
    also carries vmcnt keeps that half.
    """
    state = {"zero": False}

    def walk(container) -> None:
        kept = []
        for inst in list(container.items()):
            if _isCodeModule(inst):
                walk(inst)
                kept.append(inst)
                continue
            if _isFullLgkmDrain(inst):
                if getattr(inst, "waitAll", False):
                    state["zero"] = True
                    kept.append(inst)
                    continue
                if state["zero"]:
                    if inst.vlcnt >= 0 or inst.vscnt >= 0:
                        inst.dscnt = -1
                        inst.kmcnt = -1
                        kept.append(inst)
                    continue
                state["zero"] = True
                kept.append(inst)
                continue
            if _isWaitCnt(inst) and (inst.dscnt > 0 or inst.kmcnt > 0):
                state["zero"] = False
            elif _incrementsLgkm(inst):
                state["zero"] = False
            kept.append(inst)
        container.setItems(kept)

    walk(module)
    return module


def _bodyLeaves(module):
    """(container, instruction) for every leaf of a nested module, in order."""
    out = []
    for inst in module.items():
        if _isCodeModule(inst):
            out.extend(_bodyLeaves(inst))
        else:
            out.append((module, inst))
    return out


def _isOrderFence(inst) -> bool:
    """Nothing moves across a wait, a barrier, a branch, a label, or an
    instruction that only spaces hazards or priorities."""
    if isinstance(inst, TextBlock):
        return False
    return (_isWaitCnt(inst) or _isBarrier(inst)
            or not isinstance(inst, (CommonInstruction, LocalReadInstruction,
                                     GlobalReadInstruction, MFMAInstruction,
                                     MXMFMAInstruction))
            or type(inst).__name__.startswith(
                ("SCBranch", "SBranch", "SNop", "SSetPrio", "SSleep")))


def _operandsVisible(inst) -> bool:
    """Whether dst and srcs name every operand, so no hidden VCC or SGPR."""
    sources = getattr(inst, "srcs", None)
    if sources is None:
        return False
    dst = getattr(inst, "dst", None)
    return len(inst.getParams()) == len(sources) + (dst is not None)


def _mayReorder(first, second, scc=True) -> bool:
    """Whether second may issue ahead of first.

    Register spans, SCC unless scc is False, and the m0 a direct-to-LDS
    load reads order them. An MFMA only touches VGPRs and accumulators.
    """
    for inst in (first, second):
        if (not (_isMfma(inst) or _isDsRead(inst) or _isBufferLoad(inst))
                and not _operandsVisible(inst)):
            return False
    if _isMfma(first) or _isMfma(second):
        return not (_instructionVgprs(first) & _instructionVgprs(second))
    if scc and (_writesScc(first) and (_writesScc(second) or _readsScc(second))
                or _readsScc(first) and _writesScc(second)):
        return False

    def spans(inst):
        if not (_isDsRead(inst) or _isBufferLoad(inst)):
            found = list(_instructionRegisterSpans(inst))
        else:
            params = list(inst.getParams())
            found = [("dst", _registerSpan(params[0]))] if params else []
            found += [("src", _registerSpan(p)) for p in params[1:]]
            if _isBufferLoad(inst):
                found.append(("src", ("m", 0, 1)))
        # A named scalar's regName is a RegName, which compares by
        # identity; its text is what _spansOverlap matches lanes on.
        return [(role, span if isinstance(span[1], int)
                 else (span[0], str(span[1]), span[2]))
                for role, span in found if span is not None]

    left, right = spans(first), spans(second)
    for roleA, a in left:
        for roleB, b in right:
            if "dst" in (roleA, roleB) and _spansOverlap(a, b):
                return False
    return True


def _mfmaGaps(leaves):
    """Leaf indices of each gap after an MFMA, in body order."""
    gaps = []
    for index, (_container, inst) in enumerate(leaves):
        if _isMfma(inst):
            gaps.append([])
        elif gaps:
            gaps[-1].append(index)
    return gaps


def _code(leaves, gap):
    return [i for i in gap if not isinstance(leaves[i][1], TextBlock)]


def _moveLeaves(leaves, indices, anchor, before):
    """Move leaves next to the leaf at anchor, in their current order."""
    for index in indices:
        container, inst = leaves[index]
        container.removeItem(inst)
    container, target = leaves[anchor]
    items = list(container.items())
    at = next(i for i, item in enumerate(items) if item is target)
    at = at if before else at + 1
    container.setItems(
        items[:at] + [leaves[i][1] for i in indices] + items[at:])


def _withComments(leaves, gap, index):
    """index and the comment lines directly ahead of it in its gap."""
    start = index
    while (start - 1 in gap
           and isinstance(leaves[start - 1][1], TextBlock)):
        start -= 1
    return list(range(start, index + 1))


def _leadWithinGap(leaves, gap) -> bool:
    """Move the gap's first independent non-VALU ahead of its VALUs.

    An SCC producer takes its consumer along.
    """
    code = _code(leaves, gap)
    for position, index in enumerate(code[1:], 1):
        inst = leaves[index][1]
        if _isOrderFence(inst):
            return False
        if _isVALU(inst) or _isM0Update(inst) or _isBufferLoad(inst):
            continue
        group = [index]
        for follower in code[position + 1:]:
            if not _readsScc(leaves[follower][1]):
                break
            group.append(follower)
        ahead = code[:position]
        if all(_mayReorder(leaves[k][1], leaves[g][1])
               for k in ahead for g in group):
            moved = [i for g in group for i in _withComments(leaves, gap, g)]
            _moveLeaves(leaves, moved, _withComments(leaves, gap, code[0])[0],
                        before=True)
            return True
    return False


def _isDtlRequest(inst) -> bool:
    return _isM0Update(inst) or _isBufferLoad(inst)


def _onlyDirectToLdsVmem(leaves) -> bool:
    """Whether every VMEM op of the body is a load into LDS.

    Then a vmcnt retires nothing a VALU reads or writes.
    """
    return not any(
        _isBufferStore(inst) or (_isBufferLoad(inst)
                                 and list(inst.getParams())[0] is not None)
        for _container, inst in leaves)


def _isVmemOnlyWait(inst) -> bool:
    return (_isWaitCnt(inst) and not getattr(inst, "waitAll", False)
            and inst.vlcnt >= 0 and inst.dscnt < 0 and inst.kmcnt < 0
            and inst.vscnt < 0)


def _pullBackValus(leaves, gaps, gapIndex, run, maxInstructions) -> bool:
    """Move VALUs to the end of the nearest earlier gap that starts with
    another instruction, has room, and holds no direct-to-LDS request and
    no fence but a wait.

    A barrier orders LDS only. A VALU crosses it, and crosses a vmcnt
    when every VMEM op of the body loads into LDS.
    """
    if not all(_operandsVisible(leaves[i][1]) for i in run):
        return False
    gap = gaps[gapIndex]
    ldsOnly = _onlyDirectToLdsVmem(leaves)

    def crossable(index):
        inst = leaves[index][1]
        if _isBarrier(inst) or (ldsOnly and _isVmemOnlyWait(inst)):
            return True
        return not _isOrderFence(inst) and all(
            _mayReorder(inst, leaves[i][1]) for i in run)

    if not all(crossable(i) for i in _code(leaves, gap) if i < run[0]):
        return False
    cursor = gap[0] - 1
    for target in range(gapIndex - 1, -1, -1):
        if not crossable(cursor):
            return False
        earlier = _code(leaves, gaps[target])
        if (earlier and not _isVALU(leaves[earlier[0]][1])
                and len(earlier) + len(run) <= maxInstructions
                and not any(_isDtlRequest(leaves[i][1]) or (
                    _isOrderFence(leaves[i][1]) and not _isWaitCnt(leaves[i][1]))
                            for i in earlier)):
            moved = [i for r in run for i in _withComments(leaves, gap, r)]
            _moveLeaves(leaves, moved, gaps[target][-1], before=False)
            return True
        if not all(crossable(i) for i in earlier):
            return False
        cursor = (gaps[target][0] if gaps[target] else cursor) - 1
    return False


def keepValuBehindLeader(module, maxInstructions=_MXFP8_FOLLOWER_SLOTS):
    """Keep every VALU of a half-disjoint body behind another instruction
    and out of direct-to-LDS gaps.

    A VALU first after an MFMA waits out the MFMA's issue. The cells
    prefer a slot behind a leader, but a cell cannot see its neighbours,
    a path placed early often finds none, and a request that finds no
    legal interval falls back to one with a VALU. Once the body is
    complete, a later independent instruction of the same gap moves
    ahead of a leading VALU; otherwise the VALUs move back to the nearest
    earlier gap that has a leader, room, and no request. They never cross
    an instruction they depend on, a local-read wait, or a branch. What
    cannot move stays.
    """
    unmovable = set()
    while True:
        leaves = _bodyLeaves(module)
        gaps = _mfmaGaps(leaves)
        for gapIndex, gap in enumerate(gaps):
            code = _code(leaves, gap)
            valus = [i for i in code if _isVALU(leaves[i][1])
                     and id(leaves[i][1]) not in unmovable]
            if not valus:
                continue
            if valus[0] == code[0] and _leadWithinGap(leaves, gap):
                break
            if any(_isDtlRequest(leaves[i][1]) for i in code):
                run = valus[:1]
            elif valus[0] == code[0]:
                run = []
                for index in code:
                    if not _isVALU(leaves[index][1]):
                        break
                    run.append(index)
            else:
                continue
            if _pullBackValus(leaves, gaps, gapIndex, run, maxInstructions):
                break
            unmovable.update(id(leaves[i][1]) for i in run)
        else:
            return module


def _saluFirstRank(inst):
    """Issue class in a SALU-first gap, or None for an instruction that
    keeps its place."""
    if _isM0Update(inst) or _isBufferLoad(inst):
        return 1
    if _isScalarAlu(inst):
        return 0
    if _isDsRead(inst):
        return 2
    if _isVALU(inst):
        return 3
    return None


def _saluFirstSegments(leaves, start, end):
    """Runs of leaves[start:end] in one container with no fence inside."""
    segments, current = [], []
    for index in range(start, end):
        container, inst = leaves[index]
        if current and leaves[current[-1]][0] is not container:
            segments.append(current)
            current = []
        if isinstance(inst, TextBlock) or (
                not _isOrderFence(inst) and _saluFirstRank(inst) is not None):
            current.append(index)
        else:
            segments.append(current)
            current = []
    segments.append(current)
    return [s for s in segments if s]


def _joinAt(leaves, code, position, isHead, unitOf):
    """Append code[position] to the unit of the nearest earlier code item
    isHead accepts. A unit holds at most one load."""
    member = code[position]
    for back in range(position - 1, -1, -1):
        head = code[back]
        if not isHead(leaves[head][1]):
            continue
        if any(_isBufferLoad(leaves[i][1]) for i in unitOf[head]):
            return False
        unitOf[head].append(member)
        unitOf[member] = unitOf[head]
        return True
    return False


def _orderSaluFirst(leaves, segment, sccLiveOut, branchFollows):
    """Units of one segment in SALU-first order, or None to keep it.

    An m0 write and its buffer_load form one unit, as do an SCC producer
    and its consumers. Units keep every order their members need, and a
    segment whose units cannot be ordered stays as it is.
    """
    code = [i for i in segment if not isinstance(leaves[i][1], TextBlock)]
    if len(code) < 2:
        return None
    units, unitOf = [], {}
    sccStrict = False
    for position, index in enumerate(code):
        inst = leaves[index][1]
        joined = False
        if _readsScc(inst):
            joined = _joinAt(leaves, code, position, _writesScc, unitOf)
            sccStrict |= not joined
        elif _isBufferLoad(inst):
            joined = _joinAt(leaves, code, position, _isM0Update, unitOf)
        if not joined:
            unitOf[index] = [index]
            units.append(unitOf[index])
    order = {index: n for n, index in enumerate(code)}
    after = {id(u): set() for u in units}
    for a in units:
        for b in units:
            if a is b:
                continue
            if any(order[x] < order[y] and not _mayReorder(
                       leaves[x][1], leaves[y][1], scc=sccStrict)
                   for x in a for y in b):
                after[id(b)].add(id(a))
    sccUnits = [u for u in units if any(_writesScc(leaves[i][1]) for i in u)]
    if sccLiveOut and sccUnits:
        last = max(sccUnits, key=lambda u: max(order[i] for i in u))
        pinned = units if (branchFollows and last is units[-1]) else sccUnits
        for u in pinned:
            if u is not last:
                after[id(last)].add(id(u))
    placed, result = set(), []
    while len(result) < len(units):
        ready = [u for u in units
                 if id(u) not in placed and after[id(u)] <= placed]
        if not ready:
            return None
        pick = min(ready, key=lambda u: (_saluFirstRank(leaves[u[0]][1]),
                                         order[u[0]]))
        placed.add(id(pick))
        result.append(pick)
    if result == units and all(len(u) == 1 for u in units):
        return None
    return result


def issueSaluFirst(module):
    """Order each MFMA gap of a half-disjoint body SALU first.

    After the MFMA come the SALU, then each m0 write with its
    direct-to-LDS load, then the ds_reads, then the VALU. Only the order
    inside a gap changes. Waits, barriers, branches and labels stay in
    place and nothing crosses them, so a wait stays before what it
    guards and a barrier stays alone. An SCC producer keeps its consumer
    right behind it, the last SCC write of a run stays last when a later
    instruction reads it, and an s_cmp stays right before its branch.
    A direct-to-LDS load and a ds_read issued between the same two
    barriers touch different LDS buffers, so registers alone order them.
    """
    leaves = _bodyLeaves(module)
    bounds = [i for i, (_c, inst) in enumerate(leaves) if _isMfma(inst)]
    starts = [0] + [b + 1 for b in bounds]
    ends = bounds + [len(leaves)]
    for start, end in zip(starts, ends):
        for segment in _saluFirstSegments(leaves, start, end):
            following = next(
                (inst for _c, inst in leaves[segment[-1] + 1:]
                 if _writesScc(inst) or _readsScc(inst)), None)
            nextCode = next(
                (inst for _c, inst in leaves[segment[-1] + 1:]
                 if not isinstance(inst, TextBlock)), None)
            units = _orderSaluFirst(
                leaves, segment,
                sccLiveOut=following is not None and _readsScc(following),
                branchFollows=nextCode is not None and _readsScc(nextCode))
            if units is None:
                continue
            comments, pending = {}, []
            for index in segment:
                if isinstance(leaves[index][1], TextBlock):
                    pending.append(index)
                else:
                    comments[index] = pending
                    pending = []
            ordered = [leaves[i][1] for u in units for m in u
                       for i in comments[m] + [m]]
            ordered += [leaves[i][1] for i in pending]
            container = leaves[segment[0]][0]
            items = list(container.items())
            at = next(i for i, item in enumerate(items)
                      if item is leaves[segment[0]][1])
            if any(a is not leaves[i][1] for a, i in
                   zip(items[at:at + len(segment)], segment)):
                continue
            container.setItems(items[:at] + ordered + items[at + len(segment):])
    return module


def _schedulePartitionZeroIssueOrder(
        emittedModules, pathOrders, preMfmaOrders, mfmas,
        minGapDsReadToWait, uniformPgrEpoch,
        earliestWriteShadow=None, mfmaBase=0,
        retainAllLocalReads=False, vmemBoundary=None):
    """Issue partition-zero reads by consumer in five follower slots.

    Local reads and global-read bundles are ordered by first-consumer rank.
    The earliest cohort issues before the rest.     The partition barrier moves
    out of the pre-MFMA prefix when this epoch's wait owns the ordering.
    The direct-to-LDS stream is right-aligned to vmemBoundary, the local
    MFMA it must issue before, or to the end of the partition.
    An overwrite barrier then slides to the first A/B direct-to-LDS store.
    Each slot holds one instruction. A ds_read or VALU stays out of the
    first slot, and an interval holds at most one direct-to-LDS load and
    two ds_reads. Local-read spreading stays off.
    """
    n = len(emittedModules)
    p0Split = _splitPartitionZeroPrefetch(
        preMfmaOrders, emittedModules, retainAllLocalReads)
    p0Payload = p0Split[3] if p0Split is not None else None
    p0Sync = p0Split[4] if p0Split is not None else []
    p0PayloadReordered = False
    if p0Split is not None:
        reordered = _orderLocalReadsByFirstConsumer(p0Payload, emittedModules)
        if reordered is not None:
            p0Payload = reordered
            p0PayloadReordered = True
        p0Payload = _orderReadsAroundLiveSourceWrites(
            p0Payload, earliestWriteShadow or {}, mfmaBase)

    placementPathOrders = list(pathOrders)
    if p0Split is not None and not p0PayloadReordered:
        placementPathOrders.append(p0Split[2])

    def _emitPreMfma(result, deferSync=False):
        deferredSyncIds = {id(inst) for _, inst in p0Sync} if deferSync else set()
        for order in preMfmaOrders:
            if p0Split is not None and order == p0Split[0]:
                for _, inst in p0Split[1]:
                    if id(inst) not in deferredSyncIds:
                        result.add(inst)
                continue
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)

    if len(mfmas) < 2:
        result = Module()
        _emitPreMfma(result)
        for mfma in mfmas:
            result.add(mfma)
        if p0Split is not None:
            for _, inst in p0Payload:
                result.add(inst)
        for order in pathOrders:
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)
        return result

    paths = _classifyPaths(pathOrders, emittedModules)
    uniformEpochFrontier = uniformPgrEpoch and any(hasWait for _, hasWait in paths)
    wrapUniformEpochFrontier = uniformEpochFrontier and any(
        _isWrapPrefetchPath(order, emittedModules)
        for order, _hasWait in paths
    )
    phasePathInsts = []
    canUseThreePhase = True
    for order, _hasWait in paths:
        pathInsts = _orderReadsAroundLiveSourceWrites(
            _flattenPath(order, emittedModules),
            earliestWriteShadow or {}, mfmaBase)
        hasLR = any(_isDsRead(inst) for _, inst in pathInsts)
        hasGR = any(_isBufferLoad(inst) for _, inst in pathInsts)
        if hasLR and hasGR:
            canUseThreePhase = False
        phasePathInsts.append((order, pathInsts, hasLR, hasGR))
    if canUseThreePhase:
        placementPathOrders = []

    slotsPerInterval = _MXFP8_FOLLOWER_SLOTS
    rules = _SchedulingRules(
        totalSlots=(len(mfmas) - 1) * slotsPerInterval,
        minGapDsReadToWait=minGapDsReadToWait,
        allowReadsAcrossWait=True)
    rules.earliestWriteShadow = earliestWriteShadow or {}
    rules.mfmaBase = mfmaBase
    adjusters = [rules.spreadBufferLoads]
    if rules.earliestWriteShadow:
        adjusters.insert(0, rules.deferLiveSourceWrite)
    placer = _SlotPlacer(
        len(mfmas) - 1, n, placementPathOrders,
        validators=[
            rules.oneDsReadPerInterval, rules.oneDtlLoadPerInterval,
            rules.noDsReadOrValuInFirstSlot, rules.minGapDsReadBeforeWait,
            rules.minGapDsReadToWait, rules.noM0WithBufferLoad,
        ],
        adjusters=adjusters,
        onPlace=rules.trackPlacement,
        slotsPerInterval=slotsPerInterval,
        maxPerSlot=1,
        dataDsReadCap=(
            _MXFP8_DATA_DS_READS, _MXFP8_DATA_DS_READ_OVERFLOW),
        maxInstructionsPerInterval=slotsPerInterval)

    def _placeRegularPaths():
        for order, hasWaitGR in paths:
            if not order:
                continue
            if hasWaitGR and _isWrapPrefetchPath(order, emittedModules):
                pathInsts = _flattenPath(order, emittedModules)
                syncEnd = next(
                    (idx + 1 for idx, (_, inst) in enumerate(pathInsts)
                     if _isBarrier(inst)),
                    0)
                if syncEnd:
                    rules.resetPath()
                    placer.placePath(pathInsts[:syncEnd], multiDU=True)
                    placer.placePath(
                        pathInsts[syncEnd:], startLimit=slotsPerInterval,
                        overflowToTail=True)
                    continue
            reverse = hasWaitGR
            pathInsts = _flattenPath(order, emittedModules, reverse=reverse)
            rules.resetPath()
            if not hasWaitGR:
                rules.setupBufLoadSpreading(placer, pathInsts, order)
            placer.placePath(pathInsts, reverse=reverse)
    if canUseThreePhase:
        leading = []
        trailing = []
        grPaths = []
        otherPaths = []
        for _order, pathInsts, hasLR, hasGR in phasePathInsts:
            if hasLR:
                first, rest = _splitAfterFirstConsumerCohort(pathInsts, emittedModules)
                leading.extend(first)
                trailing.extend(rest)
            elif hasGR:
                grPaths.append(pathInsts)
            else:
                otherPaths.append(pathInsts)
        orderedGrItems = _orderGlobalReadBundlesByFirstConsumer(grPaths, emittedModules)

        overwriteReads = []
        if p0Split is not None:
            overwriteReads = [
                item for item in p0Payload
                if getattr(emittedModules[item[0]].source, "mustPrecedeP0Overwrite", False)
            ]
            regularP0Payload = [
                item for item in p0Payload
                if not getattr(emittedModules[item[0]].source, "mustPrecedeP0Overwrite", False)
            ]
            first, rest = _splitAfterFirstConsumerCohort(regularP0Payload, emittedModules)
            leading.extend(first)
            leading.extend(overwriteReads)
            trailing.extend(rest)
            for _, inst in overwriteReads:
                if _isDsRead(inst):
                    rules.p0OverwritePrerequisiteIds.add(id(inst))

        phaseLimit = 0

        def _placePhase(pathInsts, atomicDtl=False, mfmaBetweenWaitAndBarrier=False,
                        startOverride=None, overflowToTail=False):
            nonlocal phaseLimit
            if not pathInsts:
                return
            rules.resetPath()
            placer.placePath(
                pathInsts,
                startLimit=phaseLimit if startOverride is None else startOverride,
                overflowToTail=overflowToTail,
                atomicDtl=atomicDtl,
                mfmaBetweenWaitAndBarrier=mfmaBetweenWaitAndBarrier)
            phaseLimit = max(phaseLimit, _nextPlacementSlot(placer))

        _placePhase(p0Sync)
        if leading:
            if uniformEpochFrontier:
                if wrapUniformEpochFrontier:
                    _placePhase(leading + trailing, mfmaBetweenWaitAndBarrier=True)
                    _placePhase(orderedGrItems, atomicDtl=True)
                else:
                    _placePhase(orderedGrItems, atomicDtl=True)
                    _placePhase(leading + trailing, mfmaBetweenWaitAndBarrier=True)
            else:
                _placePhase(leading, mfmaBetweenWaitAndBarrier=True)
                if overwriteReads and trailing and orderedGrItems:
                    budget = max(
                        0, (rules.minGap - 1) * _MXFP8_DATA_DS_READS)
                    preWaitTail = []
                    postWaitTail = []
                    used = 0
                    index = 0
                    while index < len(trailing):
                        moduleId = trailing[index][0]
                        inst = trailing[index][1]
                        metadata = _instructionMetadata(emittedModules, moduleId, inst)
                        atomRank = metadata[0] if metadata is not None else None
                        end = index + 1
                        while end < len(trailing) and trailing[end][0] == moduleId:
                            nextMeta = _instructionMetadata(
                                emittedModules, moduleId, trailing[end][1])
                            nextRank = nextMeta[0] if nextMeta is not None else None
                            if nextRank != atomRank:
                                break
                            end += 1
                        group = trailing[index:end]
                        if (all(_isDsRead(item[1]) for item in group)
                                and used + len(group) <= budget):
                            preWaitTail.extend(group)
                            used += len(group)
                        else:
                            postWaitTail.extend(group)
                        index = end
                    for _, inst in preWaitTail:
                        if _isDsRead(inst):
                            rules.youngerP0OverwriteIds.add(id(inst))
                    for _, inst in orderedGrItems:
                        if _isWaitCnt(inst) and inst.dscnt >= 0:
                            rules.partialP0OverwriteWaitIds.add(id(inst))
                            break
                    _placePhase(preWaitTail)
                    trailing = postWaitTail
                _placePhase(orderedGrItems, atomicDtl=True)
                if overwriteReads and trailing:
                    grSlots = [
                        pos for pos, slot in enumerate(placer._placed)
                        if any(_isBufferLoad(inst) for _, inst in slot)
                    ]
                    _placePhase(
                        trailing,
                        startOverride=min(grSlots) if grSlots else None,
                        overflowToTail=True)
                else:
                    _placePhase(trailing)
            for pathInsts in otherPaths:
                _placePhase(pathInsts)
        else:
            _placePhase(trailing)
            for pathInsts in otherPaths:
                _placePhase(pathInsts)
            _placePhase(orderedGrItems, atomicDtl=True)
    else:
        _placeRegularPaths()
        if p0Split is not None:
            rules.resetPath()
            placer.placePath(p0Payload)

    # Right-align m0, direct-to-LDS loads, and the SALU that writes their
    # global or LDS address. A load and the m0 directly before it take
    # slot 0 and slot 1 of one shadow. The instructions already in that
    # shadow slide down by one slot into its last hole. With a memory
    # boundary inside the partition, the stream ends at the first MFMA
    # from the boundary on that it fits before.
    aligned = False
    if vmemBoundary is not None and 0 < vmemBoundary < len(mfmas):
        snapshot = (placer._placed, placer._firstPos, placer._lastPos)
        for boundary in range(vmemBoundary, len(mfmas)):
            placer._placed = [list(slot) for slot in snapshot[0]]
            placer._firstPos = list(snapshot[1])
            placer._lastPos = list(snapshot[2])
            if _rightAlignM0AndVmem(
                    placer, endSlot=boundary * slotsPerInterval):
                aligned = True
                break
        if not aligned:
            placer._placed, placer._firstPos, placer._lastPos = snapshot
    if not aligned:
        _rightAlignM0AndVmem(placer)

    scheduled = Module()
    _emitPreMfma(scheduled, deferSync=canUseThreePhase)
    body = placer.assemble(mfmas)
    _slideOverwriteBarriersToVmem(body)
    scheduled.add(body)

    bufLoadCount = 0
    sawOverwritePrerequisite = False
    youngerOverwriteReads = 0
    for inst in scheduled.flatitems():
        if id(inst) in rules.p0OverwritePrerequisiteIds:
            sawOverwritePrerequisite = True
            youngerOverwriteReads = 0
        elif (sawOverwritePrerequisite
              and id(inst) in rules.youngerP0OverwriteIds
              and _isDsRead(inst)):
            youngerOverwriteReads += 1
        elif id(inst) in rules.partialP0OverwriteWaitIds:
            inst.dscnt = youngerOverwriteReads
            inst.comment = (
                "Wait for p0 overwrite prerequisite; "
                f"leave {youngerOverwriteReads} younger LR requests in flight")
            sawOverwritePrerequisite = False
            youngerOverwriteReads = 0
        if _isBufferLoad(inst):
            bufLoadCount += 1
        elif _isWaitCnt(inst) and inst.vlcnt >= 0:
            if getattr(inst, "adjustVmcnt", True):
                inst.vlcnt += bufLoadCount
    return scheduled


def _feedsCell(em, cellSources) -> bool:
    """A local read whose destination one of this cell's MFMAs reads."""
    return em.opType == "lr" and any(
        set(_vgprIndices(_dsReadDstAddr(inst)[0])) & cellSources
        for inst in em.instructions if _isDsRead(inst))


def _scheduleHalfDisjointCell(
        emittedModules, pathOrders, preMfmaOrders, mfmas,
        minGapDsReadToWait, earliestWriteShadow=None, mfmaBase=0,
        tailScaleReads=0):
    """Schedule one partition of a half-disjoint body.

    The VMEM wait, local-read drain, and barrier that open a half follow
    the first MFMA, so wave skew hides behind it.  Reads the cell's own
    MFMAs consume and the MFMA's local-read wait stay ahead of that MFMA;
    when such a read needs the barrier, as the epoch's scale reads do, the
    barrier stays ahead of the MFMA too.  Every other ds_read and
    direct-to-LDS request follows the barrier.  A barrier placed after
    MFMA 0 keeps that MFMA gap to itself while the cell has intervals to
    spare, since anything sharing the gap stalls behind the wave skew.
    Reads keep their chain order and go first.  Requests keep their order
    and are spaced evenly over the MFMA gaps after the barrier that the
    reads left free, so a load issues alone.  When too few gaps are free,
    only the requests left over take gaps with the fewest reads; when no
    such placement fits, requests are spaced over every gap after the
    barrier.  tailScaleReads scale reads are appended after the cell, two
    per gap; the gaps they need stay free of requests.
    """
    spi = _MXFP8_FOLLOWER_SLOTS
    cellSources = set()
    for mfma in mfmas:
        for name in ("a", "b", "mxsa", "mxsb"):
            cellSources.update(_vgprIndices(getattr(mfma, name, None)))
    syncItems, prefix, reads, requests, others = [], [], [], [], []
    for order, preMfma in ([(o, True) for o in preMfmaOrders]
                           + [(o, False) for o in pathOrders]):
        hasRead = any(emittedModules[mid].opType == "lr" for mid in order)
        hasRequest = any(emittedModules[mid].opType == "gr" for mid in order)
        dest = reads if hasRead else (requests if hasRequest else others)
        keepUntil = -1
        inSync = False
        for i, mid in enumerate(order if preMfma else ()):
            em = emittedModules[mid]
            if em.opType == "wait_gr" or inSync:
                inSync = em.opType != "sync"
            elif _feedsCell(em, cellSources):
                keepUntil = i
        inSync = False
        for i, mid in enumerate(order):
            em = emittedModules[mid]
            items = [(mid, inst) for inst in em.instructions]
            if em.opType == "wait_gr" or inSync:
                inSync = em.opType != "sync"
                (prefix if i <= keepUntil else syncItems).extend(items)
            elif i <= keepUntil or (preMfma and em.opType == "wait_lr"):
                prefix.extend(items)
            else:
                dest.extend(items)

    if len(mfmas) < 2:
        result = Module()
        for _, inst in prefix:
            result.add(inst)
        for mfma in mfmas:
            result.add(mfma)
        for _, inst in syncItems + reads + requests + others:
            result.add(inst)
        return result

    rules = _SchedulingRules(
        totalSlots=(len(mfmas) - 1) * spi,
        minGapDsReadToWait=minGapDsReadToWait,
        allowReadsAcrossWait=True)
    rules.earliestWriteShadow = earliestWriteShadow or {}
    rules.mfmaBase = mfmaBase
    adjusters = [rules.deferLiveSourceWrite] if rules.earliestWriteShadow else []
    dtlRun = rules.maxDtlRun(_MXFP8_MAX_DTL_RUN)
    placer = _SlotPlacer(
        len(mfmas) - 1, len(emittedModules), list(preMfmaOrders) + list(pathOrders),
        validators=[
            rules.oneDsReadPerInterval, rules.oneDtlLoadPerInterval,
            rules.noDsReadOrValuInFirstSlot, rules.minGapDsReadBeforeWait,
            rules.minGapDsReadToWait, rules.noM0WithBufferLoad,
            rules.noValuWithDtl, dtlRun,
        ],
        preferences=[rules.valuNeedsLeader, rules.dtlAvoidsDsRead],
        adjusters=adjusters,
        onPlace=rules.trackPlacement,
        slotsPerInterval=spi,
        maxPerSlot=1,
        dataDsReadCap=(_MXFP8_DATA_DS_READS, _MXFP8_DATA_DS_READ_OVERFLOW),
        maxInstructionsPerInterval=spi)

    if syncItems:
        placer.placePath(syncItems, startLimit=0)
    opened = _nextPlacementSlot(placer)
    if any(_isBarrier(inst) for _, inst in syncItems) and opened % spi:
        rounded = (opened // spi + 1) * spi
        if rounded < placer.totalSlots - spi:
            opened = rounded
    if reads:
        placer.placePath(
            _orderReadsAroundLiveSourceWrites(
                reads, rules.earliestWriteShadow, mfmaBase),
            startLimit=opened)

    bundles, pending = [], []
    for item in requests:
        pending.append(item)
        if _isBufferLoad(item[1]):
            bundles.append(pending)
            pending = []
    intervals = len(mfmas) - 1
    firstInterval = opened // spi

    def intervalInsts(interval):
        return [item[1] for slot in placer._placed[interval * spi:(interval + 1) * spi]
                for item in slot if not isinstance(item[1], TextBlock)]

    def emptySlots(interval):
        return spi - sum(map(len, placer._placed[interval * spi:(interval + 1) * spi]))

    # The items after the last load need empty slots behind its m0 and
    # load, ahead of the tail scale reads.
    scaleTail = max(
        firstInterval, intervals - math.ceil(tailScaleReads / _MXFP8_DATA_DS_READS))
    tailEnd = scaleTail
    while tailEnd > firstInterval and emptySlots(tailEnd - 1) - 2 + sum(
            map(emptySlots, range(tailEnd, scaleTail))) < len(pending):
        tailEnd -= 1

    def loadGaps(reads):
        """Gaps after the barrier with this many ds_reads and room for a request."""
        return [
            interval for interval in range(firstInterval, tailEnd)
            if sum(map(_isDsRead, intervalInsts(interval))) == reads
            and not any(_isWaitCnt(x) or _isBarrier(x) or _isVALU(x)
                        for x in intervalInsts(interval))
            and len(intervalInsts(interval)) <= spi - 2]

    def evenly(gaps, count):
        return [gaps[index * len(gaps) // count] for index in range(count)]

    def placeRequestsAndOthers(targets):
        limit = opened
        for bundle, target in zip(bundles, targets):
            start = max(limit, target * spi)
            # Address updates ahead of a request issue after the previous
            # load, so only m0 and its load wait for the target gap.
            if (rules.dtlDsReadLimit is not None and len(bundle) > 2
                    and _isM0Update(bundle[-2][1])):
                lead, bundle = bundle[:-2], bundle[-2:]
                placer.placePath(lead, startLimit=limit)
                start = max(start, max(placer._lastPos[mid] for mid, _ in lead) + 1)
            placer.placePath(bundle, startLimit=start, atomicDtl=True)
            limit = max(placer._lastPos[mid] for mid, _ in bundle) + 1
        if pending:
            placer.placePath(pending, startLimit=limit)
        if others:
            placer.placePath(others, startLimit=opened)

    def overfullIntervals():
        count = 0
        for begin in range(0, placer.totalSlots, spi):
            insts = [item[1] for slot in placer._placed[begin:begin + spi]
                     for item in slot if not isinstance(item[1], TextBlock)]
            count += sum(_isDtlLoad(inst) for inst in insts) > 1
            count += len(insts) > spi
        return count

    def longestDtlRun():
        run = longest = 0
        for begin in range(0, placer.totalSlots, spi):
            run = run + 1 if placer._intervalHasDtlLoad(begin) else 0
            longest = max(longest, run)
        return longest

    # Requests first take the gaps the reads left free, and only the
    # requests left over take gaps with one read, then two, under the run
    # cap. The run cap only spreads loads that fit. When that forces a
    # second load or an extra instruction into some gap, or a longer run,
    # requests are spaced over every gap, with the cap and then without it.
    attempts, chosen = [], []
    for reads in range(_MXFP8_DATA_DS_READ_OVERFLOW + 1) if bundles else ():
        gaps, need = loadGaps(reads), len(bundles) - len(chosen)
        if len(gaps) >= need:
            attempts.append((sorted(chosen + evenly(gaps, need)), reads, True))
            break
        chosen += gaps
    spaced = evenly(
        list(range(firstInterval, max(intervals, firstInterval + 1))), len(bundles))
    attempts += [(spaced, None, True), (spaced, None, False)]
    saved = ([list(slot) for slot in placer._placed], list(placer._firstPos),
             list(placer._lastPos), dict(rules.__dict__))
    before = overfullIntervals()
    for targets, readLimit, capped in attempts:
        placer._placed = [list(slot) for slot in saved[0]]
        placer._firstPos, placer._lastPos = list(saved[1]), list(saved[2])
        rules.__dict__.update(saved[3])
        rules.dtlDsReadLimit = readLimit
        if not capped:
            placer._validators.remove(dtlRun)
        placeRequestsAndOthers(targets)
        if not capped or (overfullIntervals() <= before and (
                readLimit is None or longestDtlRun() <= _MXFP8_MAX_DTL_RUN)):
            break

    scheduled = Module()
    for _, inst in prefix:
        scheduled.add(inst)
    scheduled.add(placer.assemble(mfmas))
    bufLoadCount = 0
    for inst in scheduled.flatitems():
        if _isBufferLoad(inst):
            bufLoadCount += 1
        elif _isWaitCnt(inst) and inst.vlcnt >= 0:
            if getattr(inst, "adjustVmcnt", True):
                inst.vlcnt += bufLoadCount
    return scheduled


def extractPathsFromBeforeDeps(emittedModules) -> Tuple[int, List[List[int]], List[List[int]]]:
    """Extract non-MFMA dependency paths using only EmittedModule.before links.

    Returns:
      (mfmaIdx, paths, preMfmaPaths)
      - mfmaIdx: index of the MFMA emitted module in emittedModules
      - paths: list of non-MFMA module-index paths to interleave between MFMAs
      - preMfmaPaths: paths that must be emitted before the first MFMA
        (reachable from the MFMA's before link)
    """
    idToIdx = {em.moduleId: i for i, em in enumerate(emittedModules)}
    n = len(emittedModules)

    mfmaModuleIds = [i for i, em in enumerate(emittedModules) if em.opType == "mfma"]
    assert len(mfmaModuleIds) == 1, "extractPathsFromBeforeDeps expects exactly one MFMA emitted module"
    mfmaIdx = mfmaModuleIds[0]
    nonMfmaIds = [i for i in range(n) if i != mfmaIdx]
    nonMfmaSet = set(nonMfmaIds)

    # Identify the non-MFMA module the MFMA depends on (if any).
    mfmaBefore = emittedModules[mfmaIdx].before
    preMfmaTarget = None
    if mfmaBefore is not None:
        bi = idToIdx.get(mfmaBefore)
        if bi is not None and bi in nonMfmaSet:
            preMfmaTarget = bi

    # Each non-MFMA module has at most one predecessor (its `before` link), but a
    # predecessor may be referenced by more than one child (e.g. single-partition
    # PGR2+PLR0, where the MFMA's wait_lr and the GR-prefetch LDS-swap chain both
    # depend on the same local read). The downstream slot placer requires simple
    # linear chains, so at a branch point only ONE child may continue the chain;
    # the others must start their own paths.
    #
    # Crucially, keep the branch that leads to the MFMA's dependency
    # (preMfmaTarget) attached, so that branch — and only that branch — is
    # classified pre-MFMA. The other branches (notably the GR-prefetch
    # buffer_loads, which the MFMA does NOT depend on) are detached into their own
    # regular paths so `spreadBufferLoads` can interleave them across the MFMA
    # stream. The real wait/barrier instructions inside each branch still enforce
    # the LDS read-before-write ordering at runtime, so detaching only drops a
    # scheduling hint, not a correctness guarantee.
    raw_pred: List[int] = [-1 for _ in range(n)]
    children_of: List[List[int]] = [[] for _ in range(n)]
    for i in nonMfmaIds:
        b = emittedModules[i].before
        if b is not None:
            bi = idToIdx.get(b)
            if bi is not None and bi != i and bi in nonMfmaSet:
                raw_pred[i] = bi
                children_of[bi].append(i)

    # For each node on preMfmaTarget's ancestor chain, record which child
    # continues toward the target so we keep that branch attached.
    preferredChild = {}
    if preMfmaTarget is not None:
        c = preMfmaTarget
        guard = 0
        while raw_pred[c] != -1 and guard < n:
            preferredChild[raw_pred[c]] = c
            c = raw_pred[c]
            guard += 1

    pred: List[int] = [-1 for _ in range(n)]
    child: List[int] = [-1 for _ in range(n)]
    for p in nonMfmaIds:
        kids = children_of[p]
        if not kids:
            continue
        primary = preferredChild.get(p, kids[0])
        if primary not in kids:
            primary = kids[0]
        # Only the primary child continues p's chain; others become path heads
        # (their pred stays -1), detaching those subtrees into separate paths.
        child[p] = primary
        pred[primary] = p

    def _findHead(mid: int) -> int:
        cur = mid
        seen = [False for _ in range(n)]
        while pred[cur] != -1 and not seen[cur]:
            seen[cur] = True
            cur = pred[cur]
        return cur

    def _walkFromHead(head: int, used: List[bool]) -> List[int]:
        order: List[int] = []
        localSeen = [False for _ in range(n)]
        cur = head
        while cur != -1 and not used[cur] and not localSeen[cur]:
            order.append(cur)
            localSeen[cur] = True
            cur = child[cur]
        return order

    used = [False for _ in range(n)]
    paths: List[List[int]] = []
    for mid in nonMfmaIds:
        if used[mid]:
            continue
        head = _findHead(mid)
        order = _walkFromHead(head, used)
        assert order, f"extractPathsFromBeforeDeps produced empty path for module {mid}"
        for i in order:
            used[i] = True
        paths.append(order)

    # Separate paths that the MFMA depends on (must go before first MFMA).
    preMfmaPaths: List[List[int]] = []
    regularPaths: List[List[int]] = []
    for path in paths:
        if preMfmaTarget is not None and preMfmaTarget in path:
            preMfmaPaths.append(path)
        else:
            regularPaths.append(path)

    return mfmaIdx, regularPaths, preMfmaPaths


def instructionSchedule(emittedModules, multiDU: bool = False,
                      minGapDsReadToWait=_MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT,
                      partitionPrefetch: bool = False,
                      uniformPgrEpoch: bool = False,
                      earliestWriteShadow=None,
                      mfmaBase: int = 0,
                      retainAllLocalReads: bool = False,
                      vmemBoundary: Optional[int] = None,
                      halfDisjoint: bool = False,
                      tailScaleReads: int = 0):
    """Interleave non-MFMA instructions between MFMAs using 2 slots/interval.

    Rules (shared):
      - MFMA order is preserved.
      - Between two adjacent MFMAs there are 2 placement slots.
      - At most one ds_read (LocalReadInstruction) per interval.
      - Before dependencies are respected at module order level.
      - Minimm distance between ds_read and it waitcnt (hardcoded for now)
      - Module-internal instruction order is preserved.
      - GR path is spread as much as possible across remaining valid slots.

    When multiDU is False (single-DU / legacy path):
      - LR path containing a WAIT_GR is packed from the end backwards.

    When multiDU is True (MX multi-DU path):
      - LR path containing a WAIT_GR is placed forward (wait_gr → sync → LR insts).
      - wait_gr+barrier stay contiguous; GR m0/buffer_load defer until that path's
        post-barrier ds_reads complete.

      TODO : To be tested on multi-partition setup.
    """
    if not emittedModules:
        return Module()

    isMFMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))
    n = len(emittedModules)

    mfmaIdx, pathOrders, preMfmaOrders = extractPathsFromBeforeDeps(emittedModules)
    mfmas = [x for x in emittedModules[mfmaIdx].instructions if isMFMA(x)]
    if (halfDisjoint and partitionPrefetch and not multiDU
            and _isMx8Schedule(mfmas)):
        return _scheduleHalfDisjointCell(
            emittedModules, pathOrders, preMfmaOrders, mfmas,
            minGapDsReadToWait, earliestWriteShadow, mfmaBase, tailScaleReads)
    # Only the dual-period uniform epoch reorders partition zero. Every other
    # caller keeps the placer below, including MXFP4 and single-partition MXFP8.
    if (uniformPgrEpoch and partitionPrefetch and not multiDU
            and _isMx8Schedule(mfmas)):
        return _schedulePartitionZeroIssueOrder(
            emittedModules, pathOrders, preMfmaOrders, mfmas,
            minGapDsReadToWait, uniformPgrEpoch,
            earliestWriteShadow, mfmaBase, retainAllLocalReads,
            vmemBoundary)

    def _emitPreMfma(result):
        for order in preMfmaOrders:
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)

    # Single MFMA: no slots to interleave into — emit preMfma, MFMA, then paths.
    if len(mfmas) < 2:
        result = Module()
        _emitPreMfma(result)
        for m in mfmas:
            result.add(m)
        for order in pathOrders:
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)
        return result

    paths = _classifyPaths(pathOrders, emittedModules)
    rules = _SchedulingRules(totalSlots=(len(mfmas) - 1) * 2,
                             minGapDsReadToWait=minGapDsReadToWait)
    base_validators = [rules.oneDsReadPerInterval, rules.minGapDsReadBeforeWait,
                       rules.minGapDsReadToWait, rules.noM0WithBufferLoad]
    base_adjusters = [rules.spreadBufferLoads]
    if multiDU:
        base_validators.append(rules.grAfterWaitGrDsReads)
        base_adjusters.insert(0, rules.deferGrAfterWaitGrDsReads)
    placer = _SlotPlacer(
        len(mfmas) - 1, n, pathOrders,
        validators=base_validators,
        adjusters=base_adjusters,
        onPlace=rules.trackPlacement)

    for order, hasWaitGR in paths:
        if not order:
            continue
        reverse = (not multiDU) and hasWaitGR
        pathInsts = _flattenPath(order, emittedModules, reverse=reverse)
        rules.resetPath()
        if not hasWaitGR:
            rules.setupBufLoadSpreading(placer, pathInsts, order)
        placer.placePath(pathInsts, reverse=reverse, multiDU=multiDU)
        if multiDU and hasWaitGR:
            rules.setPostWaitGrDsReadFence(placer)

    scheduled = Module()
    _emitPreMfma(scheduled)
    scheduled.add(placer.assemble(mfmas))

    # Post-pass: adjust vmcnt of any SWaitCnt to account for buffer_loads
    # that the scheduler placed before it within this subIterK.
    bufLoadCount = 0
    for inst in scheduled.flatitems():
        if _isBufferLoad(inst):
            bufLoadCount += 1
        elif _isWaitCnt(inst) and inst.vlcnt >= 0:
            if getattr(inst, 'adjustVmcnt', True):
                inst.vlcnt += bufLoadCount

    return scheduled


def relaxWaitGrForOutstandingStores(module):
    """Stop the wait_gr waits from draining the fused epilogue's D stores.

    vmcnt is a single in-order counter over every VMEM op, loads and stores
    alike, so a count sized for the global reads also forces every store issued
    after them to retire. A wait_gr exists to order the global reads against
    the local reads that follow the barrier; it has no reason to wait on a
    store, and waiting on one costs the full trip to memory.

    The post-pass in instructionSchedule cannot see this. It runs per subIterK,
    while the staged drain is woven into the partition afterwards, so at that
    point the stores do not exist yet. Hence a second pass here, against the
    final order.

    Only the trailing stores can be spared. The counter retires in order, so a
    store issued before the last outstanding load is drained either way and
    nothing can be done about it; the stores after it are pure cost. Loads are
    still fully drained, which is the whole point of the wait.

    Left the split NGLL/NLL schedule alone in practice, since there the global
    reads in flight at each seam already pushed the count past the stores --
    protection that was accidental, and that disappeared once the four-deep
    tail stopped issuing global reads of its own.
    """
    outstanding = []                      # 'l' / 's', oldest first
    for inst in module.flatitems():
        if _isBufferLoad(inst):
            outstanding.append('l')
        elif _isBufferStore(inst):
            outstanding.append('s')
        elif _isWaitCnt(inst) and inst.vlcnt >= 0:
            if _isWaitGr(inst):
                trailing = 0
                for kind in reversed(outstanding):
                    if kind != 's':
                        break
                    trailing += 1
                inst.vlcnt += trailing
            keep = min(inst.vlcnt, len(outstanding))
            del outstanding[:len(outstanding) - keep]
    return module
