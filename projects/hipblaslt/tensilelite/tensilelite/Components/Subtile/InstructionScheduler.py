# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction scheduler for subtile-based mainloop.

Interleaves non-MFMA instructions between MFMAs using a slot-based placer
with pluggable scheduling rules.
"""

import copy
from typing import List, Tuple, Optional
from rocisa.code import Module, TextBlock
from rocisa.instruction import SWaitCnt, SBarrier, MFMAInstruction, MXMFMAInstruction, \
    LocalReadInstruction, GlobalReadInstruction, GlobalWriteInstruction, CommonInstruction


class _SlotPlacer:
    """Generic slot placement engine for interleaving instructions between MFMAs.

    The legacy schedule has 2 placement slots between adjacent MFMAs.
    MXFP8 partition prefetch uses five, one instruction each.
    Rules are injected via callbacks:
      - validators: (placer, pos, inst) -> bool — reject invalid slots
      - adjusters:  (placer, limit, inst) -> limit — shift search start
      - onPlace:    (placer, pos, inst) -> None — update rule state after placement
    """

    def __init__(self, intervals: int, numModules: int,
                 pathOrders: List[List[int]],
                 validators=None, adjusters=None, onPlace=None,
                 slotsPerInterval: int = 2,
                 maxPerSlot: int = 2,
                 dataDsReadCap: Optional[Tuple[int, int]] = None,
                 maxInstructionsPerInterval: Optional[int] = None):
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
        for pos in (range(hi, lo - 1, -1) if reverse else range(lo, hi + 1)):
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
        for setupPos in range(first, setupHi + 1):
            intervalEnd = min(
                self.totalSlots - 1,
                (setupPos // step + 1) * step - 1)
            if not self._canPlace(setupPos, setupInst):
                continue
            for loadPos in range(max(setupPos, loadLo), min(intervalEnd, loadHi) + 1):
                if loadPos == setupPos and (
                        self.maxPerSlot <= 1 or self._placed[setupPos]):
                    continue
                if not self._canPlace(loadPos, loadInst):
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
        warGap=_SCALE_SOURCE_WAR_MFMA_GAP):
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
            earliest = max(
                searchFrom,
                # The LDS write follows the target MFMA, so target-last+1 is
                # the direct-write issue distance.
                entry["lastC1Mfma"] + warGap - 1)
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
        earliest = max(
            searchFrom,
            entry["lastC1Mfma"] + warGap - 1)
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

_isDsRead = lambda x: isinstance(x, LocalReadInstruction)
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


def _isCurrentPartitionLocalRead(em) -> bool:
    """A local read consumed by the partition that issues it.

    Scale reads and the current cell's A/B reads carry the issuing
    partition as their target. Next-partition prefetch does not.
    """
    if em.opType != "lr":
        return False
    source = em.source
    return getattr(source, "targetPartition", None) == getattr(source, "partition", None)


def _splitPartitionZeroPrefetch(preMfmaOrders, emittedModules):
    """Split partition 0's prefetch into a pre-MFMA prefix and its reads.

    Reads this cell consumes stay in the prefix, ahead of the local-read
    wait. Only next-partition prefetch becomes payload. The returned sync
    list is the barrier the phase scheduler may move to after the first
    MFMA, once the epoch wait owns the ordering.
    """
    order = _findPartitionZeroPrefetchOrder(preMfmaOrders, emittedModules)
    if order is None:
        return None

    prefixInsts = []
    syncInsts = []
    payloadOrder = []
    for mid in order:
        em = emittedModules[mid]
        if _isCurrentPartitionLocalRead(em):
            for inst in em.instructions:
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


def _intervalPhysicalCount(placer, interval: int) -> int:
    begin = interval * placer.slotsPerInterval
    end = min(placer.totalSlots, begin + placer.slotsPerInterval)
    return sum(
        not isinstance(item[1], TextBlock)
        for pos in range(begin, end)
        for item in placer._placed[pos]
    )


def _chooseLateVmemInterval(
        placer, earliest, latest, setup, load, sourceSpans):
    """Pick the latest shadow whose first two slots can hold one bundle.

    Empty shadows win. A shadow that already holds another instruction is
    used only when every later legal shadow is full or illegal.
    """
    width = placer.slotsPerInterval

    def legal(interval, requireEmpty):
        if interval < earliest or interval > latest:
            return False
        slot0 = interval * width
        slot1 = slot0 + 1
        if slot1 >= placer.totalSlots:
            return False
        if requireEmpty and _intervalPhysicalCount(placer, interval) != 0:
            return False
        if (placer.maxInstructionsPerInterval is not None
                and placer._intervalInstructionCount(slot0) + 2
                > placer.maxInstructionsPerInterval):
            return False
        if not placer._canPlace(slot0, setup) or not placer._canPlace(
                slot1, load):
            return False
        begin = interval * width
        end = min(placer.totalSlots, begin + width)
        for pos in range(begin, end):
            for _mid, inst in placer._placed[pos]:
                if _vmemSlideBlocker(inst, sourceSpans):
                    return False
        return True

    for requireEmpty in (True, False):
        for interval in range(latest, earliest - 1, -1):
            if legal(interval, requireEmpty):
                return interval
    return None


def _slideVmemBundlesLate(placer):
    """Move m0 plus its direct-to-LDS load into the latest legal shadow.

    Local reads, waits, and address updates stay where the forward pass put
    them. A bundle slides later only while it stays after that pass and
    before a barrier, a VMEM wait, another load, or a write of a register
    the bundle reads. m0 occupies slot 0 and the load slot 1.
    """
    width = placer.slotsPerInterval
    if width < 2 or placer.maxPerSlot != 1 or placer.totalSlots < 2:
        return
    intervalCount = placer.totalSlots // width
    issued = []
    for pos in range(placer.totalSlots):
        for item in placer._placed[pos]:
            if not isinstance(item[1], TextBlock):
                issued.append((pos, item))
    pairs = []
    index = 0
    while index < len(issued):
        setupPos, setupItem = issued[index]
        if _isM0Update(setupItem[1]):
            loadAt = next(
                (candidate for candidate in range(index + 1, len(issued))
                 if _isDtlLoad(issued[candidate][1][1])),
                None)
            # A direct-to-LDS load can already sit across the next MFMA.
            # Keep the pair together unless another m0 or load is between.
            if loadAt is not None and not any(
                    _isM0Update(issued[candidate][1][1])
                    or _isDtlLoad(issued[candidate][1][1])
                    for candidate in range(index + 1, loadAt)):
                loadPos, loadItem = issued[loadAt]
                pairs.append((setupPos, setupItem, loadPos, loadItem))
                index = loadAt + 1
                continue
        index += 1
    if not pairs:
        return

    for setupPos, setupItem, loadPos, loadItem in pairs:
        placer._placed[setupPos].remove(setupItem)
        placer._placed[loadPos].remove(loadItem)

    nextInterval = intervalCount
    for setupPos, setupItem, loadPos, loadItem in reversed(pairs):
        sources = _vmemSourceSpans(setupItem[1], loadItem[1])
        # Slot 0 must not move m0 ahead of where it already sits, and the
        # pair stays behind any instruction the forward pass put between
        # the setup and the load.
        earliestSlot = (setupPos // width) * width
        if earliestSlot < setupPos:
            earliestSlot = (setupPos // width + 1) * width
        for pos, _between in issued:
            if setupPos < pos < loadPos and pos >= earliestSlot:
                earliestSlot = (pos // width + 1) * width
        earliest = earliestSlot // width
        limit = nextInterval * width
        for pos in range(loadPos + 1, placer.totalSlots):
            if any(
                    _vmemSlideBlocker(inst, sources)
                    for _mid, inst in placer._placed[pos]):
                limit = min(limit, pos)
                break
        latest = (limit - 2) // width if limit >= 2 else -1
        chosen = _chooseLateVmemInterval(
            placer, earliest, min(latest, nextInterval - 1),
            setupItem[1], loadItem[1], sources)
        if chosen is None:
            placer.place(setupPos, setupItem)
            placer.place(loadPos, loadItem)
            nextInterval = min(nextInterval, setupPos // width)
            continue
        slot0 = chosen * width
        placer.place(slot0, setupItem)
        placer.place(slot0 + 1, loadItem)
        nextInterval = chosen


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


def _schedulePartitionZeroIssueOrder(
        emittedModules, pathOrders, preMfmaOrders, mfmas,
        minGapDsReadToWait, uniformPgrEpoch,
        earliestWriteShadow=None, mfmaBase=0):
    """Issue partition-zero reads by consumer in five follower slots.

    Local reads and global-read bundles are ordered by first-consumer rank.
    The earliest cohort issues before the rest. The partition barrier moves
    out of the pre-MFMA prefix when this epoch's wait owns the ordering.
    Each slot holds one instruction. A ds_read or VALU stays out of the
    first slot, and an interval holds at most one direct-to-LDS load and
    two ds_reads. Local-read spreading stays off.
    """
    n = len(emittedModules)
    p0Split = _splitPartitionZeroPrefetch(preMfmaOrders, emittedModules)
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

    # Pack each direct-to-LDS bundle against the last shadow that is still
    # after its address setup and before the next wait or barrier. Slot 0
    # holds m0. Slot 1 holds the load. Another instruction shares that
    # shadow only when every later legal shadow is already taken.
    _slideVmemBundlesLate(placer)

    scheduled = Module()
    _emitPreMfma(scheduled, deferSync=canUseThreePhase)
    scheduled.add(placer.assemble(mfmas))

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
                      mfmaBase: int = 0):
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
    # Only the dual-period uniform epoch reorders partition zero. Every other
    # caller keeps the placer below, including MXFP4 and single-partition MXFP8.
    if (uniformPgrEpoch and partitionPrefetch and not multiDU
            and _isMx8Schedule(mfmas)):
        return _schedulePartitionZeroIssueOrder(
            emittedModules, pathOrders, preMfmaOrders, mfmas,
            minGapDsReadToWait, uniformPgrEpoch,
            earliestWriteShadow, mfmaBase)

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
