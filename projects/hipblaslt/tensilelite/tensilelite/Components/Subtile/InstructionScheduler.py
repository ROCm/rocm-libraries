# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction scheduler for subtile-based mainloop.

Interleaves non-MFMA instructions between MFMAs using a slot-based placer
with pluggable scheduling rules.
"""

from typing import List, Tuple, Optional
from rocisa.code import Module, TextBlock
from rocisa.instruction import SWaitCnt, SBarrier, MFMAInstruction, MXMFMAInstruction, \
    LocalReadInstruction, GlobalReadInstruction, CommonInstruction


# Advance the final partition's entry anchor by this many MFMA intervals.
# For a 16-MFMA preceding partition, the exclusive right-alignment cutoff is
# 16 - 1 - 3 = 12, leaving follower groups 12 through 15 free.  This value is
# an anchor distance, not the number of follower groups reserved.
_FINAL_ENTRY_ANCHOR_ADVANCE_MFMAS = 3
# Move the preceding partition's native DTL bundles as late as possible
# without colliding with the advanced final-partition entry anchor.
_RIGHT_ALIGN_PREVIOUS_DTL_BEFORE_ADVANCED_SUFFIX = True
# The final-partition entry LDS drain is redundant when the
# preceding partition already completed a full drain and issued no younger
# LGKM-producing instruction.  Keep its former placement as a seam anchor so
# the retained P2 DTL cohort does not otherwise move.
_DROP_REDUNDANT_FINAL_PARTITION_LGKM_WAIT = True
# Keep every real final-partition follower in that partition.  The removed
# LGKM wait still supplies the empty seam anchor for preceding-partition DTL
# right alignment.
_ADVANCE_FINAL_PARTITION_FOLLOWERS = False


class _DtlIntervalCapacityError(RuntimeError):
    """Strict DTL density could not satisfy the emitted dependency windows."""


class _SlotPlacer:
    """Generic slot placement engine for interleaving instructions between MFMAs.

    Each interval (pair of adjacent MFMAs) has a configurable number of
    4-cycle placement slots.
    Rules are injected via callbacks:
      - validators: (placer, pos, inst) -> bool — reject invalid slots
      - adjusters:  (placer, limit, inst) -> limit — shift search start
      - onPlace:    (placer, pos, inst) -> None — update rule state after placement
    """

    def __init__(self, intervals: int, numModules: int,
                 pathOrders: List[List[int]],
                 validators=None, adjusters=None, onPlace=None,
                 slotsPerInterval: int = 2,
                 oneDtlPerInterval: bool = False,
                 maxInstructionsPerInterval: Optional[int] = None):
        self.slotsPerInterval = slotsPerInterval
        self.totalSlots = intervals * slotsPerInterval
        self.oneDtlPerInterval = oneDtlPerInterval
        self.maxInstructionsPerInterval = maxInstructionsPerInterval
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

    def _intervalHasDtlLoad(self, pos: int) -> bool:
        begin = (pos // self.slotsPerInterval) * self.slotsPerInterval
        end = min(self.totalSlots, begin + self.slotsPerInterval)
        return any(
            _isDtlLoad(item[1])
            for slot in range(begin, end)
            for item in self._placed[slot]
        )

    def _intervalInstructionCount(self, pos: int) -> int:
        begin = (pos // self.slotsPerInterval) * self.slotsPerInterval
        end = min(self.totalSlots, begin + self.slotsPerInterval)
        return sum(
            sum(
                not isinstance(item[1], TextBlock)
                for item in self._placed[slot]
            )
            for slot in range(begin, end)
        )

    def _remainingCapacityWithIntervalCap(self, pos: int, cap: int) -> int:
        """Return physical follower capacity from pos under a preferred cap."""
        firstInterval = max(0, pos // self.slotsPerInterval)
        lastInterval = (self.totalSlots - 1) // self.slotsPerInterval
        return sum(
            max(
                0,
                cap - self._intervalInstructionCount(
                    interval * self.slotsPerInterval),
            )
            for interval in range(firstInterval, lastInterval + 1)
        )

    def _nextForwardLimit(self, pos: int, remainingItems: int) -> int:
        """Reuse a slot only when a two-follower schedule cannot fit."""
        if (self.denseToFit
                and remainingItems
                > self._remainingCapacityWithIntervalCap(pos, 2)):
            return pos
        return pos + 1

    def _canPlace(self, pos: int, inst) -> bool:
        if pos < 0 or pos >= self.totalSlots:
            return False
        if isinstance(inst, TextBlock):
            return True
        if sum(
                not isinstance(item[1], TextBlock)
                for item in self._placed[pos]) >= 2:
            return False
        if (self.maxInstructionsPerInterval is not None
                and self._intervalInstructionCount(pos)
                >= self.maxInstructionsPerInterval):
            return False
        return all(v(self, pos, inst) for v in self._validators)

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
        """Place final m0 setup and its DTL request before the same MFMA.

        The pair may use any two slots in one MFMA interval.  Keeping it
        interval-local ensures that moving setup earlier also ages the actual
        request instead of leaving a matrix instruction between the two.
        """
        setupMid, setupInst = setupItem
        loadMid, loadInst = loadItem
        setupLo, setupHi = self.bounds(setupMid)
        loadLo, loadHi = self.bounds(loadMid)

        first = max(limit, setupLo)
        for setupPos in range(first, setupHi + 1):
            intervalEnd = min(
                self.totalSlots - 1,
                (setupPos // self.slotsPerInterval + 1)
                * self.slotsPerInterval - 1,
            )
            if (self.oneDtlPerInterval
                    and self._intervalHasDtlLoad(setupPos)):
                continue
            if (self.maxInstructionsPerInterval is not None
                    and self._intervalInstructionCount(setupPos)
                    + 2 > self.maxInstructionsPerInterval):
                continue
            if not self._canPlace(setupPos, setupInst):
                continue
            for loadPos in range(
                    max(setupPos, loadLo),
                    min(intervalEnd, loadHi) + 1):
                if (loadPos == setupPos
                        and any(
                            not isinstance(item[1], TextBlock)
                            for item in self._placed[setupPos])):
                    continue
                if not self._canPlace(loadPos, loadInst):
                    continue
                self.place(setupPos, setupItem)
                self.place(loadPos, loadItem)
                return loadPos

        # Match the placer's normal overflow behavior: preserve the semantic
        # boundary even when the abstract slot capacity is exhausted.  This
        # may overfill a slot, but it cannot move the request across an MFMA.
        firstInterval = max(0, first // self.slotsPerInterval)
        lastInterval = (self.totalSlots - 1) // self.slotsPerInterval
        for interval in range(firstInterval, lastInterval + 1):
            intervalBegin = interval * self.slotsPerInterval
            intervalEnd = min(
                self.totalSlots - 1,
                intervalBegin + self.slotsPerInterval - 1,
            )
            setupPos = max(first, intervalBegin)
            loadPos = max(setupPos, loadLo)
            if ((not self.oneDtlPerInterval
                    or not self._intervalHasDtlLoad(setupPos))
                    and (self.maxInstructionsPerInterval is None
                         or self._intervalInstructionCount(setupPos)
                         + 2 <= self.maxInstructionsPerInterval)
                    and setupPos <= min(setupHi, intervalEnd)
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

        startLimit can reserve leading slots for an already-placed prefix.
        overflowToTail preserves forward path order by placing the first
        unplaceable item and the rest of its path after the final MFMA,
        rather than collapsing them into one overfull slot.

        atomicDtl keeps an adjacent final-m0/DTL-load pair inside one MFMA
        interval. It is used only by the partition-prefetch phase scheduler.

        mfmaBetweenWaitAndBarrier places an adjacent wait_gr/barrier pair on
        opposite sides of one MFMA whenever another interval is available.
        """
        limit = ((self.totalSlots - 1) if reverse else 0) \
            if startLimit is None else startLimit
        if not multiDU:
            idx = 0
            while idx < len(pathInsts):
                item = pathInsts[idx]
                mid, inst = item
                if (mfmaBetweenWaitAndBarrier and not reverse
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
                if (atomicDtl and not reverse
                        and idx + 1 < len(pathInsts)
                        and _isM0Update(inst)
                        and _isDtlLoad(pathInsts[idx + 1][1])):
                    pairEnd = self._placeAtomicDtlPair(
                        item, pathInsts[idx + 1], limit)
                    if pairEnd is not None:
                        limit = pairEnd + 1
                        idx += 2
                        continue
                    if self.oneDtlPerInterval:
                        raise _DtlIntervalCapacityError(
                            "unable to place one DTL bundle per MFMA interval "
                            f"(path index {idx}, start slot {limit}, "
                            f"total slots {self.totalSlots}, "
                            f"setup={getattr(inst, 'comment', '')!r}, "
                            f"load={getattr(pathInsts[idx + 1][1], 'comment', '')!r})")
                if not reverse:
                    limit = self.adjustLimit(limit, inst)
                pos = self.findSlot(mid, inst, limit, reverse=reverse)
                if pos is None:
                    if overflowToTail and not reverse:
                        self.leftovers.extend(pathInsts[idx:])
                        return
                    pos = self._forceSlot(mid, limit, reverse)
                self.place(pos, item, reverse=reverse)
                remaining = sum(
                    not isinstance(nextItem[1], TextBlock)
                    for nextItem in pathInsts[idx + 1:])
                if reverse:
                    limit = pos - 1
                elif isinstance(inst, TextBlock):
                    limit = pos
                else:
                    limit = self._nextForwardLimit(pos, remaining)
                idx += 1
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
                    if overflowToTail:
                        self.leftovers.extend(pathInsts[idx:])
                        return
                    pos = self._forceSlot(mid, limit, reverse)
                self.place(pos, item, reverse=reverse)
                sync_item = pathInsts[idx + 1]
                if self._canPlace(pos, sync_item[1]):
                    self.place(pos, sync_item, reverse=reverse)
                else:
                    sync_pos = self.findSlot(sync_item[0], sync_item[1],
                                             pos + 1, reverse=reverse)
                    if sync_pos is None:
                        if overflowToTail:
                            self.leftovers.extend(pathInsts[idx + 1:])
                            return
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
                if overflowToTail and not reverse:
                    self.leftovers.extend(pathInsts[idx:])
                    return
                pos = self._forceSlot(mid, limit, reverse)
            self.place(pos, item, reverse=reverse)
            limit = (pos - 1) if reverse else (pos + 1)
            idx += 1

    # ── Assembly ──

    def assemble(self, mfmas) -> Module:
        intervals = len(mfmas) - 1
        result = Module()
        result.add(mfmas[0])
        for i in range(intervals):
            slotBegin = self.slotsPerInterval * i
            slotEnd = slotBegin + self.slotsPerInterval
            for slot in range(slotBegin, slotEnd):
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
_MAX_LGKMCNT = 15

_isDsRead = lambda x: isinstance(x, LocalReadInstruction)
_isBufferLoad = lambda x: isinstance(x, GlobalReadInstruction)
# Compiled rocisa buffer-load bindings do not expose their MUBUF modifiers.
# In this policy an immediately preceding m0 write is the semantic DTL marker:
# ordinary VGPR buffer loads neither consume nor set m0.
_isDtlLoad = _isBufferLoad
_isWaitCnt = lambda x: isinstance(x, SWaitCnt)
_isM0Update = lambda x: isinstance(x, CommonInstruction) and hasattr(x, 'dst') and hasattr(x.dst, 'regType') and x.dst.regType == 'm'
_isVALU = lambda x: isinstance(x, CommonInstruction) \
    and not isinstance(x, (GlobalReadInstruction, LocalReadInstruction)) \
    and hasattr(x, 'dst') and hasattr(x.dst, 'regType') \
    and x.dst.regType == 'v'
# Typed op detection (see SWaitCntEx.isWaitGr / SBarrier) rather than matching
# substrings in the instruction comment, which is brittle to comment edits.
_isWaitGr = lambda x: _isWaitCnt(x) and getattr(x, "isWaitGr", False)
_isBarrier = lambda x: isinstance(x, SBarrier)


def _hasDtlIntervalCapacity(
        pathOrders, emittedModules, mfmaIntervals: int) -> bool:
    """Return whether strict one-DTL-per-MFMA placement can fit.

    Count only modules in regular interleaving paths. DTL requests in a
    pre-MFMA dependency prefix do not consume an MFMA follower interval.
    """
    scheduledModuleIndices = {
        moduleIndex for order in pathOrders for moduleIndex in order
    }
    dtlLoads = sum(
        _isDtlLoad(inst)
        for moduleIndex in scheduledModuleIndices
        for inst in emittedModules[moduleIndex].instructions
    )
    return dtlLoads <= mfmaIntervals


class _SchedulingRules:
    """Scheduling rules for slot placement: validators, adjusters, and placement hooks.

    Owns all rule state (ds_read/waitcnt tracking, buffer-load spreading).
    Bound methods are passed as callbacks to _SlotPlacer.
    """

    def __init__(self, totalSlots: int, minGapDsReadToWait: int = _MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT):
        # Cross-path state
        self.lastDsReadPos = -1
        self.earliestWaitCntPos = totalSlots
        self.waitGrBarrierPos: Optional[int] = None
        self.maxDsReadAfterWaitGr = -1
        self.p0OverwritePrerequisiteIds = set()
        self.youngerP0OverwriteIds = set()
        self.partialP0OverwriteWaitIds = set()
        self.minGap = minGapDsReadToWait
        # Per-path state
        self._resetPath()

    def _resetPath(self):
        self.firstBufLoadPos: Optional[int] = None
        self.bufLoadIdx = 0
        self.bufLoadMaxSlot = 0
        self.numBufLoads = 0
        self.localReadIdx = 0
        self.numLocalReads = 0

    # ── Validators: (placer, pos, inst) -> bool ──

    def oneDsReadPerInterval(self, placer, pos, inst):
        """Use at most the non-leading slot count for DS reads per interval."""
        if not _isDsRead(inst):
            return True
        first = (pos // placer.slotsPerInterval) * placer.slotsPerInterval
        end = min(first + placer.slotsPerInterval, placer.totalSlots)
        reads = sum(
            _isDsRead(item[1])
            for slot in range(first, end)
            for item in placer._placed[slot])
        limit = placer.slotsPerInterval \
            if placer.denseToFit else placer.slotsPerInterval - 1
        return reads < limit

    def oneDtlLoadPerInterval(self, placer, pos, inst):
        """Keep every DTL request, including an unpaired first request, isolated."""
        return not _isDtlLoad(inst) or not placer._intervalHasDtlLoad(pos)

    def noDsReadOrValuInFirstSlot(self, placer, pos, inst):
        """Reserve each p0 interval's first slot from DS reads and VALU."""
        if pos % placer.slotsPerInterval != 0:
            return True
        return not (_isDsRead(inst) or _isVALU(inst))

    def minGapDsReadBeforeWait(self, placer, pos, inst):
        """Reject ds_read too close to an already-placed waitcnt ahead."""
        if not _isDsRead(inst):
            return True
        if id(inst) in self.youngerP0OverwriteIds:
            return True
        if pos >= self.earliestWaitCntPos:
            return True
        gap = self.minGap * placer.slotsPerInterval
        return self.earliestWaitCntPos - pos >= gap

    def minGapDsReadToWait(self, placer, pos, inst):
        """Reject waitcnt too close to the last placed ds_read."""
        if not _isWaitCnt(inst) or self.lastDsReadPos < 0:
            return True
        if pos <= self.lastDsReadPos:
            return True
        gap = self.minGap * placer.slotsPerInterval
        return pos - self.lastDsReadPos >= gap

    def noM0WithBufferLoad(self, placer, pos, inst):
        """Avoid placing M0 updates and buffer_loads in the same MFMA interval."""
        if not _isM0Update(inst) and not _isBufferLoad(inst):
            return True
        peer = pos ^ 1
        slots = [pos]
        if 0 <= peer < placer.totalSlots:
            slots.append(peer)
        if _isM0Update(inst):
            return not any(_isBufferLoad(item[1]) for s in slots for item in placer._placed[s])
        return not any(_isM0Update(item[1]) for s in slots for item in placer._placed[s])

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

    def spreadLocalReads(self, placer, limit, inst):
        """Spread a selected LR path across all available 4-cycle slots."""
        if not _isDsRead(inst) or self.numLocalReads <= 1:
            return limit
        target = self.localReadIdx * (placer.totalSlots - 1) \
            // (self.numLocalReads - 1)
        self.localReadIdx += 1
        return max(limit, target)

    # ── Placement hook: (placer, pos, inst) -> None ──

    def trackPlacement(self, placer, pos, inst):
        """Update rule state after a successful placement."""
        if (_isDsRead(inst)
                and id(inst) not in self.youngerP0OverwriteIds):
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

    def setupLocalReadSpreading(self, pathInsts):
        """Enable full-range spreading for the current local-read path."""
        self.numLocalReads = sum(
            1 for _, inst in pathInsts if _isDsRead(inst)
        )

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


def _isWrapPrefetchPath(order, emittedModules):
    """Return True for a wait path that reads operands for the next MT.

    Partition-directed PLR marks the final partition's wrap local reads with
    mtIteration > 0. Those reads feed the next partition-0, not the current
    partition's MFMAs, so their payload can be scheduled after an early MFMA.
    """
    return any(
        emittedModules[mid].opType == "lr"
        and getattr(emittedModules[mid].source, "mtIteration", 0) > 0
        for mid in order
    )


def _findPartitionZeroPrefetchOrder(preMfmaOrders, emittedModules):
    """Find p0's partition-prefetch path from semantic placement metadata.

    C0 contains data and scale reads while scale-reuse bodies contain only
    data reads. The read count and tensor mix vary with the macrotile, so the
    invariant is instead that at least one emitted A/B LR in partition 0 feeds
    a later partition in the current MT. Do not use wait/sync operations as a
    recognition signature: uniform-epoch scheduling deliberately consolidates
    those operations at a later frontier while leaving this LR payload intact.
    """
    for order in preMfmaOrders:
        modules = [emittedModules[mid] for mid in order]

        readModules = [em for em in modules if em.opType == "lr"]
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
        if not hasNextPartitionData or readCount == 0:
            continue
        return order
    return None


def _splitPartitionZeroPrefetch(preMfmaOrders, emittedModules):
    """Move p0's full LR drain before its independent younger reads.

    Returns the matched order, a pre-MFMA wait/sync prefix, the payload path
    to spread through bounded scheduling slots per MXMFMA, and the sync subset
    that a phase scheduler may defer until after MFMA0.
    """
    order = _findPartitionZeroPrefetchOrder(
        preMfmaOrders, emittedModules)
    if order is None:
        return None

    prefixInsts = []
    syncInsts = []
    for opType in ("wait_gr", "sync", "wait_lr"):
        for mid in order:
            em = emittedModules[mid]
            if em.opType != opType:
                continue
            for inst in em.instructions:
                if opType == "wait_lr" and _isWaitCnt(inst):
                    inst.dscnt = 0
                    inst.comment = "Wait for preloop LR before p0 MFMA"
                item = (mid, inst)
                prefixInsts.append(item)
                if opType == "sync":
                    syncInsts.append(item)

    payloadOrder = [
        mid for mid in order
        if emittedModules[mid].opType not in ("wait_gr", "sync", "wait_lr")
    ]
    payloadInsts = _flattenPath(payloadOrder, emittedModules)
    return order, prefixInsts, payloadOrder, payloadInsts, syncInsts


def _orderLocalReadsByFirstConsumer(pathInsts, emittedModules):
    """Order independent local-read atoms by their next MFMA consumer.

    The emitter tags each instruction in an atomic read group with a rank
    derived from the configured partition traversal. Sort complete groups by
    that deadline. If two operands feed the same MFMA, issue the shorter group
    first (for example, one scale read before a two-instruction data read).
    Non-read payload such as LDS pointer updates remains after the read cohort
    in its original order.

    Return None when semantic metadata is unavailable, preserving unrelated
    schedules byte-for-byte.
    """
    reads = []
    nonReads = []
    for original_index, item in enumerate(pathInsts):
        mid, inst = item
        if not _isDsRead(inst):
            nonReads.append(item)
            continue
        instruction_index = next(
            (index for index, candidate
             in enumerate(emittedModules[mid].instructions)
             if candidate is inst),
            -1,
        )
        metadata = emittedModules[mid].instructionMetadata
        if instruction_index < 0 or instruction_index >= len(metadata):
            return None
        rank, span = metadata[instruction_index]
        rank = rank if rank is not None else float("inf")
        reads.append((rank, span, original_index, item))

    if not reads:
        return None
    reads.sort(key=lambda entry: entry[:3])
    return [item for _, _, _, item in reads] + nonReads


def _splitAfterFirstConsumerCohort(pathInsts, emittedModules):
    """Split after all consecutive LR groups sharing the earliest deadline.

    ``pathInsts`` may begin with synchronization and address setup. Those
    instructions stay attached to the leading cohort. Atomic multi-instruction
    reads share one metadata rank, so the split never bisects a read group.
    """
    first_rank = None
    split = None
    saw_first_cohort = False

    for index, (mid, inst) in enumerate(pathInsts):
        if not _isDsRead(inst):
            continue
        instruction_index = next(
            (item_index for item_index, candidate
             in enumerate(emittedModules[mid].instructions)
             if candidate is inst),
            -1,
        )
        metadata = emittedModules[mid].instructionMetadata
        if instruction_index < 0 or instruction_index >= len(metadata):
            return [], pathInsts
        rank, _span = metadata[instruction_index]
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
    """Order complete GR request bundles by their next MFMA consumer.

    The emitter assigns one metadata rank and span to every instruction in a
    request atom (address setup, final m0 setup, and DTL load). Only a
    contiguous GR prefix may move. Any SRD update or other path tail remains
    after every request, preserving its producer dependencies.

    Return the original flattened order when metadata or topology is
    incomplete so unrelated schedules remain byte-identical.
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
                if any(metadata[index] != (rank, span)
                       for index in range(offset, end)):
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


def _hasPartitionPrefetchReads(emittedModules):
    """Whether this cell belongs to partition-directed local-read prefetch."""
    return any(
        em.opType == "lr"
        and any(_isDsRead(inst) for inst in em.instructions)
        and (
            getattr(em.source, "targetPartition",
                    getattr(em.source, "partition", 0))
            != getattr(em.source, "partition", 0)
            or getattr(em.source, "mtIteration", 0) > 0
        )
        for em in emittedModules
    )


def _nextPlacementSlot(placer):
    """Return the first slot after all instructions placed so far."""
    for pos in range(placer.totalSlots - 1, -1, -1):
        if placer._placed[pos]:
            return pos + 1
    return 0


def safeOutstandingLocalReadSuffix(
        scheduled, emittedModules, immediateMfmaCount):
    """Count youngest DS reads unused by the immediate consumer partition.

    A nonzero lgkmcnt can only prove completion for the older prefix of the
    issue stream. Therefore the reads left outstanding must form a suffix in
    final physical issue order, and every read in that suffix must have its
    first consumer beyond all MFMAs in the next partition.

    Missing metadata, an unknown consumer, or an unranked read in the suffix
    conservatively terminates the count.
    """
    if immediateMfmaCount <= 0:
        return 0

    rankByInstruction = {}
    for em in emittedModules:
        if em.opType != "lr":
            continue
        if len(em.instructionMetadata) != len(em.instructions):
            continue
        for inst, metadata in zip(
                em.instructions, em.instructionMetadata):
            if _isDsRead(inst):
                rankByInstruction[id(inst)] = metadata[0]

    issuedRanks = [
        rankByInstruction.get(id(inst))
        for inst in scheduled.flatitems()
        if _isDsRead(inst)
    ]
    safe = 0
    for rank in reversed(issuedRanks):
        if rank is None or rank < immediateMfmaCount:
            break
        safe += 1
    # gfx9 encodes lgkmcnt in four bits. A smaller count is a stronger wait,
    # so clamping an unusually large safe suffix remains conservative.
    return min(safe, _MAX_LGKMCNT)


def _rightAlignDtlBundlesBefore(followers, endExclusive):
    """Right-align complete m0/DTL pairs before a reserved MFMA suffix.

    Operate on the already scheduled physical stream so the policy adapts to
    any partition width.  Non-DTL followers stay in place; complete DTL
    bundles retain request order and are moved only into intervals with room
    for both physical instructions.

    Return a rewritten follower list, or None when every bundle cannot be
    moved atomically.
    """
    rewritten = []
    bundles = []
    for group in followers:
        residual = []
        index = 0
        while index < len(group):
            if (
                    index + 1 < len(group)
                    and _isM0Update(group[index])
                    and _isDtlLoad(group[index + 1])):
                bundles.append([group[index], group[index + 1]])
                index += 2
                continue
            residual.append(group[index])
            index += 1
        rewritten.append(residual)

    if not bundles:
        return rewritten

    def physicalCount(group):
        return sum(not isinstance(inst, TextBlock) for inst in group)

    candidates = [
        index for index in range(min(endExclusive, len(rewritten)))
        if physicalCount(rewritten[index]) <= 1
        and not any(_isDtlLoad(inst) for inst in rewritten[index])
    ]
    if len(candidates) < len(bundles):
        return None

    for target, bundle in zip(candidates[-len(bundles):], bundles):
        rewritten[target].extend(bundle)
    return rewritten


def _drainsLgkm(inst) -> bool:
    """Whether inst proves that every older LDS/SMEM request completed."""
    return (
        isinstance(inst, SWaitCnt)
        and (inst.dscnt == 0 or getattr(inst, "waitAll", False))
    )


def _isLgkmOnlyDrain(inst) -> bool:
    """Whether inst is a removable full LDS/SMEM-only drain."""
    return (
        _drainsLgkm(inst)
        and inst.dscnt == 0
        and inst.vlcnt < 0
        and inst.vscnt < 0
        and inst.kmcnt < 0
        and not getattr(inst, "waitAll", False)
    )


def _isLgkmProducer(inst) -> bool:
    """Conservatively recognize instructions that can make LGKM outstanding."""
    return any(
        cls.__name__ in {
            "LocalReadInstruction",
            "LocalWriteInstruction",
            "SMemLoadInstruction",
            "SMemStoreInstruction",
        }
        for cls in type(inst).__mro__
    )


def _endsWithDrainedLgkm(schedule) -> bool:
    """Prove that no LGKM request remains after the schedule completes."""
    drained = False
    for inst in schedule.flatitems():
        if _drainsLgkm(inst):
            drained = True
        elif _isLgkmProducer(inst):
            drained = False
    return drained


def eliminateRedundantLgkmDrains(schedule) -> int:
    """Remove dominated full LGKM-only drains from a straight-line schedule.

    Logical dependency lowering may independently add a read-completion wait
    and a later consumer-readiness wait.  Once physical placement is final,
    the latter is redundant when a full drain dominates it and no intervening
    LDS/SMEM instruction can have made LGKM outstanding again.

    Keep waits that also affect another counter: their non-LGKM effect may
    still be required.  Such waits nevertheless establish the known-drained
    LGKM state.  Starting in the unknown state also deliberately preserves the
    first wait in a loop body; this pass does not reason across a backedge.
    """
    drained = False
    removed = 0

    def rewrite(module):
        nonlocal drained, removed
        rewritten = []
        for inst in module.items():
            if isinstance(inst, Module):
                rewrite(inst)
                rewritten.append(inst)
                continue

            if _isLgkmOnlyDrain(inst) and drained:
                removed += 1
                continue

            if _drainsLgkm(inst):
                drained = True
            elif _isLgkmProducer(inst):
                drained = False
            rewritten.append(inst)
        module.setItems(rewritten)

    rewrite(schedule)
    return removed


def advancePartitionInterleave(
        previousSchedule, finalSchedule, advanceMfmas,
        rightAlignPreviousDtl=False, maxPreviousDtlLoads=None,
        dropRedundantFinalLgkm=False, advanceFinalFollowers=True):
    """Move a final partition's interleaving stream earlier by whole MFMAs.

    The entry prefix is the interval immediately before the final partition's
    first MFMA.  Treating it as part of the stream preserves the physical order
    of entry wait -> VMEM -> wait/barrier -> LDS reads while moving every item
    by the same number of MFMA intervals.  This is deliberately a narrow,
    partition-local transform: only empty intervals in the preceding partition
    may receive work, no local read may cross the partition boundary, and each
    destination keeps the strict three-follower cap.

    When maxPreviousDtlLoads is set, stop advancing at the first complete
    follower group that would exceed that DTL-load count.  Keep that group and
    every younger group in the final partition so their dependency order is
    unchanged.

    When dropRedundantFinalLgkm is set, remove an LGKM-only final-partition
    entry drain only if the preceding schedule already ends in a proven
    drained state.  Its unfiltered move remains a seam-placement anchor so
    this narrow removal cannot perturb the surrounding DTL schedule.

    When advanceFinalFollowers is False, retain every post-MFMA follower in
    the final partition.  The entry prefix still provides the seam anchor used
    by the independent P2 DTL right-alignment policy.

    Returns True and mutates both Modules when the complete shift is legal.
    Returns False without changing either Module when it is not.
    """
    if advanceMfmas <= 0:
        return False

    isMfma = lambda inst: isinstance(
        inst, (MFMAInstruction, MXMFMAInstruction))

    def split(schedule):
        prefix = []
        mfmas = []
        followers = []
        for inst in schedule.flatitems():
            if isMfma(inst):
                mfmas.append(inst)
                followers.append([])
            elif not mfmas:
                prefix.append(inst)
            else:
                followers[-1].append(inst)
        return prefix, mfmas, followers

    previousPrefix, previousMfmas, previousFollowers = split(
        previousSchedule)
    finalPrefix, finalMfmas, finalFollowers = split(finalSchedule)
    if not previousMfmas or not finalMfmas:
        return False

    droppedInstructionIds = set()
    finalPrefixInstructions = [
        inst for inst in finalPrefix
        if not isinstance(inst, TextBlock)
    ]
    if (
            dropRedundantFinalLgkm
            and len(finalPrefixInstructions) == 1
            and _isLgkmOnlyDrain(finalPrefixInstructions[0])
            and _endsWithDrainedLgkm(previousSchedule)):
        droppedInstructionIds.add(id(finalPrefixInstructions[0]))

    boundary = len(previousMfmas)
    prefixTarget = boundary - 1 - advanceMfmas
    if prefixTarget < 0:
        return False

    previousDtlLoads = sum(
        _isDtlLoad(inst)
        for group in previousFollowers
        for inst in group)
    if (
            maxPreviousDtlLoads is not None
            and previousDtlLoads > maxPreviousDtlLoads):
        return False

    moves = []
    if finalPrefix:
        moves.append((prefixTarget, finalPrefix))
    holdFinalSuffix = False
    for index, group in enumerate(finalFollowers):
        if group:
            shiftedTarget = (
                boundary + index - advanceMfmas
                if advanceFinalFollowers
                else boundary + index
            )
            groupDtlLoads = sum(_isDtlLoad(inst) for inst in group)
            if (
                    maxPreviousDtlLoads is not None
                    and not holdFinalSuffix
                    and shiftedTarget < boundary
                    and previousDtlLoads + groupDtlLoads
                    > maxPreviousDtlLoads):
                holdFinalSuffix = True
            if holdFinalSuffix:
                target = boundary + index
            else:
                target = shiftedTarget
                if target < boundary:
                    previousDtlLoads += groupDtlLoads
            moves.append((target, group))

    # This prototype only borrows genuinely empty MFMA intervals from the
    # preceding partition.  Avoid inventing a new merge order at the seam.
    previousTargets = {
        target for target, group in moves
        if group and target < boundary
    }
    if any(previousFollowers[target] for target in previousTargets):
        return False

    # Keep LR accounting partition-local.  The tested one-to-three-MFMA
    # advances move waits/VMEM/barriers/offset swaps across the seam, but leave
    # the first wrap read in the final partition.
    if any(
            isinstance(inst, LocalReadInstruction)
            for target, group in moves if target < boundary
            for inst in group):
        return False

    if rightAlignPreviousDtl and previousTargets:
        previousFollowers = _rightAlignDtlBundlesBefore(
            previousFollowers, min(previousTargets))
        if previousFollowers is None:
            return False

    combinedFollowers = [
        list(group) for group in previousFollowers + finalFollowers
    ]
    for index in range(boundary, len(combinedFollowers)):
        combinedFollowers[index] = []
    for target, group in moves:
        if target < 0 or target >= len(combinedFollowers):
            return False
        combinedFollowers[target].extend(
            inst for inst in group
            if id(inst) not in droppedInstructionIds)

    def physicalCount(group):
        return sum(not isinstance(inst, TextBlock) for inst in group)

    if any(physicalCount(group) > 3 for group in combinedFollowers):
        return False

    def assemble(prefix, mfmas, followers):
        items = list(prefix)
        for mfma, group in zip(mfmas, followers):
            items.append(mfma)
            items.extend(group)
        return items

    previousSchedule.setItems(assemble(
        previousPrefix,
        previousMfmas,
        combinedFollowers[:boundary]))
    finalSchedule.setItems(assemble(
        [],
        finalMfmas,
        combinedFollowers[boundary:]))
    return True


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


def instructionSchedule(
        emittedModules, multiDU: bool = False,
        minGapDsReadToWait=_MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT,
        partitionPrefetch: bool = False,
        limitDtlBundlesPerMfma: bool = False,
        uniformPgrEpoch: bool = False):
    """Interleave non-MFMA instructions between MFMAs.

    Rules (shared):
      - MFMA order is preserved.
      - The legacy schedule uses 2 placement slots between adjacent MFMAs.
      - Strict partition-prefetch scheduling permits a third slot when needed,
        while capping physical follower instructions at three per MFMA.
      - DS reads occupy only non-leading placement slots.
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
    p0Split = None
    isMx8Cycle = bool(mfmas) and all(
            isinstance(inst, MXMFMAInstruction)
            and inst.getIssueLatency() == 8 for inst in mfmas)
    if isMx8Cycle:
        p0Split = _splitPartitionZeroPrefetch(
            preMfmaOrders, emittedModules)
    partitionPrefetchPolicy = (
        isMx8Cycle
        and not multiDU
        and (partitionPrefetch
             or _hasPartitionPrefetchReads(emittedModules))
    )

    p0Payload = p0Split[3] if p0Split is not None else None
    p0Sync = p0Split[4] if p0Split is not None else []
    p0PayloadReordered = False
    if p0Split is not None:
        reordered = _orderLocalReadsByFirstConsumer(
            p0Payload, emittedModules)
        if reordered is not None:
            p0Payload = reordered
            p0PayloadReordered = True

    strictDtlSpacing = (
        limitDtlBundlesPerMfma
        and _hasDtlIntervalCapacity(
            pathOrders, emittedModules, max(0, len(mfmas) - 1))
    )
    # The strict partition-prefetch policy normally emits two instructions
    # between adjacent MFMAs.  A third placement slot lets the leading LDS
    # cohort issue a second read when necessary, so it completes before the
    # first future-epoch DTL bundle without stacking multiple DTL bundles in
    # one MFMA interval.
    slotsPerInterval = 3 if strictDtlSpacing else 2
    placementPathOrders = list(pathOrders)
    if p0Split is not None and not p0PayloadReordered:
        placementPathOrders.append(p0Split[2])

    def _emitPreMfma(result, deferSync=False):
        deferredSyncIds = {
            id(inst) for _, inst in p0Sync
        } if deferSync else set()
        for order in preMfmaOrders:
            if p0Split is not None and order == p0Split[0]:
                for _, inst in p0Split[1]:
                    if id(inst) in deferredSyncIds:
                        continue
                    result.add(inst)
                continue
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)

    # Single MFMA: no slots to interleave into — emit preMfma, MFMA, then paths.
    if len(mfmas) < 2:
        result = Module()
        _emitPreMfma(result)
        for m in mfmas:
            result.add(m)
        if p0Split is not None:
            for _, inst in p0Payload:
                result.add(inst)
        for order in pathOrders:
            for mid in order:
                for inst in emittedModules[mid].instructions:
                    result.add(inst)
        return result

    paths = _classifyPaths(pathOrders, emittedModules)
    uniformEpochFrontier = (
        uniformPgrEpoch
        and any(hasWaitGR for _, hasWaitGR in paths)
    )
    wrapUniformEpochFrontier = (
        uniformEpochFrontier
        and any(
            _isWrapPrefetchPath(order, emittedModules)
            for order, _hasWaitGR in paths
        )
    )
    phasePathInsts = []
    canUseThreePhase = partitionPrefetchPolicy
    for order, _hasWaitGR in paths:
        pathInsts = _flattenPath(order, emittedModules)
        hasLR = any(_isDsRead(inst) for _, inst in pathInsts)
        hasGR = any(_isBufferLoad(inst) for _, inst in pathInsts)
        if hasLR and hasGR:
            canUseThreePhase = False
        phasePathInsts.append((order, pathInsts, hasLR, hasGR))
    if canUseThreePhase:
        # The phase scheduler below preserves dependencies explicitly.
        placementPathOrders = []

    rules = _SchedulingRules(
        totalSlots=(len(mfmas) - 1) * slotsPerInterval,
        minGapDsReadToWait=minGapDsReadToWait)
    base_validators = [rules.oneDsReadPerInterval, rules.minGapDsReadBeforeWait,
                       rules.minGapDsReadToWait]
    if not (partitionPrefetchPolicy or p0Split is not None):
        base_validators.append(rules.noM0WithBufferLoad)
    base_adjusters = [rules.spreadLocalReads, rules.spreadBufferLoads]
    if partitionPrefetchPolicy or p0Split is not None:
        base_validators.append(rules.noDsReadOrValuInFirstSlot)
        if strictDtlSpacing:
            base_validators.append(rules.oneDtlLoadPerInterval)
    if multiDU:
        base_validators.append(rules.grAfterWaitGrDsReads)
        base_adjusters.insert(0, rules.deferGrAfterWaitGrDsReads)
    placer = _SlotPlacer(
        len(mfmas) - 1, n, placementPathOrders,
        validators=base_validators,
        adjusters=base_adjusters,
        onPlace=rules.trackPlacement,
        slotsPerInterval=slotsPerInterval,
        oneDtlPerInterval=(
            strictDtlSpacing
            and (partitionPrefetchPolicy or p0Split is not None)),
        maxInstructionsPerInterval=(
            3 if strictDtlSpacing
            and (partitionPrefetchPolicy or p0Split is not None)
            else None))

    def _placeRegularPaths():
        for order, hasWaitGR in paths:
            if not order:
                continue
            wrapPrefetch = (not multiDU) and hasWaitGR \
                and _isWrapPrefetchPath(order, emittedModules)
            if wrapPrefetch:
                # The final partition wraps its local-read prefetch to partition 0
                # of the next MT. Legacy reverse packing runs out of valid slots and
                # force-stacks wait+barrier, LR swaps, and several ds_reads before
                # MFMA #1. Keep only the synchronization in the first interval;
                # current-partition MFMAs do not consume the wrapped LR payload.
                pathInsts = _flattenPath(order, emittedModules)
                syncEnd = next(
                    (idx + 1 for idx, (_, inst) in enumerate(pathInsts)
                     if _isBarrier(inst)),
                    0)
                if syncEnd:
                    rules.resetPath()
                    placer.placePath(
                        pathInsts[:syncEnd],
                        multiDU=True)
                    placer.placePath(
                        pathInsts[syncEnd:],
                        startLimit=2,
                        overflowToTail=True)
                    continue

            reverse = (not multiDU) and hasWaitGR
            pathInsts = _flattenPath(order, emittedModules, reverse=reverse)
            rules.resetPath()
            if not hasWaitGR:
                rules.setupBufLoadSpreading(placer, pathInsts, order)
            placer.placePath(pathInsts, reverse=reverse, multiDU=multiDU)
            if multiDU and hasWaitGR:
                rules.setPostWaitGrDsReadFence(placer)

    if canUseThreePhase:
        leading = []
        trailing = []
        grPaths = []
        otherPaths = []

        for _order, pathInsts, hasLR, hasGR in phasePathInsts:
            if hasLR:
                first, rest = _splitAfterFirstConsumerCohort(
                    pathInsts, emittedModules)
                leading.extend(first)
                trailing.extend(rest)
            elif hasGR:
                grPaths.append(pathInsts)
            else:
                otherPaths.append(pathInsts)

        orderedGrItems = _orderGlobalReadBundlesByFirstConsumer(
            grPaths, emittedModules)

        overwriteReads = []
        if p0Split is not None:
            overwriteReads = [
                item for item in p0Payload
                if getattr(
                    emittedModules[item[0]].source,
                    "mustPrecedeP0Overwrite",
                    False)
            ]
            regularP0Payload = [
                item for item in p0Payload
                if not getattr(
                    emittedModules[item[0]].source,
                    "mustPrecedeP0Overwrite",
                    False)
            ]
            first, rest = _splitAfterFirstConsumerCohort(
                regularP0Payload, emittedModules)
            leading.extend(first)
            # These current-epoch reads overlap the p0 future-write cohort.
            # Keep the normal next-partition LR prefix first, but issue every
            # overwrite prerequisite before the GR path's lgkmcnt+barrier.
            leading.extend(overwriteReads)
            trailing.extend(rest)
            for _, inst in overwriteReads:
                if _isDsRead(inst):
                    rules.p0OverwritePrerequisiteIds.add(id(inst))

        allPhaseItems = (
            p0Sync
            + leading
            + orderedGrItems
            + trailing
            + [item for path in otherPaths for item in path]
        )
        numDsReads = sum(
            _isDsRead(inst) for _, inst in allPhaseItems)
        numBufferLoads = sum(
            _isBufferLoad(inst) for _, inst in allPhaseItems)
        # Right-align non-p0 work so synchronization remains hidden behind
        # independent MFMAs. Reserve holes for DS issue-pair restrictions and
        # m0/load pairing; p0's synchronization remains before MFMA0.
        reserveSlots = (numDsReads + 1) // 2 + numBufferLoads
        # Each atomic m0/DTL-load pair consumes one abstract placement slot,
        # not two. Account for that when right-aligning the VMEM tail.
        estimatedSlots = len(allPhaseItems) - numBufferLoads
        phaseLimit = (
            0
            if (p0Split is not None or uniformPgrEpoch)
            else max(0, placer.totalSlots - estimatedSlots - reserveSlots)
        )

        def _placePhase(
                pathInsts, atomicDtl=False,
                mfmaBetweenWaitAndBarrier=False, denseToFit=False,
                startOverride=None, overflowToTail=False):
            nonlocal phaseLimit
            if not pathInsts:
                return
            rules.resetPath()
            placer.denseToFit = denseToFit
            try:
                placer.placePath(
                    pathInsts,
                    startLimit=(
                        phaseLimit
                        if startOverride is None
                        else startOverride),
                    overflowToTail=overflowToTail,
                    atomicDtl=atomicDtl,
                    mfmaBetweenWaitAndBarrier=mfmaBetweenWaitAndBarrier)
            finally:
                placer.denseToFit = False
            phaseLimit = max(phaseLimit, _nextPlacementSlot(placer))

        _placePhase(p0Sync)
        if leading:
            if uniformEpochFrontier:
                if wrapUniformEpochFrontier:
                    # At the cyclic frontier, prioritize the LDS reads that
                    # feed the next body's first MFMAs.  Future-epoch DTL
                    # requests target the alternate LDS half, so they remain
                    # legal after vmcnt -> barrier -> LR and are deferred as
                    # complete m0/load bundles.
                    _placePhase(
                        leading + trailing,
                        mfmaBetweenWaitAndBarrier=True,
                        denseToFit=True)
                    _placePhase(orderedGrItems, atomicDtl=True)
                else:
                    # Complete the future epoch first.  The following VMEM
                    # wait is adjusted by the post-pass for every buffer load
                    # placed here.
                    _placePhase(orderedGrItems, atomicDtl=True)
                    _placePhase(
                        leading + trailing,
                        mfmaBetweenWaitAndBarrier=True,
                        denseToFit=True)
            else:
                _placePhase(leading, mfmaBetweenWaitAndBarrier=True)
                if overwriteReads and trailing and orderedGrItems:
                    # Fill the MFMA intervals needed to age the prerequisite
                    # read with unrelated, younger LDS reads.  The overwrite
                    # wait is converted to lgkmcnt(N) below, so these requests
                    # may remain pending while the older overlapping read is
                    # guaranteed complete.
                    budget = max(
                        0,
                        (rules.minGap - 1)
                        * (slotsPerInterval - 1))
                    preWaitTail = []
                    postWaitTail = []
                    used = 0
                    index = 0
                    while index < len(trailing):
                        moduleId = trailing[index][0]
                        inst = trailing[index][1]
                        instructionIndex = next(
                            (itemIndex for itemIndex, candidate
                             in enumerate(
                                 emittedModules[moduleId].instructions)
                             if candidate is inst),
                            -1)
                        metadata = emittedModules[moduleId].instructionMetadata
                        atomRank = (
                            metadata[instructionIndex][0]
                            if (instructionIndex >= 0
                                and instructionIndex < len(metadata))
                            else None)
                        end = index + 1
                        while (end < len(trailing)
                               and trailing[end][0] == moduleId):
                            nextInst = trailing[end][1]
                            nextIndex = next(
                                (itemIndex for itemIndex, candidate
                                 in enumerate(
                                     emittedModules[moduleId].instructions)
                                 if candidate is nextInst),
                                -1)
                            nextRank = (
                                metadata[nextIndex][0]
                                if (nextIndex >= 0
                                    and nextIndex < len(metadata))
                                else None)
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
                    # The five-request p0 cohort leaves one physical follower
                    # slot in most MFMA intervals.  Reuse those holes for the
                    # independent LR tail instead of stacking it after the
                    # penultimate MFMA.  Dependencies still prevent any item
                    # from crossing its producer; this only relaxes the
                    # artificial phase frontier.
                    grSlots = [
                        pos
                        for pos, slot in enumerate(placer._placed)
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
            # No immediate consumer can use an LR prefix. Keep independent
            # work first and right-align the VMEM cohort near the partition
            # tail, as in the CMS schedule.
            _placePhase(trailing)
            for pathInsts in otherPaths:
                _placePhase(pathInsts)
            _placePhase(orderedGrItems, atomicDtl=True)
    else:
        _placeRegularPaths()
        if p0Split is not None:
            rules.resetPath()
            placer.placePath(p0Payload)

    scheduled = Module()
    _emitPreMfma(scheduled, deferSync=canUseThreePhase)
    scheduled.add(placer.assemble(mfmas))

    # Post-pass: adjust vmcnt of any SWaitCnt to account for buffer_loads
    # that the scheduler placed before it within this subIterK.
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
            if getattr(inst, 'adjustVmcnt', True):
                inst.vlcnt += bufLoadCount

    return scheduled
