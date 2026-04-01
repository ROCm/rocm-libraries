"""Subtile-based mainloop scheduler.

The scheduler builds an instruction schedule for the preloop, mainloop, NGLL & NLL. The GEMM is split
into Partitions, again split into subIterK steps.

Naming conventions:
  - Partition:  A rectangle of subtiles (partitionSizeA x partitionSizeB)
                processed together in one mainloop step. 
  - subIterK:   K-dimension sub-iteration within a partition. Each subtile's
                data is split along K into numSubIterK chunks (hardcoded to 2).
  - MT iteration (macrotile iteration): Which macrotile's data is being
                referenced. "n" = current iteration, "n+1" = next iteration, "n+2" = two ahead.

Scheduling pipeline:
  1. _buildPreloop     — Emit initial GR + LR to init the pipeline (preloop).
  2. _buildSubIterK    — For each partition and subIterK, build MFMA and LR
                         modules with VGPR tile assignments.
  3. _insertGROps      — Split GR loads across subIterK=0/1 within each partition.
  4. _annotateDependencies — Wire up before/after dependency edges between modules
                         (WAIT_GR, SYNC, LR_INC, GR_INC, WAIT_LR).
  5. _buildNGLL        — Derive the No-Global-Load-Loop from mainloop
                         (remove GR(n+2) and GR_INC).
  6. _buildNLL         — Derive the No-Load-Loop from mainloop
                         (remove all GR, LR(n+1), and associated sync ops).

Emission pipeline (called by the kernel writer):
  7. _buildEmittedModules — Convert AnnotatedModules into EmittedModules with
                            actual GPU instructions and before-link chains.
  8. instructionSchedule  — Interleave non-MFMA instructions between MFMAs
                            using a slot-based placer with pluggable rules.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
import math
from typing import List, Tuple, Dict, Set, Optional, Union
from Tensile.Components.SubtileBasedKernel import TileInfo
from Tensile.Components.SubtileBasedKernel import emitMfmaInstruction
from Tensile.Components.SubtileBasedKernel import emitSingleDsRead
from Tensile.Components.SubtileBasedKernel import emitSingleBufferLoad
from Tensile.Components.SubtileBasedKernel import globalReadPtrUpdates, globalReadLDSBufferSwap, localReadLDSBufferSwap
from Tensile.Components.SubtileBasedKernel import globalReadDoScaleSubtile, globalReadScalePtrUpdates
from rocisa.code import Module, Label
from rocisa.instruction import SWaitCnt, SBarrier, SCmpEQU32, SCmpLeU32, SCBranchSCC1, MFMAInstruction, \
    MXMFMAInstruction, LocalReadInstruction, GlobalReadInstruction, DSLoadB32, CommonInstruction
from rocisa.container import sgpr, vgpr, DSModifiers


class PrefetchMode(Enum):
    NO = auto()
    HALF_PREFETCH = auto()
    FULL_PREFETCH = auto()


class VGPRTileReUseStrategy(Enum):
    ACROSS_PARTITIONS = auto()


class PartitionOrdering(Enum):
    COLUMN_MAJOR = auto()
    SNAKE_COLUMN_MAJOR = auto()


@dataclass
class SchedulerConfig:
    partitionSizeA: int
    partitionSizeB: int
    prefetchMode: PrefetchMode
    reuseStrategy: VGPRTileReUseStrategy = VGPRTileReUseStrategy.ACROSS_PARTITIONS
    ordering: PartitionOrdering = PartitionOrdering.COLUMN_MAJOR


@dataclass
class Partition:
    """A rectangle (sizeA x sizeB) of subtiles processed together."""
    partitionId: int
    sizeA: int
    sizeB: int
    subtiles: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def tileAIndices(self) -> List[int]:
        return sorted(set(t[0] for t in self.subtiles))

    @property
    def tileBIndices(self) -> List[int]:
        return sorted(set(t[1] for t in self.subtiles))



# Key type for allocator: (subtileIdx, subIterK)
AllocKey = Tuple[int, int]


class _VGPRPool:
    """Free-list VGPR tile allocator with separate A/B maps and peak tracking."""

    def __init__(self):
        self._nextId: int = 0
        self._peak: int = 0
        self._freeList: List[int] = []
        self._mapA: Dict = {}
        self._mapB: Dict = {}

    def _map(self, tc: str) -> Dict:
        return self._mapA if tc == 'A' else self._mapB

    def _updatePeak(self):
        self._peak = max(self._peak, len(self._mapA) + len(self._mapB))

    def allocate(self, tc: str, key) -> int:
        if self._freeList:
            vid = self._freeList.pop(0)
        else:
            vid = self._nextId
            self._nextId += 1
        self._map(tc)[key] = vid
        self._updatePeak()
        return vid

    def release(self, tc: str, key) -> None:
        vid = self._map(tc).pop(key)
        self._freeList.append(vid)

    def isAllocated(self, tc: str, key) -> bool:
        return key in self._map(tc)

    def get(self, tc: str, key) -> int:
        return self._map(tc)[key]

    @property
    def peak(self) -> int:
        return self._peak


class VGPRTileAllocator:
    """VGPR tile allocator with free-list reuse, backed by a _VGPRPool.

    Keyed by (subtileIdx, subIterK), one allocation per subtile per K step.
    """

    def __init__(self):
        self._tiles = _VGPRPool()

    def allocate(self, tc: str, subtileIdx: int, subIterK: int) -> int:
        return self._tiles.allocate(tc, (subtileIdx, subIterK))

    def release(self, tc: str, subtileIdx: int, subIterK: int) -> None:
        self._tiles.release(tc, (subtileIdx, subIterK))

    def isAllocated(self, tc: str, subtileIdx: int, subIterK: int) -> bool:
        return self._tiles.isAllocated(tc, (subtileIdx, subIterK))

    def getVGPRTileId(self, tc: str, subtileIdx: int, subIterK: int) -> int:
        return self._tiles.get(tc, (subtileIdx, subIterK))

    def releaseAllForTile(self, tc: str, subtileIdx: int) -> None:
        """Release all subIterK allocations for a given subtile index."""
        allocMap = self._tiles._map(tc)
        keys = [k for k in allocMap if k[0] == subtileIdx]
        for k in keys:
            vid = allocMap.pop(k)
            self._tiles._freeList.append(vid)

    @property
    def totalVGPRTiles(self) -> int:
        return self._tiles.peak



@dataclass
class MFMAOp:
    mtIteration: str  # e.g. "n"
    subIterK: int
    subtiles: List[Tuple[int, int]]
    vgprTileMapA: Dict[int, int]
    vgprTileMapB: Dict[int, int]
    scaleMapA: Dict[int, int] = field(default_factory=dict)  # scaleGroupIdx → scaleVgprTileId
    scaleMapB: Dict[int, int] = field(default_factory=dict)


@dataclass
class GROp:
    mtIteration: str  # e.g. "n+1", "n+2", "0", "1"
    subtileA: List[int]
    subtileB: List[int]
    lastForMT: bool = False  # True = last partition's GR for this MT → emit ptrUpdate+swap
    firstForMT: bool = False # True = first partition's GR for this MT → emit scale loads


@dataclass
class WaitGROp:
    mtIteration: str  # e.g. "n", "n+1"
    subtileA: List[int]
    subtileB: List[int]
    inflightLoadsA: Optional[int] = None
    inflightLoadsB: Optional[int] = None
    inflightScaleLoadsA: int = 0
    inflightScaleLoadsB: int = 0


@dataclass
class WaitLROp:
    pass


@dataclass
class SyncOp:
    comment: str = ""


@dataclass
class LROp:
    mtIteration: str  # e.g. "n", "n+1", "0"
    subIterK: int
    lrLoadA: Dict[int, int]
    lrLoadB: Dict[int, int]
    lrScaleA: Dict[int, int] = field(default_factory=dict)  # scaleGroupIdx → scaleVgprTileId
    lrScaleB: Dict[int, int] = field(default_factory=dict)


@dataclass
class SkipOp:
    compare: str  # "EQ", "LE"
    value: int    # LoopCounter compared against this
    target: str   # "NLL", "NGLL"


@dataclass
class GR_INCOp:
    """Global Read increment: pointer updates + LDS buffer swap for GR."""
    pass


@dataclass
class LR_INCOp:
    """Local Read increment: LDS buffer swap for LR."""
    pass


ScheduleOp = Union[MFMAOp, GROp, WaitGROp, WaitLROp, SyncOp, LROp, SkipOp, GR_INCOp, LR_INCOp]


@dataclass
class DepEdge:
    """A synchronization/housekeeping op or module reference that acts as a dependency edge.

    Either op or module is set, not both:
    - op: a sync/housekeeping instruction to emit (WAIT_GR, WAIT_LR, SYNC, etc.)
    - module: a reference to another AnnotatedModule that must complete first (ordering constraint)
    """
    op: Union[WaitGROp, WaitLROp, SyncOp, GR_INCOp, LR_INCOp, None] = None
    module: Optional['AnnotatedModule'] = None


@dataclass
class AnnotatedModule:
    """A module with dependency edges."""
    op: ScheduleOp
    before: List[DepEdge] = field(default_factory=list)
    after: List[DepEdge] = field(default_factory=list)


@dataclass
class EmittedModule:
    """One emitted module with instructions and module-link deps."""
    moduleId: int = -1
    instructions: list = field(default_factory=list)
    before: Optional[int] = None                     # moduleId that must run before this module
    opType: str = ""


@dataclass
class PartitionGR:
    """Describes the GR (Global Read) issued by a partition during the mainloop."""
    mtIteration: str       # "n+1" or "n+2"
    targetPartitionId: int # which partition's subtiles we're loading for
    subtileA: Set[int]     # actual subtiles to load (after dedup)
    subtileB: Set[int]


@dataclass
class SubIterKSchedule:
    """Ops for one subIterK iteration within a partition."""
    subIterK: int
    modules: List[AnnotatedModule] = field(default_factory=list)


@dataclass
class PartitionSchedule:
    """Schedule for one subtile partition — contains all subIterK iterations."""
    partitionId: int
    subIterKSteps: List[SubIterKSchedule] = field(default_factory=list)


class _SlotPlacer:
    """Generic slot placement engine for interleaving instructions between MFMAs.

    Each interval (pair of adjacent MFMAs) has 2 placement slots.
    Rules are injected via callbacks:
      - validators: (placer, pos, inst) -> bool — reject invalid slots
      - adjusters:  (placer, limit, inst) -> limit — shift search start
      - onPlace:    (placer, pos, inst) -> None — update rule state after placement
    """

    def __init__(self, intervals: int, numModules: int,
                 pathOrders: List[List[int]],
                 validators=None, adjusters=None, onPlace=None):
        self.totalSlots = intervals * 2
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

        self._placed: List[List[Tuple[int, object]]] = [[] for _ in range(self.totalSlots)]
        self._firstPos: List[Optional[int]] = [None] * numModules
        self._lastPos: List[Optional[int]] = [None] * numModules
        self.leftovers: List[Tuple[int, object]] = []

    # ── Placement ──

    def _canPlace(self, pos: int, inst) -> bool:
        if pos < 0 or pos >= self.totalSlots or len(self._placed[pos]) >= 2:
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

    def placePath(self, pathInsts: List[Tuple[int, object]], reverse: bool = False):
        """Place a sequence of (moduleId, instruction) items into slots.

        Walks pathInsts in order, applying adjusters (forward only) and
        finding valid slots. When no empty slot is found, force-places at
        the closest valid position respecting dependencies (allowing >2
        items per slot).
        """
        limit = (self.totalSlots - 1) if reverse else 0
        for idx, item in enumerate(pathInsts):
            mid, inst = item
            if not reverse:
                limit = self.adjustLimit(limit, inst)
            pos = self.findSlot(mid, inst, limit, reverse=reverse)
            if pos is None:
                pos = self._forceSlot(mid, limit, reverse)
            self.place(pos, item, reverse=reverse)
            limit = (pos - 1) if reverse else (pos + 1)

    # ── Assembly ──

    def assemble(self, mfmas) -> Module:
        intervals = len(mfmas) - 1
        result = Module()
        result.add(mfmas[0])
        for i in range(intervals):
            for slot in (2 * i, 2 * i + 1):
                for item in self._placed[slot]:
                    result.add(item[1])
            result.add(mfmas[i + 1])
        for _, inst in self.leftovers:
            result.add(inst)
        return result


# ── Scheduling rules ──

# Hardcoded gap to hide ds_read latency. TODO: compute this more accurately.
_MIN_MFMA_GAP_DS_READ_TO_WAIT = 4

_isDsRead = lambda x: isinstance(x, LocalReadInstruction)
_isBufferLoad = lambda x: isinstance(x, GlobalReadInstruction)
_isWaitCnt = lambda x: isinstance(x, SWaitCnt)
_isM0Update = lambda x: isinstance(x, CommonInstruction) and hasattr(x, 'dst') and hasattr(x.dst, 'regType') and x.dst.regType == 'm'


class _SchedulingRules:
    """Scheduling rules for slot placement: validators, adjusters, and placement hooks.

    Owns all rule state (ds_read/waitcnt tracking, buffer-load spreading).
    Bound methods are passed as callbacks to _SlotPlacer.
    """

    def __init__(self, totalSlots: int):
        # Cross-path state
        self.lastDsReadPos = -1
        self.earliestWaitCntPos = totalSlots
        # Per-path state
        self._resetPath()

    def _resetPath(self):
        self.firstBufLoadPos: Optional[int] = None
        self.bufLoadIdx = 0
        self.bufLoadMaxSlot = 0
        self.numBufLoads = 0

    # ── Validators: (placer, pos, inst) -> bool ──

    def oneDsReadPerInterval(self, placer, pos, inst):
        """At most one ds_read per interval (pair of slots) to avoid same SIMD pair stalls as we have a single codepath"""
        if not _isDsRead(inst):
            return True
        peer = pos ^ 1
        return not (0 <= peer < placer.totalSlots
                    and any(_isDsRead(item[1]) for item in placer._placed[peer]))

    def minGapDsReadBeforeWait(self, placer, pos, inst):
        """Reject ds_read too close to an already-placed waitcnt ahead."""
        if not _isDsRead(inst):
            return True
        gap = _MIN_MFMA_GAP_DS_READ_TO_WAIT * 2
        return self.earliestWaitCntPos - pos >= gap

    def minGapDsReadToWait(self, placer, pos, inst):
        """Reject waitcnt too close to the last placed ds_read."""
        if not _isWaitCnt(inst) or self.lastDsReadPos < 0:
            return True
        gap = _MIN_MFMA_GAP_DS_READ_TO_WAIT * 2
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

    # ── Adjusters: (placer, limit, inst) -> limit ──

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


class SubtileBasedScheduler:
    def __init__(self, tileInfoA, tileInfoB, config: SchedulerConfig,
                 scaleTileInfoA=None, scaleTileInfoB=None):
        self.tileInfoA = tileInfoA
        self.tileInfoB = tileInfoB
        self.scaleTileInfoA = scaleTileInfoA
        self.scaleTileInfoB = scaleTileInfoB
        self.hasScale = scaleTileInfoA is not None and scaleTileInfoB is not None
        # Number of scale loads per MT (one load covers the entire MT)
        self.scaleLoadsPerMT_A = 1 if self.hasScale else 0
        self.scaleLoadsPerMT_B = 1 if self.hasScale else 0
        self.config = config

        self.MTA = tileInfoA.localSubtileGrid[0]
        self.MTB = tileInfoB.localSubtileGrid[0]

        assert self.MTA % config.partitionSizeA == 0, \
            f"MTA ({self.MTA}) must be divisible by partitionSizeA ({config.partitionSizeA})"
        assert self.MTB % config.partitionSizeB == 0, \
            f"MTB ({self.MTB}) must be divisible by partitionSizeB ({config.partitionSizeB})"

        self.numPartitionsA = self.MTA // config.partitionSizeA
        self.numPartitionsB = self.MTB // config.partitionSizeB

        self.numSubIterK = tileInfoA.subtileShape[1]
        assert self.numSubIterK == tileInfoB.subtileShape[1], \
            "A and B must have same subtileShape[1]"
        assert tileInfoA.localSubtileGrid[1] == 1, \
            f"Scheduler requires localSubtileGrid[1]==1 for A, got {tileInfoA.localSubtileGrid[1]}"
        assert tileInfoB.localSubtileGrid[1] == 1, \
            f"Scheduler requires localSubtileGrid[1]==1 for B, got {tileInfoB.localSubtileGrid[1]}"

        self.partitions: List[Partition] = self._buildPartitions()
        self.allocator = VGPRTileAllocator()

        # Scale VGPR tile IDs are deterministic: gid for A, numScaleGroupsA + gid for B.
        self.numScaleGroupsA = math.ceil(self.MTA / 2) if self.hasScale else 0
        self.numScaleGroupsB = math.ceil(self.MTB / 2) if self.hasScale else 0
        self.totalScaleVGPRTiles = self.numScaleGroupsA + self.numScaleGroupsB

        self._runSchedule()

    # ── Outputs ──────────────────────────────────────────────

    @property
    def totalVGPRTiles(self) -> int:
        return self.allocator.totalVGPRTiles

    def scaleVid(self, tc: str, subtileIdx: int) -> Tuple[int, int]:
        """Deterministic scale VGPR tile ID for the group containing subtileIdx.
        Returns (scaleGroupIdx, vgprTileId)."""
        gid = subtileIdx // 2
        vid = gid if tc == 'A' else self.numScaleGroupsA + gid
        return gid, vid

    # ── Partition construction ─────────────────────────────────

    def _generateOrder(self) -> List[Tuple[int, int]]:
        order = []
        if self.config.ordering == PartitionOrdering.COLUMN_MAJOR:
            for col in range(self.numPartitionsB):
                for row in range(self.numPartitionsA):
                    order.append((row, col))
        elif self.config.ordering == PartitionOrdering.SNAKE_COLUMN_MAJOR:
            for col in range(self.numPartitionsB):
                if col % 2 == 0:
                    for row in range(self.numPartitionsA):
                        order.append((row, col))
                else:
                    for row in range(self.numPartitionsA - 1, -1, -1):
                        order.append((row, col))
        return order

    def _buildPartitions(self) -> List[Partition]:
        order = self._generateOrder()
        sA = self.config.partitionSizeA
        sB = self.config.partitionSizeB
        partitions = []
        for partitionId, (pA, pB) in enumerate(order):
            subtiles = []
            for a in range(pA * sA, (pA + 1) * sA):
                for b in range(pB * sB, (pB + 1) * sB):
                    subtiles.append((a, b))
            partitions.append(Partition(partitionId=partitionId, sizeA=sA, sizeB=sB, subtiles=subtiles))
        return partitions

    # ── Scheduling core ──────────────────────────────────────

    def _computePartitionGRs(self, preloadedMTn1_A: Set[int], preloadedMTn1_B: Set[int]) -> Dict[int, PartitionGR]:
        """Compute each partition's GR target (mtIteration, targetPartition, subtiles).
        Current behavior is : load MT n+1, partition + 1.
        TODO. Change this to allow better GR spreading accross partition when using multi-partitions config

        Args:
            preloadedMTn1_A/B: MT n+1 subtiles already loaded by the preloop's GR(MT 1).
                These are excluded from mainloop MT n+1 GRs (dedup).
        """
        numPartitions = len(self.partitions)
        partitionGRs = {}
        loadedA = set(preloadedMTn1_A)
        loadedB = set(preloadedMTn1_B)
        for pi in range(numPartitions):
            targetPi = (pi + 1) % numPartitions
            targetPartition = self.partitions[targetPi]
            if pi == numPartitions - 1:
                # Last partition wraps to next macrotile iteration
                bufA = set(targetPartition.tileAIndices)
                bufB = set(targetPartition.tileBIndices)
                mtIter = "n+2"
            else:
                needA = set(targetPartition.tileAIndices)
                needB = set(targetPartition.tileBIndices)
                bufA = needA - loadedA
                bufB = needB - loadedB
                loadedA |= needA
                loadedB |= needB
                mtIter = "n+1"
            partitionGRs[pi] = PartitionGR(mtIteration=mtIter, targetPartitionId=targetPi,
                                            subtileA=bufA, subtileB=bufB)
        return partitionGRs

    def _buildPreloop(self) -> Tuple[Set[int], Set[int]]:
        """Allocate VGPRTile IDs for the first partition and build preloop GR ops.

        Preloop loads:
          - GR(MT 0): all subtiles
          - GR(MT 1): first partition's subtiles (1 partition worth of MT 1 data)

        Returns (preloadedMT1_A, preloadedMT1_B): what was preloaded for MT 1,
        so _computePartitionGRs can use it as the initial loaded state.
        """
        first = self.partitions[0]
        allA = list(range(self.MTA))
        allB = list(range(self.MTB))
        preloadMT1_A = list(first.tileAIndices)
        preloadMT1_B = list(first.tileBIndices)

        # Number of subIterK to preload LR for
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            numPreloadSubIterKs = 1
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            numPreloadSubIterKs = self.numSubIterK

        # Allocate VGPRs for first group and build LR maps
        lrOps = []
        for sik in range(numPreloadSubIterKs):
            lrLoadA = {}
            lrLoadB = {}
            for tA in first.tileAIndices:
                lrLoadA[tA] = self.allocator.allocate('A', tA, sik)
            for tB in first.tileBIndices:
                lrLoadB[tB] = self.allocator.allocate('B', tB, sik)

            # Allocate scale VGPRs at subIterK==0 (scale is constant across subIterK)
            lrScaleA = {}
            lrScaleB = {}
            if self.hasScale and sik == 0:
                for tA in first.tileAIndices:
                    gid, vid = self.scaleVid('A', tA)
                    lrScaleA.setdefault(gid, vid)
                for tB in first.tileBIndices:
                    gid, vid = self.scaleVid('B', tB)
                    lrScaleB.setdefault(gid, vid)

            lrOps.append(LROp(mtIteration="0", subIterK=sik,
                              lrLoadA=lrLoadA, lrLoadB=lrLoadB,
                              lrScaleA=lrScaleA, lrScaleB=lrScaleB))

        # Build preloop steps: GR(MT0) split by partition, WAIT, LR(MT0), SKIP guards, GR(MT1)
        preloopOps: List[ScheduleOp] = []
        # Split MT 0 GR by partition with dedup (same order as mainloop)
        loadedA: Set[int] = set()
        loadedB: Set[int] = set()
        for partition in self.partitions:
            grA = sorted(set(partition.tileAIndices) - loadedA)
            grB = sorted(set(partition.tileBIndices) - loadedB)
            loadedA.update(partition.tileAIndices)
            loadedB.update(partition.tileBIndices)
            if grA or grB:
                preloopOps.append(GROp(mtIteration="0",
                                       subtileA=grA, subtileB=grB,
                                       lastForMT=False))
        # Mark the first MT 0 GR as firstForMT
        for i in range(len(preloopOps)):
            if isinstance(preloopOps[i], GROp):
                preloopOps[i].firstForMT = True
                break
        # Mark the last MT 0 GR as lastForMT
        for i in range(len(preloopOps) - 1, -1, -1):
            if isinstance(preloopOps[i], GROp):
                preloopOps[i].lastForMT = True
                break
        preloopOps.append(GR_INCOp())
        preloopOps.append(WaitGROp(mtIteration="0",
                                 subtileA=allA, subtileB=allB,
                                 inflightLoadsA=0, inflightLoadsB=0))
        preloopOps.append(SyncOp(comment="Barrier: wait for GR data before LR"))
        preloopOps.extend(lrOps)
        preloopOps.append(WaitLROp())
        # With double-buffered scale VGPRs, counterL<=1 skips to a separate NLL
        # that reads from scale set 0 (where the preloop LR wrote).
        nllTarget = "NLLEarly" if self.hasScale else "NLL"
        preloopOps.append(SkipOp(compare="LE", value=1, target=nllTarget))
        mt1Complete = (set(preloadMT1_A) == set(allA) and set(preloadMT1_B) == set(allB))
        preloopOps.append(GROp(mtIteration="1",
                               subtileA=preloadMT1_A, subtileB=preloadMT1_B,
                               firstForMT=True, lastForMT=mt1Complete))
        if mt1Complete:
            preloopOps.append(GR_INCOp())
        preloopOps.append(SkipOp(compare="LE", value=2, target="NGLL"))
        preloopSik = SubIterKSchedule(subIterK=0)
        preloopSik.modules = [AnnotatedModule(op=op) for op in preloopOps]
        self.preloopSteps: List[PartitionSchedule] = [
            PartitionSchedule(partitionId=0, subIterKSteps=[preloopSik])]

        return set(preloadMT1_A), set(preloadMT1_B)

    def _buildSubIterK(self, partition, pi, sik, numPartitions):
        """Build MFMA + LR modules for one subIterK step within a partition."""
        # MFMA: map subtile indices to VGPR tile IDs
        vgprTileMapA = {tA: self.allocator.getVGPRTileId('A', tA, sik)
                        for tA in partition.tileAIndices}
        vgprTileMapB = {tB: self.allocator.getVGPRTileId('B', tB, sik)
                        for tB in partition.tileBIndices}

        # MFMA scale maps
        scaleMapA, scaleMapB = {}, {}
        if self.hasScale:
            for tA in partition.tileAIndices:
                gid, vid = self.scaleVid('A', tA)
                scaleMapA.setdefault(gid, vid)
            for tB in partition.tileBIndices:
                gid, vid = self.scaleVid('B', tB)
                scaleMapB.setdefault(gid, vid)

        # LR: load targets determined by prefetch mode
        loadATiles, loadBTiles, loadSubIterK = self._getLoadTargets(pi, sik, numPartitions)
        isWrapAround = self._isWrapAroundLoad(pi, sik, numPartitions)

        lrLoadA = {tA: v for tA in (loadATiles or [])
                   if (v := self._loadTile('A', tA, loadSubIterK, isWrapAround)) is not None}
        lrLoadB = {tB: v for tB in (loadBTiles or [])
                   if (v := self._loadTile('B', tB, loadSubIterK, isWrapAround)) is not None}

        # Scale VGPRs for loaded tiles (only for subIterK==0 loads)
        lrScaleA, lrScaleB = {}, {}
        if self.hasScale and loadSubIterK == 0:
            for tA in (loadATiles or []):
                gid, vid = self.scaleVid('A', tA)
                lrScaleA.setdefault(gid, vid)
            for tB in (loadBTiles or []):
                gid, vid = self.scaleVid('B', tB)
                lrScaleB.setdefault(gid, vid)

        # Conflict detection: MFMA reads and LR writes must not share VGPR tiles
        mfmaIds = set(vgprTileMapA.values()) | set(vgprTileMapB.values())
        loadIds = set(lrLoadA.values()) | set(lrLoadB.values())
        overlap = mfmaIds & loadIds
        if overlap:
            # Fail for now. We could support this by duplicating the loop and use different VGPR tiles.
            raise RuntimeError(
                f"VGPR tile conflict in partition {partition.partitionId} subIterK={sik}: "
                f"MFMA and LR share tile IDs {overlap}")

        # Build modules
        mfmas = [(a, b) for a in sorted(vgprTileMapA) for b in sorted(vgprTileMapB)]
        mtLoad = "n+1" if isWrapAround else "n"
        siks = SubIterKSchedule(subIterK=sik)
        siks.modules.append(AnnotatedModule(op=MFMAOp(
            mtIteration="n", subIterK=sik, subtiles=mfmas,
            vgprTileMapA=vgprTileMapA, vgprTileMapB=vgprTileMapB,
            scaleMapA=scaleMapA, scaleMapB=scaleMapB)))
        siks.modules.append(AnnotatedModule(op=LROp(
            mtIteration=mtLoad, subIterK=loadSubIterK,
            lrLoadA=lrLoadA, lrLoadB=lrLoadB,
            lrScaleA=lrScaleA, lrScaleB=lrScaleB)))

        return siks

    def _isLastGRForMT(self, pi, gr, numPartitions):
        """Check if this partition's GR is the last one that completes a full MT load."""
        if gr.mtIteration == "n+1":
            return not any(
                (self.partitionGRs[fpi].subtileA or self.partitionGRs[fpi].subtileB)
                and self.partitionGRs[fpi].mtIteration == "n+1"
                for fpi in range(pi + 1, numPartitions))
        if gr.mtIteration == "n+2":
            hasN1 = any(
                (self.partitionGRs[p].subtileA or self.partitionGRs[p].subtileB)
                and self.partitionGRs[p].mtIteration == "n+1"
                for p in range(numPartitions))
            if hasN1:
                return False
            return not any(
                (self.partitionGRs[fpi].subtileA or self.partitionGRs[fpi].subtileB)
                and self.partitionGRs[fpi].mtIteration == "n+2"
                for fpi in range(pi + 1, numPartitions))
        return False

    def _isFirstGRForMT(self, pi, gr, numPartitions):
        """Check if this partition's GR is the first one for its MT iteration."""
        if gr.mtIteration in ("n+1", "n+2"):
            return not any(
                (self.partitionGRs[fpi].subtileA or self.partitionGRs[fpi].subtileB)
                and self.partitionGRs[fpi].mtIteration == gr.mtIteration
                for fpi in range(0, pi))
        return False

    def _insertGROps(self, pss, pi, gr, numPartitions):
        """Insert GR ops for a partition, splitting across subIterK=0 and subIterK=1."""
        if not gr.subtileA and not gr.subtileB:
            return

        totalGR_A = sorted(gr.subtileA)
        totalGR_B = sorted(gr.subtileB)
        splitA = (len(totalGR_A) + 1) // 2
        splitB = (len(totalGR_B) + 1) // 2
        gr0_A, gr1_A = totalGR_A[:splitA], totalGR_A[splitA:]
        gr0_B, gr1_B = totalGR_B[:splitB], totalGR_B[splitB:]
        isLast = self._isLastGRForMT(pi, gr, numPartitions)
        isFirst = self._isFirstGRForMT(pi, gr, numPartitions)
        hasSik1 = bool(gr1_A or gr1_B)
        # handle case where gr1 is empty.
        if gr0_A or gr0_B:
            pss.subIterKSteps[0].modules.append(AnnotatedModule(op=GROp(
                mtIteration=gr.mtIteration,
                subtileA=gr0_A, subtileB=gr0_B,
                firstForMT=isFirst,
                lastForMT=isLast and not hasSik1)))
        if hasSik1:
            pss.subIterKSteps[1].modules.append(AnnotatedModule(op=GROp(
                mtIteration=gr.mtIteration,
                subtileA=gr1_A, subtileB=gr1_B, lastForMT=isLast)))

    # Generate the schedule
    # 1- build subIterK steps
    # 2- insert GR ops 
    # 3- annotate dependencies (WAIT and INC Ops)
    # 4- build NGLL & NLL using mainloop schedule
    def _runSchedule(self):
        if self.config.prefetchMode == PrefetchMode.NO:
            raise NotImplementedError("PrefetchMode.NO is not yet supported")

        preloadedMT1_A, preloadedMT1_B = self._buildPreloop()
        self.partitionGRs = self._computePartitionGRs(preloadedMT1_A, preloadedMT1_B)

        numPartitions = len(self.partitions)
        self.mainloopSteps: List[PartitionSchedule] = []

        for pi, partition in enumerate(self.partitions):
            pss = PartitionSchedule(partitionId=partition.partitionId)

            for sik in range(self.numSubIterK):
                pss.subIterKSteps.append(
                    self._buildSubIterK(partition, pi, sik, numPartitions))

            # split GR ops across subIterK steps and determine lastForMT
            self._insertGROps(pss, pi, self.partitionGRs[pi], numPartitions)
            self.mainloopSteps.append(pss)

            if self.config.reuseStrategy == VGPRTileReUseStrategy.ACROSS_PARTITIONS:
                self._releaseUnusedAfterPartition(pi)

        self._annotateDependencies(numPartitions)
        self.ngllSteps = self._buildNGLL()
        self.nllSteps = self._buildNLL()

    def _loadTile(self, tc: str, tileIdx: int, loadSubIterK: int,
                  isWrapAround: bool) -> Optional[int]:
        """Determine the VGPRTile ID for a load. Returns None if no load needed."""
        allocated = self.allocator.isAllocated(tc, tileIdx, loadSubIterK)

        if isWrapAround and allocated:
            # Wrap-around: reuse partition 0's existing VGPRTile IDs
            return self.allocator.getVGPRTileId(tc, tileIdx, loadSubIterK)

        if not allocated:
            return self.allocator.allocate(tc, tileIdx, loadSubIterK)

        # Tile stays alive, reuse in place
        return None

    def _isWrapAroundLoad(self, partitionIdx: int, subIterK: int, numPartitions: int) -> bool:
        """True when this step's load targets partition 0 for the next macrotile iteration."""
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            return partitionIdx == numPartitions - 1 and subIterK == self.numSubIterK - 1
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            return partitionIdx == numPartitions - 1
        return False

    # ── Prefetch modes ───────────────────────────────────────

    def _getLoadTargets(self, partitionIdx: int, subIterK: int,
                        numPartitions: int) -> Tuple[Optional[List[int]], Optional[List[int]], int]:
        """Returns (loadATiles, loadBTiles, targetSubIterK)."""
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            return self._loadTargetsHalfPrefetch(partitionIdx, subIterK, numPartitions)
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            return self._loadTargetsFullPrefetch(partitionIdx, subIterK, numPartitions)
        return (None, None, 0)

    def _loadTargetsHalfPrefetch(self, partitionIdx, subIterK, numPartitions):
        """HALF: subIterK=0 loads same-partition subIterK=1, subIterK=last loads next-partition subIterK=0.
        Last partition wraps around to partition 0 (next iteration)."""
        currentPartition = self.partitions[partitionIdx]
        if subIterK < self.numSubIterK - 1:
            targetSubIterK = subIterK + 1
            return (currentPartition.tileAIndices, currentPartition.tileBIndices, targetSubIterK)
        else:
            nextPartition = self.partitions[(partitionIdx + 1) % numPartitions]
            return (nextPartition.tileAIndices, nextPartition.tileBIndices, 0)

    def _loadTargetsFullPrefetch(self, partitionIdx, subIterK, numPartitions):
        """FULL: subIterK=0 loads next-partition subIterK=0, subIterK=1 loads next-partition subIterK=1.
        Last partition wraps around to partition 0 (next iteration)."""
        nextPartition = self.partitions[(partitionIdx + 1) % numPartitions]
        return (nextPartition.tileAIndices, nextPartition.tileBIndices, subIterK)

    # ── Reuse strategies ─────────────────────────────────────

    def _releaseUnusedAfterPartition(self, partitionIdx: int):
        """ACROSS_PARTITIONS: release tiles not appearing in any future partition.
        Partition 0's tiles are always considered "future" because the wrap-around
        LR at the end of the loop loads back into partition 0's vgprTile IDs."""
        currentPartition = self.partitions[partitionIdx]

        # Include partition 0 in the future set — the wrap-around LR needs
        # those tiles to still be allocated so it reuses the same vgprTile IDs.
        futureA: Set[int] = set(self.partitions[0].tileAIndices)
        futureB: Set[int] = set(self.partitions[0].tileBIndices)
        for pi in range(partitionIdx + 1, len(self.partitions)):
            futureA.update(self.partitions[pi].tileAIndices)
            futureB.update(self.partitions[pi].tileBIndices)

        for tA in currentPartition.tileAIndices:
            if tA not in futureA:
                self.allocator.releaseAllForTile('A', tA)
        for tB in currentPartition.tileBIndices:
            if tB not in futureB:
                self.allocator.releaseAllForTile('B', tB)

    def _buildGREvents(self):
        """Build GR events list from modules for inflight counting."""
        grEvents = []
        opIdx = 0
        for pss in self.mainloopSteps:
            for dus in pss.subIterKSteps:
                for mod in dus.modules:
                    if isinstance(mod.op, GROp):
                        grEvents.append((opIdx, mod.op.mtIteration,
                                         set(mod.op.subtileA), set(mod.op.subtileB),
                                         mod.op.firstForMT))
                    opIdx += 1
        return grEvents

    @staticmethod
    def _parseMTOffset(mt: str) -> Optional[int]:
        if mt == "n":
            return 0
        if mt.startswith("n+"):
            return int(mt[2:])
        return None

    def _countInflightSubtileLoads(self, grEvents, sikStart, sikEnd, waitMT, waitSubtileA, waitSubtileB):
        """Count GR subtile loads still in flight before the current subIterK.
        Excludes any loads within the current subIterK [sikStart, sikEnd).
        Returns (inflightA, inflightB, scaleLoadsA, scaleLoadsB)."""
        waitOffset = self._parseMTOffset(waitMT)
        if waitOffset is None:
            return 0, 0, 0, 0

        targetA, targetB = set(waitSubtileA), set(waitSubtileB)
        before = [(mt, a, b, first) for (idx, mt, a, b, first) in grEvents if idx < sikStart]
        after  = [(mt, a, b, first) for (idx, mt, a, b, first) in grEvents if idx >= sikEnd]
        totalA, totalB, scaleA, scaleB = 0, 0, 0, 0

        for (grMT, grA, grB, firstForMT) in reversed(before):
            grOffset = self._parseMTOffset(grMT)
            if grOffset is None:
                continue
            if grOffset == waitOffset and grA == targetA and grB == targetB:
                return totalA, totalB, scaleA, scaleB
            totalA += len(grA)
            totalB += len(grB)
            if firstForMT and self.hasScale:
                scaleA += self.scaleLoadsPerMT_A
                scaleB += self.scaleLoadsPerMT_B

        for (grMT, grA, grB, firstForMT) in reversed(after):
            grOffset = self._parseMTOffset(grMT)
            if grOffset is None:
                continue
            if grOffset - 1 == waitOffset and grA == targetA and grB == targetB:
                return totalA, totalB, scaleA, scaleB
            totalA += len(grA)
            totalB += len(grB)
            if firstForMT and self.hasScale:
                scaleA += self.scaleLoadsPerMT_A
                scaleB += self.scaleLoadsPerMT_B

        return totalA, totalB, scaleA, scaleB

    def _buildWaitGROp(self, lrOp, pendingA, pendingB, sikStart, sikEnd, grEvents):
        """Determine if a WAIT_GR is needed before this LR. Returns (waitGROp, waitA, waitB)."""
        if not lrOp or lrOp.subIterK != 0:
            return None, set(), set()

        waitA = set(lrOp.lrLoadA.keys()) & pendingA
        waitB = set(lrOp.lrLoadB.keys()) & pendingB
        if not waitA and not waitB:
            return None, set(), set()

        #TODO. fix _countInflightSubtileLoads . Not working well in 1x4,4x1 or multi-parition configs
        inflightA, inflightB, scaleA, scaleB = self._countInflightSubtileLoads(
            grEvents, sikStart, sikEnd, lrOp.mtIteration, sorted(waitA), sorted(waitB))
        waitGROp = WaitGROp(
            mtIteration=lrOp.mtIteration,
            subtileA=sorted(waitA), subtileB=sorted(waitB),
            inflightLoadsA=inflightA, inflightLoadsB=inflightB,
            inflightScaleLoadsA=scaleA, inflightScaleLoadsB=scaleB)
        pendingA -= waitA
        pendingB -= waitB
        return waitGROp, waitA, waitB

    @staticmethod
    def _findMatchingGR(lrOp, priorGRMods, grMods, waitA, waitB):
        """Find the GR module that the LR should depend on (matching MT iteration)."""
        targetMT = lrOp.mtIteration
        for g in reversed(priorGRMods + grMods):
            if g.op.mtIteration != targetMT:
                continue
            gA, gB = set(g.op.subtileA), set(g.op.subtileB)
            if (not waitA or waitA.issubset(gA)) and (not waitB or waitB.issubset(gB)):
                return g
        # Fallback: any GR with matching MT
        return next((g for g in reversed(priorGRMods + grMods)
                     if g.op.mtIteration == targetMT), None)

    @staticmethod
    def _annotateGRInc(grMods):
        """Append GR_INC to the last GR module for this MT iteration."""
        for grMod in grMods:
            if grMod.op.lastForMT:
                grMod.after.append(DepEdge(op=GR_INCOp()))

    @staticmethod
    def _annotateLRInc(lrMod, lrOp, lastLRmt):
        """Prepend LR_INC if the LR's MT iteration changed."""
        if lastLRmt is not None and lrOp.mtIteration != lastLRmt:
            lrMod.before.append(DepEdge(op=LR_INCOp()))

    def _annotateLRDependsOnGR(self, lrMod, lrOp, grMods, priorGRMods, waitGROp, waitA, waitB):
        """LR depends on GR — GR must complete before LR can read from LDS."""
        grMatch = self._findMatchingGR(lrOp, priorGRMods, grMods, waitA, waitB)
        if grMatch is not None:
            lrMod.before.append(DepEdge(module=grMatch))
        lrMod.before.append(DepEdge(op=waitGROp))
        lrMod.before.append(DepEdge(op=SyncOp(comment="Barrier: wait for GR data")))

    @staticmethod
    def _annotateGRDependsOnLR(lrMod, grMods):
        """GR(n+2) depends on LR — LR must complete before GR writes to LDS. Limited due to LDS double-buffering."""
        for grMod in grMods:
            grMod.before.append(DepEdge(module=lrMod))
            grMod.before.append(DepEdge(op=WaitLROp()))
            grMod.before.append(DepEdge(op=SyncOp(comment="Barrier: all waves done with LR before GR(n+2) writes")))

    def _annotateDependencies(self, numPartitions: int):
        """Annotate each AnnotatedModule with before/after dependency edges."""
        grEvents = self._buildGREvents()
        pendingA = set()
        pendingB = set()
        lastLRmt = None
        opIdx = 0
        priorGRMods: List[AnnotatedModule] = []

        for pss in self.mainloopSteps:
            gr = self.partitionGRs[pss.partitionId]
            # PendingA/B are subtiles issues not been waited on yet.
            pendingA |= gr.subtileA
            pendingB |= gr.subtileB

            for dus in pss.subIterKSteps:
                lrMods = [m for m in dus.modules if isinstance(m.op, LROp)]
                grMods = [m for m in dus.modules if isinstance(m.op, GROp)]
                lrMod = lrMods[0] if lrMods else None
                lrOp = lrMod.op if lrMod else None
                hasGRn2 = any(m.op.mtIteration == "n+2" for m in grMods)

                # build WAIT_GR is the LROp needs subtile that are still pending.
                waitGROp, waitA, waitB = self._buildWaitGROp(
                    lrOp, pendingA, pendingB, opIdx, opIdx + len(dus.modules), grEvents)

                self._annotateGRInc(grMods)

                if waitGROp and lrMod:
                    self._annotateLRDependsOnGR(
                        lrMod, lrOp, grMods, priorGRMods, waitGROp, waitA, waitB)
                elif hasGRn2 and lrMod:
                    self._annotateGRDependsOnLR(lrMod, grMods)

                if lrMod:
                    self._annotateLRInc(lrMod, lrOp, lastLRmt)
                    lrMod.after.append(DepEdge(op=WaitLROp()))

                if lrOp:
                    lastLRmt = lrOp.mtIteration
                priorGRMods.extend(grMods)
                opIdx += len(dus.modules)


    @staticmethod
    def _filterDepEdges(edges: List[DepEdge], remove_types: tuple) -> List[DepEdge]:
        """Filter dependency edges, removing ops of specified types."""
        return [e for e in edges if not isinstance(e.op, remove_types)]

    # TODO. Re-test with multi-partitions
    def _buildNGLL(self) -> List[PartitionSchedule]:
        """NGLL (No Global Load Loop): mainloop without GR(n+2) and GR_INC."""
        ngll = []
        for pss in self.mainloopSteps:
            newPss = PartitionSchedule(partitionId=pss.partitionId)
            for dus in pss.subIterKSteps:
                newDus = SubIterKSchedule(subIterK=dus.subIterK)
                for mod in dus.modules:
                    if isinstance(mod.op, GROp) and mod.op.mtIteration == "n+2":
                        continue
                    newBefore = []
                    for e in mod.before:
                        if e.module and isinstance(e.module.op, GROp) and e.module.op.mtIteration == "n+2":
                            continue
                        if e.op and isinstance(e.op, WaitGROp):
                            # TODO. Check counts here. 
                            newBefore.append(DepEdge(op=WaitGROp(
                                mtIteration=e.op.mtIteration,
                                subtileA=e.op.subtileA, subtileB=e.op.subtileB,
                                inflightLoadsA=0, inflightLoadsB=0)))
                        else:
                            newBefore.append(e)
                    newAfter = self._filterDepEdges(mod.after, (GR_INCOp,))
                    newDus.modules.append(AnnotatedModule(
                        op=mod.op, before=newBefore, after=newAfter))
                newPss.subIterKSteps.append(newDus)
            ngll.append(newPss)
        return ngll

    # TODO. Re-test with multi-partitions
    def _buildNLL(self) -> List[PartitionSchedule]:
        """NLL (No Load Loop): mainloop without GR, GR_INC, LR_INC, LR(n+1),
        WaitGR(n+1) and their associated SyncOps. Keeps WaitGR(n) and its SYNC."""
        nll = []
        for pss in self.mainloopSteps:
            newPss = PartitionSchedule(partitionId=pss.partitionId)
            for dus in pss.subIterKSteps:
                newDus = SubIterKSchedule(subIterK=dus.subIterK)
                # Track which modules are being removed (for filtering module refs)
                removedMods = set()
                for mod in dus.modules:
                    if isinstance(mod.op, GROp):
                        removedMods.add(id(mod))
                    elif isinstance(mod.op, LROp) and mod.op.mtIteration == "n+1":
                        removedMods.add(id(mod))

                for mod in dus.modules:
                    if id(mod) in removedMods:
                        continue
                    # Filter deps: remove module refs to removed modules, GR_INC, LR_INC,
                    # WaitGR(n+1) and its paired SYNC
                    newBefore = []
                    for e in mod.before:
                        if e.module and id(e.module) in removedMods:
                            continue
                        if e.op and isinstance(e.op, (GR_INCOp, LR_INCOp)):
                            continue
                        if e.op and isinstance(e.op, WaitGROp) and e.op.mtIteration == "n+1":
                            continue
                        if e.op and isinstance(e.op, SyncOp):
                            # Skip SYNC if paired with a removed WaitGR(n+1)
                            idx = mod.before.index(e)
                            prevEdge = mod.before[idx - 1] if idx > 0 else None
                            if prevEdge and prevEdge.op and isinstance(prevEdge.op, WaitGROp) and prevEdge.op.mtIteration == "n+1":
                                continue
                        if e.op and isinstance(e.op, WaitGROp):
                            newBefore.append(DepEdge(op=WaitGROp(
                                mtIteration=e.op.mtIteration,
                                subtileA=e.op.subtileA, subtileB=e.op.subtileB,
                                inflightLoadsA=0, inflightLoadsB=0)))
                        else:
                            newBefore.append(e)
                    newAfter = self._filterDepEdges(mod.after, (GR_INCOp,))
                    newDus.modules.append(AnnotatedModule(
                        op=mod.op, before=newBefore, after=newAfter))
                # Remove WaitLROp from after when no LR exists in this subIterK
                # (the WaitLROp was for the removed LR(n+1))
                hasLR = any(isinstance(m.op, LROp) for m in newDus.modules)
                if not hasLR:
                    for m in newDus.modules:
                        m.after = self._filterDepEdges(m.after, (WaitLROp,))
                newPss.subIterKSteps.append(newDus)
            nll.append(newPss)
        return nll


    # ── Debug ────────────────────────────────────────────────

    @staticmethod
    def _printOp(op: ScheduleOp, indent: str = "",
                 showVgpr: bool = False, showSubtiles: bool = False):
        if isinstance(op, MFMAOp):
            print(f"{indent}MFMAs (MT {op.mtIteration}, subIterK {op.subIterK}):")
            if showSubtiles:
                print(f"{indent}  - {op.subtiles}")
            if showVgpr:
                print(f"{indent}  - USING  A: {op.vgprTileMapA}  B: {op.vgprTileMapB}")
                if op.scaleMapA or op.scaleMapB:
                    print(f"{indent}  - SCALE  A: {op.scaleMapA}  B: {op.scaleMapB}")
        elif isinstance(op, GROp):
            print(f"{indent}GR (MT {op.mtIteration}):  A: {op.subtileA}  B: {op.subtileB}")
        elif isinstance(op, WaitGROp):
            if op.inflightLoadsA is not None:
                inflight = f" — inflight SubtileLoads A={op.inflightLoadsA} B={op.inflightLoadsB} scaleA={op.inflightScaleLoadsA} scaleB={op.inflightScaleLoadsB}"
            else:
                inflight = ""
            print(f"{indent}WAIT_GR (MT {op.mtIteration}) A: {op.subtileA}  B: {op.subtileB}{inflight}")
        elif isinstance(op, WaitLROp):
            print(f"{indent}WAIT_LR")
        elif isinstance(op, SyncOp):
            print(f"{indent}SYNC")
        elif isinstance(op, LROp):
            sikLabel = f", subIterK {op.subIterK}" if op.subIterK >= 0 else ""
            aKeys = sorted(op.lrLoadA.keys())
            bKeys = sorted(op.lrLoadB.keys())
            print(f"{indent}LR (MT {op.mtIteration}{sikLabel}) A: {aKeys}  B: {bKeys}")
            if showVgpr:
                print(f"{indent}  - LOAD  A: {op.lrLoadA}  B: {op.lrLoadB}")
                if op.lrScaleA or op.lrScaleB:
                    print(f"{indent}  - SCALE  A: {op.lrScaleA}  B: {op.lrScaleB}")
        elif isinstance(op, SkipOp):
            print(f"{indent}SKIP_IF_{op.compare}({op.value}, {op.target})")
        elif isinstance(op, GR_INCOp):
            print(f"{indent}GR_INC")
        elif isinstance(op, LR_INCOp):
            print(f"{indent}LR_INC")

    @staticmethod
    def _depEdgeLabel(e: DepEdge) -> str:
        if e.module:
            op = e.module.op
            if isinstance(op, LROp):
                return f"LR(MT {op.mtIteration}, sik {op.subIterK})"
            elif isinstance(op, GROp):
                return f"GR(MT {op.mtIteration})"
            return type(op).__name__
        op = e.op
        if isinstance(op, WaitGROp) and op.inflightLoadsA is not None:
            return f"WaitGROp(A={op.inflightLoadsA} B={op.inflightLoadsB} SA={op.inflightScaleLoadsA} SB={op.inflightScaleLoadsB})"
        return type(op).__name__

    def _printModules(self, modules: List[AnnotatedModule], indent: str,
                      showVgpr: bool = False, showDeps: bool = False,
                      showSubtiles: bool = False):
        for mod in modules:
            self._printOp(mod.op, indent=indent, showVgpr=showVgpr,
                          showSubtiles=showSubtiles)
            if showDeps:
                before_str = ", ".join(self._depEdgeLabel(e) for e in mod.before) if mod.before else "none"
                after_str = ", ".join(self._depEdgeLabel(e) for e in mod.after) if mod.after else "none"
                print(f"{indent}  before: [{before_str}]  after: [{after_str}]")

    def _printLoopSteps(self, loopSteps: List[PartitionSchedule], indent: str,
                        showVgpr: bool = False, showDeps: bool = False,
                        showSubtiles: bool = False):
        for partition in loopSteps:
            print(f"{indent}Partition {partition.partitionId}:")
            for dus in partition.subIterKSteps:
                print(f"{indent}  subIterK={dus.subIterK}:")
                self._printModules(dus.modules, indent=f"{indent}    ",
                                   showVgpr=showVgpr, showDeps=showDeps,
                                   showSubtiles=showSubtiles)

    def printSchedule(self, showVgpr: bool = False, showDeps: bool = False,
                      showSubtiles: bool = False):
        """Print the schedule.

        Args:
            showVgpr: show VGPR tile assignments and scale maps.
            showDeps: show before/after dependency edges on each module.
            showSubtiles: show MFMA subtile coordinate lists.
        """
        print(f"SubtileGridA={self.MTA}, SubtileGridB={self.MTB}")
        print(f"Partition grid: {self.numPartitionsA} x {self.numPartitionsB}")
        print(f"Partition size: {self.config.partitionSizeA} x {self.config.partitionSizeB}")
        print(f"Prefetch: {self.config.prefetchMode.name}")
        print(f"Reuse: {self.config.reuseStrategy.name}")
        print(f"totalVGPRTiles: {self.totalVGPRTiles} ({self.totalVGPRTiles * 4} VGPRs)")
        print(f"totalScaleVGPRTiles: {self.totalScaleVGPRTiles}")
        print(f"hasScale: {self.hasScale}")
        print()

        grid = [[None] * self.numPartitionsB for _ in range(self.numPartitionsA)]
        sA = self.config.partitionSizeA
        sB = self.config.partitionSizeB
        for partition in self.partitions:
            pA = partition.subtiles[0][0] // sA
            pB = partition.subtiles[0][1] // sB
            grid[pA][pB] = partition.partitionId
        print(f"Ordering grid ({self.config.ordering.name}):")
        for row in grid:
            print("  " + "  ".join(f"{v:2d}" if v is not None else "  " for v in row))
        print()

        opts = dict(showVgpr=showVgpr, showDeps=showDeps, showSubtiles=showSubtiles)

        print("PRELOOP:")
        for pss in self.preloopSteps:
            for dus in pss.subIterKSteps:
                self._printModules(dus.modules, indent="  ",
                                   showVgpr=showVgpr, showSubtiles=showSubtiles)
        print()

        print("MAINLOOP:")
        self._printLoopSteps(self.mainloopSteps, indent="  ", **opts)

        print()
        print("NGLL (No Global Load Loop):")
        self._printLoopSteps(self.ngllSteps, indent="  ", **opts)

        print()
        print("NLL (No Load Loop):")
        self._printLoopSteps(self.nllSteps, indent="  ", **opts)

    # Allocate totalVGPRTiles vpgrTile
    def allocVgprTiles(self, writer):
        """Allocate a shared VGPR tile array for A and B, indexed by the scheduler's vgprTileId.
        Also allocates scale VGPRs (1 VGPR each) when MX block scaling is active."""
        self.vgprTiles = []
        mmaTileRegCount = int(math.ceil(self.tileInfoA.mmaTileRegCount))
        for _ in range(self.totalVGPRTiles):
            tile = TileInfo.RegisterTileInfo(writer.vgprPool)
            for j in range(0, mmaTileRegCount, 4):
                vstart = writer.vgprPool.checkOutAligned(4, 4)
                for k in range(4):
                    tile.append(vstart + k)
            self.vgprTiles.append(tile)

        # Allocate scale VGPRs: 1 VGPR per scale tile (each covers 2 M-adjacent subtiles)
        # Double-buffer: two sets (ping/pong) so MFMA can read one set while ds_read writes the other
        self.scaleVgprTiles = []     # set 0
        self.scaleVgprTilesAlt = []  # set 1
        for _ in range(self.totalScaleVGPRTiles):
            self.scaleVgprTiles.append(writer.vgprPool.checkOut(1))
            self.scaleVgprTilesAlt.append(writer.vgprPool.checkOut(1))


    def deallocVgprTiles(self, writer):
        """Deallocate VGPR tiles allocated by allocVgprTiles."""
        for tile in self.vgprTiles:
            pool = tile.regList.regPool
            for val in tile:
                if tile.index(val) % 4 == 0:
                    pool.checkIn(val)
        self.vgprTiles = []

        for v in self.scaleVgprTiles:
            writer.vgprPool.checkIn(v)
        self.scaleVgprTiles = []
        for v in self.scaleVgprTilesAlt:
            writer.vgprPool.checkIn(v)
        self.scaleVgprTilesAlt = []


    def emitMFMA(self, writer, kernel, op, dtileInfo, scaleSet=0):
        """Emit MFMA instructions for a single MFMAOp."""
        module = Module()
        scaleTiles = self.scaleVgprTiles if scaleSet == 0 else self.scaleVgprTilesAlt

        for (a, b) in op.subtiles:
            aTile = self.vgprTiles[op.vgprTileMapA[a]]
            bTile = self.vgprTiles[op.vgprTileMapB[b]]
            dTile = dtileInfo.vgprTiles[a + b * dtileInfo.localMMATileGrid[0]]

            if self.hasScale:
                scaleGroupA = a // 2
                scaleGroupB = b // 2
                scaleAVgpr = scaleTiles[op.scaleMapA[scaleGroupA]]
                scaleBVgpr = scaleTiles[op.scaleMapB[scaleGroupB]]
                sAsel = (a % 2) + 2 * (op.subIterK % 2)
                sBsel = (b % 2) + 2 * (op.subIterK % 2)
            else:
                scaleAVgpr = scaleBVgpr = -1
                sAsel = sBsel = 0

            module.add(emitMfmaInstruction(
                writer, kernel, aTile, bTile, dTile, dTile,
                scaleAVgpr=scaleAVgpr, scaleBVgpr=scaleBVgpr,
                scaleAsel=sAsel, scaleBsel=sBsel,
                comment=f"MFMA C[{a},{b}] += A[{a},subIterK{op.subIterK}] * B[{b},subIterK{op.subIterK}]"))

        return module

    def emitLR(self, writer, kernel, op, scaleSet=0):
        """Emit LR (Local Read) ds_load instructions for a single LROp.
        Scale LRs (DSLoadB32) are also emitted here, using scheduler-managed VGPRs.
        scaleSet selects which scale VGPR set the ds_reads write to."""
        module = Module()
        for tA, vgprTileId in op.lrLoadA.items():
            dstTile = self.vgprTiles[vgprTileId]
            # Using 0 for subtile ID1 for now
            module.add(emitSingleDsRead(
                self.tileInfoA, tA, 0, op.subIterK, dstTile))
        for tB, vgprTileId in op.lrLoadB.items():
            dstTile = self.vgprTiles[vgprTileId]
            # Using 0 for subtile ID1 for now
            module.add(emitSingleDsRead(
                self.tileInfoB, tB, 0, op.subIterK, dstTile))
        if op.lrScaleA:
            self._emitScaleDsReads(module, writer, 'MXSA', op.lrScaleA, scaleSet=scaleSet)
        if op.lrScaleB:
            self._emitScaleDsReads(module, writer, 'MXSB', op.lrScaleB, scaleSet=scaleSet)
        return module

    def _emitScaleDsReads(self, module, writer, tc, lrScale, scaleSet=0):
        """Emit DSLoadB32 for scale groups using scheduler-managed VGPRs."""
        tileInfo = self.scaleTileInfoA if tc == 'MXSA' else self.scaleTileInfoB
        scaleTiles = self.scaleVgprTiles if scaleSet == 0 else self.scaleVgprTilesAlt
        # Each scale group covers 2 M-adjacent [1,2] subtiles = 4 bytes per lane.
        # dsOffset stride per group = 2 * tileInfo.subtileSize (since [1,2] subtileSize
        # covers 1 subtile, and a group is 2 subtiles).
        groupStride = 2 * tileInfo.subtileSize
        for scaleGroupIdx, scaleVgprTileId in lrScale.items():
            dsOffset = groupStride * scaleGroupIdx
            vdst = scaleTiles[scaleVgprTileId]
            module.add(DSLoadB32(dst=vgpr(vdst),
                                 src=vgpr(tileInfo.sharedVgprLROffset[0]),
                                 ds=DSModifiers(offset=dsOffset),
                                 comment="scale%s[group%u]: load 4B from LDS" % (tc, scaleGroupIdx)))

    def emitWaitGR(self, inflightLoadsA, inflightLoadsB,
                   inflightScaleLoadsA=0, inflightScaleLoadsB=0):
        """Emit SWaitCnt for GR (buffer_load) based on inflight GR counts.

        Args:
            inflightLoadsA: Number of A subtile loads still inflight.
            inflightLoadsB: Number of B subtile loads still inflight.
            inflightScaleLoadsA: Number of scale A loads still inflight.
            inflightScaleLoadsB: Number of scale B loads still inflight.
        """
        module = Module()
        grCnt = int(inflightLoadsA / self.tileInfoA.loadRatioGR) + \
                int(inflightLoadsB / self.tileInfoB.loadRatioGR) + \
                inflightScaleLoadsA + inflightScaleLoadsB
        module.add(SWaitCnt(vlcnt=grCnt, vscnt=-1,
                            comment=f"Wait GR: A={inflightLoadsA} B={inflightLoadsB} sA={inflightScaleLoadsA} sB={inflightScaleLoadsB} => vlcnt={grCnt}"))
        return module

    def emitGR(self, writer, kernel, op):
        """Emit GR (Global Read) buffer_load instructions for a single GROp."""
        module = Module()
        # Scale DTL loads: emitted on the first GR of an MT to maximize overlap
        if op.firstForMT and self.hasScale:
            module.add(globalReadDoScaleSubtile('MXSA', writer, kernel))
            module.add(globalReadDoScaleSubtile('MXSB', writer, kernel))
        # A and B data loads.  When loadRatioGR > 1, multiple subtiles share
        # the same globalReadMap (one buffer_load covers both).  Skip duplicates
        # using a grTracker set, matching the PGR=0 path in globalReadDoSubtile.
        for subtileList, tileInfo in [(op.subtileA, self.tileInfoA),
                                      (op.subtileB, self.tileInfoB)]:
            grTracker = set()
            for sId0 in subtileList:
                subtileInfo = tileInfo.localSubtiles[tileInfo.getLocalSubtileLinearId(sId0, 0)]
                grIds = subtileInfo.globalReadMap
                if not set(grIds).issubset(grTracker):
                    grTracker.update(grIds)
                    module.add(emitSingleBufferLoad(tileInfo, writer, kernel, sId0, 0))
        return module

    def _emitOp(self, writer, kernel, op, dtileInfo, scaleSet=0, scaleLRSet=0):
        """Emit a single ScheduleOp into a list of instructions."""
        module = Module()
        if isinstance(op, GROp):
            module.add(self.emitGR(writer, kernel, op))
        elif isinstance(op, GR_INCOp):
            module.add(globalReadPtrUpdates('A', writer, kernel))
            module.add(globalReadPtrUpdates('B', writer, kernel))
            module.add(globalReadLDSBufferSwap('A', writer, kernel))
            module.add(globalReadLDSBufferSwap('B', writer, kernel))
            if self.hasScale:
                module.add(globalReadLDSBufferSwap('MXSA', writer, kernel))
                module.add(globalReadLDSBufferSwap('MXSB', writer, kernel))
                module.add(globalReadScalePtrUpdates('MXSA', writer, kernel))
                module.add(globalReadScalePtrUpdates('MXSB', writer, kernel))
        elif isinstance(op, MFMAOp):
            module.add(self.emitMFMA(writer, kernel, op, dtileInfo, scaleSet=scaleSet))
        elif isinstance(op, WaitGROp):
            module.add(self.emitWaitGR(op.inflightLoadsA, op.inflightLoadsB,
                                         op.inflightScaleLoadsA, op.inflightScaleLoadsB))
        elif isinstance(op, WaitLROp):
            module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
        elif isinstance(op, SyncOp):
            module.add(SBarrier(comment=op.comment))
        elif isinstance(op, LR_INCOp):
            module.add(localReadLDSBufferSwap('A', writer, kernel))
            module.add(localReadLDSBufferSwap('B', writer, kernel))
            if self.hasScale:
                module.add(localReadLDSBufferSwap('MXSA', writer, kernel))
                module.add(localReadLDSBufferSwap('MXSB', writer, kernel))
        elif isinstance(op, LROp):
            module.add(self.emitLR(writer, kernel, op, scaleSet=scaleLRSet))
        elif isinstance(op, SkipOp):
            skipLabel = Label(f"SkipTo{op.target}", "")
            cmpMap = {"EQ": SCmpEQU32, "LE": SCmpLeU32}
            module.add(cmpMap[op.compare](
                src0=sgpr("LoopCounterL"), src1=op.value,
                comment=f"LoopCounter {op.compare} {op.value}?"))
            module.add(SCBranchSCC1(
                labelName=skipLabel.getLabelName(),
                comment=f"skip to {op.target}"))
        return module.flatitems()

    @staticmethod
    def _opType(op):
        if isinstance(op, MFMAOp):
            return "mfma"
        if isinstance(op, LROp):
            return "lr"
        if isinstance(op, GROp):
            return "gr"
        if isinstance(op, WaitGROp):
            return "wait_gr"
        if isinstance(op, WaitLROp):
            return "wait_lr"
        if isinstance(op, SyncOp):
            return "sync"
        if isinstance(op, GR_INCOp):
            return "gr_inc"
        if isinstance(op, LR_INCOp):
            return "lr_inc"
        if isinstance(op, SkipOp):
            return "skip"
        return "other"

    def _buildEmittedModules(self, writer, kernel, modules, dtileInfo, scaleSet=0, scaleLRSet=0):
        """Build EmittedModules with instructions + before module links."""
        emitted: List[EmittedModule] = []
        modToEmittedId: Dict[int, int] = {}
        suppressAfterWaitLRForMod: Set[int] = set()

        def addEmitted(op) -> Optional[int]:
            insts = self._emitOp(writer, kernel, op, dtileInfo,
                                scaleSet=scaleSet, scaleLRSet=scaleLRSet)
            if not insts:
                return None
            emId = len(emitted)
            emitted.append(EmittedModule(moduleId=emId, instructions=insts, opType=self._opType(op)))
            return emId

        def setBefore(moduleId: int, beforeId: Optional[int]) -> None:
            if beforeId is None or beforeId == moduleId:
                return
            curBefore = emitted[moduleId].before
            if curBefore is None:
                emitted[moduleId].before = beforeId
                return
            assert curBefore == beforeId, \
                f"EmittedModule {moduleId} has multiple before deps: {curBefore} and {beforeId}"

        # Primary modules first (MFMA/LR/GR)
        for mod in modules:
            emId = addEmitted(mod.op)
            if emId is not None:
                modToEmittedId[id(mod)] = emId

        # If another module has before=[module-ref-to-X, WaitLROp, ...],
        # suppress standalone X.after WaitLROp emission to avoid duplicates.
        for mod in modules:
            hasWaitLRInBefore = any(isinstance(e.op, WaitLROp) for e in mod.before)
            if not hasWaitLRInBefore:
                continue
            for e in mod.before:
                if e.module is not None:
                    suppressAfterWaitLRForMod.add(id(e.module))

        # Dependency-op links for emitted debug/scheduling.
        for mod in modules:
            curId = modToEmittedId.get(id(mod))
            if curId is None:
                continue

            # before ops: chain from module refs / deps, then before points to
            # the last non-standalone dep.
            prevId: Optional[int] = None
            lastDepId: Optional[int] = None
            for edge in mod.before:
                if edge.module:
                    prevId = modToEmittedId.get(id(edge.module), prevId)
                    continue
                if edge.op is None:
                    continue
                depId = addEmitted(edge.op)
                if depId is None:
                    continue
                if isinstance(edge.op, WaitGROp):
                    # Keep WAIT_GR standalone (no links), but allow later deps to
                    # chain from it.
                    prevId = depId
                    continue
                setBefore(depId, prevId)
                prevId = depId
                lastDepId = depId
            if lastDepId is not None:
                setBefore(curId, lastDepId)
            elif prevId is not None:
                # before had only module refs and/or standalone deps
                setBefore(curId, prevId)

            # after ops: append deps as standalone modules and chain them via
            # before links so before-only path extraction can follow them.
            depIds: List[int] = []
            for edge in mod.after:
                if edge.module:
                    mId = modToEmittedId.get(id(edge.module))
                    if mId is not None:
                        depIds.append(mId)
                    continue
                if edge.op is None:
                    continue
                if isinstance(edge.op, WaitLROp) and id(mod) in suppressAfterWaitLRForMod:
                    continue
                depId = addEmitted(edge.op)
                if depId is None:
                    continue
                depIds.append(depId)
            prevAfterId = curId
            for depId in depIds:
                setBefore(depId, prevAfterId)
                prevAfterId = depId

        return emitted

    def _emitSubIterK(self, writer, kernel, pss, dus, scaleSet=0, scaleLRSet=0):
        """Emit a single subIterK step into a Module.
        scaleSet: which scale VGPR set MFMA reads from.
        scaleLRSet: which scale VGPR set LR writes to.

        If modules contain MFMAs, emits via instructionSchedule for
        dependency-aware interleaving. Otherwise emits sequentially.
        """
        dtileInfo = writer.states.d.tileInfo
        module = Module()
        module.addComment0(f"Partition {pss.partitionId}: subIterK={dus.subIterK}")

        hasMFMA = any(isinstance(m.op, MFMAOp) for m in dus.modules)
        if hasMFMA:
            emitted = self._buildEmittedModules(writer, kernel, dus.modules, dtileInfo,
                                                scaleSet=scaleSet, scaleLRSet=scaleLRSet)
            merged = self.instructionSchedule(emitted)
            module.add(merged)
        else:
            # Special case for preloop (MFMA free)
            for m in dus.modules:
                for inst in self._emitOp(writer, kernel, m.op, dtileInfo,
                                         scaleSet=scaleSet, scaleLRSet=scaleLRSet):
                    module.add(inst)
        return module

    @staticmethod
    def _extractPathsFromBeforeDeps(emittedModules: List['EmittedModule']) -> Tuple[int, List[List[int]]]:
        """Extract non-MFMA dependency paths using only EmittedModule.before links.

        Returns:
          (mfmaIdx, paths)
          - mfmaIdx: index of the MFMA emitted module in emittedModules
          - paths: list of non-MFMA module-index paths
        """
        idToIdx = {em.moduleId: i for i, em in enumerate(emittedModules)}
        n = len(emittedModules)

        mfmaModuleIds = [i for i, em in enumerate(emittedModules) if em.opType == "mfma"]
        assert len(mfmaModuleIds) == 1, "_extractPathsFromBeforeDeps expects exactly one MFMA emitted module"
        mfmaIdx = mfmaModuleIds[0]
        nonMfmaIds = [i for i in range(n) if i != mfmaIdx]
        nonMfmaSet = set(nonMfmaIds)

        # Each non-MFMA module has at most one predecessor, and each predecessor
        # has at most one child, so paths are simple chains.
        pred: List[int] = [-1 for _ in range(n)]
        child: List[int] = [-1 for _ in range(n)]
        for i in nonMfmaIds:
            parent = -1
            b = emittedModules[i].before
            if b is not None:
                bi = idToIdx.get(b)
                if bi is not None and bi != i and bi in nonMfmaSet:
                    parent = bi
            pred[i] = parent
            if parent != -1:
                assert child[parent] == -1, \
                    f"_extractPathsFromBeforeDeps expects unique child per predecessor, got {child[parent]} and {i} for {parent}"
                child[parent] = i

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
            assert order, f"_extractPathsFromBeforeDeps produced empty path for module {mid}"
            for i in order:
                used[i] = True
            paths.append(order)

        return mfmaIdx, paths

    @staticmethod
    def instructionSchedule(emittedModules: List['EmittedModule']):
        """Interleave non-MFMA instructions between MFMAs using 2 slots/interval.

        Rules:
          - MFMA order is preserved.
          - Between two adjacent MFMAs there are 2 placement slots.
          - At most one ds_read (LocalReadInstruction) per interval.
          - Before dependencies are respected at module order level.
          - Minimm distance between ds_read and it waitcnt (hardcoded for now)
          - Module-internal instruction order is preserved.
          - LR path containing a WAIT_GR is packed from the end backwards. We want WAIT_GR to be done as late as possible.
          - GR path is spread as much as possible across remaining valid slots. No backwards here as we want GRs to be done as early as possible.

          TODO : To be tested on multi-partition setup.
        """
        if not emittedModules:
            return Module()

        isMFMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))
        n = len(emittedModules)

        mfmaIdx, pathOrders = SubtileBasedScheduler._extractPathsFromBeforeDeps(emittedModules)
        mfmas = [x for x in emittedModules[mfmaIdx].instructions if isMFMA(x)]

        # Single MFMA: no slots to interleave into — emit MFMA then all paths.
        if len(mfmas) < 2:
            result = Module()
            for m in mfmas:
                result.add(m)
            for order in pathOrders:
                for mid in order:
                    for inst in emittedModules[mid].instructions:
                        result.add(inst)
            return result

        paths = _classifyPaths(pathOrders, emittedModules)
        rules = _SchedulingRules(totalSlots=(len(mfmas) - 1) * 2)
        placer = _SlotPlacer(
            len(mfmas) - 1, n, pathOrders,
            validators=[rules.oneDsReadPerInterval, rules.minGapDsReadBeforeWait, rules.minGapDsReadToWait, rules.noM0WithBufferLoad],
            adjusters=[rules.spreadBufferLoads],
            onPlace=rules.trackPlacement)

        for order, hasWaitGR in paths:
            if not order:
                continue
            pathInsts = _flattenPath(order, emittedModules, reverse=hasWaitGR)
            rules.resetPath()
            if not hasWaitGR:
                rules.setupBufLoadSpreading(placer, pathInsts, order)
            placer.placePath(pathInsts, reverse=hasWaitGR)

        scheduled = placer.assemble(mfmas)

        # Post-pass: adjust vmcnt of any SWaitCnt to account for buffer_loads
        # that the scheduler placed before it within this subIterK.
        bufLoadCount = 0
        for inst in scheduled.items():
            if _isBufferLoad(inst):
                bufLoadCount += 1
            elif _isWaitCnt(inst) and inst.vlcnt >= 0:
                inst.vlcnt += bufLoadCount

        return scheduled

    def _emitLoop(self, writer, kernel, label, steps, scaleSet=0, scaleLRSet=None):
        """Emit a loop section (preloop, mainloop, NGLL, or NLL).

        scaleSet: which scale VGPR set MFMA reads from (starting set for first partition).
        scaleLRSet: which scale VGPR set LR writes to (defaults to 1-scaleSet if None).
            Both rotate per partition so each partition's MFMA reads the scales
            that the previous partition's LR loaded.
        """
        if scaleLRSet is None:
            scaleLRSet = 1 - scaleSet if self.hasScale else scaleSet
        module = Module(label)
        module.addComment0(f"{label} start")
        for pss in steps:
            for dus in pss.subIterKSteps:
                subModule = self._emitSubIterK(writer, kernel, pss, dus,
                                               scaleSet=scaleSet, scaleLRSet=scaleLRSet)
                module.add(subModule)
            if self.hasScale:
                scaleSet, scaleLRSet = scaleLRSet, scaleSet
        module.addComment0(f"{label} end")
        return module

