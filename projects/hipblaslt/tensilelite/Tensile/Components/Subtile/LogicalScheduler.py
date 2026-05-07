# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MFMATile-based logical scheduler.

Builds a logical schedule using MFMA tile indices as the core primitive,
with explicit per-operation load granularity for GR/LR on A, B, SA, SB.

The schedule is built in these passes:
  place_LRs                — place LRs based on their granularities
  assign_vgpr_tiles        — assign physical vgprTileIds with per-tensor free-lists
  place_GRs                — place GRs
  annotate_deps            — annotate raw per-op dependencies
  remove_unnecessary_gr_deps — remove redundant LR→GR deps
  remove_unnecessary_lr_deps — remove redundant GR→LR deps covered by MFMA syncs
  remove_cross_deps        — replace cross-subIterK deps with wait preOps
  insert_gr_lr_inc         — insert lr_inc/gr_inc preOps at MT transitions
  group                    — serialize and group (produce paths for instructionSchedule)
  remove_wait_lr_sync      — remove redundant wait_lr_sync after grouping
  emit                     — produce List[EmittedModule] with before-link chains

  TODO: add a pass to remove redundant wait_gr_sync on multi-partition configs
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union
import copy
import io
import math


class Pass(IntEnum):
    """Scheduler passes in dependency order.

    The numeric value defines topological order. The main pipeline is linear
    (each pass depends on the previous), except VGPR_TILES which forks off
    LR independently of GR.
    """
    LR                  = 0
    VGPR_TILES          = 1
    GR                  = 2
    DEPS                = 3
    REMOVE_GR_DEPS      = 4
    REMOVE_LR_DEPS      = 5
    REMOVE_DEPS         = 6
    GR_INC              = 7
    GROUP_LR_GR         = 8
    REMOVE_WAIT_LR_SYNC = 9
    EMIT                = 10
    BUILD               = 11
    POPULATE            = 12


_PASS_PIPELINE = {
    Pass.LR:                   ('place_LRs',                        []),
    Pass.VGPR_TILES:           ('assign_vgpr_tiles',                [Pass.LR]),
    Pass.GR:                   ('place_GRs',                        [Pass.LR]),
    Pass.DEPS:                 ('annotate_deps',                    [Pass.GR]),
    Pass.REMOVE_GR_DEPS:       ('remove_unnecessary_gr_deps',       [Pass.DEPS]),
    Pass.REMOVE_LR_DEPS:       ('remove_unnecessary_lr_deps',       [Pass.REMOVE_GR_DEPS]),
    Pass.REMOVE_DEPS:          ('remove_cross_deps',                [Pass.REMOVE_LR_DEPS]),
    Pass.GR_INC:               ('insert_gr_lr_inc',                 [Pass.REMOVE_DEPS]),
    Pass.GROUP_LR_GR:          ('group_lr_gr',                      [Pass.GR_INC]),
    Pass.REMOVE_WAIT_LR_SYNC:  ('remove_unnecessary_wait_lr_sync', [Pass.GROUP_LR_GR]),
    Pass.EMIT:                 ('emit',                             [Pass.REMOVE_WAIT_LR_SYNC]),
    Pass.BUILD:                ('build',                            [Pass.EMIT]),
    Pass.POPULATE:             ('populate_instructions',            []),
}


TENSOR_SIDE = {'A': 'A', 'B': 'B', 'SA': 'A', 'SB': 'B'}

def fmt_mt(mt: int) -> str:
    """Format MT iteration integer as display string: 0 → 'n', 1 → 'n+1', 2 → 'n+2'."""
    return "n" if mt == 0 else f"n+{mt}"

# ── Core primitives ─────────────────────────────────────────

@dataclass
class MFMATileRange:
    """A rectangular range of MFMA tile coordinates for one read."""
    subIterK_start: int
    subIterK_end: int          # exclusive
    tileId_start: int
    tileId_end: int            # exclusive

    @property
    def subIterK_list(self) -> List[int]:
        return list(range(self.subIterK_start, self.subIterK_end))

    @property
    def tileId_list(self) -> List[int]:
        return list(range(self.tileId_start, self.tileId_end))

    def fmt_k(self) -> str:
        ids = self.subIterK_list
        if len(ids) == 1:
            return f"[{ids[0]}]"
        return f"[{ids[0]},{ids[-1]}]"

    def fmt_tiles(self) -> str:
        return f"[{self.tileId_start}-{self.tileId_end - 1}]"


# ── Config ──────────────────────────────────────────────────

@dataclass
class ReadGranularity:
    """Load granularity for one operation on one tensor, measured in MFMA tiles.

    mn: how many MFMA tiles in the M (for A/SA) or N (for B/SB) dimension
    k:  how many subIterK steps one read covers
    """
    mn: int
    k: int


@dataclass
class SchedulerConfig:
    """Configuration for the MFMATile-based scheduler."""
    numMFMATilesM: int    # MFMA tiles in M dimension (for A)
    numMFMATilesN: int    # MFMA tiles in N dimension (for B)
    numSubIterK: int      # subIterK steps within the macrotile
    lrA: ReadGranularity
    lrB: ReadGranularity
    grA: ReadGranularity
    grB: ReadGranularity
    lrSA: Optional[ReadGranularity] = None
    lrSB: Optional[ReadGranularity] = None
    grSA: Optional[ReadGranularity] = None
    grSB: Optional[ReadGranularity] = None
    numPartitionsM: int = 1   # partition grid in M dimension
    numPartitionsN: int = 1   # partition grid in N dimension

    @property
    def hasScale(self) -> bool:
        return self.lrSA is not None and self.lrSB is not None

    @property
    def numPartitions(self) -> int:
        return self.numPartitionsM * self.numPartitionsN

    @property
    def partitionSizeM(self) -> int:
        assert self.numMFMATilesM % self.numPartitionsM == 0
        return self.numMFMATilesM // self.numPartitionsM

    @property
    def partitionSizeN(self) -> int:
        assert self.numMFMATilesN % self.numPartitionsN == 0
        return self.numMFMATilesN // self.numPartitionsN

    @staticmethod
    def get_partition_candidates(tileInfoA, tileInfoB) -> list:
        """Return partition candidates as [(numPartitionsM, numPartitionsN), ...].

        Enumerates all divisors of MAX(M, N) in ascending order and
        partitions the larger dimension. Starts with (1, 1).
        This will only produces 1xN or Nx1 partitions to allow VGPR pressure reduction.
        """
        M = tileInfoA.localMMATileGrid[0]
        N = tileInfoB.localMMATileGrid[0]
        maxDim = max(M, N)

        divisors = sorted(d for d in range(1, maxDim + 1) if maxDim % d == 0)

        candidates = []
        for d in divisors:
            if N >= M:
                candidates.append((1, d))
            else:
                candidates.append((d, 1))

        return candidates



# ── Schedule operation types ────────────────────────────────

@dataclass
class Emittable:
    """Base for anything placed in an EmittedModule."""
    kind: str = field(init=False, default="")


@dataclass
class MFMAPlacement(Emittable):
    """MFMA operation consuming data for one subIterK."""
    subIterK: int
    tileA: MFMATileRange       # A tiles consumed
    tileB: MFMATileRange       # B tiles consumed
    deps: List['Dep'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['BaseOp'] = field(default_factory=list)     # populated by remove_cross_deps()
    vgpr_tile_maps: Dict[str, List[dict]] = field(default_factory=dict)  # {tensor: [{groupIdx: vgprTileId}]} per unroll iter

    def __post_init__(self):
        self.kind = 'mfma'

    def __str__(self):
        return (f"MFMAs (MT n, subIterK {self.subIterK}  ) "
                f"A : {self.tileA.fmt_tiles()} , B : {self.tileB.fmt_tiles()}")


@dataclass
class LRPlacement(Emittable):
    """Local Read placement for one tensor in one subIterK slot."""
    tensor: str                # 'A', 'B', 'SA', 'SB'
    mtIteration: int           # 0 = current MT, 1 = next MT
    tiles: MFMATileRange
    subIterK_slot: int         # which subIterK this LR is placed in
    partition: int = 0         # which partition this LR belongs to
    deps: List['Dep'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['BaseOp'] = field(default_factory=list)     # populated by remove_cross_deps()
    vgpr_tile_map: List[dict] = field(default_factory=list)  # [{tileId: vgprTileId}] per unroll iter

    def __post_init__(self):
        self.kind = 'lr'

    def __str__(self):
        return (f"LR {self.tensor.ljust(2)} (MT {fmt_mt(self.mtIteration)}, "
                f"subIterK {self.tiles.fmt_k()}) {self.tiles.fmt_tiles()}")


@dataclass
class GRPlacement(Emittable):
    """Global Read placement for one tensor in one subIterK slot."""
    tensor: str                # 'A', 'B', 'SA', 'SB'
    mtIteration: int           # 1 = next MT, 2 = two MTs ahead
    tiles: MFMATileRange
    subIterK_slot: int         # which subIterK this GR is placed in
    partition: int = 0         # which partition this GR belongs to
    deps: List['Dep'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['BaseOp'] = field(default_factory=list)     # populated by remove_cross_deps()

    def __post_init__(self):
        self.kind = 'gr'

    def __str__(self):
        return (f"GR {self.tensor} (MT {fmt_mt(self.mtIteration)}, "
                f"subIterK {self.tiles.fmt_k()}) ids {self.tiles.fmt_tiles()}")


# ── Per-subIterK container ──────────────────────────────────

@dataclass
class SubIterKSlot:
    """All operations placed in one subIterK step."""
    subIterK: int
    mfma: Optional[MFMAPlacement] = None
    lrs: List[LRPlacement] = field(default_factory=list)
    grs: List[GRPlacement] = field(default_factory=list)


# ── Dependency types ────────────────────────────────────────

@dataclass
class WaitGRCounts:
    """Per-tensor inflight load counts for wait_gr preOp."""
    A: int = 0
    B: int = 0
    SA: int = 0
    SB: int = 0

    def __str__(self):
        parts = []
        for t in ('A', 'B', 'SA', 'SB'):
            v = getattr(self, t)
            if v:
                parts.append(f"{t}={v}")
        return ",".join(parts) if parts else "0"


@dataclass
class BaseOp(Emittable):
    """Base class for typed dependency operations in a before-chain."""

    def __str__(self):
        return self.kind


@dataclass
class WaitGROp(BaseOp):
    """Wait for global reads to complete. Optionally includes a sync barrier."""
    wait_gr_counts: Optional[WaitGRCounts] = None
    has_sync: bool = False

    def __post_init__(self):
        self.kind = 'wait_gr'

    def __str__(self):
        if self.wait_gr_counts:
            return f"{self.kind}({self.wait_gr_counts})"
        return self.kind


@dataclass
class WaitLROp(BaseOp):
    """Wait for local reads to complete. Optionally includes a sync barrier."""
    has_sync: bool = False

    def __post_init__(self):
        self.kind = 'wait_lr'


@dataclass
class SyncOp(BaseOp):
    """Standalone sync barrier."""
    def __post_init__(self):
        self.kind = 'sync'


@dataclass
class LRIncOp(BaseOp):
    """LDS buffer swap for local reads on a specific tensor."""
    tensor: str = ""

    def __post_init__(self):
        self.kind = 'lr_inc'

    def __str__(self):
        return f"lr_inc({self.tensor})"


@dataclass
class GRIncOp(BaseOp):
    """Pointer update + LDS swap for global reads on a specific tensor."""
    tensor: str = ""

    def __post_init__(self):
        self.kind = 'gr_inc'

    def __str__(self):
        return f"gr_inc({self.tensor})"


@dataclass
class SkipOp(BaseOp):
    """Skip guard: compare LoopCounter and branch."""
    compare: str = ""
    value: int = 0
    target: str = ""

    def __post_init__(self):
        self.kind = 'skip'

    @property
    def tensor(self) -> str:
        return f"{self.compare}:{self.value}:{self.target}"

    def __str__(self):
        return f"skip({self.tensor})"


@dataclass
class Dep:
    """Dependency on another placement (annotate_deps output)."""
    ref: Union[LRPlacement, GRPlacement]
    mt_offset: int = 0  # 0 = same MT, -1 = prev MT, -2 = two MTs back, ...




# ── Emitted output ─────────────────────────────────────────

@dataclass
class EmittedModule:
    """One emitted module with before-link for instruction scheduling.

    Compatible with SubtileBasedInstructionScheduler.instructionSchedule().
    Instructions are left empty at the logical level — filled during emission.
    """
    moduleId: int = -1
    instructions: list = field(default_factory=list)
    before: Optional[int] = None   # moduleId that must complete before this module
    source: Optional[Emittable] = None

    @property
    def opType(self) -> str:
        return self.source.kind if self.source else ""


# ── Main scheduler class ───────────────────────────────────

class LogicalScheduler:
    """Subtile-based logical scheduler.

    Builds the schedule in 6 passes, each producing testable intermediate output.
    Each pass auto-runs its prerequisites if needed (tracked via self._completed).
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.tensors: List[str] = ['A', 'B'] + (['SA', 'SB'] if config.hasScale else [])
        self._completed: set = set()   # tracks which passes have run (Pass enum members)
        self._partitions: Optional[List[List[SubIterKSlot]]] = None  # shared mutable state across passes
        self._emitted: Optional[List[List[EmittedModule]]] = None
        self._preloop_emitted: Optional[List[List[List[EmittedModule]]]] = None
        self._ngll_emitted: Optional[List[List[List[EmittedModule]]]] = None
        self._nll_emitted: Optional[List[List[List[EmittedModule]]]] = None

    def _ensure_pass(self, *prerequisites: Pass) -> None:
        for p in prerequisites:
            if p not in self._completed:
                getattr(self, _PASS_PIPELINE[p][0])()

    # ── Place LRs ─────────────────────────────────────────

    def _partition_tile_range(self, pi: int) -> dict:
        """Return {'A': (start, end), 'B': (start, end)} for partition pi.

        Uses COLUMN_MAJOR ordering: M (A) varies fastest, N (B) varies slowest.
        """
        cfg = self.config
        # COLUMN_MAJOR: M is inner (pi % M), N is outer (pi // M)
        piM = pi % cfg.numPartitionsM
        piN = pi // cfg.numPartitionsM
        a0 = piM * cfg.partitionSizeM
        b0 = piN * cfg.partitionSizeN
        return {'A': (a0, a0 + cfg.partitionSizeM),
                'B': (b0, b0 + cfg.partitionSizeN)}

    def place_LRs(self) -> List[List[SubIterKSlot]]:
        """Place MFMAs and LRs based on read granularities.

        Returns a list of partitions, each containing a list of SubIterKSlots.

        Each LR prefetches data for the next subIterK group. Within-partition
        prefetches use current partition tiles; cross-partition prefetches
        (wrapping) use next partition tiles.

        Two tracking mechanisms:
        - loaded_ranges: tracks tile ranges in VGPR per side. Wrapping LRs
          are only placed when the next partition's tiles aren't already loaded.
        - placed: tracks (tensor, k-range, tile-range) of non-wrapping LRs
          placed so far across partitions. Skips redundant K-prefetch when
          the same data was already loaded by an earlier partition.
        """
        cfg = self.config
        numP = cfg.numPartitions
        part_ranges = [self._partition_tile_range(pi) for pi in range(numP)]

        # Track which tile ranges are currently loaded in VGPR (for wrapping decisions).
        loaded_ranges = {'A': {part_ranges[0]['A']},
                         'B': {part_ranges[0]['B']}}

        # Track placed K-prefetch LRs across partitions (for dedup).
        placed = set()

        partitions = []
        for pi in range(numP):
            cur, nxt = part_ranges[pi], part_ranges[(pi + 1) % numP]
            is_last = (pi == numP - 1)

            load = {}
            for side in ('A', 'B'):
                load[side] = is_last or nxt[side] not in loaded_ranges[side]

            slots = self._place_LRs_for_partition(cur, nxt, is_last, load, placed)
            for slot in slots:
                for lr in slot.lrs:
                    lr.partition = pi
            partitions.append(slots)

            for side in ('A', 'B'):
                if load[side]:
                    loaded_ranges[side] = {cur[side], nxt[side]}

        self._partitions = partitions
        self._completed.add(Pass.LR)
        return partitions

    def _place_LRs_for_partition(self, cur: tuple, nxt: tuple,
                                  is_last: bool,
                                  load: dict,
                                  placed: set) -> List[SubIterKSlot]:
        """Place MFMAs and LRs for one partition."""
        cfg = self.config
        numK = cfg.numSubIterK
        multi_part = cfg.numPartitions > 1

        slots = [SubIterKSlot(subIterK=k) for k in range(numK)]
        slot_mt = {}  # slot_k → lr_mt string, for MT-homogeneity enforcement

        # MFMAs
        for k in range(numK):
            slots[k].mfma = MFMAPlacement(
                subIterK=k,
                tileA=MFMATileRange(k, k + 1, cur['A'][0], cur['A'][1]),
                tileB=MFMATileRange(k, k + 1, cur['B'][0], cur['B'][1]),
            )

        # All tensors that can participate.
        all_tensors = [('A', cfg.lrA), ('B', cfg.lrB)]
        if cfg.hasScale:
            all_tensors.append(('SA', cfg.lrSA))
            all_tensors.append(('SB', cfg.lrSB))

        # Place LRs grouped by k_gran.
        # - Non-wrapping (K-prefetch): all tensors, deduped by placed set.
        # - Wrapping (cross-partition): only tensors whose side needs loading.
        for k_gran in sorted(set(g.k for _, g in all_tensors)):
            group_all = [(t, g) for t, g in all_tensors if g.k == k_gran]
            num_chunks = numK // k_gran
            for chunk_idx in range(num_chunks):
                next_chunk = (chunk_idx + 1) % num_chunks
                is_wrap = (next_chunk == 0)
                lr_mt = 1 if is_last and is_wrap else 0
                lr_k_start = next_chunk * k_gran
                lr_k_end = lr_k_start + k_gran
                base_slot = chunk_idx * k_gran

                # For wrapping chunks, only include tensors whose side is
                # loading so that slot assignment reflects active tensors.
                # A and B always participate (their wrapping is gated inside
                # the loop) to keep slot indices stable for their k_gran group.
                if is_wrap and multi_part:
                    group = [(t, g) for t, g in group_all
                             if t in ('A', 'B') or load['A' if t in ('A', 'SA') else 'B']]
                else:
                    group = group_all

                # Group by side (A/SA together, B/SB together) for slot assignment
                sides = [[(t, g) for t, g in group if t in ('A', 'SA')],
                         [(t, g) for t, g in group if t in ('B', 'SB')]]
                sides = [s for s in sides if s]

                for side_idx, side in enumerate(sides):
                    slot_k = base_slot + (side_idx % k_gran)
                    # Redirect LRs away from slots committed to a different MT,
                    # keeping each slot MT-homogeneous.
                    # This reduce the number of wait_gr_sync needed as all LRs 
                    # in the same subIterK wait for the same MT iterration.
                    committed = slot_mt.get(slot_k)
                    if committed is not None and committed != lr_mt:
                        slot_k = numK - 1

                    for tensor, gran in side:
                        tile_range = nxt if (is_wrap or not multi_part) else cur
                        side_key = 'A' if tensor in ('A', 'SA') else 'B'
                        ts, te = tile_range[side_key]

                        # Wrapping: use load dict. Non-wrapping: use placed set.
                        if is_wrap and multi_part:
                            if not load[side_key]:
                                continue
                        else:
                            lr_key = (tensor, lr_k_start, lr_k_end, ts, te)
                            if lr_key in placed:
                                continue
                            placed.add(lr_key)

                        lr = LRPlacement(
                            tensor=tensor,
                            mtIteration=lr_mt,
                            tiles=MFMATileRange(lr_k_start, lr_k_end, ts, te),
                            subIterK_slot=slot_k,
                        )
                        slots[slot_k].lrs.append(lr)
                        slot_mt[slot_k] = lr_mt

        return slots

    # ── Assign VGPR tile IDs (free-list allocation) ──────

    def assign_vgpr_tiles(self):
        """Assign physical vgprTileIds to all placements (A, B, SA, SB).

        Free-list allocator with per-tensor FIFO queues, iterated until
        convergence (or max 4 unroll iterations).

        Three phases:
          1. Scan all MFMAs to find last read position for each
             (tensor, tileId, k_data_group) key.
          2. Walk execution order in a loop: each iteration feeds the
             previous next_iter as the starting active state.  Appends
             one tile-map dict per iteration to each placement's list.
             Stops when next_iter matches the seeded state (convergence).
          3. Record unroll_factor, needs_unrolling, and max tile_peaks.

        Keys use a unified formula parameterized by ReadGranularity:
          key = (tensor, (tileId // lr_gran.mn) * lr_gran.mn, (k // lr_gran.k) * lr_gran.k)

        Sets self.tile_peaks (per-tensor max across unrolls),
        self.needs_unrolling, self.unroll_factor.
        """
        self._ensure_pass(Pass.LR)

        cfg = self.config
        numK = cfg.numSubIterK
        MAX_UNROLL = 8

        lr_grans = {'A': cfg.lrA, 'B': cfg.lrB}
        if cfg.hasScale:
            lr_grans['SA'] = cfg.lrSA
            lr_grans['SB'] = cfg.lrSB

        # ── Phase 1: find last MFMA read for each key ──
        last_read = {}  # key -> flat position
        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                if not slot.mfma:
                    continue
                pos = pi * numK + slot.subIterK
                k = slot.subIterK
                for tensor in self.tensors:
                    side = TENSOR_SIDE[tensor]
                    tileRange = slot.mfma.tileA if side == 'A' else slot.mfma.tileB
                    gran = lr_grans[tensor]
                    for t in tileRange.tileId_list:
                        group = (t // gran.mn) * gran.mn
                        k_chunk = (k // gran.k) * gran.k
                        last_read[(tensor, group, k_chunk)] = pos

        # ── Phase 2: iterate until convergence ──
        from collections import deque

        class _FreeList:
            __slots__ = ('free', 'next_id', 'active_count', 'peak')
            def __init__(self):
                self.free = deque()
                self.next_id = 0
                self.active_count = 0
                self.peak = 0
            def alloc(self):
                if self.free:
                    vid = self.free.popleft()  # FIFO for convergence
                else:
                    vid = self.next_id
                    self.next_id += 1
                self.active_count += 1
                self.peak = max(self.peak, self.active_count)
                return vid
            def release(self, vid):
                self.free.append(vid)
                self.active_count -= 1

        max_peaks = {t: 0 for t in self.tensors}
        carry_active = {}
        all_next_iters = []     # next_iter from each iteration, for cycle detection

        pools = {t: _FreeList() for t in self.tensors}

        for unroll_iter in range(MAX_UNROLL):
            if unroll_iter == 0:
                active = {}
            else:
                active = dict(carry_active)
                # Reset active_count to match carry_active (tiles that survived
                # as live from the previous iteration's wrapping LRs).
                for t in self.tensors:
                    pools[t].active_count = sum(
                        1 for key in active if key[0] == t)

            next_iter = {}

            for pi, slots in enumerate(self._partitions):
                for slot in slots:
                    pos = pi * numK + slot.subIterK
                    k = slot.subIterK

                    # ── MFMA reads: look up or seed ──
                    if slot.mfma:
                        for tensor in self.tensors:
                            side = TENSOR_SIDE[tensor]
                            tileRange = slot.mfma.tileA if side == 'A' else slot.mfma.tileB
                            gran = lr_grans[tensor]
                            tile_map = {}
                            for t in tileRange.tileId_list:
                                group = (t // gran.mn) * gran.mn
                                k_chunk = (k // gran.k) * gran.k
                                key = (tensor, group, k_chunk)
                                if key not in active:
                                    active[key] = pools[tensor].alloc()
                                tile_map[group] = active[key]
                            slot.mfma.vgpr_tile_maps.setdefault(tensor, []).append(tile_map)

                    # ── LR writes: allocate new tiles ──
                    for lr in slot.lrs:
                        tensor = lr.tensor
                        is_wrapping = lr.mtIteration != 0
                        target = next_iter if is_wrapping else active

                        gran = lr_grans[tensor]
                        tile_map = {}
                        seen_keys = set()
                        for t in lr.tiles.tileId_list:
                            group = (t // gran.mn) * gran.mn
                            for lk in lr.tiles.subIterK_list:
                                k_chunk = (lk // gran.k) * gran.k
                                key = (tensor, group, k_chunk)
                                if key in seen_keys:
                                    continue
                                seen_keys.add(key)
                                if key in target:
                                    pools[tensor].release(target[key])
                                vid = pools[tensor].alloc()
                                target[key] = vid
                                tile_map[group] = vid
                        lr.vgpr_tile_map.append(tile_map)

                    # ── Release tiles whose last read was at this position ──
                    to_release = [key for key, lr_pos in last_read.items()
                                  if lr_pos == pos and key in active]
                    for key in to_release:
                        pools[key[0]].release(active[key])
                        del active[key]

            # Track max peaks across iterations
            for t in self.tensors:
                max_peaks[t] = max(max_peaks[t], pools[t].peak)

            # Check convergence: if this iteration's next_iter matches
            # any previous iteration's next_iter, we found a cycle.
            # The cycle period is (current_iter - matching_iter).
            # All iterations from matching_iter to current_iter-1 form
            # the repeating pattern; iterations before that are prologue.
            converged = False
            for prev_idx, prev_ni in enumerate(all_next_iters):
                if next_iter == prev_ni:
                    # Strip tile maps from the redundant convergence iteration.
                    for pi2, slots2 in enumerate(self._partitions):
                        for slot2 in slots2:
                            if slot2.mfma:
                                for tensor in self.tensors:
                                    if tensor in slot2.mfma.vgpr_tile_maps:
                                        slot2.mfma.vgpr_tile_maps[tensor].pop()
                            for lr2 in slot2.lrs:
                                lr2.vgpr_tile_map.pop()
                    converged = True
                    break
            if converged:
                break

            # Carry next_iter forward as active for next iteration
            all_next_iters.append(next_iter)
            carry_active = next_iter
        else:
            assert False, (f"assign_vgpr_tiles did not converge after "
                           f"{MAX_UNROLL} unroll iterations")

        # ── Phase 3: record results ──
        # unroll_factor = number of unique iterations (convergence iteration excluded)
        self.unroll_factor = unroll_iter
        self.needs_unrolling = self.unroll_factor > 1
        self.tile_peaks = max_peaks

        self._completed.add(Pass.VGPR_TILES)

    # ── Place GRs ─────────────────────────────────────────

    def _build_gr_list(self, part_ranges, offsetMT, offsetPartition,
                             debug=False):
        """Phase 1: Build ordered GR list from placed MFMAs.

        For each partition × subIterK, derive target partition/MT from
        the MFMA and offsets. Add GRs (A, B, SA, SB) with tile and K
        ranges snapped to GR granularity. Dedup within same MT level,
        then remove n+1 entries that also appear at n+2 (cross-MT dedup).
        
        For each subIterK, we apply offsetMT on MT and offsetPartition on partition.

        Returns list of (tensor, mt_str, tile_start, tile_end,
                         k_start, k_end, gr_gran).
        """
        cfg = self.config
        numP = cfg.numPartitions

        seen = set()
        gr_list = []

        for pi in range(numP):
            partition_slots = self._partitions[pi]

            target_pi = (pi + offsetPartition) % numP
            wraps = (pi + offsetPartition) >= numP
            mt_val = offsetMT + (1 if wraps else 0)

            target_range = part_ranges[target_pi]

            for slot in partition_slots:
                k = slot.mfma.subIterK

                items = [('A', target_range['A'], cfg.grA),
                         ('B', target_range['B'], cfg.grB)]
                if cfg.hasScale:
                    items.append(('SA', target_range['A'], cfg.grSA))
                    items.append(('SB', target_range['B'], cfg.grSB))

                for tensor, (t_start, t_end), gr_gran in items:
                    mn = gr_gran.mn
                    k_gran = gr_gran.k

                    gr_tile_start = (t_start // mn) * mn
                    gr_tile_end = ((t_end + mn - 1) // mn) * mn

                    gr_k_start = (k // k_gran) * k_gran
                    gr_k_end = gr_k_start + k_gran

                    key = (tensor, mt_val, gr_tile_start, gr_tile_end,
                           gr_k_start, gr_k_end)
                    if key in seen:
                        continue
                    seen.add(key)
                    gr_list.append((tensor, mt_val, gr_tile_start,
                                    gr_tile_end, gr_k_start, gr_k_end,
                                    gr_gran))

        # Cross-MT dedup: if a tile/k range appears at both n+1 and n+2,
        # the n+1 load is redundant — the previous iteration's n+2 already
        # wrote the same data into LDS.  Remove the n+1 duplicate.
        base_mt = offsetMT
        n2_keys = {(t, ts, te, ks, ke)
                   for t, mt, ts, te, ks, ke, _ in gr_list
                   if mt != base_mt}
        gr_list = [entry for entry in gr_list
                   if entry[1] != base_mt or
                   (entry[0], entry[2], entry[3], entry[4], entry[5])
                   not in n2_keys]

        if debug:
            print(f"Phase 1: {len(gr_list)} GR entries")
            for i, (t, mt, ts, te, ks, ke, g) in enumerate(gr_list):
                loads = ((te - ts) // g.mn) * ((ke - ks) // g.k)
                print(f"  [{i}] {t:2s} {fmt_mt(mt)} tiles[{ts},{te - 1}] k[{ks},{ke - 1}] "
                      f"gr_gran(mn={g.mn},k={g.k}) loads={loads}")

        return gr_list

    def _build_lr_conflict_map(self):
        """Build per-partition LR(MT n) info for LDS conflict checking.

        Returns dict: (partition_idx, tensor) -> list of
                      (subIterK_slot, k_start, k_end).
        """
        lr_mt_n_info = {}
        for pi, partition_slots in enumerate(self._partitions):
            for slot in partition_slots:
                for lr in slot.lrs:
                    if lr.mtIteration == 0:
                        lr_mt_n_info.setdefault((pi, lr.tensor), []).append(
                            (slot.subIterK,
                             lr.tiles.subIterK_start,
                             lr.tiles.subIterK_end))
        return lr_mt_n_info

    @staticmethod
    def _has_lr_conflict(lr_mt_n_info, tensor, mt_val, pi, subIterK,
                         gr_k_start, gr_k_end):
        """Return True if placing GR(mt_val) at (pi, subIterK) conflicts.

        GR(MT n+2) writes the same LDS buffer as MT n, so it conflicts
        only if a later LR(MT n) in the same partition accesses an
        overlapping subIterK range.
        """
        if mt_val != 2:
            return False
        for lr_slot, lr_ks, lr_ke in lr_mt_n_info.get((pi, tensor), []):
            if lr_slot > subIterK and gr_k_start < lr_ke and lr_ks < gr_k_end:
                return True
        return False

    def _distribute_grs(self, gr_list, lr_mt_n_info, debug=False):
        """Phase 2: Distribute GR atoms across partition × subIterK slots.

        Explodes GR entries into atomic loads, distributes them into flat
        buckets respecting LDS conflict constraints and load balance,
        then remerges consecutive atoms and places them into partitions.
        """
        cfg = self.config
        numK = cfg.numSubIterK
        numP = cfg.numPartitions
        numSlots = numP * numK

        # 2a. Explode GR entries into atomic loads (1 load each)
        atoms = []
        for tensor, mt_val, t_start, t_end, k_start, k_end, gr_gran in gr_list:
            mn = gr_gran.mn
            for pos in range(t_start, t_end, mn):
                atoms.append((tensor, mt_val, pos, pos + mn, k_start, k_end))

        loads_per_slot = len(atoms) // numSlots

        # 2b. Distribute atoms into flat buckets [0..numSlots),
        #     each bucket maps to (partition=flat//numK, subIterK=flat%numK)
        buckets = [[] for _ in range(numSlots)]
        for atom in atoms:
            tensor, mt_val, _, _, ks, ke = atom
            cur = 0
            while cur < numSlots - 1:
                pi = cur // numK
                subK = cur % numK
                if (not self._has_lr_conflict(lr_mt_n_info, tensor, mt_val,
                                              pi, subK, ks, ke) and
                        len(buckets[cur]) < loads_per_slot):
                    break
                cur += 1
            buckets[cur].append(atom)

        if debug:
            print(f"Phase 2b: {len(atoms)} atoms, {numSlots} slots, "
                  f"{loads_per_slot} per slot")
            for flat, bucket in enumerate(buckets):
                pi = flat // numK
                si = flat % numK
                if bucket:
                    items = ", ".join(
                        f"{t} {fmt_mt(mt)} tile[{ts},{te-1}] k[{ks},{ke-1}]"
                        for t, mt, ts, te, ks, ke in bucket)
                    print(f"  P{pi} s{si}: {len(bucket)} atoms — {items}")
                else:
                    print(f"  P{pi} s{si}: empty")

        # 2c. Remerge consecutive atoms and place into partitions
        for flat, bucket in enumerate(buckets):
            pi = flat // numK
            si = flat % numK
            target_slot = self._partitions[pi][si]
            for atom in bucket:
                tensor, mt_val, ts, te, ks, ke = atom
                if target_slot.grs:
                    prev = target_slot.grs[-1]
                    if (prev.tensor == tensor and
                            prev.mtIteration == mt_val and
                            prev.tiles.subIterK_start == ks and
                            prev.tiles.subIterK_end == ke and
                            prev.tiles.tileId_end == ts):
                        prev.tiles = MFMATileRange(ks, ke, prev.tiles.tileId_start, te)
                        continue
                target_slot.grs.append(GRPlacement(
                    tensor=tensor, mtIteration=mt_val,
                    tiles=MFMATileRange(ks, ke, ts, te),
                    subIterK_slot=si,
                    partition=pi))

    def place_GRs(self) -> List[SubIterKSlot]:
        """Place Global Reads by iterating MFMAs across partitions.

        Phase 1: Build ordered GR list from partition traversal respecting gr granularities.
        Phase 2: Distribute evenly GR atoms across all (partition, subIterK) slots. GR atoms being the smallest load granularity for a specific tensor.

        This should give a sheduling respecting the following rules:
         - GR are in the order we expect them from the LR pov
         - we respect the GR granularities (can change the above rule a bit)
         - Overall loads are spread accross all subIterKs of all partitions.

        """
        self._ensure_pass(Pass.LR)

        part_ranges = [self._partition_tile_range(pi)
                       for pi in range(self.config.numPartitions)]

        # TODO: cover PGR3 (offsetMT and offsetPartition may differ)
        offsetMT = 1
        offsetPartition = 1
        # Build ordered list of GRs to place for the entire MT based on the partitioning ordering and the GR granularities.
        gr_list = self._build_gr_list(part_ranges, offsetMT, offsetPartition)
        # Map to keep track of LR(MT n) for each partiion and tensor, used for LDS double buffer conflict checking when placing GRs.
        lr_mt_n_info = self._build_lr_conflict_map()
        # Distribute GRs accross partition.
        self._distribute_grs(gr_list, lr_mt_n_info)

        self._completed.add(Pass.GR)
        return self._partitions[0]

    # ── Annotate dependencies ─────────────────────────────

    def annotate_deps(self):
        """Annotate each placement with its raw before-dependencies.

        Populates the `before` field on MFMAPlacement, LRPlacement, and
        GRPlacement objects in self._partitions. Each lr_ref/gr_ref BaseOp
        is resolved to point at the specific placement it depends on.

        Iterates all partitions. Two-pass per partition:
        - Pass 1: build lookups from existing placements
        - Pass 2: populate .before on each placement

        Rules:
        - MFMA(subIterK=k) depends on all LRs that loaded subIterK=k data
          (cross-partition: LRs for a tensor may be in any partition)
        - LR depends on GR for same tensor (data must be in LDS)
        - GR depends on collision LR for same tensor (LDS double-buffer)
        """
        self._ensure_pass(Pass.GR)
        cfg = self.config
        numK = cfg.numSubIterK

        # Build global lr_by_data across all partitions (MFMA deps are cross-partition)
        # lr_by_data[data_k][tensor] → list of LRPlacements loading subIterK=data_k
        lr_by_data = [{} for _ in range(numK)]
        # gr_by_tensor[tensor] → list of all GRPlacements (LR→GR deps are cross-partition)
        gr_by_tensor = {}
        # lr_by_tensor[tensor] → list of all LRPlacements (GR→LR collision is cross-partition)
        lr_by_tensor = {}
        for slots in self._partitions:
            for slot in slots:
                for lr in slot.lrs:
                    for data_k in lr.tiles.subIterK_list:
                        lr_by_data[data_k].setdefault(lr.tensor, []).append(lr)
                    lr_by_tensor.setdefault(lr.tensor, []).append(lr)
                for gr in slot.grs:
                    gr_by_tensor.setdefault(gr.tensor, []).append(gr)

        for pi, slots in enumerate(self._partitions):
            self._annotate_deps_partition(pi, slots, cfg, lr_by_data,
                                          gr_by_tensor, lr_by_tensor)

        self._completed.add(Pass.DEPS)

    def _annotate_deps_partition(self, pi: int, slots: List[SubIterKSlot],
                                 cfg: SchedulerConfig, lr_by_data: list,
                                 gr_by_tensor: dict, lr_by_tensor: dict):
        """Annotate deps for a single partition (in-place on placements)."""
        numK = len(slots)

        # Clear any previous annotations (idempotent re-runs)
        for slot in slots:
            if slot.mfma:
                slot.mfma.deps.clear()
            for lr in slot.lrs:
                lr.deps.clear()
            for gr in slot.grs:
                gr.deps.clear()

        # ── Pass 1: build per-partition lookups ──
        # lr_by_slot[k][tensor] → LRPlacement at subIterK=k
        # gr_by_slot[k][tensor] → GRPlacement at subIterK=k
        # (lr_by_data, gr_by_tensor, lr_by_tensor are built globally in annotate_deps)
        lr_by_slot = [{} for _ in range(numK)]
        gr_by_slot = [{} for _ in range(numK)]

        for k, slot in enumerate(slots):
            for lr in slot.lrs:
                lr_by_slot[k][lr.tensor] = lr

            for gr in slot.grs:
                gr_by_slot[k][gr.tensor] = gr

        # ── Pass 2: populate deps on each placement ──
        # mt_offset: 0 = same MT, -1 = prev MT, -2 = two MTs back, etc.
        # Within one iteration, execution order per slot is MFMA → LR → GR,
        # and slots run in order 0, 1, 2, ...
        _order = {'MFMA': 0, 'LR': 1, 'GR': 2}

        def _slot_offset(consumer_partition, consumer_slot, consumer_type, producer):
            """Offset from partition+slot ordering: 0 if producer ran first, -1 otherwise."""
            prod_partition = producer.partition
            if prod_partition < consumer_partition:
                return 0
            if prod_partition > consumer_partition:
                return -1
            prod_slot = producer.subIterK_slot
            if prod_slot < consumer_slot:
                return 0
            if prod_slot > consumer_slot:
                return -1
            prod_type = 'LR' if isinstance(producer, LRPlacement) else 'GR'
            return -1 if _order[prod_type] >= _order[consumer_type] else 0

        def _mt_offset(consumer_partition, consumer_slot, consumer_type, producer, consumer=None):
            # MFMA→LR: MFMA always consumes mt=0 (current).
            if consumer_type == 'MFMA' and isinstance(producer, LRPlacement):
                if producer.mtIteration > 0:
                    return -producer.mtIteration
            # LR→GR: mt difference determines how many iterations back.
            if consumer_type == 'LR' and isinstance(producer, GRPlacement) and consumer:
                diff = producer.mtIteration - consumer.mtIteration
                if diff != 0:
                    return -diff
            # Same effective mt: partition+slot ordering decides.
            return _slot_offset(consumer_partition, consumer_slot, consumer_type, producer)

        def _tiles_overlap(mfma, lr_tensor, lr_tiles):
            """Check if LR tile range overlaps with MFMA's tile range for that tensor."""
            # SA/SB follow A/B tile ranges respectively
            if lr_tensor in ('A', 'SA'):
                mfma_range = mfma.tileA
            else:
                mfma_range = mfma.tileB
            return (lr_tiles.tileId_start < mfma_range.tileId_end and
                    lr_tiles.tileId_end > mfma_range.tileId_start)

        def _range_overlaps(a: MFMATileRange, b: MFMATileRange) -> bool:
            """Check if two tile ranges overlap on both tile ids and subIterK."""
            return (a.tileId_start < b.tileId_end and
                    a.tileId_end > b.tileId_start and
                    a.subIterK_start < b.subIterK_end and
                    a.subIterK_end > b.subIterK_start)

        def _dedup_deps(deps):
            if len(deps) <= 1:
                return deps
            def _exec_order(dep):
                return (dep.mt_offset, dep.ref.partition, dep.ref.subIterK_slot)
            return [max(deps, key=_exec_order)]

        for k, slot in enumerate(slots):
            # MFMA: depends on the most recent LR per tensor (tile-overlapping).
            # Uses lr_by_tensor (all LRs across partitions) so that a more recent
            # LR loading a different subIterK still subsumes older data deps.
            if slot.mfma:
                for t in self.tensors:
                    deps_for_t = []
                    for lr in lr_by_tensor.get(t, []):
                        if _tiles_overlap(slot.mfma, t, lr.tiles):
                            deps_for_t.append(Dep(
                                ref=lr, mt_offset=_mt_offset(pi, k, 'MFMA', lr)))
                    slot.mfma.deps.extend(_dedup_deps(deps_for_t))

            # LR: depends on GR (data must be in LDS before reading)
            # Cross-partition: the GR that loaded the matching tiles may be
            # in a different partition. Filter by tile overlap.
            for lr in slot.lrs:
                for gr in gr_by_tensor.get(lr.tensor, []):
                    if _range_overlaps(lr.tiles, gr.tiles):
                        lr.deps.append(Dep(
                            ref=gr, mt_offset=_mt_offset(pi, k, 'LR', gr, consumer=lr)))

            # GR: depends on collision LR (LDS double-buffer)
            # GR(n+x) collides with LR(n+x-2) — same buffer, period 2.
            # target_data = gr.mtIteration - 2. For each LR of same tensor,
            # mt_offset = target_data - lr.mtIteration. Dedup keeps latest.
            #   GR(2)→LR(0):  mt_offset = 0   (same iteration)
            #   GR(2)→LR(1):  mt_offset = -1  (prev iter LR(1) handled n)
            #   GR(1)→LR(0):  mt_offset = -1  (prev iter LR(0) handled n-1)
            for gr in slot.grs:
                target_data = gr.mtIteration - 2
                for lr in lr_by_tensor.get(gr.tensor, []):
                    if _range_overlaps(lr.tiles, gr.tiles):
                        mt_off = target_data - lr.mtIteration
                        gr.deps.append(Dep(ref=lr, mt_offset=mt_off))
                if not gr.deps:
                    raise ValueError(
                        f"GR {gr.tensor} mt={fmt_mt(gr.mtIteration)} at slot {k} "
                        f"has no overlapping LR(n) dependency")

        for slot in slots:
            for lr in slot.lrs:
                lr.deps = _dedup_deps(lr.deps)
            for gr in slot.grs:
                gr.deps = _dedup_deps(gr.deps)

    # ── Remove unnecessary GR deps ────────────────────────

    def remove_unnecessary_gr_deps(self):
        """Remove GR deps on LRs that are already guaranteed by an earlier LR's wait.

        Per tensor, walks LR placements in execution order. If an earlier LR
        already waits for a GR with equal or higher exec_order, the later LR's
        dep is redundant and removed.

        Wraps around: the first LR's dep is compared against the last from the
        previous MT iteration (max dep exec_order shifted by mt_offset -1).
        """
        self._ensure_pass(Pass.DEPS)

        def _dep_exec_order(dep):
            return (dep.mt_offset, dep.ref.partition, dep.ref.subIterK_slot)

        for tensor in self.tensors:
            lr_with_gr_deps = []
            for pi, slots in enumerate(self._partitions):
                for slot in slots:
                    for lr in slot.lrs:
                        if lr.tensor == tensor and lr.deps:
                            dep = lr.deps[0]
                            if isinstance(dep.ref, GRPlacement):
                                lr_with_gr_deps.append((lr, dep))

            if len(lr_with_gr_deps) <= 1:
                continue

            max_eo = max(_dep_exec_order(dep) for _, dep in lr_with_gr_deps)
            max_guaranteed = (max_eo[0] - 1, max_eo[1], max_eo[2])

            for lr, dep in lr_with_gr_deps:
                eo = _dep_exec_order(dep)
                if eo <= max_guaranteed:
                    lr.deps.clear()
                else:
                    max_guaranteed = eo

        self._completed.add(Pass.REMOVE_GR_DEPS)

    # ── Remove unnecessary LR deps ────────────────────────

    def remove_unnecessary_lr_deps(self):
        """Remove GR→LR collision deps already covered by an earlier sync.

        A GR with an LR dep creates a sync point. 
        We get the latest LR guaranted at this sync point (prevLRDep), which is the max of:
         - the GR's own LR dep 
         - the MFMA's same-tensor LR dep
        
        If the current GR's LR dep (currLRDep) has exec order <= prevLRDep, it is already guaranteed
        and can be removed.

        Exec order is (mt_offset, partition, subIterK_slot).
        On wrap-around the exec order is shifted by MT-1.
        """
        self._ensure_pass(Pass.REMOVE_GR_DEPS)

        def _dep_exec_order(dep):
            return (dep.mt_offset, dep.ref.partition, dep.ref.subIterK_slot)

        gr_with_lr_deps = []
        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                for gr in slot.grs:
                    if gr.deps:
                        dep = gr.deps[0]
                        if isinstance(dep.ref, LRPlacement):
                            gr_with_lr_deps.append((pi, slot.subIterK, gr, dep))

        if len(gr_with_lr_deps) <= 1:
            self._completed.add(Pass.REMOVE_LR_DEPS)
            return

        mfma_by_pos = {}
        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                if slot.mfma:
                    mfma_by_pos[(pi, slot.subIterK)] = slot.mfma

        gr_with_lr_deps.sort(key=lambda x: (x[0], x[1]))

        last_sync = (gr_with_lr_deps[-1][0], gr_with_lr_deps[-1][1])
        # Per-tensor max LR exec order guaranteed at last_sync.
        last_sync_eo = {}

        def _update_sync_eo(pos, shift):
            """Collect per-tensor max LR exec order at pos (MFMA deps)."""
            eo_map = {}
            mfma = mfma_by_pos.get(pos)
            if mfma and mfma.deps:
                for d in mfma.deps:
                    if isinstance(d.ref, LRPlacement):
                        t = d.ref.tensor
                        d_eo = _dep_exec_order(d)
                        if shift:
                            d_eo = (d_eo[0] - 1, d_eo[1], d_eo[2])
                        if t not in eo_map or d_eo > eo_map[t]:
                            eo_map[t] = d_eo
            return eo_map

        # Seed from last position (previous MT → shift by -1).
        last_sync_eo = _update_sync_eo(last_sync, shift=True)

        for pi, subIterK, gr, dep in gr_with_lr_deps:
            curr_eo = _dep_exec_order(dep)
            tensor = dep.ref.tensor

            prev_lr_eo = last_sync_eo.get(tensor)
            if prev_lr_eo is not None and curr_eo <= prev_lr_eo:
                gr.deps.clear()
                continue

            last_sync = (pi, subIterK)
            last_sync_eo = _update_sync_eo(last_sync, shift=False)
            # The GR's own dep is also a sync point at this slot.
            if tensor not in last_sync_eo or curr_eo > last_sync_eo[tensor]:
                last_sync_eo[tensor] = curr_eo

        self._completed.add(Pass.REMOVE_LR_DEPS)

    # ── Remove cross-subIterK deps ─────────────────────────

    def _gr_granularity(self, tensor: str) -> ReadGranularity:
        """Return GR granularity for a tensor."""
        return {'A': self.config.grA, 'B': self.config.grB,
                'SA': self.config.grSA, 'SB': self.config.grSB}[tensor]

    def _compute_inflight_loads(self, consumer_pi: int, consumer_slot: int,
                                tensor: str, dep_ref: Dep) -> WaitGRCounts:
        """Count inflight GR atomic loads between a dep GR and the consumer.

        Walks backward through the flattened schedule (all partitions x subIterK)
        from the consumer position, counting atomic GR loads for all tensors.
        Stops when reaching the dependency GR (dep_ref.ref) after accounting
        for mt_offset wraps.

        Returns per-tensor inflight load counts.
        """
        numP = len(self._partitions)
        numK = len(self._partitions[0])
        flat_len = numP * numK

        consumer_flat = consumer_pi * numK + consumer_slot

        wraps_needed = abs(dep_ref.mt_offset)

        counts = WaitGRCounts()
        wraps_completed = 0
        pos = consumer_flat

        max_steps = (wraps_needed + 1) * flat_len
        for _ in range(max_steps):
            pos = (pos - 1) % flat_len
            if pos == flat_len - 1 and _ > 0:
                wraps_completed += 1

            pi = pos // numK
            slot_k = pos % numK
            slot = self._partitions[pi][slot_k]

            for gr in slot.grs:
                if gr.tensor == tensor and gr is dep_ref.ref and wraps_completed >= wraps_needed:
                    return counts
                gr_gran = self._gr_granularity(gr.tensor)
                tiles = gr.tiles
                n_tile = (tiles.tileId_end - tiles.tileId_start) // gr_gran.mn
                n_k = (tiles.subIterK_end - tiles.subIterK_start) // gr_gran.k
                cur = getattr(counts, gr.tensor)
                setattr(counts, gr.tensor, cur + n_tile * n_k)

        return counts

    def remove_cross_deps(self):
        """Replace cross-subIterK deps with wait preOps.

        For each placement, separates deps into same-subIterK (kept) and
        cross-subIterK (converted to preOps):
          - MFMA depending on LRs → single wait_lr
          - GR depending on LRs   → single wait_lr_sync
          - LR depending on GRs   → single wait_gr_sync with per-tensor inflight counts
        """
        self._ensure_pass(Pass.REMOVE_LR_DEPS)

        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                # ── MFMA ──
                if slot.mfma:
                    same, cross = self._split_deps(slot.mfma.deps, pi, slot.subIterK)
                    slot.mfma.deps = same
                    slot.mfma.preOps = []
                    if cross:
                        slot.mfma.preOps.append(WaitLROp())

                # ── LRs ──
                for lr in slot.lrs:
                    same, cross = self._split_deps(lr.deps, pi, lr.subIterK_slot)
                    lr.deps = same
                    lr.preOps = []
                    if cross:
                        dep = cross[0]
                        counts = self._compute_inflight_loads(
                            pi, lr.subIterK_slot, dep.ref.tensor, dep)
                        lr.preOps.append(WaitGROp(wait_gr_counts=counts,
                                                  has_sync=True))

                # ── GRs ──
                for gr in slot.grs:
                    same, cross = self._split_deps(gr.deps, pi, gr.subIterK_slot)
                    gr.deps = same
                    has_lr_dep = any(
                        isinstance(d.ref, LRPlacement)
                        for d in same + cross)
                    gr.preOps = [WaitLROp(has_sync=True)] if has_lr_dep else []

        self._completed.add(Pass.REMOVE_DEPS)

    def insert_gr_lr_inc(self):
        """Insert gr_inc/lr_inc preOps at MacroTile iteration transitions.

        Walks all LR and GR placements in global execution order
        (partition 0 slots → partition 1 slots → ..., within each slot: LR then GR).
        Tracks per-tensor the last-seen mtIteration. When a tensor's mtIteration
        changes, inserts a BaseOp into that placement's preOps:
          - lr_inc for LR placements
          - gr_inc for GR placements
        """
        self._ensure_pass(Pass.REMOVE_DEPS)

        last_lr_mt = {}  # tensor -> mtIteration for LR only
        last_gr_mt = {}  # tensor -> mtIteration for GR only
        first_lr = {}  # tensor -> first LR placement seen
        lr_inc_tensors = set()  # tensors that already received lr_inc

        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                for lr in slot.lrs:
                    tensor = lr.tensor
                    mt = lr.mtIteration
                    if tensor not in first_lr:
                        first_lr[tensor] = lr
                    if tensor in last_lr_mt and last_lr_mt[tensor] != mt:
                        lr.preOps.append(LRIncOp(tensor=tensor))
                        lr_inc_tensors.add(tensor)
                    last_lr_mt[tensor] = mt
                for gr in slot.grs:
                    tensor = gr.tensor
                    mt = gr.mtIteration
                    prev_mt = last_gr_mt.get(tensor, last_lr_mt.get(tensor))
                    if prev_mt is not None and prev_mt != mt:
                        if gr.tiles.tileId_start == 0:
                            gr.preOps.append(GRIncOp(tensor=tensor))
                    last_gr_mt[tensor] = mt

        # Handle wrap-around: tensors with a single LR per iteration (e.g. SA, SB)
        # still need lr_inc because the GR at end-of-iteration writes to the other
        # LDS buffer, and the next iteration's LR must swap to read from it.
        # Safe: preOps are consumed by emit(), not during this walk.
        for tensor, lr in first_lr.items():
            if tensor not in lr_inc_tensors:
                last = last_gr_mt.get(tensor, last_lr_mt.get(tensor))
                if last is not None and last != lr.mtIteration:
                    lr.preOps.append(LRIncOp(tensor=tensor))

        self._completed.add(Pass.GR_INC)

    # ── Group LR/GR chains ─────────────────────────────────────

    _LR_GR_ORDER = ['A', 'B', 'SA', 'SB']

    @staticmethod
    def _merge_preops(all_preops: List[List['BaseOp']]) -> List['BaseOp']:
        """Merge preOps from multiple placements.

        Combines wait_gr/wait_gr_sync counts into a single BaseOp, deduplicates barrier ops
        (wait_lr_sync, wait_lr), and collects the rest.
        """
        wait_gr_ops = []
        has_wait_gr_sync = False
        seen_wait_lr = False
        others = []
        for preops in all_preops:
            for op in preops:
                if isinstance(op, WaitGROp) and op.wait_gr_counts:
                    if op.has_sync:
                        has_wait_gr_sync = True
                    wait_gr_ops.append(op.wait_gr_counts)
                elif isinstance(op, WaitLROp):
                    if not seen_wait_lr:
                        seen_wait_lr = True
                        others.append(op)
                else:
                    others.append(op)
        result = []
        if wait_gr_ops:
            merged_counts = WaitGRCounts()
            for t in ('A', 'B', 'SA', 'SB'):
                setattr(merged_counts, t, min(getattr(c, t) for c in wait_gr_ops))
            result.append(WaitGROp(wait_gr_counts=merged_counts,
                                   has_sync=has_wait_gr_sync))
        result.extend(others)
        return result

    def group_lr_gr(self):
        """Group LR and GR placements into chains within each subIterK.

        Phase 1 — LR chain:
          Sort LRs by tensor order (A, B, SA, SB).  Build a dep chain so each
          LR depends on the previous one.  Merge all preOps onto the first LR
          (wait_gr counts are combined, other preOps are collected).

        Phase 2 — GR chain:
          Sort GRs by tensor order (A, B, SA, SB).  Build a dep chain.  If any
          GR originally had same-subIterK deps, replace the first GR's deps with
          a single dep on the last LR of the phase-1 chain.  Each GR keeps its
          own preOps; only redundant wait_lr_sync ops are removed (keep the
          first occurrence only).
        """
        self._ensure_pass(Pass.GR_INC)

        order = self._LR_GR_ORDER

        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                # ── Phase 1: LR chain ──
                ordered_lrs = sorted(
                    slot.lrs,
                    key=lambda lr: order.index(lr.tensor))

                if len(ordered_lrs) > 1:
                    # Merge preOps onto first LR
                    merged = self._merge_preops(
                        [lr.preOps for lr in ordered_lrs])
                    ordered_lrs[0].preOps = merged
                    for lr in ordered_lrs[1:]:
                        lr.preOps = []

                    # Build chain: each LR depends on the previous
                    for i in range(1, len(ordered_lrs)):
                        ordered_lrs[i].deps = [
                            Dep(ref=ordered_lrs[i - 1], mt_offset=0)]

                last_lr = ordered_lrs[-1] if ordered_lrs else None

                # ── Phase 2: GR chain ──
                ordered_grs = sorted(
                    slot.grs,
                    key=lambda gr: order.index(gr.tensor))

                if len(ordered_grs) > 1:
                    # Check if any GR has same-subIterK deps
                    any_deps = any(gr.deps for gr in ordered_grs)

                    # Remove redundant wait_lr_sync (keep only the first)
                    seen_wait_lr_sync = False
                    for gr in ordered_grs:
                        if seen_wait_lr_sync:
                            gr.preOps = [
                                op for op in gr.preOps
                                if not (isinstance(op, WaitLROp) and op.has_sync)]
                        elif any(isinstance(op, WaitLROp) and op.has_sync
                                 for op in gr.preOps):
                            seen_wait_lr_sync = True

                    # First GR: if any GR had deps, point to last LR
                    if any_deps and last_lr is not None:
                        ordered_grs[0].deps = [
                            Dep(ref=last_lr, mt_offset=0)]
                    else:
                        ordered_grs[0].deps = []

                    # Build chain: each GR depends on the previous
                    for i in range(1, len(ordered_grs)):
                        ordered_grs[i].deps = [
                            Dep(ref=ordered_grs[i - 1], mt_offset=0)]
                elif len(ordered_grs) == 1:
                    # Single GR: still consolidate dep to last LR if it had deps
                    if ordered_grs[0].deps and last_lr is not None:
                        ordered_grs[0].deps = [
                            Dep(ref=last_lr, mt_offset=0)]

        self._completed.add(Pass.GROUP_LR_GR)

    def remove_unnecessary_wait_lr_sync(self):
        """Remove redundant wait_lr_sync from GRs after grouping.
        Given that we always use wait_lr cnt=0, grouping can guarantee future wait_lr_sync.

        A GR's wait_lr_sync is unnecessary when:
          1. The GR has no same-subIterK deps (deps is empty after grouping)
          2. The previous subIterK's GRs already have a wait_lr_sync
          3. That previous wait_lr_sync is ordered after all LRs in the
             previous subIterK (the GR has deps on the LR chain)

        In that case, all prior LR reads were already synced by the previous
        subIterK's barrier, and the current GR doesn't conflict with any LRs
        in its own subIterK, so the second wait_lr_sync is redundant.

        Finally, any remaining wait_lr_sync on a GR with no deps is downgraded
        to just sync — the wait_lr is already guaranteed by the MFMA op in the
        same subIterK.
        """
        self._ensure_pass(Pass.GROUP_LR_GR)

        for pi, slots in enumerate(self._partitions):
            for si, slot in enumerate(slots):
                if not slot.grs:
                    continue
                first_gr = slot.grs[0]
                has_wait_lr_sync = any(
                    isinstance(op, WaitLROp) and op.has_sync for op in first_gr.preOps)
                if not has_wait_lr_sync:
                    continue
                has_deps = bool(first_gr.deps)
                if has_deps:
                    continue
                # Check previous subIterK in the same partition
                if si == 0:
                    continue
                prev_slot = slots[si - 1]
                if not prev_slot.grs:
                    continue
                prev_first_gr = prev_slot.grs[0]
                prev_has_wait_lr_sync = any(
                    isinstance(op, WaitLROp) and op.has_sync for op in prev_first_gr.preOps)
                prev_deps_on_lrs = bool(prev_first_gr.deps)
                if prev_has_wait_lr_sync and prev_deps_on_lrs:
                    first_gr.preOps = [
                        op for op in first_gr.preOps
                        if not (isinstance(op, WaitLROp) and op.has_sync)]

        # Downgrade remaining wait_lr_sync → sync on GRs with no LR deps.
        # The MFMA in the same subIterK already ensures wait_lr.
        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                for gr in slot.grs:
                    if not any(isinstance(op, WaitLROp) and op.has_sync for op in gr.preOps):
                        continue
                    has_lr_dep = False
                    node = gr
                    while node and node.deps:
                        ref = node.deps[0].ref
                        if isinstance(ref, LRPlacement):
                            has_lr_dep = True
                            break
                        node = ref
                    if has_lr_dep:
                        continue
                    gr.preOps = [
                        SyncOp() if (isinstance(op, WaitLROp) and op.has_sync) else op
                        for op in gr.preOps]

        self._completed.add(Pass.REMOVE_WAIT_LR_SYNC)

    def _split_deps(self, deps: List[Dep], consumer_pi: int,
                    consumer_slot: int) -> Tuple[List[Dep], List[Dep]]:
        """Split deps into same-subIterK and cross-subIterK lists.

        A dep is "same subIterK" if mt_offset == 0 AND the producer is in the
        same partition and same subIterK slot as the consumer.
        """
        same, cross = [], []
        for dep in deps:
            if (dep.mt_offset == 0 and
                    dep.ref.partition == consumer_pi and
                    dep.ref.subIterK_slot == consumer_slot):
                same.append(dep)
            else:
                cross.append(dep)
        return same, cross

    def emit(self) -> List[List[List[EmittedModule]]]:
        """Convert placements into EmittedModule chains per partition per subIterK.

        Returns [partition][subIterK][EmittedModule].

        Each subIterK list contains:
          - Primary modules (MFMA, LRs, GRs)
          - Dependency modules (wait_gr, wait_lr, sync, lr_inc, gr_inc)
            emitted from preOps, chained via before-links

        The before-link topology:
          - wait_gr is standalone (no incoming before-link), but later deps chain from it
          - WaitGROp with has_sync expands to two modules: wait_gr then sync
          - WaitLROp with has_sync expands to two modules: wait_lr then sync
          - Same-subIterK Dep deps become ordering constraints (no new module)
        """
        self._ensure_pass(Pass.REMOVE_WAIT_LR_SYNC)

        all_partitions = []
        for pi, slots in enumerate(self._partitions):
            partition_emitted = []
            for slot in slots:
                emitted: List[EmittedModule] = []
                placement_to_id = {}

                def add(source: Emittable) -> int:
                    mid = len(emitted)
                    emitted.append(EmittedModule(moduleId=mid, source=source))
                    return mid

                def setBefore(moduleId: int, beforeId: int) -> None:
                    if beforeId is None or beforeId == moduleId:
                        return
                    cur = emitted[moduleId].before
                    if cur is None:
                        emitted[moduleId].before = beforeId
                        return
                    assert cur == beforeId, \
                        f"EmittedModule {moduleId} has multiple before deps: {cur} and {beforeId}"

                # Step 1: emit primary modules
                placements = []
                if slot.mfma:
                    placements.append(slot.mfma)
                for lr in slot.lrs:
                    placements.append(lr)
                for gr in slot.grs:
                    placements.append(gr)

                for placement in placements:
                    mid = add(placement)
                    placement_to_id[id(placement)] = mid

                # Step 2: wire before-chains from preOps + deps
                for placement in placements:
                    curId = placement_to_id[id(placement)]
                    prevId = None
                    lastDepId = None
                    firstPreOpId = None

                    # preOps
                    for preOp in placement.preOps:
                        if isinstance(preOp, WaitGROp):
                            depId = add(preOp)
                            prevId = depId
                            if firstPreOpId is None:
                                firstPreOpId = depId
                            if preOp.has_sync:
                                depId = add(SyncOp())
                                setBefore(depId, prevId)
                                prevId = depId
                                lastDepId = depId
                            continue
                        elif isinstance(preOp, WaitLROp) and preOp.has_sync:
                            depId = add(WaitLROp())
                            setBefore(depId, prevId)
                            prevId = depId
                            lastDepId = depId
                            if firstPreOpId is None:
                                firstPreOpId = depId
                            depId = add(SyncOp())
                            setBefore(depId, prevId)
                            prevId = depId
                            lastDepId = depId
                            continue
                        else:
                            depId = add(preOp)
                            setBefore(depId, prevId)
                            prevId = depId
                            lastDepId = depId
                            if firstPreOpId is None:
                                firstPreOpId = depId

                    # deps (same-subIterK Deps — ordering constraints)
                    # Wire dep refs as roots of the preOp chain so the
                    # dependency is not lost when preOps are present.
                    for dep in placement.deps:
                        ref_id = placement_to_id.get(id(dep.ref))
                        if ref_id is not None:
                            if firstPreOpId is not None:
                                setBefore(firstPreOpId, ref_id)
                            else:
                                prevId = ref_id

                    # Final link: primary module points to last dep
                    if lastDepId is not None:
                        setBefore(curId, lastDepId)
                    elif prevId is not None:
                        setBefore(curId, prevId)

                partition_emitted.append(emitted)
            all_partitions.append(partition_emitted)

        self._emitted = all_partitions
        self._completed.add(Pass.EMIT)
        return all_partitions

    def build(self):
        """Build mainloop """
        self.emit()
        self._completed.add(Pass.BUILD)

    # ── Loop variant derivation ────────────────────────────

    @staticmethod
    def _rewire_before(emitted: List[EmittedModule],
                       removed_ids: set) -> List[EmittedModule]:
        """Rewire before-links that point to removed modules.

        If em.before points to a removed module, follow that module's own
        before link until we find a non-removed module (or None).
        """
        id_to_em = {em.moduleId: em for em in emitted}
        for em in emitted:
            if em.moduleId in removed_ids:
                continue
            b = em.before
            while b is not None and b in removed_ids:
                b = id_to_em[b].before
            em.before = b
        return [em for em in emitted if em.moduleId not in removed_ids]

    def build_ngll(self) -> List[List[List[EmittedModule]]]:
        """NGLL (No Global Load Loop): mainloop without GR(n+2), GR_INC.

        WaitGR inflight counts are zeroed since no new GRs are in flight.
        """
        self._ensure_pass(Pass.EMIT)

        ngll = []
        for partition_emitted in self._emitted:
            part_ngll = []
            for emitted in partition_emitted:
                new_emitted = copy.deepcopy(emitted)
                removed = set()
                for em in new_emitted:
                    src = em.source
                    if em.opType == 'gr' and src.mtIteration == 2:
                        removed.add(em.moduleId)
                    elif em.opType == 'gr_inc':
                        removed.add(em.moduleId)
                    elif em.opType == 'wait_gr':
                        if src.wait_gr_counts is not None:
                            src.wait_gr_counts = WaitGRCounts()
                part_ngll.append(self._rewire_before(new_emitted, removed))
            ngll.append(part_ngll)

        self._ngll_emitted = ngll
        return ngll

    def build_nll(self) -> List[List[List[EmittedModule]]]:
        """NLL (No Load Loop): mainloop without GR, LR(n+1), GR_INC, LR_INC,
        WaitGR(n+1)+Sync. Keeps LR(n), MFMAs, WaitGR(n) with zeroed counts."""
        self._ensure_pass(Pass.EMIT)

        nll = []
        for partition_emitted in self._emitted:
            part_nll = []
            for emitted in partition_emitted:
                new_emitted = copy.deepcopy(emitted)
                removed = set()

                for em in new_emitted:
                    src = em.source
                    if em.opType == 'gr':
                        removed.add(em.moduleId)
                    elif em.opType == 'lr' and src.mtIteration == 1:
                        removed.add(em.moduleId)
                    elif em.opType in ('gr_inc', 'lr_inc'):
                        removed.add(em.moduleId)

                # Zero inflight counts on remaining WaitGR.
                for em in new_emitted:
                    if em.opType == 'wait_gr' and em.moduleId not in removed:
                        em.source.wait_gr_counts = WaitGRCounts()

                # Find Sync modules paired with removed wait_gr
                for em in new_emitted:
                    if em.opType == 'sync' and em.before is not None \
                            and em.before in removed:
                        removed.add(em.moduleId)

                # Remove WaitLR if no LR remains in this subIterK
                has_lr = any(em.opType == 'lr' and em.moduleId not in removed
                             for em in new_emitted)
                if not has_lr:
                    for em in new_emitted:
                        if em.opType == 'wait_lr':
                            removed.add(em.moduleId)

                part_nll.append(self._rewire_before(new_emitted, removed))
            nll.append(part_nll)

        self._nll_emitted = nll
        return nll

    @staticmethod
    def _to_emitted(ops) -> List[EmittedModule]:
        """Wrap Emittable objects (Placements / BaseOps) into EmittedModules."""
        return [EmittedModule(moduleId=mid, source=op) for mid, op in enumerate(ops)]

    def _preloop_make_gr(self, mt: str, tiles: dict) -> List[GRPlacement]:
        """Create GR placements for all tensors at the given MT iteration.

        tiles: {'A': MFMATileRange, 'B': MFMATileRange}
        """
        return [GRPlacement(tensor=tensor, mtIteration=mt,
                            tiles=tiles['A' if tensor in ('A', 'SA') else 'B'],
                            subIterK_slot=0)
                for tensor in self.tensors]

    def _preloop_make_lr(self, tiles: dict) -> List[LRPlacement]:
        """Create LR placements for first partition.

        tiles: per-tensor MFMATileRange, e.g. {'A': MFMATileRange(0, k, mn0, mn1), ...}

        Uses the first MFMA's vgpr tile maps (the preloop loads data consumed
        by the first MFMA, not the next subIterK like mainloop LRs).
        """
        first_mfma = self._partitions[0][0].mfma

        placements = []
        for tensor in self.tensors:
            lr = LRPlacement(
                tensor=tensor, mtIteration=0,
                tiles=tiles[tensor],
                subIterK_slot=0, partition=0)
            if tensor in first_mfma.vgpr_tile_maps:
                lr.vgpr_tile_map = copy.deepcopy(first_mfma.vgpr_tile_maps[tensor])
            placements.append(lr)
        return placements

    def _make_tensor_depops(self, cls) -> List[BaseOp]:
        """Create a BaseOp subclass instance for each tensor."""
        return [cls(tensor=tensor) for tensor in self.tensors]

    def build_preloop(self) -> List[List[List[EmittedModule]]]:
        """Build preloop: pipeline initialization sequence before mainloop.

        High-level sequence (waits/syncs auto-inserted by _insert_preloop_waits):
          GR(MT 0)  — all tensors, all tiles
          GR_INC
          LR        — first partition, subIterK=0
          LR_INC
          skip(LE 1, NLLEarly/NLL)
          GR(MT 1)  — first partition tiles
          GR_INC
          skip(LE 2, NGLL)

        Returns [1 partition][1 subIterK][EmittedModules] to match emit() shape.
        """
        cfg = self.config
        numK = cfg.numSubIterK
        part0 = self._partition_tile_range(0)
        all_tiles = {
            'A': MFMATileRange(0, numK, 0, cfg.numMFMATilesM),
            'B': MFMATileRange(0, numK, 0, cfg.numMFMATilesN),
        }
        part0_tiles = {
            'A': MFMATileRange(0, numK, *part0['A']),
            'B': MFMATileRange(0, numK, *part0['B']),
        }
        lr_tiles = {
            'A':  MFMATileRange(0, cfg.lrA.k, *part0['A']),
            'B':  MFMATileRange(0, cfg.lrB.k, *part0['B']),
        }
        if cfg.hasScale:
            lr_tiles['SA'] = MFMATileRange(0, cfg.lrSA.k, *part0['A'])
            lr_tiles['SB'] = MFMATileRange(0, cfg.lrSB.k, *part0['B'])

        emitted = self._to_emitted([
            *self._preloop_make_gr(0, all_tiles),
            *self._make_tensor_depops(GRIncOp),
            WaitGROp(wait_gr_counts=WaitGRCounts()),
            SyncOp(),
            *self._preloop_make_lr(lr_tiles),
            WaitLROp(),
            SkipOp(compare='LE', value=1, target='NLL'),
            *self._preloop_make_gr(1, part0_tiles),
            # *self._make_tensor_depops(GRIncOp),
            SkipOp(compare='LE', value=2, target='NGLL'),
        ])

        self._preloop_emitted = [[emitted]]
        return self._preloop_emitted

    def _emitLoop(self, writer, kernel, label, emitted_3d):
        """Emit a loop section from a 3D emitted structure.

        emitted_3d: [partition][subIterK][EmittedModule]

        For subIterKs with MFMAs: calls instructionSchedule for interleaving.
        For subIterKs without MFMAs (preloop): emits instructions sequentially.
        """
        from Tensile.Components.Subtile.InstructionScheduler import instructionSchedule
        from rocisa.code import Module

        module = Module(label)
        module.addComment0(f"{label} start")
        for pi, partition_emitted in enumerate(emitted_3d):
            for k, em_list in enumerate(partition_emitted):
                module.addComment0(f"partition={pi} subIterK={k}")
                hasMFMA = any(em.opType == 'mfma' for em in em_list)
                if hasMFMA:
                    scheduled = instructionSchedule(em_list)
                    module.add(scheduled)
                else:
                    for em in em_list:
                        for inst in em.instructions:
                            module.add(inst)
        module.addComment0(f"{label} end")
        return module

    def emitAllLoops(self, writer, kernel):
        """Emit complete loop structure: preloop + mainloop + NGLL + NLL.

        Owns all control flow (labels, branches, counter management).
        For unroll_factor > 1, emits per-unroll copies with correct vgpr tiles.
        Each mainloop exit jumps to its corresponding NGLL→NLL pair.
        """
        from rocisa.code import Module, Label
        from rocisa.instruction import (SSubU32, SCmpEQU32, SCBranchSCC0,
                                        SCBranchSCC1, SBranch)
        from rocisa.container import sgpr

        assert Pass.POPULATE in self._completed, \
            "populate_instructions() must be called before emitAllLoops()"

        module = Module("AllLoops")
        uf = self.unroll_factor

        # ── Preloop ──
        module.add(self._emitLoop(writer, kernel, "PRELOOP",
                                  self._preloop_emitted))

        # ── Mainloop ──
        module.addComment0("MAINLOOP")
        loopBegin = Label("LoopBeginL", "")

        if uf == 1:
            module.add(loopBegin)
            module.add(self._emitLoop(writer, kernel, "MAINLOOP",
                                      self._emitted_per_unroll[0]))
            module.add(SSubU32(dst=sgpr("LoopCounterL"),
                               src0=sgpr("LoopCounterL"), src1=1,
                               comment="dec counterL"))
            module.add(SCmpEQU32(src0=sgpr("LoopCounterL"), src1=2,
                                 comment="counterL == 2?"))
            module.add(SCBranchSCC0(labelName=loopBegin.getLabelName(),
                                    comment="restart mainloop"))
        else:
            exitLabels = [Label(f"ExitC{ui}", "") for ui in range(uf - 1)]
            module.add(loopBegin)
            for ui in range(uf):
                module.add(self._emitLoop(writer, kernel, f"MAINLOOP_C{ui}",
                                          self._emitted_per_unroll[ui]))
                module.add(SSubU32(dst=sgpr("LoopCounterL"),
                                   src0=sgpr("LoopCounterL"), src1=1,
                                   comment=f"dec counterL (copy {ui})"))
                module.add(SCmpEQU32(src0=sgpr("LoopCounterL"), src1=2,
                                     comment=f"counterL == 2? (copy {ui} exit)"))
                if ui < uf - 1:
                    module.add(SCBranchSCC1(
                        labelName=exitLabels[ui].getLabelName(),
                        comment=f"copy {ui} exit → NGLL_C{ui}"))
                else:
                    module.add(SCBranchSCC0(
                        labelName=loopBegin.getLabelName(),
                        comment="restart mainloop"))

        # ── NGLL + NLL exit paths ──
        endLabel = Label("SkipToEnd", "")
        module.add(Label("SkipMainloop", ""))
        module.add(Label("SkipToNGLL", ""))

        if uf == 1:
            module.addComment0("NGLL")
            module.add(self._emitLoop(writer, kernel, "NGLL",
                                      self._ngll_per_unroll[0]))
            module.addComment0("NLL")
            module.add(Label("SkipToNLL", ""))
            module.add(self._emitLoop(writer, kernel, "NLL",
                                      self._nll_per_unroll[0]))
        else:
            # Fall-through from last mainloop copy
            last = uf - 1
            module.addComment0(f"NGLL_C{last}")
            module.add(self._emitLoop(writer, kernel, f"NGLL_C{last}",
                                      self._ngll_per_unroll[last]))
            module.addComment0(f"NLL_C{last}")
            module.add(self._emitLoop(writer, kernel, f"NLL_C{last}",
                                      self._nll_per_unroll[last]))
            module.add(SBranch(labelName=endLabel.getLabelName(),
                               comment="skip other exit paths"))

            for ui in range(uf - 1):
                module.add(exitLabels[ui])
                module.addComment0(f"NGLL_C{ui}")
                module.add(self._emitLoop(writer, kernel, f"NGLL_C{ui}",
                                          self._ngll_per_unroll[ui]))
                module.addComment0(f"NLL_C{ui}")
                module.add(self._emitLoop(writer, kernel, f"NLL_C{ui}",
                                          self._nll_per_unroll[ui]))
                if ui < uf - 2:
                    module.add(SBranch(labelName=endLabel.getLabelName(),
                                       comment="skip other exit paths"))

            # NLLEarly: reached from preloop when LoopCounterL <= 1
            module.add(SBranch(labelName=endLabel.getLabelName(),
                               comment="skip NLLEarly"))
            module.addComment0("NLLEarly")
            module.add(Label("SkipToNLL", ""))
            module.add(self._emitLoop(writer, kernel, "NLLEarly",
                                      self._nll_per_unroll[0]))
            module.add(endLabel)

        return module

    # ── VGPR tile allocation ──────────────────────────────

    def getNumVgpr(self, tileInfoA, tileInfoB,
                        scaleTileInfoA=None, scaleTileInfoB=None) -> int:
        """Return the total number of VGPRs needed across all tensors (A, B, SA, SB)
        without performing any allocation.

        Must be called after scheduling is complete.
        """
        self._ensure_pass(Pass.VGPR_TILES)

        cfg = self.config

        def _tile_vgpr_count(tileInfo, lrGran):
            return int(math.ceil(tileInfo.mmaTileRegCount * lrGran.k * lrGran.mn))

        total = self.tile_peaks.get('A', 0) * _tile_vgpr_count(tileInfoA, cfg.lrA) \
              + self.tile_peaks.get('B', 0) * _tile_vgpr_count(tileInfoB, cfg.lrB)

        if cfg.hasScale and scaleTileInfoA and scaleTileInfoB:
            total += self.tile_peaks.get('SA', 0) * _tile_vgpr_count(scaleTileInfoA, cfg.lrSA) \
                   + self.tile_peaks.get('SB', 0) * _tile_vgpr_count(scaleTileInfoB, cfg.lrSB)

        return total

    def allocVgprTiles(self, writer, tileInfoA, tileInfoB,
                       scaleTileInfoA=None, scaleTileInfoB=None):
        """Allocate physical VGPR tiles based on assign_vgpr_tiles() peaks.

        Each vgprTile holds one LR granularity worth of data:
          size = ceil(mmaTileRegCount * lrGranularity.k * lrGranularity.mn)

        Ex: 4 VGPRs for A/B for 1 MFMATile, and 1 VGPR for a 2x2 MFMA tile for SA/SB if hasScale.

        Produces per-tensor lists indexed by vgprTileId:
          vgprTilesA/B:   List[RegisterTileInfo]
          vgprTilesSA/SB: List[RegisterTileInfo]
        """
        self._ensure_pass(Pass.VGPR_TILES)

        from Tensile.Components.Subtile.Kernel import RegisterTileInfo

        cfg = self.config

        def _tile_vgpr_count(tileInfo, lrGran):
            return int(math.ceil(tileInfo.mmaTileRegCount * lrGran.k * lrGran.mn))

        def _alloc_tiles(count, numRegs):
            tiles = []
            for _ in range(count):
                tile = RegisterTileInfo(writer.vgprPool)
                for j in range(0, numRegs, 4):
                    blockSize = min(4, numRegs - j)
                    vstart = writer.vgprPool.checkOutAligned(blockSize, blockSize)
                    for k in range(blockSize):
                        tile.append(vstart + k)
                tiles.append(tile)
            return tiles

        self.vgprTilesA = _alloc_tiles(self.tile_peaks.get('A', 0),
                                       _tile_vgpr_count(tileInfoA, cfg.lrA))
        self.vgprTilesB = _alloc_tiles(self.tile_peaks.get('B', 0),
                                       _tile_vgpr_count(tileInfoB, cfg.lrB))

        if cfg.hasScale and scaleTileInfoA and scaleTileInfoB:
            self.vgprTilesSA = _alloc_tiles(self.tile_peaks.get('SA', 0),
                                            _tile_vgpr_count(scaleTileInfoA, cfg.lrSA))
            self.vgprTilesSB = _alloc_tiles(self.tile_peaks.get('SB', 0),
                                            _tile_vgpr_count(scaleTileInfoB, cfg.lrSB))
        else:
            self.vgprTilesSA = []
            self.vgprTilesSB = []

    def deallocVgprTiles(self, writer):
        """Deallocate VGPR tiles allocated by allocVgprTiles."""
        def _dealloc_tiles(tiles):
            for tile in tiles:
                pool = tile.regList.pool
                for val in tile:
                    if tile.index(val) % 4 == 0:
                        pool.checkIn(val)

        _dealloc_tiles(self.vgprTilesA)
        _dealloc_tiles(self.vgprTilesB)
        _dealloc_tiles(self.vgprTilesSA)
        _dealloc_tiles(self.vgprTilesSB)
        self.vgprTilesA = []
        self.vgprTilesB = []
        self.vgprTilesSA = []
        self.vgprTilesSB = []

    # ── Populate instructions ──────────────────────────────

    def populate_instructions(self, writer, kernel,
                              tileInfoA, tileInfoB, dtileInfo,
                              scaleTileInfoA=None, scaleTileInfoB=None) -> None:
        """Populate EmittedModule.instructions from placements and preOps.

        Uses per-tensor VGPR tile lists (vgprTilesA/B/SA/SB) indexed by
        vgprTileId from placement tile maps.
        """
        if self._preloop_emitted is None or self._ngll_emitted is None \
                or self._nll_emitted is None:
            self.build()

        from Tensile.Components.Subtile.InstructionEmitter import InstructionEmitter

        emitter = InstructionEmitter(
            writer, kernel, self.config,
            tileInfoA, tileInfoB, dtileInfo,
            self.vgprTilesA, self.vgprTilesB,
            scaleTileInfoA, scaleTileInfoB,
            self.vgprTilesSA, self.vgprTilesSB,
        )

        # Rebuild all loop variants from current _emitted (which now has
        # vgpr_tile_maps populated by assign_vgpr_tiles, unlike the stale
        # copies from build()).
        self.build_preloop()
        self.build_ngll()
        self.build_nll()

        emitter.populate(self._preloop_emitted, unroll_iter=0)

        self._emitted_per_unroll = []
        self._ngll_per_unroll = []
        self._nll_per_unroll = []
        for ui in range(self.unroll_factor):
            em_copy = copy.deepcopy(self._emitted)
            emitter.populate(em_copy, unroll_iter=ui)
            self._emitted_per_unroll.append(em_copy)

            ngll_copy = copy.deepcopy(self._ngll_emitted)
            ngll_ui = (ui + 1) % self.unroll_factor
            emitter.populate(ngll_copy, unroll_iter=ngll_ui)
            self._ngll_per_unroll.append(ngll_copy)

            nll_copy = copy.deepcopy(self._nll_emitted)
            nll_ui = (ui + 2) % self.unroll_factor
            emitter.populate(nll_copy, unroll_iter=nll_ui)
            self._nll_per_unroll.append(nll_copy)

        self._completed.add(Pass.POPULATE)

    # ── Print helpers ───────────────────────────────────────

    @staticmethod
    def _fmt_tensor(tensor: str) -> str:
        """Pad tensor name to 2 chars for alignment: 'A' -> 'A ', 'SA' -> 'SA'."""
        return tensor.ljust(2)


    def print_lr(self, partitions: List[List[SubIterKSlot]] = None) -> str:
        """Print place_LRs output in design doc format."""
        if partitions is None:
            partitions = self._partitions
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(partitions):
            buf.write(f"  Partition {pi}:\n")
            self._print_lr_partition(buf, slots)
        return buf.getvalue()

    def _print_lr_partition(self, buf, slots):
        for slot in slots:
            buf.write(f"    subIterK={slot.subIterK}:\n")
            if slot.mfma:
                m = slot.mfma
                buf.write(f"      MFMAs (MT n, subIterK {m.subIterK}  ) "
                          f"A : {m.tileA.fmt_tiles()} , B : {m.tileB.fmt_tiles()}\n")
            for lr in slot.lrs:
                t = self._fmt_tensor(lr.tensor)
                buf.write(f"      LR {t} (MT {fmt_mt(lr.mtIteration)}, "
                          f"subIterK {lr.tiles.fmt_k()}) "
                          f"{lr.tiles.fmt_tiles()}\n")
        return buf.getvalue()

    def print_vgpr(self) -> str:
        """Print assign_vgpr_tiles output: LRs + MFMAs with vgprTileId annotations."""
        partitions = self._partitions
        buf = io.StringIO()
        buf.write(f"needsUnrolling: {self.needs_unrolling}, "
                  f"unrollFactor: {self.unroll_factor}\n")
        peaks_str = ", ".join(f"{t}: {cnt}" for t, cnt in sorted(self.tile_peaks.items()))
        buf.write(f"vgprTiles: {peaks_str}\n")
        for ui in range(self.unroll_factor):
            if self.unroll_factor > 1:
                buf.write(f"MAINLOOP (unroll {ui}):\n")
            else:
                buf.write("MAINLOOP:\n")
            for pi, slots in enumerate(partitions):
                buf.write(f"  Partition {pi}:\n")
                for slot in slots:
                    buf.write(f"    subIterK={slot.subIterK}:\n")
                    if slot.mfma:
                        m = slot.mfma
                        tiles_str = ""
                        parts = []
                        for tensor in self.tensors:
                            maps = m.vgpr_tile_maps.get(tensor)
                            if maps:
                                parts.append(f"{tensor}:" + str(maps[ui]))
                        if parts:
                            tiles_str = " " + ", ".join(parts)
                        buf.write(f"      MFMAs (MT n, subIterK {m.subIterK}  ) "
                                  f"A : {m.tileA.fmt_tiles()} , "
                                  f"B : {m.tileB.fmt_tiles()}{tiles_str}\n")
                    for lr in slot.lrs:
                        tile_str = ""
                        if lr.vgpr_tile_map:
                            tile_str = f" tiles:{lr.vgpr_tile_map[ui]}"
                        t = self._fmt_tensor(lr.tensor)
                        buf.write(f"      LR {t} (MT {fmt_mt(lr.mtIteration)}, "
                                  f"subIterK {lr.tiles.fmt_k()}) "
                                  f"{lr.tiles.fmt_tiles()}{tile_str}\n")
        return buf.getvalue()

    def print_gr(self) -> str:
        """Print place_GRs output: LRs + MFMAs + GR placements, all partitions."""
        partitions = self._partitions
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(partitions):
            buf.write(f"  Partition {pi}:\n")
            for slot in slots:
                buf.write(f"    subIterK={slot.subIterK}:\n")
                if slot.mfma:
                    m = slot.mfma
                    buf.write(f"      MFMAs (MT n, subIterK {m.subIterK}  ) "
                              f"A : {m.tileA.fmt_tiles()} , "
                              f"B : {m.tileB.fmt_tiles()}\n")
                for lr in slot.lrs:
                    t = self._fmt_tensor(lr.tensor)
                    buf.write(f"      LR {t} (MT {fmt_mt(lr.mtIteration)}, "
                              f"subIterK {lr.tiles.fmt_k()}) "
                              f"{lr.tiles.fmt_tiles()}\n")
                for gr in slot.grs:
                    buf.write(f"      GR {gr.tensor} (MT {fmt_mt(gr.mtIteration)}, "
                              f"subIterK {gr.tiles.fmt_k()}) "
                              f"ids {gr.tiles.fmt_tiles()}\n")
        return buf.getvalue()

    def print_deps(self) -> str:
        """Print annotate_deps output: placements with their before-dependencies."""
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(self._partitions):
            buf.write(f"  Partition {pi}:\n")
            for slot in slots:
                buf.write(f"    subIterK={slot.subIterK}:\n")
                if slot.mfma:
                    self._print_placement_with_deps(buf, slot.mfma, slot)
                for lr in slot.lrs:
                    self._print_placement_with_deps(buf, lr, slot)
                for gr in slot.grs:
                    self._print_placement_with_deps(buf, gr, slot)
        return buf.getvalue()

    def _print_placement_with_deps(self, buf, placement, slot: SubIterKSlot):
        """Print a placement label followed by its deps."""
        buf.write(f"      {placement}\n")
        if placement.deps:
            buf.write("        deps:\n")
            for dep in placement.deps:
                dep_str = self._format_dep_ref(dep)
                buf.write(f"            - {dep_str}\n")

    def print_remove_deps(self) -> str:
        """Print remove_cross_deps output: placements with preOps and remaining deps."""
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(self._partitions):
            buf.write(f"  Partition {pi}:\n")
            for slot in slots:
                buf.write(f"    subIterK={slot.subIterK}:\n")
                if slot.mfma:
                    self._print_placement_with_preops(buf, slot.mfma, slot)
                for lr in slot.lrs:
                    self._print_placement_with_preops(buf, lr, slot)
                for gr in slot.grs:
                    self._print_placement_with_preops(buf, gr, slot)
        return buf.getvalue()

    def print_group_lr_gr(self) -> str:
        """Print group_lr_gr output: placements with chained deps and merged preOps."""
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(self._partitions):
            buf.write(f"  Partition {pi}:\n")
            for slot in slots:
                buf.write(f"    subIterK={slot.subIterK}:\n")
                if slot.mfma:
                    self._print_placement_with_preops(buf, slot.mfma, slot)
                for lr in slot.lrs:
                    self._print_placement_with_preops(buf, lr, slot)
                for gr in slot.grs:
                    self._print_placement_with_preops(buf, gr, slot)
        return buf.getvalue()

    def _print_placement_with_preops(self, buf, placement, slot: SubIterKSlot):
        """Print a placement label followed by its preOps and remaining deps."""
        buf.write(f"      {placement}\n")
        if placement.preOps:
            buf.write("        preOps:\n")
            for op in placement.preOps:
                buf.write(f"            - {op}\n")
        if placement.deps:
            buf.write("        deps:\n")
            for dep in placement.deps:
                dep_str = self._format_dep_ref(dep)
                buf.write(f"            - {dep_str}\n")
    

    def _format_dep_ref(self, dep: Dep) -> str:
        """Format a Dep for display."""
        p = dep.ref
        slot = p.subIterK_slot if hasattr(p, 'subIterK_slot') else '?'
        part = p.partition if hasattr(p, 'partition') else 0
        kind = 'LR' if isinstance(p, LRPlacement) else 'GR'
        mt = f" (MT{dep.mt_offset})" if dep.mt_offset != 0 else ""
        return f"{kind} {p.tensor} @P{part}:subIterK={slot}{mt}"


    def print_emit(self, all_partitions: List[List[List[EmittedModule]]] = None) -> str:
        """Print emit output: EmittedModule list with before-links."""
        if all_partitions is None:
            all_partitions = self._emitted
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, partition_emitted in enumerate(all_partitions):
            buf.write(f"  Partition {pi}:\n")
            for k, emitted in enumerate(partition_emitted):
                buf.write(f"    subIterK={k}:\n")
                for em in emitted:
                    before_str = f" <- [{em.before}]" if em.before is not None else ""
                    buf.write(f"      [{em.moduleId:2d}] {em.opType:10s} {em.source}{before_str}\n")
        return buf.getvalue()

    def print_emit_dep_order(self, all_partitions: List[List[List[EmittedModule]]] = None) -> str:
        """Print emit output as dependency paths (same decomposition as _extractPathsFromBeforeDeps)."""
        from Tensile.Components.Subtile.InstructionScheduler import extractPathsFromBeforeDeps
        if all_partitions is None:
            all_partitions = self._emitted
        buf = io.StringIO()
        buf.write("MAINLOOP (dependency paths):\n")
        for pi, partition_emitted in enumerate(all_partitions):
            buf.write(f"  Partition {pi}:\n")
            for k, emitted in enumerate(partition_emitted):
                buf.write(f"    subIterK={k}:\n")
                mfmaIdx, paths, preMfmaPaths = extractPathsFromBeforeDeps(emitted)
                em = emitted[mfmaIdx]
                buf.write(f"      MFMA: [{em.moduleId:2d}] {em.source}")
                if em.before is not None:
                    buf.write(f" <- [{em.before}]")
                buf.write("\n")
                for i, path in enumerate(preMfmaPaths):
                    buf.write(f"      preMFMA path {i}:\n")
                    for idx in path:
                        buf.write(f"        [{emitted[idx].moduleId:2d}] {emitted[idx].opType:10s} {emitted[idx].source}\n")
                for i, path in enumerate(paths):
                    buf.write(f"      path {i}:\n")
                    for idx in path:
                        buf.write(f"        [{emitted[idx].moduleId:2d}] {emitted[idx].opType:10s} {emitted[idx].source}\n")
        return buf.getvalue()
