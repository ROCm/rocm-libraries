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
from typing import Callable, Dict, List, Optional, Tuple, Union
from bisect import bisect_left
import copy
import io
import math

from rocisa.code import Module

from ...Common.GlobalParameters import globalParameters

# ds_load_b128 reads 4 contiguous VGPRs.
DS_B128_VGPRS = 4

def _checkout_tile(pool, numRegs, tag):
    """Check out one VGPR tile as a single contiguous, min(numRegs, 4)-aligned block (b128-aligned when numRegs >= 4)."""
    from Tensile.Components.Subtile.Kernel import RegisterTileInfo
    # min(): full b128 tiles get 4-VGPR alignment; smaller tiles aren't padded
    # up to 4 (which would waste registers and can break occupancy).
    align = min(numRegs, DS_B128_VGPRS)
    base = pool.checkOutAligned(numRegs, align, tag=tag)
    tile = RegisterTileInfo(pool)
    for k in range(numRegs):
        tile.append(base + k)
    return tile


def _checkin_tile(tile):
    """Return a contiguous tile to its pool via its base-register handle."""
    tile.regList.pool.checkIn(tile.regList.indices[0])


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

    def tile_range(self, k: int, t_start: int, t_end: int) -> 'MFMATileRange':
        """Snap subIterK and tile indices to this granularity, return MFMATileRange."""
        ks = (k // self.k) * self.k
        ts = (t_start // self.mn) * self.mn
        te = ((t_end + self.mn - 1) // self.mn) * self.mn
        return MFMATileRange(ks, ks + self.k, ts, te)


class GRPlacementStrategy(IntEnum):
    """How Global Reads are spread across (partition, subIterK) slots.

    SPREAD   — distribute GR atoms across all slots weighted by partition
               MFMA count. Default for buffer-load GR (gfx9xx): spreading
               avoids issue-slot pressure on the SIMD.
    BUNCHED  — pin every GR atom to partition 0, subIterK 0. Suitable when
               the GR instruction does not contend for SIMD issue slots
               (e.g. TDM tensor_load_to_lds on gfx1250).
    """
    SPREAD  = 0
    BUNCHED = 1


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
    partitionSizeM: Union[int, List[int]] = 0  # partition size(s) in M dimension (0 = full dim)
    partitionSizeN: Union[int, List[int]] = 0  # partition size(s) in N dimension (0 = full dim)
    pgr: int = 2              # Prefetch Global Read
    grPlacement: GRPlacementStrategy = GRPlacementStrategy.SPREAD

    # Resolve a partition spec into per-partition sizes along one dimension.
    # spec is either:
    #  - an explicit list (must sum to total)
    #  - a single tile size (0 means full dim).
    # Uneven splits place the remainder in the middle so the smaller partition is bracketed by full ones.
    #
    # Every partition size must be a multiple of `mn` (LR read granularity) to avoid emitting under-sized LRs.
    # Single-int specs are rounded DOWN to an mn-multiple (smaller partition, less VGPR usage).
    # If no solution exists, we return [total] (single partition).
    @staticmethod
    def _normalize_partition_sizes(spec: Union[int, List[int]], total: int, dim: str, mn: int = 1) -> List[int]:
        if isinstance(spec, (list, tuple)):
            assert sum(spec) == total, \
                f"partition sizes for {dim} must sum to {total}, got {sum(spec)}"
            assert all(s >= 1 for s in spec), \
                f"all partition sizes for {dim} must be >= 1"
            assert all(s % mn == 0 for s in spec), \
                f"partition sizes for {dim} must be multiples of mn={mn}, got {list(spec)}"
            return list(spec)
        s = spec if spec != 0 else total
        assert 1 <= s <= total, \
            f"partition size for {dim} must be in [1, {total}], got {s}"
        if total % mn != 0:
            return [total]
        s = max(mn, (s // mn) * mn)
        if s > total:
            return [total]
        num_full = total // s
        remainder = total - num_full * s
        if remainder == 0:
            return [s] * num_full
        if num_full == 1:
            return [s, remainder]
        mid = num_full // 2
        return [s] * mid + [remainder] + [s] * (num_full - mid)

    @staticmethod
    def _build_prefix(sizes: List[int]) -> List[int]:
        prefix = [0]
        for s in sizes:
            prefix.append(prefix[-1] + s)
        return prefix

    def __post_init__(self):
        assert self.pgr in (0, 1, 2), f"pgr must be 0, 1, or 2, got {self.pgr}"
        mn_M = max((g.mn for g in (self.lrA, self.lrSA) if g is not None), default=1)
        mn_N = max((g.mn for g in (self.lrB, self.lrSB) if g is not None), default=1)
        self._partitionSizesM = self._normalize_partition_sizes(
            self.partitionSizeM, self.numMFMATilesM, 'M', mn_M)
        self._partitionSizesN = self._normalize_partition_sizes(
            self.partitionSizeN, self.numMFMATilesN, 'N', mn_N)
        self._prefixM = self._build_prefix(self._partitionSizesM)
        self._prefixN = self._build_prefix(self._partitionSizesN)
        self.plr = 0 if self.pgr == 0 else 1
        # Forcing offsetPartition to 1.
        self.offsetPartition = 1 if self.pgr >= 2 else 0
        if self.pgr == 0:
            assert self.numPartitions == 1, "pgr=0 requires numPartitions=1"

    @property
    def partitionSizesM(self) -> List[int]:
        return self._partitionSizesM

    @property
    def partitionSizesN(self) -> List[int]:
        return self._partitionSizesN

    @property
    def hasScale(self) -> bool:
        return self.lrSA is not None and self.lrSB is not None

    @property
    def numPartitionsM(self) -> int:
        return len(self._partitionSizesM)

    @property
    def numPartitionsN(self) -> int:
        return len(self._partitionSizesN)

    @property
    def numPartitions(self) -> int:
        return self.numPartitionsM * self.numPartitionsN

    @staticmethod
    def get_partition_candidates(tileInfoA, tileInfoB) -> list:
        """Return partition candidates as [(partitionSizeM, partitionSizeN), ...].

        For the smaller dimension, uses a single partition (full size).
        For the larger dimension, starts at full size then jumps to divUp(dim,2)
        and decrements from there, skipping unbalanced 2-partition sizes.
        """
        M = tileInfoA.localMMATileGrid[0]
        N = tileInfoB.localMMATileGrid[0]

        def divUp(n, d):
            return (n + d - 1) // d

        def partitionSizes(dim):
            return [dim] + list(range(divUp(dim, 2), 0, -1))

        if N >= M:
            candidates = [(M, s) for s in partitionSizes(N)]
        else:
            candidates = [(s, N) for s in partitionSizes(M)]

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
    postOps: List['BaseOp'] = field(default_factory=list)    # populated by insert_gr_lr_inc()
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
    postOps: List['BaseOp'] = field(default_factory=list)    # populated by insert_gr_lr_inc()
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
    mtIteration: int           # 0 = current MT, 1 = next MT, 2 = two MTs ahead
    tiles: MFMATileRange
    subIterK_slot: int         # which subIterK this GR is placed in
    partition: int = 0         # which partition this GR belongs to
    deps: List['Dep'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['BaseOp'] = field(default_factory=list)     # populated by remove_cross_deps()
    postOps: List['BaseOp'] = field(default_factory=list)    # populated by insert_gr_lr_inc()

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
    adjustVmcnt: bool = True

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

    def __str__(self):
        return 'wait_lr_sync' if self.has_sync else 'wait_lr'


@dataclass
class SyncOp(BaseOp):
    """Standalone sync barrier."""
    def __post_init__(self):
        self.kind = 'sync'


@dataclass
class MaskKOp(BaseOp):
    """Zero A and B vgprs whose K-index >= remaining
    tail K, for one subIterK group.
    """
    subIterK: int = 0
    vgpr_tile_map: dict = field(default_factory=dict)

    def __post_init__(self):
        self.kind = 'mask_k'

    def __str__(self):
        return f"mask_k(k={self.subIterK})"


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
    """Skip guard: compare LoopCounter and branch.

    target is normally a short name (e.g. 'NLL'); the emitter prefixes 'SkipTo'.
    Set rawLabel=True to pass the label name through verbatim
    (e.g. 'SkipTailLoopL'). branchComment overrides the default."""
    compare: str = ""
    value: int = 0
    target: str = ""
    rawLabel: bool = False
    branchComment: str = ""

    def __post_init__(self):
        self.kind = 'skip'

    @property
    def tensor(self) -> str:
        return f"{self.compare}:{self.value}:{self.target}"

    def __str__(self):
        return f"skip({self.tensor})"


@dataclass
class InlineModuleOp(BaseOp):
    """Inline a writer-built Module at this point in the schedule.

    The callback receives the InstructionEmitter (so it can reach writer,
    kernel, tensorParametersMap, etc.) and must return a rocisa Module.
    Use this for one-off boilerplate that doesn't deserve its own Op class."""
    build: Optional[Callable] = None
    label: str = "inline"

    def __post_init__(self):
        self.kind = 'inline'

    def __str__(self):
        return f"inline({self.label})"


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
        # Tail-loop tile bookkeeping. Tail loop only use a subset of tiles, so we track which tileIds are
        # unused or freed for reuse within the tail loop.
        self._tail_unused_tile_ids: Dict[str, set] = {'A': set(), 'B': set(),
                                                      'SA': set(), 'SB': set()}
        self._tail_freed_tile_ids: Dict[str, set] = {'A': set(), 'B': set(),
                                                     'SA': set(), 'SB': set()}

    def _ensure_pass(self, *prerequisites: Pass) -> None:
        for p in prerequisites:
            if p not in self._completed:
                getattr(self, _PASS_PIPELINE[p][0])()

    # ── Place LRs ─────────────────────────────────────────

    def _partition_tile_range(self, pi: int) -> dict:
        """Return {'A': (start, end), 'B': (start, end)} for partition pi.

        Uses COLUMN_MAJOR ordering: M (A) varies fastest, N (B) varies slowest.
        Tile ranges are derived from prefix sums of partition sizes.
        """
        cfg = self.config
        piM = pi % cfg.numPartitionsM
        piN = pi // cfg.numPartitionsM
        return {'A': (cfg._prefixM[piM], cfg._prefixM[piM + 1]),
                'B': (cfg._prefixN[piN], cfg._prefixN[piN + 1])}

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
        if self.config.plr == 0:
            return self._place_LRs_PLR0()

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

    def _create_partition_slots(self, cur: dict) -> List[SubIterKSlot]:
        """Create SubIterKSlots with MFMAs placed for one partition."""
        numK = self.config.numSubIterK
        slots = [SubIterKSlot(subIterK=k) for k in range(numK)]
        for k in range(numK):
            slots[k].mfma = MFMAPlacement(
                subIterK=k,
                tileA=MFMATileRange(k, k + 1, cur['A'][0], cur['A'][1]),
                tileB=MFMATileRange(k, k + 1, cur['B'][0], cur['B'][1]),
            )
        return slots

    def _lr_tensors(self) -> list:
        """Return list of (tensor_name, ReadGranularity) for all LR tensors."""
        cfg = self.config
        tensors = [('A', cfg.lrA), ('B', cfg.lrB)]
        if cfg.hasScale:
            tensors.append(('SA', cfg.lrSA))
            tensors.append(('SB', cfg.lrSB))
        return tensors

    def _place_LRs_PLR0(self) -> List[List[SubIterKSlot]]:
        """Place MFMAs and LRs for PLR=0: no prefetching.

        Each LR loads data for its own subIterK. All LRs have mtIteration=0.
        Single partition only (enforced by config validation).
        """
        cfg = self.config
        numK = cfg.numSubIterK
        cur = self._partition_tile_range(0)
        slots = self._create_partition_slots(cur)

        for tensor, gran in self._lr_tensors():
            side_key = 'A' if tensor in ('A', 'SA') else 'B'
            ts, te = cur[side_key]
            k_gran = gran.k
            num_chunks = numK // k_gran
            for chunk_idx in range(num_chunks):
                lr_k_start = chunk_idx * k_gran
                lr_k_end = lr_k_start + k_gran
                slot_k = lr_k_start
                lr = LRPlacement(
                    tensor=tensor,
                    mtIteration=0,
                    tiles=MFMATileRange(lr_k_start, lr_k_end, ts, te),
                    subIterK_slot=slot_k,
                )
                slots[slot_k].lrs.append(lr)

        self._partitions = [slots]
        self._completed.add(Pass.LR)
        return self._partitions

    def _place_LRs_for_partition(self, cur: tuple, nxt: tuple,
                                  is_last: bool,
                                  load: dict,
                                  placed: set) -> List[SubIterKSlot]:
        """Place MFMAs and LRs for one partition."""
        cfg = self.config
        numK = cfg.numSubIterK
        multi_part = cfg.numPartitions > 1

        slots = self._create_partition_slots(cur)
        slot_mt = {}  # slot_k → lr_mt string, for MT-homogeneity enforcement

        all_tensors = self._lr_tensors()

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

        Deterministic double-buffer allocator.  Each tensor gets two sets
        of vgprTiles, each of size max_groups.  The active set alternates
        based on the K-group index and macro-tile iteration:

          set = (mt_iter * num_k_groups + k // gran.k) % 2

        The position within a set depends on the number of K-chunks per tensor
        (num_k_groups = numSubIterK // gran.k):

        - Multi-K-chunk tensors (e.g. BF16, nkg≥2): per-partition positions.
          Groups in each partition are indexed 0, 1, … locally; max_groups is
          the largest partition group count.  Aliasing across partitions is safe
          because the LR and MFMA in the same slot always use different set_idx.

        - Single-K-chunk tensors (e.g. FP8, nkg=1) with multiple partitions:
          global positions.  Every unique group across all partitions gets a
          distinct position index.  This is required because every LR is a
          "wrapping" LR (loads the next partition's tiles) and shares the same
          set_idx as the MFMA in its slot; per-partition aliasing would map
          the LR write onto the same VGPRs the MFMA is reading.

        Unrolling (factor 2) is applied when num_k_groups is odd for any
        tensor, because the set parity would flip across macro-tile boundaries.
        PGR=0 suppresses unrolling and collapses to a single set.

        Sets self.tile_peaks, self.needs_unrolling, self.unroll_factor.
        """
        self._ensure_pass(Pass.LR)

        cfg = self.config
        numK = cfg.numSubIterK
        numP = cfg.numPartitions

        lr_grans = {'A': cfg.lrA, 'B': cfg.lrB}
        if cfg.hasScale:
            lr_grans['SA'] = cfg.lrSA
            lr_grans['SB'] = cfg.lrSB

        # ── Precompute group mappings across partitions ──
        part_ranges = [self._partition_tile_range(pi) for pi in range(numP)]

        # When any tensor has exactly one K-chunk (e.g. FP8 with numSubIterK=1,
        # gran.k=1 → nkg=1), every LR is a "wrapping" LR (is_wrap=True in
        # _place_LRs_for_partition), so the LR in slot Pi always loads the next
        # partition's tiles.  That LR and the MFMA in the same slot share the
        # same set_idx, so per-partition aliasing (different partitions reusing
        # position 0, 1, …) would map the next-partition LR data onto the same
        # VGPR tile IDs as the current-partition MFMA data → silent corruption.
        # Fix: use globally unique positions so the LR writes to different VGPRs.
        #
        # For tensors with multiple K-chunks (e.g. BF16, nkg≥2) the LR and MFMA
        # in the same slot always differ in set_idx, so per-partition aliasing is
        # safe and saves VGPRs — keep it for those cases.
        any_single_k_chunk = any(
            numK // lr_grans[t].k == 1 for t in self.tensors)
        use_global_pos = (numP > 1) and any_single_k_chunk

        # group_to_pos[tensor][group] = position (globally unique or local-within-partition)
        group_to_pos = {t: {} for t in self.tensors}
        max_groups = {t: 0 for t in self.tensors}

        if use_global_pos:
            # Global positions: each unique group across all partitions gets a
            # distinct position index, so LR and MFMA tile IDs never collide.
            for pi in range(numP):
                for tensor in self.tensors:
                    side = TENSOR_SIDE[tensor]
                    start, end = part_ranges[pi][side]
                    gran = lr_grans[tensor]
                    groups = sorted(set(
                        (t // gran.mn) * gran.mn for t in range(start, end)))
                    for g in groups:
                        if g not in group_to_pos[tensor]:
                            group_to_pos[tensor][g] = max_groups[tensor]
                            max_groups[tensor] += 1
        else:
            # Per-partition positions: each partition assigns local indices 0, 1, …
            # to its own groups, and max_groups is the largest partition size.
            # Groups shared across partitions (e.g. A-tensor tiles) get the same
            # position in every partition, which is safe because any LR that touches
            # those groups uses a different set_idx than the concurrent MFMA.
            for pi in range(numP):
                for tensor in self.tensors:
                    side = TENSOR_SIDE[tensor]
                    start, end = part_ranges[pi][side]
                    gran = lr_grans[tensor]
                    groups = sorted(set(
                        (t // gran.mn) * gran.mn for t in range(start, end)))
                    local_pos = 0
                    for g in groups:
                        if g not in group_to_pos[tensor]:
                            group_to_pos[tensor][g] = local_pos
                        local_pos += 1
                    max_groups[tensor] = max(max_groups[tensor], local_pos)

        # ── Compute per-tensor K-groups and unroll factor ──
        num_k_groups = {}
        for tensor in self.tensors:
            num_k_groups[tensor] = numK // lr_grans[tensor].k

        unroll_factor = 1
        for tensor in self.tensors:
            if num_k_groups[tensor] % 2 != 0:
                unroll_factor = 2
                break
        # PGR=0 has no prefetch and no wrapping LRs — each K-group's data is
        # fully consumed before the next LR overwrites it, so no double-buffering
        # is needed.  Force a single set (set_idx=0) and one unroll body.
        pgr0 = cfg.pgr == 0
        if pgr0:
            unroll_factor = 1

        # ── Deterministic tile assignment ──
        for unroll_iter in range(unroll_factor):
            for pi, slots in enumerate(self._partitions):
                for slot in slots:
                    k = slot.subIterK

                    if slot.mfma:
                        for tensor in self.tensors:
                            gran = lr_grans[tensor]
                            nkg = num_k_groups[tensor]
                            set_idx = 0 if pgr0 else (unroll_iter * nkg + k // gran.k) % 2
                            side = TENSOR_SIDE[tensor]
                            tileRange = (slot.mfma.tileA if side == 'A'
                                         else slot.mfma.tileB)
                            tile_map = {}
                            for t in tileRange.tileId_list:
                                group = (t // gran.mn) * gran.mn
                                pos = group_to_pos[tensor][group]
                                tile_map[group] = (set_idx * max_groups[tensor]
                                                   + pos)
                            slot.mfma.vgpr_tile_maps.setdefault(
                                tensor, []).append(tile_map)

                    for lr in slot.lrs:
                        tensor = lr.tensor
                        gran = lr_grans[tensor]
                        nkg = num_k_groups[tensor]
                        target_mt = unroll_iter + lr.mtIteration
                        target_k = lr.tiles.subIterK_start
                        set_idx = 0 if pgr0 else (target_mt * nkg + target_k // gran.k) % 2

                        tile_map = {}
                        for t in lr.tiles.tileId_list:
                            group = (t // gran.mn) * gran.mn
                            if group in tile_map:
                                continue
                            pos = group_to_pos[tensor][group]
                            tile_map[group] = (set_idx * max_groups[tensor]
                                               + pos)
                        lr.vgpr_tile_map.append(tile_map)

        # ── Record results ──
        # PGR=0: no prefetch — each K-group's LR data is consumed before the
        #        next LR overwrites it, so only 1 VGPR tile set is needed.
        # PGR≥1: next-iteration LRs are issued while current MFMAs run, so two
        #        iterations' tile data coexist in VGPRs → 2 VGPR tile sets needed.
        num_sets = 1 if pgr0 else 2
        self.tile_peaks = {t: num_sets * max_groups[t] for t in self.tensors}
        self.unroll_factor = unroll_factor
        self.needs_unrolling = unroll_factor > 1

        self._completed.add(Pass.VGPR_TILES)

    # ── Place GRs ─────────────────────────────────────────

    def _build_gr_list(self, part_ranges, offsetMT, offsetPartition):
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
                    tr = gr_gran.tile_range(k, t_start, t_end)

                    key = (tensor, mt_val, tr.tileId_start, tr.tileId_end,
                           tr.subIterK_start, tr.subIterK_end)
                    if key in seen:
                        continue
                    seen.add(key)
                    gr_list.append((tensor, mt_val, tr.tileId_start,
                                    tr.tileId_end, tr.subIterK_start,
                                    tr.subIterK_end, gr_gran))

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

        return gr_list

    def _build_gr_slot_bounds(self):
        """Build lower and upper slot bounds for GR placement.

        lower: tensor -> [(flat, t_start, t_end, k_start, k_end)] for LR(mt=0).
               GR(mt=2) writes the same LDS buffer as MT n, so it can't be
               placed at/before a flat slot where a later LR(mt=0) in *any*
               partition reads an overlapping tile/k-range (LDS conflict).
               Collisions span partitions: an LR(n) read for tensor B may sit
               in partition pi+1 while the GR(n+2) overwrite is emitted in
               partition pi, so the check must be in flat execution order.
        upper: (tensor, mt) -> first flat slot index with LR(tensor, mt).
               GR(tensor, mt) must be placed strictly before this slot.
        """
        numK = self.config.numSubIterK
        lower = {}
        upper = {}
        for pi, partition_slots in enumerate(self._partitions):
            for slot in partition_slots:
                flat = pi * numK + slot.subIterK
                for lr in slot.lrs:
                    if lr.mtIteration == 0:
                        lower.setdefault(lr.tensor, []).append(
                            (flat,
                             lr.tiles.tileId_start,
                             lr.tiles.tileId_end,
                             lr.tiles.subIterK_start,
                             lr.tiles.subIterK_end))
                    key = (lr.tensor, lr.mtIteration)
                    if key not in upper or flat < upper[key]:
                        upper[key] = flat
        return lower, upper

    @staticmethod
    def _has_lr_conflict(lr_lower, tensor, mt_val, flat,
                         gr_t_start, gr_t_end, gr_k_start, gr_k_end):
        """Return True if placing GR(mt_val) at flat slot conflicts.

        GR(MT n+2) writes the same LDS buffer as MT n, so it conflicts if a
        later LR(MT n) — in flat execution order across all partitions —
        still reads an overlapping tile/subIterK range from that buffer.
        """
        if mt_val != 2:
            return False
        for lr_flat, lr_ts, lr_te, lr_ks, lr_ke in lr_lower.get(tensor, []):
            if (lr_flat > flat and
                    gr_t_start < lr_te and lr_ts < gr_t_end and
                    gr_k_start < lr_ke and lr_ks < gr_k_end):
                return True
        return False

    def _distribute_grs(self, gr_list, gr_slot_bounds):
        """Phase 2: Distribute GR atoms across partition × subIterK slots.

        Explodes GR entries into atomic loads, distributes them into flat
        buckets respecting LDS conflict constraints and load balance,
        then remerges consecutive atoms and places them into partitions.
        """
        cfg = self.config
        numK = cfg.numSubIterK
        numP = cfg.numPartitions
        numSlots = numP * numK
        lower, upper = gr_slot_bounds

        # 2a. Explode GR entries into atomic loads (1 load each)
        atoms = []
        for tensor, mt_val, t_start, t_end, k_start, k_end, gr_gran in gr_list:
            mn = gr_gran.mn
            last = max(0, min(upper.get((tensor, mt_val), numSlots) - 1,
                            numSlots - 1))
            for pos in range(t_start, t_end, mn):
                atoms.append((tensor, mt_val, pos, pos + mn, k_start, k_end, last))

        # 2b. Place atoms across [0..numSlots) weighted by partition MFMA count.
        #     Each partition's slots get a share proportional to its MFMAs,
        #     so larger partitions receive more GR loads.
        nAtoms = len(atoms)
        buckets = [[] for _ in range(numSlots)]

        mfma_per_partition = []
        for pi in range(numP):
            piM = pi % cfg.numPartitionsM
            piN = pi // cfg.numPartitionsM
            mfma_per_partition.append(cfg.partitionSizesM[piM] * cfg.partitionSizesN[piN])

        weight_prefix = [0]
        for s in range(numSlots):
            weight_prefix.append(weight_prefix[-1] + mfma_per_partition[s // numK])
        total_weight = weight_prefix[numSlots]
        slot_boundaries = [p * nAtoms for p in weight_prefix[1:]]

        for i, (tensor, mt_val, ts, te, ks, ke, last) in enumerate(atoms):
            if cfg.grPlacement == GRPlacementStrategy.BUNCHED:
                # TDM: pin every GR atom to partition 0, subIterK 0.
                slot = 0
            else:
                slot = min(bisect_left(slot_boundaries, i * total_weight + 1),
                           last) if nAtoms else 0
            while (slot < last and
                   self._has_lr_conflict(lower, tensor, mt_val,
                                         slot, ts, te, ks, ke)):
                slot += 1
            buckets[slot].append((tensor, mt_val, ts, te, ks, ke))

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

        pgr = self.config.pgr
        offsetMT = 0 if pgr == 0 else 1
        gr_list = self._build_gr_list(part_ranges, offsetMT, self.config.offsetPartition)
        gr_slot_bounds = self._build_gr_slot_bounds()
        self._distribute_grs(gr_list, gr_slot_bounds)

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
                return -producer.mtIteration
            # LR→GR: mt difference determines how many iterations back.
            if consumer_type == 'LR' and isinstance(producer, GRPlacement) and consumer:
                return consumer.mtIteration - producer.mtIteration
            # Fallback: partition+slot ordering decides.
            return _slot_offset(consumer_partition, consumer_slot, consumer_type, producer)

        def _tiles_overlap(mfma, lr_tensor, lr_tiles):
            """Check if LR tile range overlaps with MFMA's tile range for that tensor."""
            if lr_tensor in ('A', 'SA'):
                mfma_range = mfma.tileA
            else:
                mfma_range = mfma.tileB
            return (lr_tiles.tileId_start < mfma_range.tileId_end and
                    lr_tiles.tileId_end > mfma_range.tileId_start and
                    lr_tiles.subIterK_start < mfma_range.subIterK_end and
                    lr_tiles.subIterK_end > mfma_range.subIterK_start)

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

    def _make_gr_dep_exec_order(self, tensor):
        """Return a key fn ordering deps by (mt_offset, partition, slot, intra-slot rank).

        Two GRs sharing a (mtIteration, partition, subIterK_slot) collapse to one
        (mt_offset, partition, slot) key, so an intra-slot rank (by _gr_sort_key,
        the order the wait emitter walks) is needed to keep them distinguishable:
        dropping a dep on the slot's last GR is unsafe even if a dep on an
        earlier-rank GR in the same slot is kept.
        """
        slot_members = {}
        for slots in self._partitions:
            for slot in slots:
                for gr in slot.grs:
                    if gr.tensor == tensor:
                        key = (gr.mtIteration, gr.partition, gr.subIterK_slot)
                        slot_members.setdefault(key, []).append(gr)

        gr_intra_rank = {}
        for grs in slot_members.values():
            for rank, gr in enumerate(sorted(grs, key=self._gr_sort_key)):
                gr_intra_rank[id(gr)] = rank

        def _dep_exec_order(dep):
            gr = dep.ref
            return (dep.mt_offset,
                    gr.partition,
                    gr.subIterK_slot,
                    gr_intra_rank[id(gr)])
        return _dep_exec_order

    def remove_unnecessary_gr_deps(self):
        """Remove GR deps on LRs that are already guaranteed by an earlier LR's wait.

        Per tensor, walks LR placements in execution order. If an earlier LR
        already waits for a GR with equal or higher exec_order, the later LR's
        dep is redundant and removed.

        Wraps around: the first LR's dep is compared against the last from the
        previous MT iteration (max dep exec_order shifted by mt_offset -1).
        """
        self._ensure_pass(Pass.DEPS)

        for tensor in self.tensors:
            _dep_exec_order = self._make_gr_dep_exec_order(tensor)

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
            max_guaranteed = (max_eo[0] - 1, max_eo[1], max_eo[2], 0)

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

        A slot is a sync point when it contains any GR-with-LR-dep or any
        LR-with-GR-deps. At a sync slot, the per-tensor last LR guaranteed
        is the max exec_order of:
          - the MFMA's LR deps at that slot
          - any GR-with-LR-dep's own LR dep at that slot

        For each LR dep on a GR (curLR), find the previous sync (in exec
        order, skipping the current slot) providing a last LR for the same
        tensor. If that last LR's exec_order >= curLR's, the dep is redundant.

        Exec order: (mt_offset, partition, subIterK_slot). On wrap-around
        the mt_offset is shifted by -1.
        """
        self._ensure_pass(Pass.REMOVE_GR_DEPS)

        def _dep_exec_order(dep):
            return (dep.mt_offset, dep.ref.partition, dep.ref.subIterK_slot)

        # Step 1: collect one sync entry per sync slot.
        # Each entry: (pos, last_lr_by_tensor, [grs_to_check])
        sync_slots = []
        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                grs_with_lr = [
                    gr for gr in slot.grs
                    if gr.deps and isinstance(gr.deps[0].ref, LRPlacement)]
                lr_with_gr_exists = any(
                    lr.deps and isinstance(lr.deps[0].ref, GRPlacement)
                    for lr in slot.lrs)
                if not grs_with_lr and not lr_with_gr_exists:
                    continue

                last_lr = {}
                if slot.mfma:
                    for d in slot.mfma.deps:
                        if isinstance(d.ref, LRPlacement):
                            t = d.ref.tensor
                            eo = _dep_exec_order(d)
                            if t not in last_lr or eo > last_lr[t]:
                                last_lr[t] = eo
                for gr in grs_with_lr:
                    dep = gr.deps[0]
                    t = dep.ref.tensor
                    eo = _dep_exec_order(dep)
                    if t not in last_lr or eo > last_lr[t]:
                        last_lr[t] = eo

                sync_slots.append(((pi, slot.subIterK), last_lr, grs_with_lr))

        if not sync_slots:
            self._completed.add(Pass.REMOVE_LR_DEPS)
            return

        sync_slots.sort(key=lambda x: x[0])
        n = len(sync_slots)

        # Step 2 & 3: for each GR with LR dep, walk backward (with
        # wrap-around) to find the previous sync slot providing a last LR
        # for this tensor.
        for i in range(n):
            _, _, grs_to_check = sync_slots[i]
            for gr in grs_to_check:
                if not gr.deps:
                    continue
                dep = gr.deps[0]
                tensor = dep.ref.tensor
                cur_eo = _dep_exec_order(dep)

                prev_eo = None
                cur_pos = sync_slots[i][0]
                for j in range(1, n + 1):
                    idx = (i - j) % n
                    wrapped = j > i  # crossed iteration boundary
                    # Same-slot in same iteration is concurrent, skip.
                    if not wrapped and sync_slots[idx][0] == cur_pos:
                        continue
                    prev_last_lr = sync_slots[idx][1]
                    if tensor in prev_last_lr:
                        eo = prev_last_lr[tensor]
                        if wrapped:
                            eo = (eo[0] - 1, eo[1], eo[2])
                        prev_eo = eo
                        break

                if prev_eo is not None and prev_eo >= cur_eo:
                    gr.deps.clear()

        self._completed.add(Pass.REMOVE_LR_DEPS)

    # ── Remove cross-subIterK deps ─────────────────────────

    def _gr_granularity(self, tensor: str) -> ReadGranularity:
        """Return GR granularity for a tensor."""
        return {'A': self.config.grA, 'B': self.config.grB,
                'SA': self.config.grSA, 'SB': self.config.grSB}[tensor]

    def _count_gr_atoms(self, gr: GRPlacement) -> int:
        """Count the number of atomic loads for a single GR placement."""
        gr_gran = self._gr_granularity(gr.tensor)
        tiles = gr.tiles
        n_tile = (tiles.tileId_end - tiles.tileId_start) // gr_gran.mn
        n_k = (tiles.subIterK_end - tiles.subIterK_start) // gr_gran.k
        return n_tile * n_k

    def _compute_inflight_loads(self, consumer_pi: int, consumer_slot: int,
                                tensor: str, dep_ref: Dep) -> WaitGRCounts:
        """Count inflight GR atomic loads between a dep GR and the consumer.

        Walks backward through the flattened schedule (all partitions x subIterK)
        from the consumer position, counting atomic GR loads for all tensors.
        Stops when reaching the dependency GR (dep_ref.ref) after accounting
        for mt_offset wraps.

        Within a slot, GRs are walked in reverse _gr_sort_key order (matching
        hardware issue order: last emitted = most recent = walked first).
        GRs emitted after the dep in the same slot are still inflight but
        do not need to be waited for, so they are added to the count.

        Returns per-tensor inflight load counts.
        """
        numP = len(self._partitions)
        numK = len(self._partitions[0])
        flat_len = numP * numK

        consumer_flat = consumer_pi * numK + consumer_slot
        wraps_needed = abs(dep_ref.mt_offset)

        # Locate dep_flat: the flat position of the dependency GR in the schedule.
        dep_flat = None
        for p_idx, pslots in enumerate(self._partitions):
            for k_idx, slot in enumerate(pslots):
                if any(gr is dep_ref.ref for gr in slot.grs):
                    dep_flat = p_idx * numK + k_idx
                    break
            if dep_flat is not None:
                break

        if dep_flat is None:
            return WaitGRCounts()

        # Exact number of backward steps from consumer to dep's slot.
        # Forward distance from dep (exclusive) to consumer (inclusive) =
        #   wraps_needed full iterations + (consumer_flat - dep_flat) slots.
        # Walking backward covers the same count of slots.
        # When wraps_needed==0 and dep_flat >= consumer_flat, the dep GR is at or
        # after the consumer in the same iteration — nothing is inflight yet.
        # When wraps_needed >= 1, total_steps is always >= 1 by construction
        # (wraps_needed*flat_len >= flat_len > flat_len-1 >= dep_flat-consumer_flat).
        total_steps = wraps_needed * flat_len + consumer_flat - dep_flat
        if total_steps <= 0:
            assert wraps_needed == 0, (
                f"_compute_inflight_loads: total_steps={total_steps} < 1 "
                f"(wraps_needed={wraps_needed}, consumer_flat={consumer_flat}, dep_flat={dep_flat}); "
                "unexpected negative total_steps with wraps_needed >= 1"
            )
            return WaitGRCounts()

        counts = WaitGRCounts()
        pos = consumer_flat
        for step in range(total_steps):
            pos = (pos - 1) % flat_len
            pi = pos // numK
            slot_k = pos % numK
            slot = self._partitions[pi][slot_k]

            # On the final step we are at dep's slot: stop when we reach the dep GR.
            # GRs emitted after the dep (encountered first in reverse order) are in-flight
            # and are counted before we hit the dep.
            is_final = (step == total_steps - 1)

            # Walk GRs in reverse emission order (most recently issued first)
            sorted_grs = sorted(slot.grs, key=self._gr_sort_key, reverse=True)
            for gr in sorted_grs:
                if is_final and gr.tensor == tensor and gr is dep_ref.ref:
                    return counts
                atoms = self._count_gr_atoms(gr)
                cur = getattr(counts, gr.tensor)
                setattr(counts, gr.tensor, cur + atoms)

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
                    has_lr_dep = any(
                        isinstance(d.ref, LRPlacement) for d in same + cross)
                    if has_lr_dep:
                        slot.mfma.preOps.append(WaitLROp())

                # ── LRs ──
                for lr in slot.lrs:
                    gr_deps = [d for d in lr.deps
                               if isinstance(d.ref, GRPlacement)]
                    same, cross = self._split_deps(lr.deps, pi, lr.subIterK_slot)
                    lr.deps = same
                    lr.preOps = []
                    if gr_deps:
                        dep = gr_deps[0]
                        cross_set = set(id(d) for d in cross)
                        is_cross = id(dep) in cross_set
                        counts = self._compute_inflight_loads(
                            pi, lr.subIterK_slot, dep.ref.tensor, dep)
                        lr.preOps.append(WaitGROp(wait_gr_counts=counts,
                                                  has_sync=True,
                                                  adjustVmcnt=is_cross))

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
        last_lr = {}  # tensor -> last LR placement seen
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
                    last_lr[tensor] = lr
                    last_lr_mt[tensor] = mt
                for gr in slot.grs:
                    tensor = gr.tensor
                    mt = gr.mtIteration
                    if tensor in last_gr_mt:
                        prev_mt = last_gr_mt[tensor]
                    else:
                        prev_mt = 0
                    if prev_mt != mt:
                        if gr.tiles.tileId_start == 0:
                            gr.preOps.append(GRIncOp(tensor=tensor))
                    last_gr_mt[tensor] = mt

        if self.config.pgr == 0:
            last_gr_per_tensor = {}
            for slots in self._partitions:
                for slot in slots:
                    for gr in slot.grs:
                        last_gr_per_tensor[gr.tensor] = gr
            for tensor in self._LR_GR_ORDER:
                if tensor in last_lr and tensor in last_lr_mt:
                    last_lr[tensor].postOps.append(LRIncOp(tensor=tensor))
                if tensor in last_gr_per_tensor and tensor in last_gr_mt:
                    last_gr_per_tensor[tensor].postOps.append(GRIncOp(tensor=tensor))
        else:
            for tensor, lr in first_lr.items():
                if tensor not in lr_inc_tensors:
                    lr.preOps.append(LRIncOp(tensor=tensor))

        self._completed.add(Pass.GR_INC)

    # ── Group LR/GR chains ─────────────────────────────────────

    _TENSOR_ORDER = {'A': 0, 'B': 1, 'SA': 2, 'SB': 3}
    _LR_GR_ORDER = ['A', 'B', 'SA', 'SB']

    @staticmethod
    def _gr_sort_key(gr: GRPlacement) -> tuple:
        """Sort key for deterministic GR ordering within a subIterK slot.

        Priority (most significant first):
          1. MT iteration      — earlier MT loads first (n+1 before n+2)
          2. subIterK start    — lower min K first
          3. Tensor            — A, B, SA, SB (hardcoded order)
          4. Tile id start     — lower tile range first
        """
        return (gr.mtIteration,
                gr.tiles.subIterK_start,
                LogicalScheduler._TENSOR_ORDER[gr.tensor],
                gr.tiles.tileId_start)

    @staticmethod
    def _merge_preops(all_preops: List[List['BaseOp']]) -> List['BaseOp']:
        """Merge preOps from multiple placements.

        Combines wait_gr/wait_gr_sync counts into a single BaseOp, deduplicates barrier ops
        (wait_lr_sync, wait_lr), and collects the rest.
        """
        wait_gr_ops_full = []
        has_wait_gr_sync = False
        seen_wait_lr = False
        others = []
        for preops in all_preops:
            for op in preops:
                if isinstance(op, WaitGROp) and op.wait_gr_counts:
                    if op.has_sync:
                        has_wait_gr_sync = True
                    wait_gr_ops_full.append(op)
                elif isinstance(op, WaitLROp):
                    if not seen_wait_lr:
                        seen_wait_lr = True
                        others.append(op)
                else:
                    others.append(op)
        result = []
        if wait_gr_ops_full:
            merged_counts = WaitGRCounts()
            for t in ('A', 'B', 'SA', 'SB'):
                setattr(merged_counts, t,
                        min(getattr(op.wait_gr_counts, t) for op in wait_gr_ops_full))
            adjust = all(op.adjustVmcnt for op in wait_gr_ops_full)
            result.append(WaitGROp(wait_gr_counts=merged_counts,
                                   has_sync=has_wait_gr_sync,
                                   adjustVmcnt=adjust))
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

        Phase 3 — Cross-group merge:
          If any LR has a dep on a GR in the same slot, merge the two chains
          into one: GR chain → LR chain (first LR points to last GR, LR's
          original GR dep is removed).  This avoids two nodes sharing the
          same parent.
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
                    key=self._gr_sort_key)

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

                # ── Phase 3: Cross-group merge ──
                # If any LR depends on a GR in this slot, merge into one
                # chain: GR_group → LR_group to avoid shared parents.
                if ordered_grs and ordered_lrs:
                    slot_gr_set = set(id(gr) for gr in ordered_grs)
                    lr_has_gr_dep = any(
                        any(id(d.ref) in slot_gr_set for d in lr.deps)
                        for lr in ordered_lrs if lr.deps)
                    if lr_has_gr_dep:
                        last_gr = ordered_grs[-1]
                        # Clear LR deps that point to GRs in this slot
                        for lr in ordered_lrs:
                            lr.deps = [d for d in lr.deps
                                       if id(d.ref) not in slot_gr_set]
                        # First LR points to last GR
                        ordered_lrs[0].deps = [
                            Dep(ref=last_gr, mt_offset=0)]

                # ── Phase 4: Consolidate MFMA deps ──
                # After chaining, MFMA only needs the tail of its dep chain.
                if slot.mfma and last_lr is not None:
                    slot_lr_set = set(id(lr) for lr in ordered_lrs)
                    lr_deps = [d for d in slot.mfma.deps
                               if id(d.ref) in slot_lr_set]
                    if len(lr_deps) > 1:
                        other_deps = [d for d in slot.mfma.deps
                                      if id(d.ref) not in slot_lr_set]
                        slot.mfma.deps = other_deps + [
                            Dep(ref=last_lr, mt_offset=lr_deps[0].mt_offset)]

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

                placement_tail_id = {}
                for placement in placements:
                    mid = add(placement)
                    placement_to_id[id(placement)] = mid
                    placement_tail_id[id(placement)] = mid

                # Step 1b: add postOps and update tail ids so that
                # deps on a placement with postOps resolve to the last postOp.
                for placement in placements:
                    if not placement.postOps:
                        continue
                    curId = placement_to_id[id(placement)]
                    postPrevId = curId
                    for postOp in placement.postOps:
                        postId = add(postOp)
                        setBefore(postId, postPrevId)
                        postPrevId = postId
                    placement_tail_id[id(placement)] = postPrevId

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
                        ref_id = placement_tail_id.get(id(dep.ref))
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

        if self.config.pgr in (0, 1):
            self._ngll_emitted = [[[]]]
            return self._ngll_emitted

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

        if self.config.pgr == 0:
            self._nll_emitted = [[[]]]
            return self._nll_emitted

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
                    elif em.opType == 'gr_inc' and self.config.pgr == 2:
                        # PGR=2: NGLL already swapped LW via its kept gr_inc,
                        # so NLL must drop gr_inc to avoid swapping it back.
                        # PGR=1: keep gr_inc — it advances SRD + swaps LW for
                        # tail entry (PRELOOP's single GR did neither).
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
                # but keep WaitLR ops that non-removed modules depend on
                # (e.g. MFMAs waiting for LRs issued in a previous subIterK)
                has_lr = any(em.opType == 'lr' and em.moduleId not in removed
                             for em in new_emitted)
                if not has_lr:
                    depended_on = {em.before for em in new_emitted
                                   if em.moduleId not in removed
                                   and em.before is not None}
                    for em in new_emitted:
                        if em.opType == 'wait_lr' \
                                and em.moduleId not in depended_on:
                            removed.add(em.moduleId)

                part_nll.append(self._rewire_before(new_emitted, removed))
            nll.append(part_nll)

        self._nll_emitted = nll
        return nll

    def build_tailloop_pgr0(self) -> List[List[List[EmittedModule]]]:
        """Template for Tailloop based on PGR0 schedule.

        Returns [partition][groups] where each group has at most one MFMA.

        The tail loop runs flat (no partitioning): per subIterK we emit one
        LR pass covering every unique (tensor, tile_range), one boundary
        mask, then every partition's MFMAs back-to-back. This requires the
        flat tile-id layout from _compute_flat_tail_tile_state (and the
        matching vgpr realloc in _realloc_tail_tiles_flat) so each unique
        partition group has its own vgpr range — the mainloop's per-
        partition tile budget multiplexes vgprs across pi and cannot hold
        all partitions' tiles live at once.
        """
        cfg = self.config
        numK = cfg.numSubIterK

        # Flat tile layout: every unique (tensor, partition_group) gets its
        # own vgpr tile id. _compute_tail_tile_state's old per-partition
        # tile_maps would reuse vgprs across pi and break a flat loop.
        tile_maps, self._flat_tail_peaks = self._compute_flat_tail_tile_state()
        # Legacy unused-tile bookkeeping: in the flat path we replace the
        # vgpr tiles wholesale at tail entry, so nothing here.
        self._tail_unused_tile_ids = {'A': set(), 'B': set(),
                                      'SA': set(), 'SB': set()}

        preamble = []

        # GRs entire MT at once for all tensors.
        all_tiles = {
            'A': MFMATileRange(0, numK, 0, cfg.numMFMATilesM),
            'B': MFMATileRange(0, numK, 0, cfg.numMFMATilesN),
        }
        preamble.extend(self._make_gr_all_tensors(0, all_tiles))
        # bf16-only: an OOB dwordx4 load can corrupt the trailing 16-bit
        # element at the K-boundary (buffer instructions enforce dword
        # granularity on OOB). We patch it with a 16-bit DTL load. Wider
        # dtypes (e.g. fp4 read at K=32 granularity) don't have this issue,
        # so we skip emission entirely for them.
        # TDM uses tensor_load_to_lds, not buffer instructions, so the
        # dword-granularity OOB corruption does not apply.
        hasTDM = self._kernel.get("enableTDMA") and self._kernel.get("enableTDMB")
        if self._kernel["ProblemType"]["DataTypeA"].isBFloat16() and not hasTDM:
            # We need to wait for other SIMD before placing the DTL load
            # (as we'll write twice to this address : OOB Zero then fixup load)
            preamble.append(SyncOp())
            preamble.append(InlineModuleOp(
                build=lambda em: em.writer.tailLoopBoundaryDtlLoadAB(
                    em.kernel,
                    em.tensorParametersMap['A'],
                    em.tensorParametersMap['B']),
                label="tail_boundary_ab"))
        preamble.append(WaitGROp(wait_gr_counts=WaitGRCounts()))
        preamble.append(SyncOp())


        # Flat per-subIterK emission. The K-boundary mask depends only on k,
        # and with flat tile ids the per-partition tile_maps reference
        # disjoint vgpr ranges per (tensor, group). So per k we can:
        #   1. emit each unique (tensor, tile_range) LR exactly once
        #   2. wait + mask once (single VAnd per unique flat vgpr)
        #   3. run every partition's MFMA back-to-back
        # The returned shape is still [partition][group][ops]; we use a
        # single outer "partition" holding all per-k groups.
        miK = int(self._kernel["MatrixInstK"])
        groups = [self._to_emitted(preamble)]

        # Build a merged tile_map covering every partition's tiles, used by
        # MaskKOp to enumerate the live flat vgpr ids.
        merged_tile_map: dict = {}
        for pi in range(cfg.numPartitions):
            for tensor in ('A', 'B', 'SA', 'SB'):
                src = tile_maps[pi].get(tensor)
                if not src:
                    continue
                dst = merged_tile_map.setdefault(tensor, [{}])
                while len(dst) < len(src):
                    dst.append({})
                for ui, m in enumerate(src):
                    dst[ui].update(m)

        for k in range(numK):
            ops = []
            # Dedup LRs across partitions by tileId range — with flat tile
            # ids, same range ⇒ same vgprs, so one LR populates all readers.
            seen_lr = set()
            for pi in range(cfg.numPartitions):
                cur = self._partition_tile_range(pi)
                for tensor, gran in self._lr_tensors():
                    if k % gran.k != 0:
                        continue
                    side_key = 'A' if tensor in ('A', 'SA') else 'B'
                    tiles = gran.tile_range(k, *cur[side_key])
                    lr_key = (tensor,
                              tiles.tileId_start, tiles.tileId_end,
                              tiles.subIterK_start, tiles.subIterK_end)
                    if lr_key in seen_lr:
                        continue
                    seen_lr.add(lr_key)
                    lr = LRPlacement(tensor=tensor, mtIteration=0,
                                     tiles=tiles,
                                     subIterK_slot=k, partition=pi)
                    lr.vgpr_tile_map = copy.deepcopy(tile_maps[pi].get(tensor, []))
                    ops.append(lr)
            ops.append(WaitLROp())
            ops.append(MaskKOp(subIterK=k,
                               vgpr_tile_map=copy.deepcopy(merged_tile_map)))
            # All partitions' MFMAs for this k, back-to-back.
            for pi in range(cfg.numPartitions):
                cur = self._partition_tile_range(pi)
                mfma_tileA = MFMATileRange(k, k + 1, *cur['A'])
                mfma_tileB = MFMATileRange(k, k + 1, *cur['B'])
                mfma = MFMAPlacement(subIterK=k, tileA=mfma_tileA, tileB=mfma_tileB)
                mfma.vgpr_tile_maps = copy.deepcopy(tile_maps[pi])
                ops.append(mfma)
            # Early-exit: after subIterK=k completes for every partition,
            # skip ahead if no more valid K remains. Omit on the last k.
            if k != numK - 1:
                ops.append(SkipOp(
                    compare='LE', value=miK * (k + 1),
                    target='SkipTailLoopL', rawLabel=True,
                    branchComment=f"early-exit tail after subIterK={k} (no valid K left)"))
            groups.append(self._to_emitted(ops))

        self._tailloop_emitted = [groups]
        return self._tailloop_emitted

    @staticmethod
    def _to_emitted(ops) -> List[EmittedModule]:
        """Wrap Emittable objects (Placements / BaseOps) into EmittedModules."""
        return [EmittedModule(moduleId=mid, source=op) for mid, op in enumerate(ops)]

    def _make_gr_all_tensors(self, mt: int, tiles: dict) -> List[GRPlacement]:
        """Create GR placements for all tensors at the given MT iteration.

        tiles: {'A': MFMATileRange, 'B': MFMATileRange}
        """
        return [GRPlacement(tensor=tensor, mtIteration=mt,
                            tiles=tiles['A' if tensor in ('A', 'SA') else 'B'],
                            subIterK_slot=0)
                for tensor in self.tensors]

    def _make_lr_all_tensors(self, tiles: dict) -> List[LRPlacement]:
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

    def _make_depops_all_tensors(self, cls) -> List[BaseOp]:
        """Create a BaseOp subclass instance for each tensor."""
        return [cls(tensor=tensor) for tensor in self.tensors]

    def _make_preloop_mt1_grs(self) -> List[GRPlacement]:
        """Create MT1 GRs for the PGR=2 preloop, ordered to match the mainloop.

        Covers partitions 0..offsetPartition-1 with proper deduplication.
        Each unique (tensor, tile-range, k-range) appears exactly once.
        """
        self._ensure_pass(Pass.LR)
        cfg = self.config

        seen = set()
        result = []
        for pi in range(cfg.offsetPartition):
            target_range = self._partition_tile_range(pi)
            for slot in self._partitions[0]:
                k = slot.mfma.subIterK
                items = [('A', target_range['A'], cfg.grA),
                         ('B', target_range['B'], cfg.grB)]
                if cfg.hasScale:
                    items.append(('SA', target_range['A'], cfg.grSA))
                    items.append(('SB', target_range['B'], cfg.grSB))
                for tensor, (t_start, t_end), gr_gran in items:
                    tr = gr_gran.tile_range(k, t_start, t_end)
                    key = (tensor, tr.tileId_start, tr.tileId_end,
                           tr.subIterK_start, tr.subIterK_end)
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(GRPlacement(
                        tensor=tensor,
                        mtIteration=1,
                        tiles=tr,
                        subIterK_slot=k,
                        partition=pi,
                    ))
        return result

    def build_preloop(self) -> List[List[List[EmittedModule]]]:
        """Build preloop: pipeline initialization sequence before mainloop.

        PGR=0: no preloop (mainloop only).

        PGR=1 sequence:
          GR(MT 0)  — all tensors, all tiles
          LR        — first partition, subIterK=0
          skip(LE 1, NLL)

        PGR=2 sequence:
          GR(MT 0)  — all tensors, all tiles
          LR        — first partition, subIterK=0
          skip(LE 1, NLL)
          GR(MT 1)  — first partition tiles
          skip(LE 2, NGLL)

        Returns [1 partition][1 subIterK][EmittedModules] to match emit() shape.
        """
        if self.config.pgr == 0:
            self._preloop_emitted = [[[]]]
            return self._preloop_emitted

        cfg = self.config
        numK = cfg.numSubIterK
        part0 = self._partition_tile_range(0)
        all_tiles = {
            'A': MFMATileRange(0, numK, 0, cfg.numMFMATilesM),
            'B': MFMATileRange(0, numK, 0, cfg.numMFMATilesN),
        }
        lr_tiles = {
            'A':  MFMATileRange(0, cfg.lrA.k, *part0['A']),
            'B':  MFMATileRange(0, cfg.lrB.k, *part0['B']),
        }
        if cfg.hasScale:
            lr_tiles['SA'] = MFMATileRange(0, cfg.lrSA.k, *part0['A'])
            lr_tiles['SB'] = MFMATileRange(0, cfg.lrSB.k, *part0['B'])

        if cfg.pgr == 1:
            emitted = self._to_emitted([
                *self._make_gr_all_tensors(0, all_tiles),
                WaitGROp(wait_gr_counts=WaitGRCounts()),
                SyncOp(),
                *self._make_lr_all_tensors(lr_tiles),
                SkipOp(compare='LE', value=1, target='NLL'),
            ])
        else:
            emitted = self._to_emitted([
                *self._make_gr_all_tensors(0, all_tiles),
                *self._make_depops_all_tensors(GRIncOp),
                WaitGROp(wait_gr_counts=WaitGRCounts()),
                SyncOp(),
                *self._make_lr_all_tensors(lr_tiles),
                SkipOp(compare='LE', value=1, target='NLL'),
                *self._make_preloop_mt1_grs(),
                SkipOp(compare='LE', value=2, target='NGLL'),
            ])

        self._preloop_emitted = [[emitted]]
        return self._preloop_emitted

    def _emitLoop(self, writer, kernel, label, emitted_3d, schedule=True):
        """Emit a loop section from a 3D emitted structure.

        emitted_3d: [partition][subIterK][EmittedModule]

        When schedule=True and a group has MFMAs, calls instructionSchedule
        for interleaving. When schedule=False, emits instructions sequentially.
        """
        from Tensile.Components.Subtile.InstructionScheduler import (
            instructionSchedule,
            _MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT,
            _MIN_MFMA_GAP_DS_READ_TO_WAIT_GFX1250,
        )
        from Tensile.Components.Subtile.WaitAluInsertion import insertLRSwapWaitAlu,setMatrixAReuse
        from rocisa.code import Module, Label
        from rocisa.container import sgpr
        from rocisa.instruction import SCmpEQU32, SCBranchSCC0, SMovB32

        # gfx1250 needs a larger ds_read->waitcnt gap.
        isGfx1250 = writer.states.archCaps.get("HasWmmaArbStallBit", False)
        minGapDsReadToWait = (_MIN_MFMA_GAP_DS_READ_TO_WAIT_GFX1250
                              if isGfx1250
                              else _MIN_MFMA_GAP_DS_READ_TO_WAIT_DEFAULT)

        module = Module(label)
        module.addComment0(f"{label} start")
        if kernel.get("ClusterBarrier"):
            from Tensile.Components.Subtile.ClusterBarrier import subtileClusterBarrier
            for inst in subtileClusterBarrier(writer, kernel, label=label).flatitems():
                module.add(inst)
        use_pap_preloop_skip = (
            label == "PRELOOP"
            and kernel.get("UseSubtileImpl")
            and kernel.get("PrefetchAcrossPersistent")
            and kernel.get("StreamK") == 3
        )
        pap_merge_label = Label("SubtilePAPPreloopFirstGRMerge", "") if use_pap_preloop_skip else None
        skipping_first_gr_group = False
        first_gr_group_done = False
        for pi, partition_emitted in enumerate(emitted_3d):
            for k, em_list in enumerate(partition_emitted):
                module.addComment0(f"partition={pi} subIterK={k}")
                if schedule and em_list:
                    scheduled = instructionSchedule(em_list, minGapDsReadToWait=minGapDsReadToWait)
                    module.add(scheduled)
                else:
                    for em in em_list:
                        if use_pap_preloop_skip and not first_gr_group_done:
                            if em.opType == 'gr':
                                if not skipping_first_gr_group:
                                    module.add(SCmpEQU32(src0=sgpr("SkPrefetchPrimed"), src1=0,
                                                         comment="Subtile PAP: first PRELOOP GR already issued?"))
                                    module.add(SCBranchSCC0(labelName=pap_merge_label.getLabelName(),
                                                            comment="skip first PRELOOP GR group if primed"))
                                    skipping_first_gr_group = True
                            elif skipping_first_gr_group:
                                module.add(pap_merge_label)
                                module.add(SMovB32(dst=sgpr("SkPrefetchPrimed"), src=0,
                                                   comment="Subtile PAP: clear after first PRELOOP GR merge"))
                                first_gr_group_done = True
                        for inst in em.instructions:
                            module.add(inst)
        if use_pap_preloop_skip and skipping_first_gr_group and not first_gr_group_done:
            module.add(pap_merge_label)
            module.add(SMovB32(dst=sgpr("SkPrefetchPrimed"), src=0,
                               comment="Subtile PAP: clear after first PRELOOP GR merge"))
        module.addComment0(f"{label} end")
        # SCHED_MODE 2: guard the LR offset-swap -> ds_read RAW hazard once, against
        # the final post-schedule order (no-op on other archs).
        module = insertLRSwapWaitAlu(module, writer, kernel)
        # gfx1250: enable WMMA matrix-A reuse on the final post-schedule order.
        module = setMatrixAReuse(module, writer, kernel)
        return module

    def emitMainAndExitLoops(self, writer, kernel, tensorParametersA=None, tensorParametersB=None):
        """Emit preloop + mainloop + NGLL + NLL exit paths (no tail).

        Owns all control flow (labels, branches, counter management) for the
        main unrolled pipeline. For unroll_factor > 1, emits per-unroll copies
        with correct vgpr tiles. Each mainloop exit jumps to its corresponding
        NGLL→NLL pair. The tail loop is emitted separately by emitTailLoop()
        so the orchestrator (Subtile.Kernel.mainLoop) can wrap it with the
        runtime K%DU counter setup and skip branch.
        """
        from rocisa.code import Module, Label
        from rocisa.instruction import (SSubU32, SCmpEQU32, SCBranchSCC0,
                                        SCBranchSCC1, SBranch)
        from rocisa.container import sgpr

        assert Pass.POPULATE in self._completed, \
            "populate_instructions() must be called before emitMainAndExitLoops()"

        module = Module("MainAndExitLoops")
        uf = self.unroll_factor

        def make_subtile_pap_module(skip_barrier=False):
            if (kernel.get("UseSubtileImpl")
                and kernel.get("PrefetchAcrossPersistent")
                and kernel.get("StreamK") == 3
                and hasattr(writer, "prefetchAcrossPersistentSubtile")):
                preloop_gr = Module("Subtile PAP first PRELOOP GR")
                for em in self._preloop_emitted[0][0]:
                    if em.opType != 'gr':
                        break
                    for inst in em.instructions:
                        preloop_gr.add(copy.deepcopy(inst))
                return writer.prefetchAcrossPersistentSubtile(
                    kernel, tensorParametersA, tensorParametersB, preloop_gr,
                    skipBarrier=skip_barrier)
            return None

        def inject_pap_after_nll_drain(emitted_3d):
            emitted_3d = copy.deepcopy(emitted_3d)

            def insert_after_drain(em_list):
                wait_gr = next((em for em in em_list if em.opType == 'wait_gr'), None)
                if wait_gr is None:
                    return False

                tail_id = wait_gr.moduleId
                sync = next((em for em in em_list
                             if em.opType == 'sync' and em.before == tail_id), None)
                if sync is not None:
                    tail_id = sync.moduleId

                pap_module = make_subtile_pap_module(skip_barrier=(sync is not None))
                if pap_module is None:
                    return False

                pap_id = max(em.moduleId for em in em_list) + 1
                consumers = [em for em in em_list if em.before == tail_id]
                em_list.append(EmittedModule(moduleId=pap_id,
                                             instructions=[pap_module],
                                             before=tail_id,
                                             source=None))
                # Place PAP between the final drain and the first NLL work that
                # was waiting on that drain. The existing before-link graph is a
                # chain, so this preserves scheduler ordering while moving PAP
                # past the vlcnt(0) that would otherwise drain it immediately.
                for consumer in consumers:
                    consumer.before = pap_id
                return True

            for partition_emitted in emitted_3d:
                for em_list in partition_emitted:
                    if insert_after_drain(em_list):
                        return emitted_3d

            return emitted_3d

        # ── Skip preloop/mainloop/NGLL/NLL when K < DepthU ──
        endLabel = Label("SkipToEnd", "")
        if not kernel["NoTailLoop"]:
            module.add(SCmpEQU32(src0=sgpr("LoopCounterL"), src1=0,
                                 comment="K < DepthU? skip to tail loop"))
            module.add(SCBranchSCC1(labelName=endLabel.getLabelName(),
                                    comment="K < DepthU: only tail loop runs"))

        # ── Preloop ──
        module.add(self._emitLoop(writer, kernel, "PRELOOP",
                                  self._preloop_emitted, schedule=False))

        # ── Mainloop ──
        module.addComment0("MAINLOOP")
        loopBegin = Label("LoopBeginL", "")

        exitValue = self.config.pgr

        exitLabels = [Label(f"ExitC{ui}", "") for ui in range(uf - 1)]
        module.add(loopBegin)
        # Debug: emit `s_mov_b32 m0, LoopCounterL; s_ttracedata` at the start of
        # every mainloop iteration so SQTT / trace decoders can identify iterations
        # (adds 2 instructions per iter). Gated by the EmitMainloopTraceMarker global.
        emitTraceMarker = globalParameters.get("EmitMainloopTraceMarker", False)
        if emitTraceMarker:
            from rocisa.container import mgpr
            from rocisa.instruction import SMovB32 as _SMovB32
            from rocisa.instruction import STtraceData as _STtraceData
        for ui in range(uf):
            if emitTraceMarker:
                # Mainloop iteration marker for SQTT / trace decoder: write
                # LoopCounterL into M0 then emit it via s_ttracedata. Decoder
                # only uses low 8 bits, so M0 wrap past 256 is fine.
                module.add(_SMovB32(dst=mgpr(0), src=sgpr("LoopCounterL"),
                                    comment="trace: M0 = LoopCounterL"))
                module.add(_STtraceData(comment="trace: emit M0 to SQTT"))
            module.add(self._emitLoop(writer, kernel, f"MAINLOOP_C{ui}",
                                      self._emitted_per_unroll[ui]))
            module.add(SSubU32(dst=sgpr("LoopCounterL"),
                               src0=sgpr("LoopCounterL"), src1=1,
                               comment=f"dec counterL (copy {ui})"))
            module.add(SCmpEQU32(src0=sgpr("LoopCounterL"), src1=exitValue,
                                 comment=f"counterL == {exitValue}? (copy {ui} exit)"))
            if ui < uf - 1:
                module.add(SCBranchSCC1(
                    labelName=exitLabels[ui].getLabelName(),
                    comment=f"copy {ui} exit → NGLL_C{ui}"))
            else:
                module.add(SCBranchSCC0(
                    labelName=loopBegin.getLabelName(),
                    comment="restart mainloop"))

        # ── NGLL + NLL exit paths ──
        hasNGLL = self.config.pgr >= 2
        module.add(Label("SkipMainloop", ""))
        if hasNGLL:
            module.add(Label("SkipToNGLL", ""))

        # _per_unroll[i] has tiles for unroll_iter=i.
        # After mainloop C{ui}, data in LDS/vgprs corresponds to
        # unroll_iter = (ui + pgr) % uf for NLL, (ui + 1) % uf for NGLL.
        # NLLEarly (preloop skip) needs unroll_iter=0, i.e. _nll_per_unroll[0].
        # We place SkipToNLL before whichever NLL block uses index 0.
        pgr = self.config.pgr
        last = uf - 1

        # Fall-through from last mainloop copy
        nll_ft = (last + pgr) % uf
        if hasNGLL:
            module.addComment0(f"NGLL_C{last}")
            module.add(self._emitLoop(writer, kernel, f"NGLL_C{last}",
                                      self._ngll_per_unroll[(last + 1) % uf]))
        if nll_ft == 0:
            module.add(Label("SkipToNLL", ""))
        module.addComment0(f"NLL_C{last}")
        module.add(self._emitLoop(writer, kernel, f"NLL_C{last}",
                                  inject_pap_after_nll_drain(self._nll_per_unroll[nll_ft])))
        module.add(SBranch(labelName=endLabel.getLabelName(),
                           comment="skip other exit paths"))

        for ui in range(uf - 1):
            nll_idx = (ui + pgr) % uf
            module.add(exitLabels[ui])
            if hasNGLL:
                module.addComment0(f"NGLL_C{ui}")
                module.add(self._emitLoop(writer, kernel, f"NGLL_C{ui}",
                                          self._ngll_per_unroll[(ui + 1) % uf]))
            if nll_idx == 0:
                module.add(Label("SkipToNLL", ""))
            module.addComment0(f"NLL_C{ui}")
            module.add(self._emitLoop(writer, kernel, f"NLL_C{ui}",
                                      inject_pap_after_nll_drain(self._nll_per_unroll[nll_idx])))
            if ui < uf - 2:
                module.add(SBranch(labelName=endLabel.getLabelName(),
                                   comment="skip other exit paths"))

        module.add(endLabel)

        return module

    def emitTailLoop(self, writer, kernel):
        """Emit the tail loop body only (no counter setup, no skip branch).

        Returns an empty Module when NoTailLoop is set. The caller is
        responsible for emitting calculateLoopNumIter(-1) before this and
        closeLoop(emitEndLabelOnly=True) after, mirroring the legacy
        KernelWriter pattern.
        """
        assert Pass.POPULATE in self._completed, \
            "populate_instructions() must be called before emitTailLoop()"

        module = Module("TailLoop")

        if kernel["NoTailLoop"]:
            return module

        module.addComment0("TAILLOOP")
        # Swap to the flat tail vgpr tile layout. Frees the mainloop's
        # per-partition tiles back to the pool and reallocates a flat set
        # sized by _compute_flat_tail_tile_state (already invoked by
        # build_tailloop_pgr0; peaks stashed on self._flat_tail_peaks).
        self._realloc_tail_tiles_flat(writer, self._flat_tail_peaks)
        # init must run before populate so each MaskKOp in the body can read
        # the mask vgprs (kReg, vDiff, …) that init allocates.
        for inst in self._emitter.emit_mask_k_init():
            module.add(inst)
        self._emitter.populate(self._tailloop_emitted, unroll_iter=0)
        module.add(self._emitLoop(writer, kernel, "TAILLOOP",
                                  self._tailloop_emitted,
                                  schedule=False))
        for inst in self._emitter.emit_mask_k_done():
            module.add(inst)
        return module

    # ── VGPR tile allocation ──────────────────────────────

    def getNumVgpr(self, tileInfoA, tileInfoB,
                        scaleTileInfoA=None, scaleTileInfoB=None) -> int:
        """Return the total number of VGPRs needed across all tensors (A, B, SA, SB)
        without performing any allocation.

        Returns max(mainloop_peak, flat_tail_peak) — the two layouts don't
        coexist (the tail frees and reallocates at entry), so the kernel
        budget is the larger of them.

        Must be called after scheduling is complete.
        """
        self._ensure_pass(Pass.VGPR_TILES)

        cfg = self.config

        def _tile_vgpr_count(tileInfo, lrGran):
            return int(math.ceil(tileInfo.mmaTileRegCount * lrGran.k * lrGran.mn))

        def _total_for(peaks):
            t = peaks.get('A', 0) * _tile_vgpr_count(tileInfoA, cfg.lrA) \
              + peaks.get('B', 0) * _tile_vgpr_count(tileInfoB, cfg.lrB)
            if cfg.hasScale and scaleTileInfoA and scaleTileInfoB:
                t += peaks.get('SA', 0) * _tile_vgpr_count(scaleTileInfoA, cfg.lrSA) \
                   + peaks.get('SB', 0) * _tile_vgpr_count(scaleTileInfoB, cfg.lrSB)
            return t

        mainloop_total = _total_for(self.tile_peaks)
        _, flat_peaks = self._compute_flat_tail_tile_state()
        tail_total = _total_for(flat_peaks)
        return max(mainloop_total, tail_total)

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

        cfg = self.config

        def _tile_vgpr_count(tileInfo, lrGran):
            return int(math.ceil(tileInfo.mmaTileRegCount * lrGran.k * lrGran.mn))

        def _alloc_tiles(count, numRegs):
            return [_checkout_tile(writer.vgprPool, numRegs, "allocVgprTiles_vstart")
                    for _ in range(count)]

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

        # Stash tile-info so _realloc_tail_tiles_flat can reallocate the
        # tail's flat tile set without the caller plumbing them in again.
        self._alloc_tile_info = {
            'tileInfoA': tileInfoA, 'tileInfoB': tileInfoB,
            'scaleTileInfoA': scaleTileInfoA, 'scaleTileInfoB': scaleTileInfoB}

    def deallocVgprTiles(self, writer):
        """Deallocate VGPR tiles allocated by allocVgprTiles.

        Skips tile ids in self._tail_freed_tile_ids — those were already
        returned to the pool by _release_unused_tail_tiles.
        """
        def _dealloc_tiles(tiles, freed):
            for tid, tile in enumerate(tiles):
                if tid in freed:
                    continue
                _checkin_tile(tile)

        _dealloc_tiles(self.vgprTilesA,  self._tail_freed_tile_ids['A'])
        _dealloc_tiles(self.vgprTilesB,  self._tail_freed_tile_ids['B'])
        _dealloc_tiles(self.vgprTilesSA, self._tail_freed_tile_ids['SA'])
        _dealloc_tiles(self.vgprTilesSB, self._tail_freed_tile_ids['SB'])
        self.vgprTilesA = []
        self.vgprTilesB = []
        self.vgprTilesSA = []
        self.vgprTilesSB = []
        self._tail_freed_tile_ids = {'A': set(), 'B': set(),
                                     'SA': set(), 'SB': set()}

    def _compute_tail_tile_state(self):
        """Single source of truth for tail-loop tile usage.

        Returns (tile_maps, unused) where
          - tile_maps[pi] = self._partitions[pi][0].mfma.vgpr_tile_maps,
            reused by build_tailloop_pgr0 to wire LR/MFMA/MaskK ops.
          - unused[tensor] = {tid} for tile slots the tail loop never
            references (the PGR>=1 prefetch half). Consumed by
            _release_unused_tail_tiles to reclaim their vgprs.

        """
        tile_maps = [self._partitions[pi][0].mfma.vgpr_tile_maps
                     for pi in range(self.config.numPartitions)]
        used = {t: set() for t in ('A', 'B', 'SA', 'SB')}
        for pi_map in tile_maps:
            for tensor in used:
                m = pi_map.get(tensor, [{}])[0]   # unroll_iter=0 only
                used[tensor].update(m.values())
        tiles_by_tensor = {'A':  self.vgprTilesA, 'B':  self.vgprTilesB,
                           'SA': self.vgprTilesSA, 'SB': self.vgprTilesSB}
        unused = {
            tensor: {tid for tid in range(len(tile_list))
                     if tid not in used[tensor]}
            for tensor, tile_list in tiles_by_tensor.items()
        }
        return tile_maps, unused

    def _compute_flat_tail_tile_state(self):
        """Tile-id remap for a non-partitioned ("flat") tail loop.

        The mainloop's tile_peaks are per-partition (each pi reuses the same
        vgprs across its subIterKs). A flat tail loop holds every partition's
        tiles live at once and needs one vgpr range per unique (tensor,
        partition_group). This method assigns each such group a fresh flat
        tile id in 0..flat_peaks[T)-1.

        Returns (tile_maps, peaks) where
          - tile_maps[pi][tensor] = [{group_key: flat_tile_id}] (single-entry
            list mirroring the mainloop's per-unroll_iter shape; tail always
            uses unroll_iter=0).
          - peaks[tensor] = count of distinct flat tile ids for tensor.
        """
        cfg = self.config
        numP = cfg.numPartitions
        lr_grans = {'A': cfg.lrA, 'B': cfg.lrB}
        if cfg.hasScale:
            lr_grans['SA'] = cfg.lrSA
            lr_grans['SB'] = cfg.lrSB

        part_ranges = [self._partition_tile_range(pi) for pi in range(numP)]
        # group_id[(tensor, group_key)] = flat tile id
        group_id: dict = {t: {} for t in lr_grans}
        # tile_maps[pi][tensor] = [{group_key: flat_tile_id}]
        tile_maps: list = [{} for _ in range(numP)]
        for pi in range(numP):
            for tensor, gran in lr_grans.items():
                side = TENSOR_SIDE[tensor]
                start, end = part_ranges[pi][side]
                groups = sorted({(t // gran.mn) * gran.mn
                                 for t in range(start, end)})
                m = {}
                for g in groups:
                    if g not in group_id[tensor]:
                        group_id[tensor][g] = len(group_id[tensor])
                    m[g] = group_id[tensor][g]
                tile_maps[pi][tensor] = [m]
        peaks = {t: len(group_id[t]) for t in lr_grans}
        return tile_maps, peaks

    def _release_unused_tail_tiles(self, writer):
        """Return tile slots dead for the tail loop to the vgpr pool.

        Consumes self._tail_unused_tile_ids (populated by
        _compute_tail_tile_state in build_tailloop_pgr0). The freed tids
        are recorded in self._tail_freed_tile_ids so deallocVgprTiles
        skips them.
        """
        assert not any(self._tail_freed_tile_ids[t]
                       for t in self._tail_freed_tile_ids), \
            "_release_unused_tail_tiles called twice"

        tiles_by_tensor = {'A':  self.vgprTilesA, 'B':  self.vgprTilesB,
                           'SA': self.vgprTilesSA, 'SB': self.vgprTilesSB}
        for tensor, tile_list in tiles_by_tensor.items():
            for tid in self._tail_unused_tile_ids.get(tensor, ()):
                _checkin_tile(tile_list[tid])
                self._tail_freed_tile_ids[tensor].add(tid)

    def _realloc_tail_tiles_flat(self, writer, peaks):
        """Free mainloop's per-partition tiles and reallocate flat tiles for
        the non-partitioned tail loop.

        `peaks[tensor]` comes from _compute_flat_tail_tile_state and is the
        number of distinct flat tile ids per tensor. The new flat tiles
        replace self.vgprTilesA/B/SA/SB; _tail_freed_tile_ids is cleared so
        deallocVgprTiles drops the flat set wholesale at kernel end.
        """
        cfg = self.config
        info = self._alloc_tile_info

        def _tile_vgpr_count(tileInfo, lrGran):
            return int(math.ceil(tileInfo.mmaTileRegCount * lrGran.k * lrGran.mn))

        def _dealloc_all(tiles):
            for tile in tiles:
                _checkin_tile(tile)

        def _alloc_tiles(count, numRegs):
            return [_checkout_tile(writer.vgprPool, numRegs, "reallocTailTilesFlat_vstart")
                    for _ in range(count)]

        def _swap(target, new_tiles):
            # In-place swap so the InstructionEmitter's references stay valid.
            target.clear()
            target.extend(new_tiles)

        _dealloc_all(self.vgprTilesA)
        _dealloc_all(self.vgprTilesB)
        _dealloc_all(self.vgprTilesSA)
        _dealloc_all(self.vgprTilesSB)

        _swap(self.vgprTilesA,
              _alloc_tiles(peaks.get('A', 0),
                           _tile_vgpr_count(info['tileInfoA'], cfg.lrA)))
        _swap(self.vgprTilesB,
              _alloc_tiles(peaks.get('B', 0),
                           _tile_vgpr_count(info['tileInfoB'], cfg.lrB)))
        if cfg.hasScale and info['scaleTileInfoA'] and info['scaleTileInfoB']:
            _swap(self.vgprTilesSA,
                  _alloc_tiles(peaks.get('SA', 0),
                               _tile_vgpr_count(info['scaleTileInfoA'], cfg.lrSA)))
            _swap(self.vgprTilesSB,
                  _alloc_tiles(peaks.get('SB', 0),
                               _tile_vgpr_count(info['scaleTileInfoB'], cfg.lrSB)))
        else:
            _swap(self.vgprTilesSA, [])
            _swap(self.vgprTilesSB, [])

        # Flat tiles are freed wholesale by deallocVgprTiles at kernel end;
        # there are no pre-freed tids to skip.
        self._tail_freed_tile_ids = {'A': set(), 'B': set(),
                                     'SA': set(), 'SB': set()}

    # ── Populate instructions ──────────────────────────────

    def populate_instructions(self, writer, kernel,
                              tileInfoA, tileInfoB, dtileInfo,
                              scaleTileInfoA=None, scaleTileInfoB=None,
                              tensorParametersA=None,
                              tensorParametersB=None) -> None:
        """Populate EmittedModule.instructions from placements and preOps.

        Uses per-tensor VGPR tile lists (vgprTilesA/B/SA/SB) indexed by
        vgprTileId from placement tile maps.
        """
        if self._preloop_emitted is None or self._ngll_emitted is None \
                or self._nll_emitted is None:
            self.build()

        self._kernel = kernel

        from Tensile.Components.Subtile.InstructionEmitter import InstructionEmitter

        emitter = InstructionEmitter(
            writer, kernel, self.config,
            tileInfoA, tileInfoB, dtileInfo,
            self.vgprTilesA, self.vgprTilesB,
            scaleTileInfoA, scaleTileInfoB,
            self.vgprTilesSA, self.vgprTilesSB,
            tensorParametersA=tensorParametersA,
            tensorParametersB=tensorParametersB,
        )

        # Rebuild all loop variants from current _emitted (which now has
        # vgpr_tile_maps populated by assign_vgpr_tiles, unlike the stale
        # copies from build()).
        self.build_preloop()
        self.build_ngll()
        self.build_nll()
        self.build_tailloop_pgr0()

        emitter.populate(self._preloop_emitted, unroll_iter=0)

        self._emitted_per_unroll = []
        self._ngll_per_unroll = []
        self._nll_per_unroll = []
        for ui in range(self.unroll_factor):
            em_copy = copy.deepcopy(self._emitted)
            emitter.populate(em_copy, unroll_iter=ui)
            self._emitted_per_unroll.append(em_copy)

            ngll_copy = copy.deepcopy(self._ngll_emitted)
            emitter.populate(ngll_copy, unroll_iter=ui)
            self._ngll_per_unroll.append(ngll_copy)

            nll_copy = copy.deepcopy(self._nll_emitted)
            emitter.populate(nll_copy, unroll_iter=ui)
            self._nll_per_unroll.append(nll_copy)

        self._emitter = emitter
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
        needs = getattr(self, 'needs_unrolling', None)
        factor = getattr(self, 'unroll_factor', 1)
        peaks = getattr(self, 'tile_peaks', {})
        buf.write(f"needsUnrolling: {needs}, "
                  f"unrollFactor: {factor}\n")
        peaks_str = ", ".join(f"{t}: {cnt}" for t, cnt in sorted(peaks.items()))
        buf.write(f"vgprTiles: {peaks_str}\n")
        for ui in range(factor):
            if factor > 1:
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
        """Print a placement label followed by its preOps, deps, and postOps."""
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
        if placement.postOps:
            buf.write("        postOps:\n")
            for op in placement.postOps:
                buf.write(f"            - {op}\n")
    

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
