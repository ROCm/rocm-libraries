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
import copy
import io
import math

from rocisa.code import Module


################################################################################
# C++-backed value/config helpers
#
# The pure data/config helpers below (``fmt_mt`` and
# ``SchedulerConfig.get_partition_candidates``) delegate unconditionally to the
# compiled ``tensile_writer.subtile.logical_scheduler`` nanobind extension,
# mirroring SubtileGeometry / TileInfo / InstructionScheduler. There is no
# opt-in flag and no Python mirror of this ported value/config math. The Python
# dataclasses remain the canonical scheduler objects and the scheduling passes
# stay in Python; only the ported value/config math lives in C++.
################################################################################

from tensile_writer.subtile import logical_scheduler as _cppls


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
    return _cppls.fmt_mt(mt)

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
        # NOTE: use raise (not assert) so validation survives `python -O`, which
        # disables asserts. The C++ counterpart always throws std::invalid_argument;
        # raising here keeps parity with the ported value layer under `python -O`.
        if isinstance(spec, (list, tuple)):
            if sum(spec) != total:
                raise ValueError(
                    f"partition sizes for {dim} must sum to {total}, got {sum(spec)}")
            if not all(s >= 1 for s in spec):
                raise ValueError(
                    f"all partition sizes for {dim} must be >= 1")
            if not all(s % mn == 0 for s in spec):
                raise ValueError(
                    f"partition sizes for {dim} must be multiples of mn={mn}, got {list(spec)}")
            return list(spec)
        s = spec if spec != 0 else total
        if not (1 <= s <= total):
            raise ValueError(
                f"partition size for {dim} must be in [1, {total}], got {s}")
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
        return _cppls.get_partition_candidates(tileInfoA, tileInfoB)



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


# ── C++ pass-pipeline delegation helpers ───────────────────
#
# The writer-free pass pipeline (place_LRs … remove_unnecessary_wait_lr_sync,
# plus assign_vgpr_tiles) is computed by the ported C++ passes::LogicalScheduler
# (tensile_writer.subtile.logical_scheduler.LogicalScheduler). The Python
# LogicalScheduler keeps the rocisa writer integration (emit / build_* /
# populate_instructions / alloc / Kernel.mainLoop emission) and its placement /
# op dataclasses, which the converter below rebuilds from the C++ value objects
# after each delegated pass. Placement object identity is kept stable across
# passes (so cached slot/placement references and Dep.ref identity hold) by
# reusing dataclass instances keyed on their immutable coordinates.

def _rg_to_cpp(g: Optional[ReadGranularity]):
    return _cppls.ReadGranularity(g.mn, g.k) if g is not None else None


def _config_to_cpp(cfg: 'SchedulerConfig'):
    """Build a C++ SchedulerConfig mirroring a Python SchedulerConfig."""
    return _cppls.SchedulerConfig(
        numMFMATilesM=cfg.numMFMATilesM,
        numMFMATilesN=cfg.numMFMATilesN,
        numSubIterK=cfg.numSubIterK,
        lrA=_rg_to_cpp(cfg.lrA), lrB=_rg_to_cpp(cfg.lrB),
        grA=_rg_to_cpp(cfg.grA), grB=_rg_to_cpp(cfg.grB),
        lrSA=_rg_to_cpp(cfg.lrSA), lrSB=_rg_to_cpp(cfg.lrSB),
        grSA=_rg_to_cpp(cfg.grSA), grSB=_rg_to_cpp(cfg.grSB),
        partitionSizeM=cfg.partitionSizeM,
        partitionSizeN=cfg.partitionSizeN,
        pgr=cfg.pgr,
    )


def _range_to_py(r) -> MFMATileRange:
    return MFMATileRange(r.subIterK_start, r.subIterK_end,
                         r.tileId_start, r.tileId_end)


def _ref_key(p) -> tuple:
    """Coordinate identity key for an LR/GR placement.

    Works for both C++ value placements and Python placement dataclasses; the
    full coordinate tuple uniquely identifies a placement across the schedule,
    so it re-establishes Dep.ref object identity after a C++ → Python rebuild.
    """
    t = p.tiles
    return (p.kind, p.tensor, p.partition, p.subIterK_slot, p.mtIteration,
            t.subIterK_start, t.subIterK_end, t.tileId_start, t.tileId_end)


def _op_to_py(op):
    """Rebuild a Python before-chain op dataclass from a C++ value op."""
    kind = op.kind
    if kind == 'wait_gr':
        c = op.wait_gr_counts
        counts = WaitGRCounts(c.A, c.B, c.SA, c.SB) if c is not None else None
        return WaitGROp(wait_gr_counts=counts, has_sync=op.has_sync,
                        adjustVmcnt=op.adjustVmcnt)
    if kind == 'wait_lr':
        return WaitLROp(has_sync=op.has_sync)
    if kind == 'sync':
        return SyncOp()
    if kind == 'mask_k':
        return MaskKOp(subIterK=op.subIterK)
    if kind == 'lr_inc':
        return LRIncOp(tensor=op.tensor)
    if kind == 'gr_inc':
        return GRIncOp(tensor=op.tensor)
    raise ValueError(f"unexpected C++ before-chain op kind: {kind!r}")


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
        # C++ pass-pipeline delegate + the coordinate→dataclass registry used to
        # keep placement object identity stable across delegated passes.
        self._cpp = None
        self._placement_reg: Dict[tuple, object] = {}
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

    def _lr_tensors(self) -> list:
        """Return list of (tensor_name, ReadGranularity) for all LR tensors."""
        cfg = self.config
        tensors = [('A', cfg.lrA), ('B', cfg.lrB)]
        if cfg.hasScale:
            tensors.append(('SA', cfg.lrSA))
            tensors.append(('SB', cfg.lrSB))
        return tensors

    # ── C++ pass-pipeline delegation ──────────────────────
    #
    # place_LRs … remove_unnecessary_wait_lr_sync (and the assign_vgpr_tiles
    # fork) are computed by the C++ passes::LogicalScheduler. Each method runs
    # the corresponding C++ pass (which auto-runs its prerequisites) and then
    # rebuilds the Python dataclass self._partitions from the C++ value objects.
    # The duplicate Python pass logic was removed; emit()/build_*/the rocisa
    # writer integration below still operate on self._partitions unchanged.

    def _ensure_cpp(self):
        if self._cpp is None:
            self._cpp = _cppls.LogicalScheduler(_config_to_cpp(self.config))
        return self._cpp

    def _mark_completed(self, pass_enum: Pass) -> None:
        """Mark a pass and all its transitive prerequisites complete.

        Mirrors the original ``_ensure_pass``-driven chain so downstream
        ``_ensure_pass`` checks see the same completed set the Python passes
        produced.
        """
        stack = [pass_enum]
        while stack:
            p = stack.pop()
            if p in self._completed:
                continue
            self._completed.add(p)
            stack.extend(_PASS_PIPELINE[p][1])

    def _run_cpp_pass(self, pass_enum: Pass) -> None:
        """Delegate one pass to the C++ scheduler, then rebuild self._partitions."""
        cpp = self._ensure_cpp()
        getattr(cpp, _PASS_PIPELINE[pass_enum][0])()
        self._sync_from_cpp()
        self._mark_completed(pass_enum)

    @staticmethod
    def _reuse_placement(reg: dict, cpp_p, cls):
        """Return the registered Python placement for cpp_p, creating it once."""
        key = _ref_key(cpp_p)
        p = reg.get(key)
        if p is None:
            p = cls(tensor=cpp_p.tensor, mtIteration=cpp_p.mtIteration,
                    tiles=_range_to_py(cpp_p.tiles),
                    subIterK_slot=cpp_p.subIterK_slot,
                    partition=cpp_p.partition)
            reg[key] = p
        return p

    def _refresh_placement(self, py_p, cpp_p, mfma: bool = False,
                           lr: bool = False) -> None:
        """Refresh the pass-populated (mutable) fields of a Python placement."""
        reg = self._placement_reg
        py_p.deps = [Dep(reg[_ref_key(d.ref)], d.mt_offset) for d in cpp_p.deps]
        py_p.preOps = [_op_to_py(o) for o in cpp_p.preOps]
        py_p.postOps = [_op_to_py(o) for o in cpp_p.postOps]
        if mfma:
            py_p.vgpr_tile_maps = {t: [dict(m) for m in maps]
                                   for t, maps in cpp_p.vgpr_tile_maps.items()}
        elif lr:
            py_p.vgpr_tile_map = [dict(m) for m in cpp_p.vgpr_tile_map]

    def _sync_from_cpp(self) -> None:
        """Rebuild self._partitions (Python dataclasses) from the C++ schedule.

        Placement and SubIterKSlot objects are reused across passes (keyed on
        immutable coordinates) so cached references and Dep.ref object identity
        stay valid; only the mutable fields (deps / preOps / postOps / vgpr
        maps) are refreshed. Dep.ref identity is re-established by coordinate
        match against the persistent placement registry.
        """
        cpp_parts = self._cpp.value_partitions()
        reg = self._placement_reg

        if self._partitions is None:
            self._partitions = []
        # Match the partition / slot structure, reusing SubIterKSlot objects.
        while len(self._partitions) < len(cpp_parts):
            self._partitions.append([])
        del self._partitions[len(cpp_parts):]

        for pi, cpp_slots in enumerate(cpp_parts):
            pyslots = self._partitions[pi]
            while len(pyslots) < len(cpp_slots):
                pyslots.append(SubIterKSlot(subIterK=len(pyslots)))
            del pyslots[len(cpp_slots):]
            for si, cs in enumerate(cpp_slots):
                slot = pyslots[si]
                slot.subIterK = cs.subIterK
                # MFMA: one per slot, keyed by slot position (never a dep ref).
                if cs.mfma is not None:
                    key = ('mfma', pi, si)
                    m = reg.get(key)
                    if m is None:
                        m = MFMAPlacement(subIterK=cs.mfma.subIterK,
                                          tileA=_range_to_py(cs.mfma.tileA),
                                          tileB=_range_to_py(cs.mfma.tileB))
                        reg[key] = m
                    slot.mfma = m
                else:
                    slot.mfma = None
                slot.lrs = [self._reuse_placement(reg, clr, LRPlacement)
                            for clr in cs.lrs]
                slot.grs = [self._reuse_placement(reg, cgr, GRPlacement)
                            for cgr in cs.grs]

        # Second walk: refresh mutable fields and resolve Dep.ref identity now
        # that every placement exists in the registry.
        for pi, cpp_slots in enumerate(cpp_parts):
            for si, cs in enumerate(cpp_slots):
                slot = self._partitions[pi][si]
                if slot.mfma is not None and cs.mfma is not None:
                    self._refresh_placement(slot.mfma, cs.mfma, mfma=True)
                for lr_p, clr in zip(slot.lrs, cs.lrs):
                    self._refresh_placement(lr_p, clr, lr=True)
                for gr_p, cgr in zip(slot.grs, cs.grs):
                    self._refresh_placement(gr_p, cgr)

    # ── Passes (delegated to C++) ─────────────────────────

    def place_LRs(self) -> List[List[SubIterKSlot]]:
        """Place MFMAs and LRs based on read granularities (C++-backed)."""
        self._run_cpp_pass(Pass.LR)
        return self._partitions

    def assign_vgpr_tiles(self):
        """Assign physical vgprTileIds to all placements (C++-backed).

        Sets self.tile_peaks, self.needs_unrolling, self.unroll_factor.
        """
        self._run_cpp_pass(Pass.VGPR_TILES)
        self.tile_peaks = dict(self._cpp.tile_peaks)
        self.unroll_factor = self._cpp.unroll_factor
        self.needs_unrolling = self._cpp.needs_unrolling

    def place_GRs(self) -> List[SubIterKSlot]:
        """Place Global Reads across partitions (C++-backed)."""
        self._run_cpp_pass(Pass.GR)
        return self._partitions[0]

    def annotate_deps(self):
        """Annotate each placement with its raw before-dependencies (C++-backed)."""
        self._run_cpp_pass(Pass.DEPS)

    def remove_unnecessary_gr_deps(self):
        """Remove redundant LR→GR deps already guaranteed by an earlier wait (C++-backed)."""
        self._run_cpp_pass(Pass.REMOVE_GR_DEPS)

    def remove_unnecessary_lr_deps(self):
        """Remove GR→LR collision deps already covered by an earlier sync (C++-backed)."""
        self._run_cpp_pass(Pass.REMOVE_LR_DEPS)

    def remove_cross_deps(self):
        """Replace cross-subIterK deps with wait preOps (C++-backed)."""
        self._run_cpp_pass(Pass.REMOVE_DEPS)

    def insert_gr_lr_inc(self):
        """Insert gr_inc/lr_inc preOps at MacroTile iteration transitions (C++-backed)."""
        self._run_cpp_pass(Pass.GR_INC)

    def group_lr_gr(self):
        """Group LR and GR placements into chains within each subIterK (C++-backed)."""
        self._run_cpp_pass(Pass.GROUP_LR_GR)

    def remove_unnecessary_wait_lr_sync(self):
        """Remove redundant wait_lr_sync from GRs after grouping (C++-backed)."""
        self._run_cpp_pass(Pass.REMOVE_WAIT_LR_SYNC)

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
        if self._kernel["ProblemType"]["DataTypeA"].isBFloat16():
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
        from Tensile.Components.Subtile.InstructionScheduler import instructionSchedule
        from rocisa.code import Module

        module = Module(label)
        module.addComment0(f"{label} start")
        for pi, partition_emitted in enumerate(emitted_3d):
            for k, em_list in enumerate(partition_emitted):
                module.addComment0(f"partition={pi} subIterK={k}")
                if schedule and em_list:
                    scheduled = instructionSchedule(em_list)
                    module.add(scheduled)
                else:
                    for em in em_list:
                        for inst in em.instructions:
                            module.add(inst)
        module.addComment0(f"{label} end")
        return module

    def emitMainAndExitLoops(self, writer, kernel):
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
        for ui in range(uf):
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
                                  self._nll_per_unroll[nll_ft]))
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
                                      self._nll_per_unroll[nll_idx]))
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
                pool = tile.regList.pool
                for val in tile:
                    if tile.index(val) % 4 == 0:
                        pool.checkIn(val)

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
                tile = tile_list[tid]
                pool = tile.regList.pool
                for j, v in enumerate(tile):
                    if j % 4 == 0:                # match _alloc_tiles block stride
                        pool.checkIn(v)
                self._tail_freed_tile_ids[tensor].add(tid)

    def _realloc_tail_tiles_flat(self, writer, peaks):
        """Free mainloop's per-partition tiles and reallocate flat tiles for
        the non-partitioned tail loop.

        `peaks[tensor]` comes from _compute_flat_tail_tile_state and is the
        number of distinct flat tile ids per tensor. The new flat tiles
        replace self.vgprTilesA/B/SA/SB; _tail_freed_tile_ids is cleared so
        deallocVgprTiles drops the flat set wholesale at kernel end.
        """
        from Tensile.Components.Subtile.Kernel import RegisterTileInfo

        cfg = self.config
        info = self._alloc_tile_info

        def _tile_vgpr_count(tileInfo, lrGran):
            return int(math.ceil(tileInfo.mmaTileRegCount * lrGran.k * lrGran.mn))

        def _dealloc_all(tiles):
            for tile in tiles:
                pool = tile.regList.pool
                for j, v in enumerate(tile):
                    if j % 4 == 0:
                        pool.checkIn(v)

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
