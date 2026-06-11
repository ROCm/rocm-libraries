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
  group                    — serialize and group (produce paths for instructionScheduleFromLists)
  remove_wait_lr_sync      — remove redundant wait_lr_sync after grouping
  emit                     — produce List[EmittedModule] with before-link chains

  TODO: add a pass to remove redundant wait_gr_sync on multi-partition configs
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Union
import io
import math

from rocisa.code import Module, Label
from rocisa.instruction import (
    SWaitCnt,
    SCmpEQU32, SCmpLeU32,
    SCBranchSCC1, SMovB32, VAndB32, VCmpGTI32, VCmpLeI32,
    VCmpLtI32, VCndMaskB32, VLShiftLeftB32, VLShiftRightB32, VMovB32, VSubI32,
)
from rocisa.container import vgpr, sgpr


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
from tensile_writer.subtile import instruction_scheduler as _cppsched
from tensile_writer.subtile.loop_orchestrator import (
    emit_loop as _emit_loop,
    emit_main_and_exit_loops as _emit_main_and_exit_loops,
    emit_tail_loop as _emit_tail_loop,
)
from tensile_writer.subtile.module_builder import ModuleBuilder as _ModuleBuilder

from Tensile.Components.Subtile.SubtileGREmit import (
    emitSingleBufferLoad, globalReadPtrUpdates, globalReadLDSBufferSwap,
    globalReadDoScaleSubtile, globalReadScalePtrUpdates,
)
from Tensile.Components.Subtile.SubtileLREmit import (
    emitSingleDsRead, localReadLDSBufferSwap,
    emitScaleDsRead,
)


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

    Compatible with SubtileBasedInstructionScheduler.instructionScheduleFromLists().
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

        The before-link topology (wait_gr standalone with later deps chaining
        from it, WaitGROp/WaitLROp has_sync expanding into a sync module, and
        same-subIterK Dep deps becoming ordering constraints) is computed by the
        ported C++ passes::LogicalScheduler emit pass; this method rebuilds the
        Python EmittedModule dataclasses from the exported value modules. There
        is no second Python implementation of the before-link wiring.

        Placement sources reuse the persistent placement dataclasses (keyed on
        their immutable coordinates via the registry built by the delegated
        passes) so emit sources stay identical to self._partitions — keeping
        Dep.ref identity and letting assign_vgpr_tiles' vgpr maps flow through to
        emission. Before-chain op sources are rebuilt as fresh op dataclasses.
        """
        self._ensure_pass(Pass.REMOVE_WAIT_LR_SYNC)

        cpp = self._ensure_cpp()
        cpp.emit()

        all_partitions = []
        for pi, partition_emitted in enumerate(cpp.value_emitted()):
            py_partition = []
            for k, cpp_mods in enumerate(partition_emitted):
                emitted: List[EmittedModule] = [
                    EmittedModule(moduleId=cm.moduleId, before=cm.before,
                                  source=self._emit_source_to_py(cm.source, pi, k))
                    for cm in cpp_mods
                ]
                py_partition.append(emitted)
            all_partitions.append(py_partition)

        self._emitted = all_partitions
        self._completed.add(Pass.EMIT)
        return all_partitions

    def _emit_source_to_py(self, cpp_src, pi: int, k: int) -> Emittable:
        """Map a C++ EmittedModule source (value Emittable) to its Python source.

        Placement sources are looked up in the persistent placement registry
        (so they share identity with self._partitions and pick up the vgpr maps
        assigned by assign_vgpr_tiles); before-chain op sources are rebuilt as
        fresh op dataclasses. emit() only ever produces placement, wait_gr,
        wait_lr, sync, lr_inc and gr_inc sources.
        """
        kind = cpp_src.kind
        if kind == 'mfma':
            return self._placement_reg[('mfma', pi, k)]
        if kind in ('lr', 'gr'):
            return self._placement_reg[_ref_key(cpp_src)]
        return _op_to_py(cpp_src)

    def build(self):
        """Build mainloop (emit delegates the before-link graph to C++)."""
        self.emit()
        self._completed.add(Pass.BUILD)

    # ── Loop variant derivation ────────────────────────────

    # The loop-variant schedule rewrites (NGLL / NLL / preloop / tail-PGR0)
    # live in the ported C++ passes::LogicalScheduler. The wrappers below run
    # the corresponding C++ builder and rebuild the Python EmittedModule
    # dataclasses from the exported value modules; they hold no duplicate
    # schedule rewrite logic. NGLL/NLL reuse the mainloop placement registry
    # (coordinate-only sources, like emit()); preloop/tail synthesize fresh
    # placements, so their value sources carry full placement data.

    def _variant_from_cpp(self, cpp_partitions, full: bool):
        """Rebuild [partition][subIterK][EmittedModule] from C++ value modules.

        ``full=False`` resolves placement sources through the persistent
        placement registry (NGLL/NLL share the mainloop placements);
        ``full=True`` rebuilds fresh placement dataclasses from the value
        source (preloop/tail use synthesized placements with their own vgpr
        tile maps).
        """
        out = []
        for pi, partition_emitted in enumerate(cpp_partitions):
            py_partition = []
            for k, cpp_mods in enumerate(partition_emitted):
                emitted = [
                    EmittedModule(
                        moduleId=cm.moduleId, before=cm.before,
                        source=(self._full_source_to_py(cm.source) if full
                                else self._emit_source_to_py(cm.source, pi, k)))
                    for cm in cpp_mods
                ]
                py_partition.append(emitted)
            out.append(py_partition)
        return out

    def _full_source_to_py(self, cpp_src) -> Emittable:
        """Rebuild a fresh Python source dataclass from a C++ value Emittable.

        Used for the synthesized preloop/tail placements (which are not in the
        mainloop placement registry): placement vgpr tile maps and op fields are
        copied straight from the value source. The InlineModuleOp ``build``
        callback stays None here and is attached by the tail-loop wrapper.
        """
        kind = cpp_src.kind
        if kind == 'mfma':
            m = MFMAPlacement(subIterK=cpp_src.subIterK,
                              tileA=_range_to_py(cpp_src.tileA),
                              tileB=_range_to_py(cpp_src.tileB))
            m.vgpr_tile_maps = {t: [dict(x) for x in maps]
                                for t, maps in cpp_src.vgpr_tile_maps.items()}
            return m
        if kind == 'lr':
            lr = LRPlacement(tensor=cpp_src.tensor,
                             mtIteration=cpp_src.mtIteration,
                             tiles=_range_to_py(cpp_src.tiles),
                             subIterK_slot=cpp_src.subIterK_slot,
                             partition=cpp_src.partition)
            lr.vgpr_tile_map = [dict(x) for x in cpp_src.vgpr_tile_map]
            return lr
        if kind == 'gr':
            return GRPlacement(tensor=cpp_src.tensor,
                               mtIteration=cpp_src.mtIteration,
                               tiles=_range_to_py(cpp_src.tiles),
                               subIterK_slot=cpp_src.subIterK_slot,
                               partition=cpp_src.partition)
        if kind == 'mask_k':
            op = MaskKOp(subIterK=cpp_src.subIterK)
            op.vgpr_tile_map = {t: [dict(x) for x in maps]
                                for t, maps in cpp_src.vgpr_tile_map.items()}
            return op
        if kind == 'skip':
            return SkipOp(compare=cpp_src.compare, value=cpp_src.value,
                          target=cpp_src.target, rawLabel=cpp_src.rawLabel,
                          branchComment=cpp_src.branchComment)
        if kind == 'inline':
            return InlineModuleOp(label=cpp_src.label)
        return _op_to_py(cpp_src)

    def build_ngll(self) -> List[List[List[EmittedModule]]]:
        """NGLL (No Global Load Loop): mainloop without GR(n+2), GR_INC.

        WaitGR inflight counts are zeroed since no new GRs are in flight.
        The schedule rewrite is performed by the C++ scheduler; this wrapper
        rebuilds the Python EmittedModule dataclasses from the value modules.
        """
        self._ensure_pass(Pass.EMIT)
        cpp = self._ensure_cpp()
        cpp.build_ngll()
        self._ngll_emitted = self._variant_from_cpp(cpp.value_ngll(), full=False)
        return self._ngll_emitted

    def build_nll(self) -> List[List[List[EmittedModule]]]:
        """NLL (No Load Loop): mainloop without GR, LR(n+1), GR_INC, LR_INC,
        WaitGR(n+1)+Sync. Keeps LR(n), MFMAs, WaitGR(n) with zeroed counts.

        The schedule rewrite is performed by the C++ scheduler; this wrapper
        rebuilds the Python EmittedModule dataclasses from the value modules.
        """
        self._ensure_pass(Pass.EMIT)
        cpp = self._ensure_cpp()
        cpp.build_nll()
        self._nll_emitted = self._variant_from_cpp(cpp.value_nll(), full=False)
        return self._nll_emitted

    def build_tailloop_pgr0(self) -> List[List[List[EmittedModule]]]:
        """Template for Tailloop based on PGR0 schedule.

        Returns [partition][groups] where each group has at most one MFMA.

        The tail loop runs flat (no partitioning): per subIterK the C++
        scheduler emits one LR pass covering every unique (tensor, tile_range),
        one boundary mask, then every partition's MFMAs back-to-back. This
        relies on the flat tile-id layout from _compute_flat_tail_tile_state
        (and the matching vgpr realloc in _realloc_tail_tiles_flat) so each
        unique partition group has its own vgpr range.

        The schedule construction itself is performed in C++; this wrapper
        feeds it the flat tile layout plus the BF16 / MatrixInstK boundary
        inputs, rebuilds the Python EmittedModule dataclasses, and attaches the
        writer-built BF16 boundary-fixup callback (which cannot live in C++).
        """
        # Flat tile layout: every unique (tensor, partition_group) gets its own
        # vgpr tile id. Stays in Python — it is also consumed by getNumVgpr and
        # _realloc_tail_tiles_flat.
        tile_maps, self._flat_tail_peaks = self._compute_flat_tail_tile_state()
        # Legacy unused-tile bookkeeping: in the flat path we replace the vgpr
        # tiles wholesale at tail entry, so nothing here.
        self._tail_unused_tile_ids = {'A': set(), 'B': set(),
                                      'SA': set(), 'SB': set()}

        # bf16-only: an OOB dwordx4 load can corrupt the trailing 16-bit element
        # at the K-boundary; the C++ builder inserts a sync + boundary
        # InlineModuleOp placeholder, and we attach the writer callback below.
        bf16 = bool(self._kernel["ProblemType"]["DataTypeA"].isBFloat16())
        miK = int(self._kernel["MatrixInstK"])

        cpp = self._ensure_cpp()
        cpp.build_tailloop_pgr0(tile_maps, bf16, miK)
        self._tailloop_emitted = self._variant_from_cpp(
            cpp.value_tailloop(), full=True)
        if bf16:
            self._attach_tail_boundary_build()
        return self._tailloop_emitted

    def _attach_tail_boundary_build(self) -> None:
        """Attach the writer-built BF16 boundary DTL-load callback to the tail
        InlineModuleOp emitted by the C++ builder (kind 'inline',
        label 'tail_boundary_ab'). The callback captures writer state and so
        stays in Python."""
        def _boundary_build(em):
            return em.writer.tailLoopBoundaryDtlLoadAB(
                em.kernel,
                em.tensorParametersMap['A'],
                em.tensorParametersMap['B'])

        for partition_emitted in self._tailloop_emitted:
            for group in partition_emitted:
                for em in group:
                    src = em.source
                    if getattr(src, 'kind', None) == 'inline' \
                            and getattr(src, 'label', None) == 'tail_boundary_ab':
                        src.build = _boundary_build

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
        The schedule construction is performed by the C++ scheduler; this
        wrapper rebuilds the Python EmittedModule dataclasses from the value
        modules.
        """
        self._ensure_pass(Pass.VGPR_TILES)
        cpp = self._ensure_cpp()
        cpp.build_preloop()
        self._preloop_emitted = self._variant_from_cpp(
            cpp.value_preloop(), full=True)
        return self._preloop_emitted

    def _emitLoop(self, writer, kernel, label, emitted_3d, unroll_iter=0,
                  schedule=True):
        """Emit a loop section from a 3D emitted structure.

        emitted_3d: [partition][subIterK][EmittedModule]

        Delegates to the C++ loop_orchestrator.emit_loop, which iterates the
        3D structure and calls self._emitter.emit_module per EmittedModule,
        routing through instructionScheduleFromLists when schedule=True.
        """
        builder = _ModuleBuilder()
        return _emit_loop(
            builder, self._emitter.emit_module, instructionScheduleFromLists,
            emitted_3d, label, unroll_iter, schedule)

    def emitMainAndExitLoops(self, writer, kernel):
        """Emit preloop + mainloop + NGLL + NLL exit paths (no tail).

        Owns all control flow (labels, branches, counter management) for the
        main unrolled pipeline. For unroll_factor > 1, emits per-unroll copies
        with correct vgpr tiles. Each mainloop exit jumps to its corresponding
        NGLL→NLL pair. The tail loop is emitted separately by emitTailLoop()
        so the orchestrator (Subtile.Kernel.mainLoop) can wrap it with the
        runtime K%DU counter setup and skip branch.

        Delegates structural control flow to the C++ loop_orchestrator.
        """
        assert Pass.POPULATE in self._completed, \
            "populate_instructions() must be called before emitMainAndExitLoops()"

        builder = _ModuleBuilder()
        return _emit_main_and_exit_loops(
            builder,
            self._emitter.emit_module,
            instructionScheduleFromLists,
            self._preloop_emitted,
            self._emitted,
            self._ngll_emitted,
            self._nll_emitted,
            bool(kernel["NoTailLoop"]),
            self.config.pgr,
            self.unroll_factor,
        )

    def emitTailLoop(self, writer, kernel):
        """Emit the tail loop body only (no counter setup, no skip branch).

        Returns an empty Module when NoTailLoop is set. The caller is
        responsible for emitting calculateLoopNumIter(-1) before this and
        closeLoop(emitEndLabelOnly=True) after, mirroring the legacy
        KernelWriter pattern.

        Delegates structural emission to the C++ loop_orchestrator.
        """
        assert Pass.POPULATE in self._completed, \
            "populate_instructions() must be called before emitTailLoop()"

        if kernel["NoTailLoop"]:
            return Module("TailLoop")

        # Swap to the flat tail vgpr tile layout. Frees the mainloop's
        # per-partition tiles back to the pool and reallocates a flat set
        # sized by _compute_flat_tail_tile_state (already invoked by
        # build_tailloop_pgr0; peaks stashed on self._flat_tail_peaks).
        self._realloc_tail_tiles_flat(writer, self._flat_tail_peaks)

        # emit_mask_k_init allocates the mask VGPRs (_tail_vDiff etc.) and
        # yields instructions that load per-lane invariants.  These VGPRs
        # must stay live through the loop body (emit_loop calls emit_module
        # for each MaskKOp which reads _tail_vDiff).
        # emit_mask_k_done always returns [] but frees those VGPRs; it MUST
        # run AFTER the loop body, not before.
        mask_k_init_items = list(self._emitter.emit_mask_k_init())

        builder = _ModuleBuilder()
        result = _emit_tail_loop(
            builder,
            self._emitter.emit_module,
            self._tailloop_emitted,
            mask_k_init_items,
            [],  # emit_mask_k_done always returns [] — cleanup runs below
        )

        # Release the tail-loop mask VGPRs after the loop body has used them.
        self._emitter.emit_mask_k_done()

        return result

    # ── VGPR tile allocation ──────────────────────────────

    def getNumVgpr(self, tileInfoA, tileInfoB,
                        scaleTileInfoA=None, scaleTileInfoB=None) -> int:
        """Return the total number of VGPRs needed across all tensors (A, B, SA, SB).

        Delegates to C++ passes::LogicalScheduler.get_num_vgpr which computes
        max(mainloop_peak_vgprs, flat_tail_peak_vgprs).  The two layouts
        never coexist (the tail loop frees and reallocates at entry) so the
        kernel VGPR budget is the larger of the two peaks.

        Must be called after scheduling is complete (assign_vgpr_tiles).
        """
        self._ensure_pass(Pass.VGPR_TILES)
        cpp = self._ensure_cpp()
        return cpp.get_num_vgpr(
            tileInfoA.mmaTileRegCount,
            tileInfoB.mmaTileRegCount,
            scaleTileInfoA.mmaTileRegCount if scaleTileInfoA else 0.0,
            scaleTileInfoB.mmaTileRegCount if scaleTileInfoB else 0.0,
        )

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
        """Create the InstructionEmitter used during loop emission.

        Rebuilds all loop-variant EmittedModule graphs (preloop, NGLL, NLL,
        tail) so they reflect the latest vgpr_tile_maps from assign_vgpr_tiles.
        Instructions are emitted on demand during _emitLoop via
        InstructionEmitter.emit_module — no deep-copy or pre-population loop.
        """
        if self._preloop_emitted is None or self._ngll_emitted is None \
                or self._nll_emitted is None:
            self.build()

        self._kernel = kernel

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


################################################################################
# Instruction scheduling helpers
# (formerly Tensile/Components/Subtile/InstructionScheduler.py)
#
# External callers: LogicalScheduler._emitLoop / emitMainAndExitLoops pass
# ``instructionScheduleFromLists`` as a callback to the C++ loop orchestrator.
# ``extractPathsFromBeforeDeps`` is used by LogicalScheduler.print_emit_dep_order
# for diagnostics.
################################################################################

def extractPathsFromBeforeDeps(emittedModules) -> tuple:
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

    mfmaBefore = emittedModules[mfmaIdx].before
    preMfmaTarget = None
    if mfmaBefore is not None:
        bi = idToIdx.get(mfmaBefore)
        if bi is not None and bi in nonMfmaSet:
            preMfmaTarget = bi

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
                f"extractPathsFromBeforeDeps expects unique child per predecessor, got {child[parent]} and {i} for {parent}"
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
        assert order, f"extractPathsFromBeforeDeps produced empty path for module {mid}"
        for i in order:
            used[i] = True
        paths.append(order)

    preMfmaPaths: List[List[int]] = []
    regularPaths: List[List[int]] = []
    for path in paths:
        if preMfmaTarget is not None and preMfmaTarget in path:
            preMfmaPaths.append(path)
        else:
            regularPaths.append(path)

    return mfmaIdx, regularPaths, preMfmaPaths


def instructionSchedule(emittedModules):
    """Interleave non-MFMA instructions between MFMAs (slot-based placement).

    Thin adapter over the C++ slot-placement algorithm. Converts the live
    rocisa emitted-module objects to the data-only C++ model, runs the C++
    scheduler, and returns a rocisa ``Module`` in the resulting emission order
    with the waitcnt vmcnt post-pass applied.
    """
    return _cppsched.instructionSchedule(emittedModules)


def instructionScheduleFromLists(emittedModules, instruction_lists):
    """C++-backed scheduler driven by caller-supplied instruction lists.

    Like ``instructionSchedule`` but instructions are provided externally
    (on-demand emission) instead of reading from ``em.instructions``.
    Thin adapter over the C++ slot-placement algorithm.
    """
    return _cppsched.instructionScheduleFromLists(emittedModules, instruction_lists)


################################################################################
# Instruction emitter
# (formerly Tensile/Components/Subtile/InstructionEmitter.py)
#
# External callers: LogicalScheduler.populate_instructions creates an
# InstructionEmitter and stores it on self._emitter. The C++ loop orchestrator
# calls back into self._emitter.emit_module per EmittedModule.
################################################################################

class SWaitCntEx(SWaitCnt):
    """SWaitCnt subclass carrying the adjustVmcnt flag for the vmcnt post-pass.

    Externally justified facade: rocisa.SWaitCnt is a C++ extension type that
    does not support dynamic attributes, so the adjustVmcnt flag (consumed by
    classifyInstruction via duck-typing) must live on a Python subclass. C++
    owns the post-pass *computation* (instruction_scheduler.hpp); this class is
    the minimal Python surface required for the shim to read the flag back from
    live rocisa objects.
    """
    def __init__(self, adjustVmcnt=True, **kwargs):
        super().__init__(**kwargs)
        self._adjustVmcnt = adjustVmcnt

    @property
    def adjustVmcnt(self):
        return self._adjustVmcnt

    def __deepcopy__(self, memo):
        return SWaitCntEx(
            adjustVmcnt=self._adjustVmcnt,
            vlcnt=self.vlcnt, vscnt=self.vscnt,
            dscnt=self.dscnt, kmcnt=self.kmcnt,
            comment=self.comment)


class InstructionEmitter:
    """Emits GPU instructions for each opType in the LogicalScheduler output.

    VGPR tile indexing uses placement-level tile maps (tileId → vgprTileId)
    set by assign_vgpr_tiles(). Per-tensor VGPR tile lists are indexed by
    vgprTileId. All tensors (A, B, SA, SB) use the same tile-map approach.
    """

    def __init__(self, writer, kernel, config,
                 tileInfoA, tileInfoB, dtileInfo,
                 vgprTilesA, vgprTilesB,
                 scaleTileInfoA=None, scaleTileInfoB=None,
                 vgprTilesSA=None, vgprTilesSB=None,
                 tensorParametersA=None, tensorParametersB=None):
        self.writer = writer
        self.kernel = kernel
        self.config = config
        self.tileInfoA = tileInfoA
        self.tileInfoB = tileInfoB
        self.dtileInfo = dtileInfo
        self.vgprTilesA = vgprTilesA
        self.vgprTilesB = vgprTilesB
        self.vgprTilesSA = vgprTilesSA or []
        self.vgprTilesSB = vgprTilesSB or []
        self.tensorParametersMap = {}
        if tensorParametersA is not None:
            self.tensorParametersMap['A'] = tensorParametersA
        if tensorParametersB is not None:
            self.tensorParametersMap['B'] = tensorParametersB

        self.hasScale = scaleTileInfoA is not None and scaleTileInfoB is not None
        self.subtileShapeK = tileInfoA.subtileShape[1]
        self.tileInfoMap = {'A': tileInfoA, 'B': tileInfoB}
        if self.hasScale:
            self.tileInfoMap['SA'] = scaleTileInfoA
            self.tileInfoMap['SB'] = scaleTileInfoB

        self._dispatch = {
            'mfma':         lambda em, ui: self.emit_mfma(em.source, ui),
            'lr':           lambda em, ui: self.emit_lr(em.source, ui),
            'gr':           lambda em, ui: self.emit_gr(em.source),
            'wait_gr':      lambda em, ui: self.emit_wait_gr(em.source),
            'wait_lr':      lambda em, ui: self.emit_wait_lr(),
            'sync':         lambda em, ui: self.emit_sync(),
            'lr_inc':           lambda em, ui: self.emit_lr_inc(em.source),
            'gr_inc':           lambda em, ui: self.emit_gr_inc(em.source),
            'skip':             lambda em, ui: self.emit_skip(em.source),
            'mask_k':       lambda em, ui: self.emit_mask_k(em.source),
            'inline':       lambda em, ui: self.emit_inline(em.source),
        }

        self._tail_vDiff = None

    def emit_mfma(self, placement, unroll_iter=0):
        """Emit MFMA instructions from MFMAPlacement."""
        # Lazy import to break the Kernel.py → LogicalScheduler.py → Kernel.py
        # circular import (Kernel.py imports LogicalScheduler at module load).
        from Tensile.Components.Subtile.Kernel import emitMfmaInstruction
        module = Module()
        subIterK = placement.subIterK
        tile_maps = {t: placement.vgpr_tile_maps[t][unroll_iter]
                     for t in placement.vgpr_tile_maps}

        for a in placement.tileA.tileId_list:
            for b in placement.tileB.tileId_list:
                groupA = (a // self.config.lrA.mn) * self.config.lrA.mn
                groupB = (b // self.config.lrB.mn) * self.config.lrB.mn
                aTile = self.vgprTilesA[tile_maps['A'][groupA]]
                bTile = self.vgprTilesB[tile_maps['B'][groupB]]
                dTile = self.dtileInfo.vgprTiles[a + b * self.dtileInfo.localMMATileGrid[0]]

                if self.hasScale:
                    scaleGroupA = (a // self.config.lrSA.mn) * self.config.lrSA.mn
                    scaleGroupB = (b // self.config.lrSB.mn) * self.config.lrSB.mn
                    scaleATile = self.vgprTilesSA[tile_maps['SA'][scaleGroupA]]
                    scaleBTile = self.vgprTilesSB[tile_maps['SB'][scaleGroupB]]
                    scaleAVgpr = next(iter(scaleATile))
                    scaleBVgpr = next(iter(scaleBTile))
                    mShapeA = self.tileInfoMap['SA'].lrSubtileShape[0]
                    mShapeB = self.tileInfoMap['SB'].lrSubtileShape[0]
                    kShapeA = self.tileInfoMap['SA'].lrSubtileShape[1]
                    kShapeB = self.tileInfoMap['SB'].lrSubtileShape[1]
                    sAsel = (a % mShapeA) + mShapeA * (subIterK % kShapeA)
                    sBsel = (b % mShapeB) + mShapeB * (subIterK % kShapeB)
                else:
                    scaleAVgpr = scaleBVgpr = -1
                    sAsel = sBsel = 0

                module.add(emitMfmaInstruction(
                    self.writer, self.kernel, aTile, bTile, dTile, dTile,
                    scaleAVgpr=scaleAVgpr, scaleBVgpr=scaleBVgpr,
                    scaleAsel=sAsel, scaleBsel=sBsel,
                    comment=f"MFMA C[{a},{b}] += A[{a},K={subIterK}] * B[{b},K={subIterK}]"))
        return list(module.flatitems())

    def emit_lr(self, placement, unroll_iter=0):
        """Emit LR (ds_read) instructions from LRPlacement."""
        module = Module()
        tensor = placement.tensor
        tile_map = placement.vgpr_tile_map[unroll_iter] if placement.vgpr_tile_map else {}

        if tensor in ('A', 'B'):
            ti = self.tileInfoMap[tensor]
            vgprTiles = self.vgprTilesA if tensor == 'A' else self.vgprTilesB
            lrGran = self.config.lrA if tensor == 'A' else self.config.lrB
            for tileId in range(placement.tiles.tileId_start, placement.tiles.tileId_end, lrGran.mn):
                for k in range(placement.tiles.subIterK_start, placement.tiles.subIterK_end, lrGran.k):
                    subtileK = k // self.subtileShapeK
                    subIterK_within = k % self.subtileShapeK
                    dstTile = vgprTiles[tile_map[tileId]]
                    module.add(emitSingleDsRead(
                        ti, tileId, subtileK, subIterK_within, dstTile))
        elif tensor in ('SA', 'SB'):
            tc = 'MXSA' if tensor == 'SA' else 'MXSB'
            ti = self.tileInfoMap[tensor]
            lrGran = self.config.lrSA if tensor == 'SA' else self.config.lrSB
            vgprTilesScale = self.vgprTilesSA if tensor == 'SA' else self.vgprTilesSB
            for tileId in range(placement.tiles.tileId_start, placement.tiles.tileId_end, lrGran.mn):
                scaleGroupIdx = tileId // lrGran.mn
                groupKey = scaleGroupIdx * lrGran.mn
                kGroupIdx = placement.tiles.subIterK_start // ti.lrSubtileShape[1]
                numKGroups = ti.lrLocalSubtileGrid[1]
                dsOffset = int(ti.lrSubtileSize) * (scaleGroupIdx * numKGroups + kGroupIdx)
                vdst = next(iter(vgprTilesScale[tile_map[groupKey]]))
                module.add(emitScaleDsRead(
                    tc, vdst, ti.sharedVgprLROffset[0], dsOffset, scaleGroupIdx,
                    placement.tiles.subIterK_start))
        return list(module.flatitems())

    def emit_gr(self, placement):
        """Emit GR (buffer_load) instructions from GRPlacement."""
        module = Module()
        tensor = placement.tensor
        if tensor in ('A', 'B'):
            ti = self.tileInfoMap[tensor]
            grGran = self.config.grA if tensor == 'A' else self.config.grB
            for tileId in range(placement.tiles.tileId_start, placement.tiles.tileId_end, grGran.mn):
                for k in range(placement.tiles.subIterK_start, placement.tiles.subIterK_end, grGran.k):
                    subtileK = k // self.subtileShapeK
                    module.add(emitSingleBufferLoad(ti, self.kernel, tileId, subtileK))
        elif tensor in ('SA', 'SB'):
            tc = 'MXSA' if tensor == 'SA' else 'MXSB'
            module.add(globalReadDoScaleSubtile(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_wait_gr(self, source):
        """Emit SWaitCnt for wait_gr from BaseOp with wait_gr_counts.

        The count arithmetic uses tileInfoA/B.loadRatioGR (Python-side geometry).
        SWaitCntEx carries the adjustVmcnt flag needed by classifyInstruction
        (see SWaitCntEx docstring for why this subclass cannot move to C++).

        Note: a C++ ModuleBuilder::wait_gr_swait() helper was considered but
        removed (sio). SWaitCntEx must remain a Python-only subclass because
        rocisa.SWaitCnt is a C++ extension type that does not support dynamic
        attributes; constructing SWaitCntEx directly here is both correct and
        avoids an unnecessary intermediate allocation.
        """
        counts = source.wait_gr_counts
        if counts is None:
            return []
        grMap = {'A': max(1, int(1.0 / self.tileInfoA.loadRatioGR)),
                 'B': max(1, int(1.0 / self.tileInfoB.loadRatioGR)),
                 'SA': 1,
                 'SB': 1}
        grCnt = (counts.A * grMap['A'] + counts.B * grMap['B'] +
                 counts.SA * grMap['SA'] + counts.SB * grMap['SB'])
        return [SWaitCntEx(
            adjustVmcnt=source.adjustVmcnt, vlcnt=grCnt, vscnt=-1,
            comment=(f"Wait GR (per-subIterK): "
                     f"A={counts.A} B={counts.B} SA={counts.SA} SB={counts.SB}"))]

    def emit_wait_lr(self):
        """Emit SWaitCnt(dscnt=0) — delegated to C++ ModuleBuilder."""
        return [_ModuleBuilder().wait_lr()]

    def emit_sync(self):
        """Emit SBarrier — delegated to C++ ModuleBuilder."""
        return [_ModuleBuilder().barrier()]

    def emit_inline(self, source):
        """Emit a writer-built Module supplied by an InlineModuleOp callback."""
        if source.build is None:
            return []
        mod = source.build(self)
        return list(mod.flatitems()) if mod is not None else []

    def emit_lr_inc(self, source):
        """Emit localReadLDSBufferSwap for a single tensor."""
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        module = Module()
        module.add(localReadLDSBufferSwap(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_gr_inc(self, source):
        """Emit globalReadPtrUpdates + globalReadLDSBufferSwap for a single tensor."""
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        module = Module()
        if tensor in ('SA', 'SB'):
            module.add(globalReadScalePtrUpdates(tc, self.writer, self.kernel))
        else:
            module.add(globalReadPtrUpdates(tc, self.writer, self.kernel))
        module.add(globalReadLDSBufferSwap(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_skip(self, source):
        """Emit skip guard: compare LoopCounterL and branch."""
        labelName = source.target if source.rawLabel else f"SkipTo{source.target}"
        skipLabel = Label(labelName, "")
        cmpMap = {"EQ": SCmpEQU32, "LE": SCmpLeU32}
        cmpCls = cmpMap[source.compare]
        module = Module()
        if -16 <= source.value <= 64:
            module.add(cmpCls(
                src0=sgpr("LoopCounterL"), src1=source.value,
                comment=f"LoopCounter {source.compare} {source.value}?"))
        else:
            with self.writer.allocTmpSgpr(1) as litSgprInfo:
                litSgpr = litSgprInfo.idx
                module.add(SMovB32(
                    dst=sgpr(litSgpr), src=hex(source.value),
                    comment=f"stage literal {source.value} (non-inline) for cmp src1"))
                module.add(cmpCls(
                    src0=sgpr("LoopCounterL"), src1=sgpr(litSgpr),
                    comment=f"LoopCounter {source.compare} {source.value}?"))
        module.add(SCBranchSCC1(
            labelName=skipLabel.getLabelName(),
            comment=source.branchComment or f"skip to {source.target}"))
        return list(module.flatitems())

    def _mfma_K_constants(self):
        """Constants used by both mask emitters.

        Returns (numMIInUnroll, dividerFortidInK):
          * numMIInUnroll    = MI_M * MI_K / WavefrontSize — per-lane K chunk
            consumed by one MFMA call (contiguous K-elements held by each lane).
          * dividerFortidInK = MI_M — lanes with the same
            (Serial % WavefrontSize) / MI_M share the same K-position.
        """
        kernel = self.kernel
        MI_M             = kernel["MatrixInstM"]
        MI_K             = kernel["MatrixInstK"]
        waveSize         = kernel["WavefrontSize"]
        numMIInUnroll    = MI_M * MI_K // waveSize
        dividerFortidInK = MI_M
        return numMIInUnroll, dividerFortidInK

    def emit_mask_k_init(self):
        """Stage the per-subIterK invariants for emit_mask_k."""
        numMIInUnroll, dividerFortidInK = self._mfma_K_constants()
        writer = self.writer
        module = Module()

        waveSize = self.kernel["WavefrontSize"]
        divLog2  = (dividerFortidInK).bit_length() - 1
        mulLog2  = (numMIInUnroll).bit_length() - 1

        kReg = writer.vgprPool.checkOut(1, "tail_kReg")
        module.add(VAndB32(
            dst=vgpr(kReg), src0=waveSize - 1, src1=vgpr("Serial"),
            comment=f"tail_kReg = Serial & {waveSize - 1} (Serial % {waveSize})"))
        module.add(VLShiftRightB32(
            dst=vgpr(kReg), shiftHex=divLog2, src=vgpr(kReg),
            comment=f"tail_kReg >>= {divLog2} (tail_kReg / {dividerFortidInK})"))

        workKVgpr = writer.vgprPool.checkOut(1, "tail_workK")
        self._tail_vDiff = writer.vgprPool.checkOut(1, "tail_vDiff")
        loopCounterName = writer.loopCounterName(
            self.kernel, writer.states.unrollIdx)
        module.add(VLShiftLeftB32(
            dst=vgpr(workKVgpr), shiftHex=mulLog2, src=vgpr(kReg),
            comment=f"laneK_0 = tail_kReg << {mulLog2} (tail_kReg * {numMIInUnroll})"))
        writer.vgprPool.checkIn(kReg)
        module.add(VSubI32(
            dst=vgpr(self._tail_vDiff),
            src0=sgpr(loopCounterName), src1=vgpr(workKVgpr),
            comment="diff = rem - laneK_0 (shared across all subIterK)"))
        writer.vgprPool.checkIn(workKVgpr)

        self._tail_boundaryMask = None
        laneSGPRCount = writer.states.laneSGPRCount
        if self.kernel["ProblemType"]["DataTypeA"].isBFloat16():
            halfMaskVgpr = writer.vgprPool.checkOut(1, "tail_halfMask")
            module.add(VMovB32(
                dst=vgpr(halfMaskVgpr), src="0x0000FFFF",
                comment="BF16 half-mask: keep K0 (low 16b), zero K1 (high 16b)"))

            kStride = 2
            assert numMIInUnroll % kStride == 0, \
                f"numMIInUnroll ({numMIInUnroll}) must be a multiple of kStride ({kStride})"
            numBoundaryMasks = numMIInUnroll // kStride
            vDLaneRem = writer.vgprPool.checkOut(1, "tail_vDLaneRem")
            module.add(VAndB32(
                dst=vgpr(vDLaneRem),
                src0=sgpr(loopCounterName), src1=numMIInUnroll - 1,
                comment=f"d = rem % {numMIInUnroll} (boundary-mask pattern depends only on this)"))
            self._tail_boundaryMask = [
                writer.vgprPool.checkOut(1, f"tail_boundaryMask{i}")
                for i in range(numBoundaryMasks)
            ]
            with writer.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount) as tmpSgprInfo:
                maskSgpr = tmpSgprInfo.idx
                for i in range(numBoundaryMasks):
                    bm = self._tail_boundaryMask[i]
                    hiBound = i * kStride + kStride
                    loBound = i * kStride + 1
                    module.add(VCmpLtI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(vDLaneRem), src1=hiBound,
                        comment=f"boundary[{i}]: d < {hiBound} ? halfKeep : full"))
                    module.add(VCndMaskB32(
                        dst=vgpr(bm),
                        src0=-1,
                        src1=vgpr(halfMaskVgpr),
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"boundaryMask[{i}] = (d<{hiBound}) ? halfKeep : full"))
                    module.add(VCmpLtI32(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(vDLaneRem), src1=loBound,
                        comment=f"boundary[{i}]: d < {loBound} ? 0 : prev"))
                    module.add(VCndMaskB32(
                        dst=vgpr(bm), src0=vgpr(bm), src1=0,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"boundaryMask[{i}] = (d<{loBound}) ? 0 : prev"))
            writer.vgprPool.checkIn(halfMaskVgpr)
            writer.vgprPool.checkIn(vDLaneRem)

        return list(module.flatitems())

    def emit_mask_k(self, source):
        """Per-lane K-mask for one subIterK, applied to A/B tiles via V_AND_B32."""
        assert self._tail_vDiff is not None, \
            "emit_mask_k_init must run before emit_mask_k"

        writer = self.writer
        kernel = self.kernel
        subIterK = source.subIterK
        kBaseConst = subIterK * kernel["MatrixInstK"]

        laneSGPRCount = writer.states.laneSGPRCount
        isBF16 = kernel["ProblemType"]["DataTypeA"].isBFloat16()
        kStride = 2

        module = Module()

        def _unique_ids(key):
            m = source.vgpr_tile_map.get(key, [{}])[0]
            return sorted(set(m.values()))

        aIds, bIds = _unique_ids('A'), _unique_ids('B')
        refTiles = self.vgprTilesA if aIds else self.vgprTilesB
        refIds = aIds or bIds
        vgprPerInUnroll = len(list(refTiles[refIds[0]])) if refIds else 0

        with writer.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount) as tmpSgprInfo:
            maskSgpr = tmpSgprInfo.idx

            def _emit_cmp(cmpCls, literal, comment):
                if -16 <= literal <= 64:
                    module.add(cmpCls(
                        dst=sgpr(maskSgpr, laneSGPRCount),
                        src0=vgpr(self._tail_vDiff), src1=literal,
                        comment=comment))
                else:
                    with writer.allocTmpSgpr(1) as litSgprInfo:
                        litSgpr = litSgprInfo.idx
                        module.add(SMovB32(
                            dst=sgpr(litSgpr), src=hex(literal),
                            comment=f"stage literal {literal} (non-inline)"))
                        module.add(cmpCls(
                            dst=sgpr(maskSgpr, laneSGPRCount),
                            src0=vgpr(self._tail_vDiff), src1=sgpr(litSgpr),
                            comment=comment))

            if isBF16:
                numMIInUnroll = vgprPerInUnroll * kStride
                fullLit = kBaseConst + numMIInUnroll - 1
                zeroLit = kBaseConst
                maskVgprs = [writer.vgprPool.checkOut(1, f"mask_k_msk{i}_k{subIterK}")
                             for i in range(vgprPerInUnroll)]
                _emit_cmp(VCmpGTI32, fullLit,
                          f"sFull: diff > {fullLit} (effective_diff_{subIterK} >= {numMIInUnroll})")
                for i in range(vgprPerInUnroll):
                    module.add(VCndMaskB32(
                        dst=vgpr(maskVgprs[i]),
                        src0=vgpr(self._tail_boundaryMask[i]), src1=-1,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"mask[{i}] = sFull ? full : boundary[{i}]"))
                _emit_cmp(VCmpLeI32, zeroLit,
                          f"sZero: diff <= {zeroLit} (effective_diff_{subIterK} <= 0)")
                for i in range(vgprPerInUnroll):
                    module.add(VCndMaskB32(
                        dst=vgpr(maskVgprs[i]), src0=vgpr(maskVgprs[i]), src1=0,
                        src2=sgpr(maskSgpr, laneSGPRCount),
                        comment=f"mask[{i}] = sZero ? 0 : prev"))
            else:
                literal = kBaseConst + 1
                sharedMask = writer.vgprPool.checkOut(1, f"mask_k_msk_k{subIterK}")
                maskVgprs = [sharedMask] * vgprPerInUnroll
                _emit_cmp(VCmpLtI32, literal,
                          f"mask: diff < {literal} (laneK_{subIterK} >= rem)")
                module.add(VCndMaskB32(
                    dst=vgpr(sharedMask), src0=-1, src1=0,
                    src2=sgpr(maskSgpr, laneSGPRCount),
                    comment=f"mask = (diff < {literal}) ? 0 : -1"))

            for label, ids, tilesDict in (("A", aIds, self.vgprTilesA),
                                          ("B", bIds, self.vgprTilesB)):
                for tid in ids:
                    for i, v in enumerate(list(tilesDict[tid])):
                        module.add(VAndB32(
                            dst=vgpr(v), src0=vgpr(v), src1=vgpr(maskVgprs[i]),
                            comment=f"mask {label}[{i}] (K=[{i*kStride},{i*kStride+kStride-1}])"))

            scaleStride = self.config.lrSA.k if self.hasScale else 0
            if (not isBF16) and self.hasScale \
                    and (self.vgprTilesSA or self.vgprTilesSB) \
                    and scaleStride > 0 and (subIterK % scaleStride == 0):
                for tensor, tilesList in (('SA', self.vgprTilesSA),
                                          ('SB', self.vgprTilesSB)):
                    liveIds = sorted(set(
                        source.vgpr_tile_map.get(tensor, [{}])[0].values()))
                    for tid in liveIds:
                        for v in list(tilesList[tid]):
                            module.add(VAndB32(
                                dst=vgpr(v), src0=vgpr(v),
                                src1=vgpr(maskVgprs[0]),
                                comment=f"mask scale vgpr (reuse A/B mask, subIterK={subIterK})"))

        for m in set(maskVgprs):
            writer.vgprPool.checkIn(m)
        return list(module.flatitems())

    def emit_mask_k_done(self):
        """Release the long-lived tail-loop vgprs."""
        if getattr(self, "_tail_vDiff", None) is not None:
            self.writer.vgprPool.checkIn(self._tail_vDiff)
            self._tail_vDiff = None
        if getattr(self, "_tail_boundaryMask", None) is not None:
            for bm in self._tail_boundaryMask:
                self.writer.vgprPool.checkIn(bm)
            self._tail_boundaryMask = None
        return []

    def emit_module(self, em, unroll_iter=0):
        """Emit instructions for one EmittedModule, returning a list.

        On-demand replacement for the old ``populate()`` adapter. The caller
        is responsible for ordering and scheduling the returned instruction list;
        nothing is stored on ``em.instructions``.
        """
        handler = self._dispatch.get(em.opType)
        return handler(em, unroll_iter) if handler else []
