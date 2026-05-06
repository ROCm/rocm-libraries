################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto
from typing import ClassVar, Optional

from rocisa.instruction import (
    SWaitCnt, SBarrier,
    MFMAInstruction, SMovB32, SAddU32,
    VPermB32, VOrB32, VLShiftLeftOrB32, VSwapB32,
    VCvtPkF32toBF16, PVCvtBF16toFP32, VCvtBF16toFP32, VSubF32, VDot2CF32BF16,
    BufferLoadB128, BufferLoadB64, BufferLoadB32,
)
from Tensile.Common.Utilities import printWarning
from Tensile.Components.ScheduleCapture import SchedulePosition


# Sentinel values for "infinitely far" positions. Values chosen to be well beyond
# any realistic schedule size (num_vmfma is typically ~48-200).
POSITION_INF = SchedulePosition(loop_index=9_999, vmfma_index=9_999, sub_index=9_999)
POSITION_NEG_INF = SchedulePosition(loop_index=-9_999, vmfma_index=-9_999, sub_index=-9_999)

class ValidationConcern(Enum):
    """Abstract correctness properties the validator must check.

    Each concern is a unit of coverage tracking — if isValid returns True,
    every required concern was checked by at least one rule.
    """
    # Universal (any ISA)
    INSTRUCTION_ORDERING = auto()
    SCHEDULE_COMPLETENESS = auto()
    # Data readiness
    LR_DATA_READY = auto()
    PACK_DATA_READY = auto()
    # LDS coherence
    LDS_WRITE_AFTER_READ = auto()
    LDS_READ_AFTER_WRITE = auto()
    # DTL=0 specific
    LW_ORDERING = auto()
    GR_VGPR_READY = auto()
    # Scalar safety
    SCALAR_REGISTER_SAFETY = auto()
    # Timing
    QUAD_CYCLE_TIMING = auto()


# All concerns that kernels on a given ISA can require.
ISA_CONCERN_CATALOG: dict[tuple, set[ValidationConcern]] = {
    (9, 5, 0): {    # CDNA 4 (gfx950)
        ValidationConcern.INSTRUCTION_ORDERING,
        ValidationConcern.SCHEDULE_COMPLETENESS,
        ValidationConcern.LR_DATA_READY,
        ValidationConcern.PACK_DATA_READY,
        ValidationConcern.LDS_WRITE_AFTER_READ,
        ValidationConcern.LDS_READ_AFTER_WRITE,
        ValidationConcern.SCALAR_REGISTER_SAFETY,
        ValidationConcern.QUAD_CYCLE_TIMING,
    },
}


def active_concerns(kernel: dict, idmap: dict) -> set[ValidationConcern]:
    """Determine which concerns must be covered for this specific kernel.

    Intersects kernel-derived requirements with the ISA's concern catalog.
    Returns an empty set if the ISA is not in the catalog (graceful fallback
    for ISAs that haven't been characterized yet).
    """
    if "ISA" not in kernel:
        return set()  # No ISA specified — no concern-based validation
    isa = tuple(kernel["ISA"])
    if isa not in ISA_CONCERN_CATALOG:
        return set()  # Unknown ISA — no concern-based validation
    isa_concerns = ISA_CONCERN_CATALOG[isa]

    active = {
        ValidationConcern.INSTRUCTION_ORDERING,
        ValidationConcern.SCHEDULE_COMPLETENESS,
    }

    has_local_reads = any(k.startswith("LR") and not k.startswith("LRS") for k in idmap)
    if has_local_reads:
        active.add(ValidationConcern.LR_DATA_READY)

    has_packs = any(k.startswith("Pack") for k in idmap)
    if has_packs:
        active.add(ValidationConcern.PACK_DATA_READY)

    dtl = kernel.get("DirectToLds", 0)
    if dtl:
        active.add(ValidationConcern.LDS_WRITE_AFTER_READ)
        active.add(ValidationConcern.LDS_READ_AFTER_WRITE)
    else:
        active.add(ValidationConcern.LW_ORDERING)
        active.add(ValidationConcern.GR_VGPR_READY)
        active.add(ValidationConcern.LDS_WRITE_AFTER_READ)
        active.add(ValidationConcern.LDS_READ_AFTER_WRITE)

    has_grinc = any(k.startswith("GRInc") for k in idmap)
    if has_grinc:
        active.add(ValidationConcern.SCALAR_REGISTER_SAFETY)

    if has_packs and kernel.get("UseF32XEmulation", False):
        active.add(ValidationConcern.QUAD_CYCLE_TIMING)

    return active & isa_concerns


@dataclass(frozen=True)
class PipelineStage:
    """Describes one loop stage in the CMS pipeline.

    Replaces the hardcoded [ML-1, ML, NGL, NLL] loop list with a model
    that generates the correct number of stages based on PGR.

    PGR=1: [ML, NLL] (2 stages)
    PGR=2: [ML-1, ML, NGL, NLL] (4 stages — same as today)
    PGR=3: [ML-2, ML-1, ML, NGL-2, NGL-1, NLL] (6 stages)
    """
    name: str
    has_global_reads: bool
    has_global_read_incs: bool
    has_local_reads_lr0_only: bool
    vlcnt_shift: int


def build_pipeline_stages(pgr: int, nglshift: int, nllshift: int) -> list[PipelineStage]:
    """Build the pipeline stage list for a given PGR value.

    Args:
        pgr:      PrefetchGlobalRead value (1, 2, or higher)
        nglshift: Total vlcnt shift for no-global-load stages
        nllshift: vlcnt shift for the no-local-load stage
    """
    stages = []
    # N main loop copies
    for i in range(pgr - 1, -1, -1):
        stages.append(PipelineStage(
            name=f"ML-{i}" if i > 0 else "ML",
            has_global_reads=True, has_global_read_incs=True,
            has_local_reads_lr0_only=False, vlcnt_shift=0))
    # N-1 NGLL drain stages
    for k in range(pgr - 1, 0, -1):
        shift = nglshift * (pgr - k) // (pgr - 1) if pgr > 1 else 0
        stages.append(PipelineStage(
            name=f"NGL-{k}" if pgr > 2 else "NGL",
            has_global_reads=False, has_global_read_incs=False,
            has_local_reads_lr0_only=False, vlcnt_shift=shift))
    # NLL
    stages.append(PipelineStage(
        name="NLL", has_global_reads=False, has_global_read_incs=False,
        has_local_reads_lr0_only=True, vlcnt_shift=nllshift))
    return stages


# Pack-related invariants live graph-side:
#   * LR -> Pack RAW (dscnt) and Pack -> MFMA RAW: `validate_edge_wait_coverage`
#     + `compare_graphs` over register dataflow from `_GenericALURule`.
#   * MiddlePack pair-interleaving: `validate_middle_pack_pair_interleaving`
#     (emits `OverriddenInputFailure`).
#   * Quad-cycle visibility: `_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` /
#     `_mfma_pack_to_cvt_gap_ok` (emit `TimingTooCloseFailure`).
# Migrated coverage: `test_validate_pack_graph.py`,
# `test_dataflow_graph_register_gaps.py`.

# --- Loop Names ---
MAIN_LOOP_PREV = "ML-1"
MAIN_LOOP = "ML"
NO_GLOBAL_LOAD_LOOP = "NGL"
NO_LOCAL_LOAD_LOOP = "NLL"

# --- Pack Group Sizes ---
PACK_GROUP_SIZE_TF32 = 24        # 4 CVT0 + 16 middle + 4 CVT1
PACK_GROUP_SIZE_TF32_4X4 = 10    # 4 CVT0 + 2 MFMA + 4 CVT1

# --- Quad-Cycle Timing (CDNA 4 ISA section 7.6) ---
# Quad-cycle visibility verdicts live graph-side; the constants
# (`_QUAD_CYCLES_*`, `_MFMA_TYPE_SWITCH_THRESHOLD_*`) live in
# `Tensile/Components/ScheduleCapture.py` alongside the
# `_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` /
# `_mfma_pack_to_cvt_gap_ok` helpers that consume them.

# --- TF32 Emulation ---
MFMAS_PER_TILE_TF32 = 3   # 3 MFMAs per tile pair in TF32 emulation
MFMAS_PER_TILE_BF16 = 1   # 1 MFMA per tile pair in BF16

# --- VGPRs ---
VGPRS_PER_CONVERSION_GROUP = 8   # 8 VGPRs per conversion group in TF32 emulation


@dataclass
class ValidatorInstruction(ABC):
    """Abstract base for all validator instructions."""
    name: str
    issued_at: SchedulePosition
    # The minimum number of quad-cycles that this instruction takes to issue.
    min_issue_quad_cycles_base: ClassVar[int] = 1
    # Reference to the rocisa instruction object this was created from.
    # None when constructed from mock/test data without real instructions.
    # init=False: set via _insert after construction to avoid dataclass
    # ordering issues (non-default following default in subclasses).
    rocisa_inst: object = field(default=None, init=False, repr=False, compare=False)

    @property
    def category(self) -> str:
        """Alias for `name`. Lets the typed-Failure canonical formatters in
        ScheduleCapture.py — which were written against GraphNode (which has
        both `category` and `name`) — accept ValidatorInstruction objects
        without crashing. ValidatorInstruction's `name` already carries the
        per-tensor / per-iter tag (e.g. 'LRA0', 'PackB1'), matching what
        GraphNode.category holds."""
        return self.name

    @abstractmethod
    def validate(self) -> Optional[str]:
        ...

    def done_idx(self) -> SchedulePosition:
        """Position after which this instruction is done for scheduling purposes.

        Default: instruction is done at its issue position.
        Override in subclasses where completion depends on an SWaitCnt (LocalRead, GlobalRead).
        """
        return self.issued_at

    def min_issue_quad_cycles(self) -> int:
        return self.min_issue_quad_cycles_base

@dataclass
class LocalRead(ValidatorInstruction):
    # The index in the list of Local Read instructions provided by a CMS schedule.
    # Needed to properly calculate must_start_after for Packs.
    issue_index: int
    # `guaranteed_by` is set by ``apply_swaits`` to the SWaitCnt position
    # that drains this LR; consumed by Pack-side dependency derivation
    # (``derive_pack_must_start_after``) via ``done_idx()``.
    guaranteed_by: SchedulePosition = field(default_factory=lambda: POSITION_INF)

    def done_idx(self) -> SchedulePosition:
        return self.guaranteed_by

    def validate(self) -> Optional[str]:
        """No-op. LR-data-ready (LR -> MFMA dscnt drain) lives graph-side
        via ``validate_edge_wait_coverage`` over LR -> MFMA RAW edges.
        The ``ValidatorInstruction.validate`` abstract method still
        requires an override on every subclass; this no-op preserves the
        contract without re-introducing the deleted structural rule.
        """
        return None

@dataclass
class MFMA(ValidatorInstruction):
    def validate(self) -> Optional[str]:
        return None

@dataclass
class Pack(ValidatorInstruction):
    """BF16 pack instructions (v_perm). Base class for all pack types.

    The structural-side ordering rule (`Pack.validate`,
    `Pack.must_start_after`, `Pack.needed_by`, plus all `_hook_up_packs_*`
    plumbing) was removed. Pack-related ordering invariants (LR -> Pack
    RAW dscnt drain, Pack -> MFMA RAW order, Pack -> Pack WAR ordering)
    are now enforced graph-side by `validate_edge_wait_coverage` and
    `compare_graphs` over real register dataflow edges produced by
    `_GenericALURule` (ScheduleCapture.py). See migrated coverage in
    `Tensile/Tests/unit/test_validate_pack_graph.py`.
    """
    # The index in the list of Pack instructions provided by a CMS schedule.
    # Set at construction time (the `_insert` call paths in `Timeline`).
    # Retained as a construction-time tag; the producing call site assigns
    # it unconditionally.
    issue_index: int
    # Which tile/group this pack belongs to, computed at construction time.
    # Only meaningful for TF32 subclasses (CVTPack, MiddlePack, MFMAPack); None for BF16 packs.
    group_index: Optional[int] = None

    def validate(self) -> Optional[str]:
        """No-op. Pack-ordering invariants live graph-side; the
        ValidatorInstruction abstract method still requires an override on
        every subclass, so this preserves the contract.
        """
        return None

@dataclass
class CVTPack(Pack):
    """TF32 CVT0/CVT1 packs (v_cvt_pk_bf16_f32). Type marker for isinstance dispatch."""
    pass

@dataclass
class MiddlePack(Pack):
    """Middle-16 packs in TF32 groups of 24.

    The structural-side `MiddlePack.validate`, `pair_consumer`, and
    `next_scheduled_middle_16` helpers were removed. The pair-interleaving
    invariant (`OverriddenInputFailure` on a non-pair-consumer between
    pair-leader and pair-consumer) is now enforced graph-side via
    `validate_middle_pack_pair_interleaving`; see
    `test_validate_pack_graph.py::TestMiddlePackPairInterleavingGraph`.
    """
    pass

@dataclass
class SwapPack(Pack):
    """VSwapB32 instructions that transpose registers after wider local reads (VW > 1).

    Generated by transposeLRVregs() in LocalRead.py. Count per pack group: 4 * (vw - 1).
    Always appear at the beginning of a pack sequence before CVT0/MFMA/CVT1 groups.
    """
    pass

@dataclass
class MFMAPack(Pack, MFMA):
    """A v_mfma_f32_4x4x4_16b_bf16 instruction used in TF32 4x4 emulation pack groups.

    Identified from idMap via resolve_pack_type() (isinstance MFMAInstruction).
    They are real MFMA instructions but participate in the pack dependency
    chain (CVT0 -> MFMAPack -> CVT1).

    Inherits from both Pack and MFMA:
    - isinstance(x, Pack) is True — works with pack gathering, filtering, type hints
    - isinstance(x, MFMA) is True — captures "this IS an MFMA" semantics
    """


@dataclass
class GlobalRead(ValidatorInstruction):
    swap_global_read_order: bool
    needed_by: ValidatorInstruction = field(default_factory=lambda: MFMA(name="MFMA", issued_at=POSITION_INF))
    guaranteed_by: SchedulePosition = field(default_factory=lambda: POSITION_INF)
    barriered_at: list[SchedulePosition] = field(default_factory=list)

    def done_idx(self) -> SchedulePosition:
        return self.guaranteed_by

    def validate(self):
        """Stack 1.3: returns typed Failure or None (legacy str also accepted
        for the residual defensive-fallback branches)."""
        # Check needed_by constraint (GR must finish before LR1/3)
        needed_by_error = self._validate_needed_by()
        if needed_by_error:
            return needed_by_error

        return None

    def _validate_needed_by(self):
        """Validate: GR -> SWait -> SBarrier -> LR1. Returns Failure or None."""
        from Tensile.Components.ScheduleCapture import (
            MissingWaitFailure, MissingBarrierFailure,
            cms_node_label, _cms_iter_delta,
        )

        # If needed_by is at inf, the constraint is not active (e.g. no LR1s).
        if self.needed_by.issued_at == POSITION_INF:
            return None

        if self.issued_at < self.guaranteed_by < self.needed_by.issued_at:
            if any(self.guaranteed_by < barriered_at < self.needed_by.issued_at for barriered_at in self.barriered_at):
                    return None

        issued_at = self.issued_at.vmfma_index
        needed_by = self.needed_by.issued_at.vmfma_index

        name = self._name()

        # Eager source-aware labels. ValidatorInstruction has no body capture
        # context (no tagged_inst, no body_label), so cms_node_label falls
        # through to bare-category primary — matches the pre-g4w wording for
        # this path (which previously raised under strict mode anyway).
        producer_label = cms_node_label(self, None)
        consumer_label = cms_node_label(self.needed_by, None)
        iter_delta = _cms_iter_delta(self, self.needed_by)

        # 1. No SWait
        if self.guaranteed_by == POSITION_INF:
            return MissingWaitFailure(
                producer=producer_label, consumer=consumer_label,
                iter_delta=iter_delta, counter_kind="vlcnt",
            )

        # NOTE: Must do it after the check above to guard against infinity.
        guaranteed_by = self.guaranteed_by.vmfma_index

        # 2. No Barrier
        if len(self.barriered_at) == 0:
            return MissingBarrierFailure(
                producer=producer_label, consumer=consumer_label,
                iter_delta=iter_delta, role="needed_by",
            )

        # 3. Guaranteed after needed — semantically equivalent to no wait from the
        # consumer's perspective, so surface as MissingWaitFailure.
        if self.guaranteed_by > self.needed_by.issued_at:
            return MissingWaitFailure(
                producer=producer_label, consumer=consumer_label,
                iter_delta=iter_delta, counter_kind="vlcnt",
            )

        # 4. No Barrier between SWait and LR1
        if not any(self.guaranteed_by < barriered_at < self.needed_by.issued_at for barriered_at in self.barriered_at):
            return MissingBarrierFailure(
                producer=producer_label, consumer=consumer_label,
                iter_delta=iter_delta, role="needed_by",
            )

        # Defensive fallback — string return; should not fire on valid logic.
        return (f"{name} @ idx={issued_at} is not valid. issued @ idx={issued_at}, "
                f"guaranteed @ idx={guaranteed_by}, "
                f"barriered @ idx={[b.vmfma_index for b in self.barriered_at]}, "
                f"needed @ idx={needed_by} is not valid.")

    def _name(self) -> str:
        name = self.name
        if not self.swap_global_read_order:
            return name

        if name.startswith("GRA"):
            return name + " (Swapped, loading B)"
        elif name.startswith("GRB"):
            return name + " (Swapped, loading A)"
        else:
            raise ValueError(f"Unexpected global read name: {name}")

@dataclass
class SWait(ValidatorInstruction):
    dscnt: int
    vlcnt: int
    vscnt: int
    comment: str

    def _is_valid(self) -> bool:
        return self.dscnt >= -1 and self.vlcnt >= -1 and self.vscnt >= -1 and self.issued_at.vmfma_index >= -1

    def validate(self):
        """Stack 1 of jaunty-reddy plan: returns typed Failure or None."""
        if self._is_valid():
            return None
        from Tensile.Components.ScheduleCapture import InvalidCounterValueFailure
        return InvalidCounterValueFailure(
            swait_idx=self.issued_at.vmfma_index,
            dscnt=self.dscnt,
            vlcnt=self.vlcnt,
            vscnt=self.vscnt,
        )

@dataclass
class Barrier(ValidatorInstruction):
    comment: str

    def validate(self) -> Optional[str]:
        # Sentinel range check stays as a defensive guard. No user-facing
        # Failure type — this fires only on malformed test fixtures.
        # Returning a string keeps the legacy boundary working.
        if self.issued_at.vmfma_index < -1:
            return f"Barrier at index {self.issued_at.vmfma_index} is not valid. Must be >= -1."
        return None

@dataclass
class SNop(ValidatorInstruction):
    wait_state: int

    def min_issue_quad_cycles(self) -> int:
        # Base instruction quad-cycles plus wait_state additional cycles
        return self.min_issue_quad_cycles_base + self.wait_state

    def validate(self) -> Optional[str]:
        return None

@dataclass
class GRInc(ValidatorInstruction):
    """Scalar pointer-increment instructions (GRIncA/GRIncB) that advance the
    global memory address before the next buffer_load."""

    def validate(self) -> Optional[str]:
        return None

ALL_INSTRUCTION_NAMES = [
    "LRA0", "LRB0", "LRA1", "LRB1", "LRA3", "LRB3",
    "GRA", "GRB",
    "GRIncA", "GRIncB",
    "PackA0", "PackB0", "PackA1", "PackB1", "PackA3", "PackB3",
    "SYNC", "SNOP",
]


# --- Type Resolution for idMap-driven instruction classification ---
# Maps rocisa instruction types to (ValidatorInstruction subclass, assembly label).
# Lookup: exact type match first, then isinstance fallback for subclass chains.
PACK_TYPE_MAP: dict[type, tuple[type, str]] = {
    VPermB32:          (Pack,       "v_perm_b32"),
    VOrB32:            (Pack,       "v_or_b32"),
    VLShiftLeftOrB32:  (Pack,       "v_lshlrev_or_b32"),
    VCvtPkF32toBF16:   (CVTPack,    "v_cvt_pk_bf16_f32"),
    PVCvtBF16toFP32:   (MiddlePack, "p_v_cvt_f32_bf16"),
    VCvtBF16toFP32:    (MiddlePack, "v_cvt_f32_bf16"),
    VSubF32:           (MiddlePack, "v_sub_f32"),
    VDot2CF32BF16:     (MiddlePack, "v_dot2c_f32_bf16"),
    VSwapB32:          (SwapPack,   "v_swap_b32"),
    MFMAInstruction:   (MFMAPack,   "v_mfma_*"),
}

# Maps rocisa instruction types to (is_gr_load: bool).
# Used to distinguish actual GR loads from m0 pointer writes in DTL sequences.
GR_TYPE_MAP: dict[type, bool] = {
    SMovB32:        False,  # m0 pointer setup
    SAddU32:        False,  # m0 pointer increment
    BufferLoadB128: True,   # actual GR load
    BufferLoadB64:  True,
    BufferLoadB32:  True,
}


def resolve_pack_type(rocisa_inst: object) -> tuple[type, str]:
    """Resolve a rocisa instruction to its ValidatorInstruction class and assembly label.

    Args:
        rocisa_inst: A rocisa instruction object from the idMap.

    Returns:
        Tuple of (ValidatorInstruction subclass, assembly label string).

    Raises:
        ValueError: If the instruction type is not recognized.
    """
    inst_type = type(rocisa_inst)
    # Exact match first
    if inst_type in PACK_TYPE_MAP:
        return PACK_TYPE_MAP[inst_type]
    # isinstance fallback for subclass chains (e.g. MFMAInstruction subclasses)
    for base_type, result in PACK_TYPE_MAP.items():
        if isinstance(rocisa_inst, base_type):
            return result
    raise ValueError(f"Unknown pack instruction type: {inst_type.__name__} ({rocisa_inst})")


def is_gr_load(rocisa_inst: object) -> bool:
    """Return True if the rocisa instruction is an actual GR load, False if it's a pointer operation.

    Raises:
        ValueError: If the instruction type is not recognized.
    """
    inst_type = type(rocisa_inst)
    if inst_type in GR_TYPE_MAP:
        return GR_TYPE_MAP[inst_type]
    for base_type, result in GR_TYPE_MAP.items():
        if isinstance(rocisa_inst, base_type):
            return result
    raise ValueError(f"Unknown GR instruction type: {inst_type.__name__} ({rocisa_inst})")


def detect_pack_groups(idmap_items: list) -> list[dict]:
    """Walk the idMap entries for a pack name and identify group boundaries from the type pattern.

    Returns a list of group dicts, each containing:
      - 'group_index': int (or None for ungrouped SwapPacks)
      - 'entries': list of (index, validator_cls, asm_label) tuples

    Group detection rules:
      - SwapPack instructions at the start are ungrouped (group_index=None).
      - If no CVTPack/MiddlePack/MFMAPack types appear, all entries are plain Pack (BF16, no grouping needed).
      - Otherwise, determine the group size from the type pattern (presence of MiddlePack → 24, MFMAPack → 10)
        and split the non-swap entries into fixed-size groups.
    """
    if not idmap_items:
        return []

    entries = []
    for idx, inst in enumerate(idmap_items):
        cls, label = resolve_pack_type(inst)
        entries.append((idx, cls, label))

    # Check if there are any TF32-related types
    has_middle = any(cls is MiddlePack for _, cls, _ in entries)
    has_mfma_pack = any(cls is MFMAPack for _, cls, _ in entries)
    has_cvt = any(cls is CVTPack for _, cls, _ in entries)

    if not (has_middle or has_mfma_pack or has_cvt):
        # BF16: all plain Pack, single group
        return [{"group_index": 0, "entries": entries}]

    groups = []
    # Separate leading SwapPacks
    n_swaps = 0
    for idx, cls, label in entries:
        if cls is SwapPack:
            n_swaps += 1
        else:
            break

    if n_swaps > 0:
        groups.append({"group_index": None, "entries": entries[:n_swaps]})

    remaining = entries[n_swaps:]
    if not remaining:
        return groups

    # Determine group size from the type pattern
    if has_middle:
        group_size = PACK_GROUP_SIZE_TF32       # 24: CVT0×4 + Middle×16 + CVT1×4
    elif has_mfma_pack:
        group_size = PACK_GROUP_SIZE_TF32_4X4   # 10: CVT0×4 + MFMAPack×2 + CVT1×4
    else:
        # CVTPack only (no middle or mfma) — shouldn't happen in practice, but handle gracefully
        group_size = len(remaining)

    # Split remaining entries into fixed-size groups
    for i in range(0, len(remaining), group_size):
        group_entries = remaining[i:i + group_size]
        group_idx = i // group_size
        groups.append({"group_index": group_idx, "entries": group_entries})

    return groups


# --- Register Utilities for rocisa-based dependency analysis ---

def _parse_reg_name(reg_name_str: str) -> tuple[str, int]:
    """Parse a register name string like 'ValuA_X0_I0+12' into (base_name, offset).

    Returns:
        Tuple of (base_name, offset). If no '+' offset, offset is 0.
        E.g. 'ValuA_X0_I0+12' → ('ValuA_X0_I0', 12)
             'ValuA_X0_I0' → ('ValuA_X0_I0', 0)
    """
    parts = str(reg_name_str).split('+')
    base = parts[0]
    offset = int(parts[1]) if len(parts) > 1 else 0
    return base, offset


def get_reg_range(rc) -> Optional[tuple[str, int, int]]:
    """Extract the register identity range from a RegisterContainer.

    Returns:
        Tuple of (base_name, start_offset, end_offset_exclusive), or None if
        the container is not a VGPR register or cannot be resolved.

        For numeric registers (regIdx >= 0): base_name is the regType string,
        start/end are the numeric indices.

        For named registers (regName set): base_name is the symbolic name,
        start/end are the offsets within that name.

    Examples:
        v[10:13] → ('v', 10, 14)
        v[vgprValuA_X0_I0+12:+15] → ('ValuA_X0_I0', 12, 16)
    """
    if rc is None:
        return None

    reg_num = int(rc.regNum)
    if reg_num <= 0:
        return None

    if rc.regIdx >= 0:
        # Numeric register
        return (rc.regType, rc.regIdx, rc.regIdx + reg_num)

    if rc.regName is not None:
        # Named register with symbolic name
        base, offset = _parse_reg_name(rc.regName)
        return (base, offset, offset + reg_num)

    return None


def reg_ranges_overlap(a: tuple[str, int, int], b: tuple[str, int, int]) -> bool:
    """Check if two register ranges overlap.

    Both ranges must have the same base name (register file) to overlap.
    Ranges are [start, end) — half-open intervals.
    """
    if a[0] != b[0]:
        return False
    return a[1] < b[2] and b[1] < a[2]


def get_dst_range(rocisa_inst) -> Optional[tuple[str, int, int]]:
    """Extract the destination register range from a rocisa instruction.

    Handles CommonInstruction (.dst), MFMAInstruction (.acc), and
    CompositeInstruction (.dst) via attribute inspection.
    """
    # MFMAInstruction: destination is the accumulator
    if hasattr(rocisa_inst, 'acc') and rocisa_inst.acc is not None:
        return get_reg_range(rocisa_inst.acc)
    # CommonInstruction and others
    if hasattr(rocisa_inst, 'dst') and rocisa_inst.dst is not None:
        return get_reg_range(rocisa_inst.dst)
    return None


def get_src_ranges(rocisa_inst) -> list[tuple[str, int, int]]:
    """Extract all source register ranges from a rocisa instruction.

    Returns a list of register ranges for all source operands that are
    RegisterContainer objects (skips immediates and string operands).
    """
    ranges = []
    if hasattr(rocisa_inst, 'srcs'):
        for src in rocisa_inst.srcs:
            rng = get_reg_range(src)
            if rng is not None:
                ranges.append(rng)
    # MFMAInstruction: .a and .b are sources
    if hasattr(rocisa_inst, 'a') and rocisa_inst.a is not None:
        rng = get_reg_range(rocisa_inst.a)
        if rng:
            ranges.append(rng)
    if hasattr(rocisa_inst, 'b') and rocisa_inst.b is not None:
        rng = get_reg_range(rocisa_inst.b)
        if rng:
            ranges.append(rng)
    return ranges


def create_unified_timeline(
    schedule_info: 'ScheduleInfo',
    kernel: 'Solution',
    code_path: int,
    id_map: dict,
    mfma_code: list,
) -> 'Timeline':
    """Create a single Timeline with all instruction types.

    Args:
        schedule_info:  The schedule to validate.
        kernel:         Kernel configuration dict.
        code_path:      Which code path to build the timeline for.
        id_map:         Maps schedule keys to lists of rocisa instruction objects.
        mfma_code:      Flat list of MFMA rocisa instruction objects in execution order
                        (already reordered by mfmaReorder).
    """
    available_names = set(schedule_info.optSchedule.keys())
    names_to_add = [n for n in ALL_INSTRUCTION_NAMES if n in available_names]
    return Timeline(names_to_add, code_path, schedule_info, kernel, id_map=id_map, mfma_code=mfma_code)


class Timeline:
    def __init__(self, instruction_names_to_add: list[str], code_path: int, schedule_info: 'ScheduleInfo', kernel: 'Solution',
                 id_map: dict, mfma_code: list):
        """Build a Timeline restricted to the categories in
        ``instruction_names_to_add``. Only one ``code_path`` is materialized
        per Timeline — multi-codepath validation builds N Timelines so
        per-codepath schedule disagreements are isolated.

        Asserts the per-iteration suffix scheme (LRA0/LRA1/... or 0..3 for
        ForceUnrollSubIter) matches what DepthU + matrixInstK derive. A
        mismatched schedule key here means upstream drift between the CMS
        author's iteration count and the kernel's, which would otherwise
        surface as a confusing IndexError deep in the validators."""
        
        available_keys = schedule_info.optSchedule.keys()
        has_lr1s = "LRA1" in available_keys or "LRB1" in available_keys
        has_lr3s = "LRA3" in available_keys or "LRB3" in available_keys
        assert not (has_lr1s and has_lr3s), "Can't mix LR1s and LR3s."

        # Validate that sub-iteration suffixes are consistent with the kernel configuration.
        # The valid suffixes depend on how numLoopIter is determined:
        # - ForceUnrollSubIter=True: numLoopIter = numSubTiles² = 4 (KernelWriter.py:4592)
        # - DepthU == matrixInstK (n_sub_iters == 1): split to numLoopIter = 2 (CustomSchedule.py:317)
        # - DepthU > matrixInstK: numLoopIter = DepthU / matrixInstK
        if "DepthU" in kernel and "MatrixInstruction" in kernel:
            force_unroll = kernel.get("ForceUnrollSubIter", False)
            if force_unroll:
                valid_suffixes = {0, 1, 2, 3}
            else:
                n_sub_iters = kernel["DepthU"] // kernel["MatrixInstruction"][2]
                if n_sub_iters == 1:
                    valid_suffixes = {0, 1}
                else:
                    valid_suffixes = set(range(n_sub_iters))
            for key in available_keys:
                for prefix in ("LRA", "LRB", "PackA", "PackB"):
                    if key.startswith(prefix):
                        suffix_str = key[len(prefix):]
                        if suffix_str.isdigit():
                            suffix = int(suffix_str)
                            assert suffix in valid_suffixes, (
                                f"Schedule key '{key}' has sub-iteration index {suffix}, "
                                f"but with DepthU={kernel['DepthU']} and matrixInstK={kernel['MatrixInstruction'][2]}, "
                                f"valid sub-iteration indices are {sorted(valid_suffixes)}."
                            )
                        break

        self.num_vmfma = schedule_info.numMfma
        self.id_map = id_map
        self.mfma_code = mfma_code
        self.vlcnt_shift = defaultdict(int)
        self.vlcnt_shift[NO_GLOBAL_LOAD_LOOP] = schedule_info.nglshift
        self.vlcnt_shift[NO_LOCAL_LOAD_LOOP] = schedule_info.nllshift
        self.nll_zero_dscnt = schedule_info.nllZeroDscnt

        self.loops = [MAIN_LOOP_PREV, MAIN_LOOP, NO_GLOBAL_LOAD_LOOP, NO_LOCAL_LOAD_LOOP]
        # NOTE: num_vmfma + 1 to account for special idx=-1.
        #       idx=-1 is special case that occurs BEFORE the first VMFMA but AFTER the last VMFMA.
        #       Instructions at idx=-1 happen after all instructions at idx=num_vmfma-1 and BEFORE all instructions (including the VMFMA) at idx=0.
        self._instructions_at_index: dict[str, list[list[ValidatorInstruction]]] = {loop: [[] for _ in range(self.num_vmfma+1)] for loop in self.loops}
        
        # Linear timelines for each loop.
        self._timelines: dict[str, list[ValidatorInstruction]] = {loop: [] for loop in self.loops}
        # One linear timeline that spans all loops.
        self.combined_timeline: list[ValidatorInstruction] = []

        # Lookup for all instructions in a given loop for a given name.
        # First key is the loop name, second key is the instruction name (e.g. "GRA").
        # Value is a list of tuples of (index, instruction) for the given name in the given loop.
        # Index is the index of the instruction in the loop, index in [0, len(self._timelines[loop])-1]
        self._instructions_for_name: dict[str, dict[str, list[tuple[int, ValidatorInstruction]]]] = {loop: defaultdict(list) for loop in self.loops}
        # Same as above, except for all instructions across all loops.
        # Only index by instruction name.
        # Index is the index of the instruction in the combined timeline. index in [0, len(self.combined_timeline)-1]
        self._instructions_for_name_combined: dict[str, list[tuple[int, ValidatorInstruction]]] = defaultdict(list)

        # Track which validation passes have already been applied to this timeline to avoid applying them multiple times.
        self._applied_passes: set[Callable[['Timeline', 'ValidationContext'], None]] = set()

        # Populate the timeline with instructions
        self._populate_instructions(instruction_names_to_add, code_path, schedule_info, kernel)
        self._linearize_timeline()
    
    def _populate_instructions(self, instruction_names_to_add: list[str], code_path: int, schedule_info: 'ScheduleInfo', kernel: 'Solution') -> None:
        assert kernel["DirectToLds"], "Only DirectToLds cases are supported by validator."
        assert kernel.get("LocalSplitU", 1) == 1, "Only LocalSplitU=1 cases are supported by validator."

        swap_global_read_order = kernel["SwapGlobalReadOrder"]
        is_tf32_emulation = kernel.get("UseF32XEmulation", False)
        is_4x4mfma_tf32 = kernel.get("UseMFMAF32XEmulation", False)

        # Explicitly add MFMAs to timeline.
        # Do at the top here so they are the first ones scheduled at each vmfma index.
        for i_vmfma in range(self.num_vmfma):
            # mfmaReorder[new_pos] = original_pos. i_vmfma is the new (execution) position.
            vmfma_slot = i_vmfma
            if schedule_info.mfmaReorder:
                vmfma_slot = schedule_info.mfmaReorder[i_vmfma]

            mfma_kwargs = {"name": "MFMA"}
            # mfma_code is indexed by new position (already reordered in CustomSchedule.py)
            if self.mfma_code and i_vmfma < len(self.mfma_code):
                mfma_kwargs["rocisa_inst"] = self.mfma_code[i_vmfma]
            self._insert(vmfma_slot, MFMA, mfma_kwargs, kernel)

        def _get_rocisa(name: str, idx: int) -> object:
            """Look up the rocisa instruction from id_map, or None if unavailable."""
            if self.id_map and name in self.id_map:
                items = self.id_map[name]
                if idx < len(items):
                    return items[idx]
            return None

        # NOTE: Relative ordering of instructions must be preserved.
        #       Order dictates the order in which instructions are scheduled if they are scheduled at the same vmfmaindex.
        for name in schedule_info.optSchedule.keys():
            if name not in instruction_names_to_add:
                continue

            if name == "SYNC":
                for idx_sync, (idx_vmfma, sync) in enumerate(zip(schedule_get(name, code_path, schedule_info), schedule_info.syncCode)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: SWaitCnt at index {idx_sync} is not valid. Must be >= -1."
                    ri = _get_rocisa("SYNC", idx_sync)
                    if isinstance(sync, SWaitCnt):
                        self._insert(idx_vmfma, SWait, {"name": "SWaitCnt", "dscnt": sync.dscnt, "vlcnt": sync.vlcnt, "vscnt": sync.vscnt, "comment": sync.comment, "rocisa_inst": ri}, kernel)
                    elif isinstance(sync, SBarrier):
                        self._insert(idx_vmfma, Barrier, {"name": "SBarrier", "comment": sync.comment, "rocisa_inst": ri}, kernel)
                    else:
                        raise ValueError(f"Unexpected sync instruction type: {type(sync)}")
            elif name == "SNOP":
                for idx_snop, (idx_vmfma, snop) in enumerate(zip(schedule_get(name, code_path, schedule_info), schedule_info.snopCode)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: SNop at index {idx_snop} is not valid. Must be >= -1."
                    # The waitState is stored as the first parameter in the rocisa SNop instruction
                    wait_state = snop.getParams()[0]
                    self._insert(idx_vmfma, SNop, {"name": "SNop", "wait_state": wait_state, "rocisa_inst": _get_rocisa("SNOP", idx_snop)}, kernel)
            elif name.startswith("LRA") or name.startswith("LRB"):
                for idx_LR, idx_vmfma in enumerate(schedule_get(name, code_path, schedule_info)):
                    assert idx_vmfma >= -1, f"Code path {code_path}: LocalRead {name} at index {idx_LR} is not valid. Must be >= -1."

                    # TODO: For ForceUnrollSubIter, need to account for register reuse and the fact that the LR0/LR1/LR3s must start after a certain point in the iteration.
                    self._insert(idx_vmfma, LocalRead, {"name": name, "issue_index": idx_LR, "rocisa_inst": _get_rocisa(name, idx_LR)}, kernel)
            elif name.startswith("GRInc"):
                grincs = schedule_get(name, code_path, schedule_info)
                for idx_grinc, idx_vmfma in enumerate(grincs):
                    assert idx_vmfma >= -1, f"Code path {code_path}: GRInc {name} at index {idx_grinc} is not valid. Must be >= -1."
                    self._insert(idx_vmfma, GRInc, {"name": name, "rocisa_inst": _get_rocisa(name, idx_grinc)}, kernel)
            elif name.startswith("GRA") or name.startswith("GRB"):
                global_reads = schedule_get(name, code_path, schedule_info)
                idmap_items = self.id_map[name]
                if kernel["DirectToLds"]:
                    assert len(global_reads) % 2 == 0, f"Code path {code_path}: {name} has an odd number of indices. Must be even if DirectToLds is True."
                for idx_GR, idx_vmfma in enumerate(global_reads):
                    assert idx_vmfma >= -1, f"Code path {code_path}: GlobalRead {name} at index {idx_GR} is not valid. Must be >= -1."
                    ri = idmap_items[idx_GR] if idx_GR < len(idmap_items) else None
                    if ri is not None and not is_gr_load(ri):
                        continue  # pointer setup (SMovB32/SAddU32), not an actual load
                    self._insert(idx_vmfma, GlobalRead, {"name": name, "swap_global_read_order": swap_global_read_order, "rocisa_inst": ri}, kernel)
            elif name.startswith("Pack"):
                packs = schedule_get(name, code_path, schedule_info)
                idmap_items = self.id_map[name]
                groups = detect_pack_groups(idmap_items)
                idx_to_cls: dict[int, tuple[type, Optional[int]]] = {}
                for group in groups:
                    gidx = group["group_index"]
                    for entry_idx, cls, label in group["entries"]:
                        idx_to_cls[entry_idx] = (cls, gidx)

                for idx_pack, idx_vmfma in enumerate(packs):
                    assert idx_vmfma >= -1, f"Code path {code_path}: Pack {name} at index {idx_pack} is not valid. Must be >= -1."
                    ri = idmap_items[idx_pack] if idx_pack < len(idmap_items) else None
                    pack_cls, group_idx = idx_to_cls.get(idx_pack, (Pack, None))
                    pack_kwargs = {"name": name, "issue_index": idx_pack, "rocisa_inst": ri}
                    if group_idx is not None:
                        pack_kwargs["group_index"] = group_idx
                    self._insert(idx_vmfma, pack_cls, pack_kwargs, kernel)
            else:
                raise NotImplementedError(f"Instruction {name} not implemented")

    def _insert(self, vmfma_index: int, cls: type[ValidatorInstruction], kwargs: dict, kernel: 'Solution') -> None:
        """
        Construct and add an instruction to the timeline at a given VMFMA index.
        A fresh instance is created for each applicable loop directly from
        *cls* and *kwargs*, so no copying is needed.

        Args:
            vmfma_index:  The VMFMA slot to place the instruction at.
            cls:          The instruction class to instantiate (e.g. LocalRead, MFMA).
            kwargs:       Constructor keyword arguments **excluding** ``issued_at``
                          (which is set per-loop by this method).
                          ``rocisa_inst`` is extracted and set post-construction
                          (it uses init=False on the dataclass).
            kernel:       The kernel configuration dict.
        """
        # Extract rocisa_inst before passing kwargs to constructor (init=False field).
        rocisa_inst = kwargs.pop("rocisa_inst", None)

        for loop in self.loops:
            if self._should_add(cls, kwargs.get("name", ""), loop, kernel):
                loop_index = self.loops.index(loop)
                sub_index = len(self._instructions_at_index[loop][vmfma_index + 1])
                kwargs["issued_at"] = SchedulePosition(loop_index=loop_index, vmfma_index=vmfma_index, sub_index=sub_index)

                _instruction = cls(**kwargs)
                _instruction.rocisa_inst = rocisa_inst

                # Adjust for NLL/NGL shifts.
                if isinstance(_instruction, SWait):
                    if _instruction.vlcnt != -1:
                        vlcnt = max(0, _instruction.vlcnt - self.vlcnt_shift[loop])
                        _instruction.vlcnt = vlcnt
                    if _instruction.dscnt != -1 and self.nll_zero_dscnt \
                       and loop in [NO_LOCAL_LOAD_LOOP]:
                        _instruction.dscnt = 0

                self._instructions_at_index[loop][vmfma_index+1].append(_instruction)

    @staticmethod
    def _should_add(cls: type[ValidatorInstruction], name: str, loop: str, kernel: 'Solution') -> bool:
        """Per-loop instruction filter: GR/GRInc only fire in main loops;
        only LR0 / Pack0 reach NLL (the rest of LR/Pack live in the
        main-loop bodies). Encodes the CMS pipeline-stage contract."""
        if issubclass(cls, GlobalRead):
            # No GRs issued in NGL or NLL
            return loop == MAIN_LOOP or loop == MAIN_LOOP_PREV
        elif issubclass(cls, GRInc):
            return loop == MAIN_LOOP or loop == MAIN_LOOP_PREV
        elif issubclass(cls, LocalRead):
            # Only LR0s are issued in the NLL
            if loop == NO_LOCAL_LOAD_LOOP:
                return name == "LRA0" or name == "LRB0"
            return True
        elif issubclass(cls, Pack):
            if kernel.get("UsePLRPack", False):
                # Packs1/3s correspond to the LR1/3s of this iteration.
                if loop == NO_LOCAL_LOAD_LOOP:
                    return name == "PackA0" or name == "PackB0"
            return True
        else:
            return True
   
    def __len__(self):
        return len(self._timelines)

    def __getitem__(self, index: int) -> ValidatorInstruction:
        return self._timelines[index]

    def get_instruction_names(self) -> list[str]:
        return list(self._instructions_for_name_combined.keys())

    def get_instructions(self, name: str, loop: str) -> list[tuple[int, ValidatorInstruction]]:
        return self._instructions_for_name[loop][name]

    def get_instructions_combined(self, name: str) -> list[tuple[int, ValidatorInstruction]]:
        return self._instructions_for_name_combined[name]

    def get_instructions_at(self, index: int, loop: str) -> list[ValidatorInstruction]:
        return self._instructions_at_index[loop][index+1]

    def _linearize_timeline(self) -> None:
        """Materialize the per-loop and combined linear timelines plus the
        per-name lookup tables that ``get_instructions*`` indexes into."""
        self.combined_timeline.clear()
        self._instructions_for_name_combined.clear()
        i_combined = 0
        for loop_name, loop_instructions in self._instructions_at_index.items():
            i_loop = 0
            self._timelines[loop_name].clear()
            self._instructions_for_name[loop_name].clear()

            for instructions in loop_instructions:
                for instruction in instructions:
                    self._timelines[loop_name].append(instruction)
                    self._instructions_for_name[loop_name][instruction.name].append((i_loop, instruction))
                    self._instructions_for_name_combined[instruction.name].append((i_combined, instruction))
                    i_loop += 1
                    i_combined += 1
            
            self.combined_timeline.extend(self._timelines[loop_name])


def applies_only_once(func: Callable) -> Callable:
    """Decorator: skips the function if it has already been applied to this timeline."""
    @functools.wraps(func)
    def wrapper(timeline: 'Timeline', *args, **kwargs):
        if func in timeline._applied_passes:
            return
        result = func(timeline, *args, **kwargs)
        timeline._applied_passes.add(func)
        return result
    return wrapper


def _compute_swap_register_pairs(vw: int, total_regs: int) -> list[tuple[int, int]]:
    """Compute the (src_reg, dst_reg) pairs for each VSwapB32 instruction, in issue order.

    Replicates the iteration logic of transposeLRVregs() in LocalRead.py.
    Uses the same conversion tables from getTransposeIndex() (LocalRead.py:319-320).

    The conversion tables map each register index within a block of size
    MIInputPerThread(8) * VW to its transposed position. transposeLRVregs
    iterates indices 1..totalRegs-2 (skipping first and last), and for each:
      - Looks up the target position via the conversion table
      - If neither position has been visited and they differ, emits a VSwapB32
      - Marks both positions as visited

    Args:
        vw: The vector width (lrvwTile). Must be 2 or 4.
        total_regs: Total number of registers being transposed within one block
                    (VGPRS_PER_CONVERSION_GROUP * vw).

    Returns:
        List of (src_reg, dst_reg) tuples, one per swap, in the order they are issued.
    """
    if vw <= 1:
        return []
    _CONV_TABLE = {
        2: [0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15],
        4: [0, 8, 16, 24, 4, 12, 20, 28, 1, 9, 17, 25, 5, 13, 21, 29,
            2, 10, 18, 26, 6, 14, 22, 30, 3, 11, 19, 27, 7, 15, 23, 31],
    }
    conv = _CONV_TABLE[vw]
    block_size = 8 * vw  # MIInputPerThread * lrvwTile
    start, last = 0, total_regs - 1
    done = [start, last]
    pairs = []
    for idx in range(start + 1, last):
        block_idx = idx // block_size
        new_idx = conv[idx % block_size] + block_idx * block_size
        if idx in done or idx == new_idx:
            done.append(idx)
            continue
        pairs.append((idx, new_idx))
        done.append(idx)
        done.append(new_idx)
    return pairs


def _failure_to_string(result: object) -> Optional[str]:
    """Boundary helper: a rule's validate() may return either a legacy
    string OR a typed Failure. Normalize to string for the existing
    isValid contract.

    Stack 1 of plans/then-let-s-work-on-jaunty-reddy.md migrates rules
    one at a time. Until every rule emits Failures, this helper supports
    both shapes so the boundary stays stable.

    Failure.format requires a capture; structural rules don't have one
    in scope, so pass an empty LoopBodyCapture. Formatters that read
    capture for [N] index lookup will fall through to bare category for
    these legacy rule paths — an honest acknowledgement that the
    structural-rule emitter has no per-body context.
    """
    if result is None:
        return None
    if isinstance(result, str):
        return result
    # Typed Failure (from Tensile.Components.ScheduleCapture).
    from Tensile.Components.ScheduleCapture import LoopBodyCapture
    return result.format(LoopBodyCapture(instructions=[]))


def validate_timeline(timeline: Timeline) -> Optional[str]:
    """Iterate every instruction in stream order; the first one whose
    ``validate()`` returns non-None ends the walk.

    Side effect: stashes the typed Failure (when one is returned) on
    ``timeline._last_failure`` so test infrastructure can assert on type +
    fields without parsing message text. Production callers consume only
    the returned string. Rules still on the legacy str path leave
    ``_last_failure`` None."""
    timeline._last_failure = None
    for loop in timeline.loops:
        for instruction in timeline._timelines[loop]:
            result = instruction.validate()
            message = _failure_to_string(result)
            if message is not None:
                # Surface the typed Failure to test introspection. Rules
                # that still return raw strings during migration leave
                # _last_failure as None.
                timeline._last_failure = result if not isinstance(result, str) else None
                if loop in [NO_GLOBAL_LOAD_LOOP, NO_LOCAL_LOAD_LOOP]:
                    message = f"Loop {loop}: {message}"
                return message
    return None


def schedule_get(name: str, code_path: int, schedule_info: 'ScheduleInfo') -> list[list[int]]:
    """When the schedule has only one code path, return it (broadcast for
    all callers regardless of ``code_path``); otherwise return the slice
    for ``code_path``. Lets multi-codepath rules iterate ``code_path`` in
    a loop without special-casing single-codepath schedules."""
    assert code_path >= 0, f"Code path {code_path} is not valid. Must be >= 0."
    schedules = schedule_info.optSchedule[name]
    return schedules[0] if len(schedules) == 1 else schedules[code_path]


@dataclass
class ValidationContext:
    """Typed context for CMS validation — replaces the raw context dict and ValidatorPassContext."""
    kernel: dict
    id_map: dict
    mfma_code: list
    mfma_reorder: list[int] = field(default_factory=list)
    # Phase 6 of plans/then-let-s-work-on-jaunty-reddy.md: schedule captures
    # for default-vs-CMS comparison rules. Both fields are optional and
    # populated by customMainLoopSchedule when capture is enabled. None means
    # the comparison rule is skipped (existing behavior preserved).
    default_capture: object = None
    cms_capture: object = None
    # Side-channel for typed Failure objects emitted by structural rules
    # (e.g. verify_scc_overlap). These rules return (False, str) for backward
    # compatibility but stash the Failure here so test infrastructure can
    # introspect typed fields. Reset to None at the top of each rule iteration
    # in isValid to prevent cross-rule leakage.
    _last_failure: Optional[object] = None

    @property
    def swap_global_read_order(self) -> bool:
        return self.kernel.get("SwapGlobalReadOrder", False)

    @property
    def direct_to_lds(self) -> bool:
        return self.kernel.get("DirectToLds", False)

    @property
    def use_f32x_emulation(self) -> bool:
        return self.kernel.get("UseF32XEmulation", False)

    @property
    def use_4x4mfma_tf32(self) -> bool:
        return self.kernel.get("UseMFMAF32XEmulation", False)

    @property
    def use_direct_32x_emulation(self) -> bool:
        return self.kernel.get("UseDirect32XEmulation", False)

    @property
    def use_plr_pack(self) -> bool:
        return self.kernel.get("UsePLRPack", False)

    @property
    def force_unroll_sub_iter(self) -> bool:
        return self.kernel.get("ForceUnrollSubIter", False)

    @property
    def n_tiles_a(self) -> int:
        return self.kernel["MIWaveTileA"]

    @property
    def n_tiles_b(self) -> int:
        return self.kernel["MIWaveTileB"]

    @property
    def dtl_plus_lds_buf(self) -> bool:
        return self.kernel.get("DtlPlusLdsBuf", False)

    @property
    def use_shadow_limit(self) -> bool:
        return self.kernel.get("Use64bShadowLimit", True)


# Keep backward-compatible alias during transition
ValidatorPassContext = ValidationContext


def format_kernel_string(kernel: 'Solution') -> str:
    """Format a human-readable description of the kernel's tile dimensions and transpose modes."""
    mt0 = kernel.get("MacroTile0", "?")
    mt1 = kernel.get("MacroTile1", "?")
    du = kernel.get("DepthU", "?")
    transA = "T" if kernel.get("TransA") else "N"
    transB = "T" if kernel.get("TransB") else "N"
    return f"MT0xMT1xDepthU = {mt0}x{mt1}x{du} {transA}{transB}"


def isValid(scheduleInfo: 'ScheduleInfo', context: 'ValidationContext') -> tuple[bool, str]:
    """
    Return True if all the validation rules pass, False otherwise.
    If validation fails, a string containing the reason is returned.

    Note 1: If True is returned, this is not proof that this schedule
    is valid. It may be a false negative.

    Note 2: if False is returned, this is not proof that the schedule
    is invalid. It may be a false positive.

    Rule dispatch uses the concern-based framework: active_concerns()
    determines which ValidationConcern values apply to this kernel,
    and only rules whose concerns intersect are executed.
    """
    kernel = context.kernel

    context.mfma_reorder = scheduleInfo.mfmaReorder or []

    # Determine required concerns from kernel config + ISA catalog.
    required = active_concerns(kernel, context.id_map)

    # Unknown/missing ISA (not in catalog) → run all rules (legacy behavior).
    # This ensures tests and kernels without ISA metadata get full validation.
    if not required:
        required = set(ValidationConcern)

    # Cross-scheduler comparison rule. Only fires when both default and CMS
    # captures are present in the context (i.e. capture was enabled when
    # this kernel was scheduled). Validates the CMS schedule against the
    # default via dataflow-graph equality + per-edge wait-coverage.
    if context.default_capture is not None and context.cms_capture is not None:
        from Tensile.Components.ScheduleCapture import (
            build_dataflow_graph, compare_graphs, validate_edge_wait_coverage,
            _resolve_arch_profile_for_isa,
        )
        # Attach the per-arch quad-cycle profile derived from this kernel's
        # ISA tuple so the four pair-specific gap helpers consult arch-
        # appropriate finish-cycle / settle-window values. Unknown ISAs
        # fall back to the CDNA 4 default for historical compatibility.
        arch_profile = _resolve_arch_profile_for_isa(
            tuple(kernel["ISA"]) if "ISA" in kernel else None
        )
        if context.default_capture.arch_profile is None:
            context.default_capture.arch_profile = arch_profile
        if context.cms_capture.arch_profile is None:
            context.cms_capture.arch_profile = arch_profile
        ref_graph = build_dataflow_graph(context.default_capture)
        subj_graph = build_dataflow_graph(context.cms_capture)
        graph_failures = compare_graphs(
            ref_graph, subj_graph, raise_on_unexplained=False,
        )
        if graph_failures:
            summary = "\n  ".join(
                f.format()
                for f in graph_failures
            )
            return False, (
                f"Dataflow graph comparison failed: "
                f"{len(graph_failures)} edge difference(s):\n  {summary}"
            )
        wait_failures = validate_edge_wait_coverage(
            subj_graph, raise_on_unexplained=False,
        )
        if wait_failures:
            summary = "\n  ".join(
                f.format()
                for f in wait_failures
            )
            return False, (
                f"Wait-coverage validation failed: "
                f"{len(wait_failures)} failure(s):\n  {summary}"
            )

    return True, ""


def findValidPositions(
    scheduleInfo: 'ScheduleInfo',
    context: 'ValidationContext',
    inst_name: str,
    inst_issue_idx: int,
) -> list[tuple[int, int]]:
    """Find all vmfma indices where an instruction can be moved while keeping the schedule valid.

    For each candidate vmfma index in [-1, numMfma-1], mutates the target
    instruction's entry in scheduleInfo.optSchedule in-place, calls isValid(),
    and restores. Collects all positions that pass and merges adjacent indices
    into ranges.

    Args:
        scheduleInfo:   The schedule to query.
        context:        Validation context dict (must contain "kernel", optionally "idMap"/"mfmaCode").
        inst_name:      The schedule key of the instruction to move (e.g. "LRA0", "GRA", "PackA0").
        inst_issue_idx: The index within that instruction's schedule list (0-based).

    Returns:
        A list of (start, end) tuples representing inclusive ranges of valid vmfma indices.
        E.g. [(-1, 2), (5, 8)] means indices -1, 0, 1, 2, 5, 6, 7, 8 are valid.
        An empty list means the instruction cannot be moved anywhere.
    """
    assert inst_name in scheduleInfo.optSchedule, f"Unknown instruction: {inst_name}"

    num_code_paths = len(scheduleInfo.optSchedule[inst_name])

    # Save original values across all code paths.
    original_values = []
    for cp in range(num_code_paths):
        original_values.append(scheduleInfo.optSchedule[inst_name][cp][inst_issue_idx])

    valid_indices = []
    for candidate in range(-1, scheduleInfo.numMfma):
        # Mutate in-place.
        for cp in range(num_code_paths):
            scheduleInfo.optSchedule[inst_name][cp][inst_issue_idx] = candidate

        valid, _ = isValid(scheduleInfo, context)
        if valid:
            valid_indices.append(candidate)

    # Restore original values.
    for cp in range(num_code_paths):
        scheduleInfo.optSchedule[inst_name][cp][inst_issue_idx] = original_values[cp]

    # Merge adjacent indices into ranges.
    ranges = []
    for idx in valid_indices:
        if ranges and idx == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], idx)
        else:
            ranges.append((idx, idx))

    return ranges

