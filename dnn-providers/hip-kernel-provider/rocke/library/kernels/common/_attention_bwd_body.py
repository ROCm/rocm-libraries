# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared inner body of the attention backward kernels (dQ, dK, dV).

Family glue for the backward kernels (Python-only, no C++ twin; packs pin
``ROCKE_BACKEND=python``). One :class:`BwdTileStep` emits the work of one q
step (``kM0`` query rows of one query head) against the CTA's resident key
tile (``kN0`` keys), on a multi-wave CTA:

* **G0** ``S = Q K^T`` and **G2** ``dP = dO V^T`` on the warp grid ``(1, W)``
  (waves split the keys; with ``pt_route = "lds"`` any grid of
  :mod:`._attention_bwd_tiles`, including replicated grids of fewer waves,
  whose extra waves compute copies and do not store P / dS);
* ``P = select(keep, exp2(S * scale_log2 - lse2), 0)`` and
  ``dS = select(keep && lse2 < +inf, P * (dP - Dsum), 0)``: selects, never
  multiplies, so a dead row (``lse2 = +inf``) or a masked cell contributes an
  exact 0 whatever its inputs;
* **G1** ``dV += P^T dO`` and **G3** ``dK += dS^T Q`` on ``(W, 1)``: the
  ``P^T`` / ``dS^T`` A operands are a checked register relabel of the G0 / G2
  accumulators (``pt_route = "relabel"``) or are read back from LDS
  (``"lds"``); ``dO^T`` and ``Q^T`` are K-outer reads of the staged tiles
  through the transpose source (element reads for ``lds_plain`` /
  ``wmma_lds``, ``ds_read_tr16_b64`` for ``tr_read``, K-inner vector reads
  of the transposed XT images ``dot`` / ``qt`` for ``xt_lds``);
* **G4** ``dQ_part = dS K`` with dS written to LDS once per step and K^T from
  registers (``kt_source = "reg"``, read once per CTA) or LDS (under
  ``xt_lds`` both from the XT image of K); dS is read one atom-K step at a
  time or in ``kK4`` key slices (``g4_slicing``, single or double register
  buffers, optionally the first slice under G3); the partial is
  handed to the dQ sink (fp32 workspace atomics, branch-free). A sink that
  takes no dQ (``NullSink``, the main kernel of ``dq_mode = "split"``, whose
  dQ comes from the separate Q-parallel kernel) emits no G4, no dS tile and
  no K^T fragments; G1 and G3 keep their order.

XT transposes (``transpose_source = "xt_lds"``): :func:`write_xt` copies a
published row-major tile into its transposed image (``rows = D``, ``cols =``
the sequence extent; per-buffer ``lds_swizzle``, by default the conflict
predictor's choice), so the K-outer operand becomes a K-inner one. K^T is written once per CTA after the K
staging barrier (``kt_stage`` / ``kt_res``), then a barrier; Q^T / dO^T are
written every step right after the barrier that publishes Q / dO, and are
published by the next barrier before G1 / G3 (the P / dS barrier of the LDS
route, the dS barrier when G4 issues first, else one more barrier). The
images move bytes only, so every MMA sees the operands of the plain reader in
the same order: dK, dV and every dQ partial are bit for bit those of
``lds_plain``.

G1, G3 and G4 issue in the order of ``stage_order`` (default ``"g1g3g4"``;
for example ``"g4g1g3"`` issues G4 and its atomics right after dS is
published, so they overlap the G1 / G3 matrix ops). Each accumulator's MMA
chain is unchanged by the order, so dK, dV and every dQ partial are bit for
bit those of the default order. Under the inline-asm G1 / G3 route
(``agpr_route = "asm_g13"``) with ``asm_load_hoist`` the LDS operands of the
second of G1 / G3 are read before the first one's asm cluster (the asm block
is a scheduling barrier for memory operations, so the backend cannot do it);
the MMAs and their operands are unchanged. Every asm cluster opens with the
ISA's VALU-to-MFMA lead wait states (``BlockGemm.mma``), which the backend
cannot insert for an MFMA it does not see.

Barriers per q step: three by default (Q / dO staged, dS visible before G4,
step end). ``barrier_merge = "on"`` (relabel route, ``ring_depth = 2``) keeps
one: Q / dO / lse2 / Dsum live in a 2-slot LDS ring and dS in 2 slots, a step
reads slot ``i % 2`` and stages the next step into the other slot before its
pre-G4 barrier; the next step's global loads are issued ahead of this step's
dQ atomics (:meth:`BwdTileStep._ring_views` holds the hazard proof).

Scheduling (``sched``, ``setprio``; :mod:`._attention_bwd_sched`): the step
runs as stages A (G0), B (G2 + G1), C (G3) and D (G4). The fenced values load
the stats in A and read each register-fed operand one stage ahead, the stage
tables add ``sched_group_barrier`` groups per stage, and
``stage_table_atomics`` defers each step's dQ atomics to stage A of the next
step (the partial is loop-carried; :meth:`BwdTileStep.flush_dq` issues the
last one after the loops). Only placement changes: every value is the
default's. With a ring (the prefetch ring of ``_attention_bwd_ring`` or the
merged barrier) the fenced stats come from the ring's slot and a stage table
issues the register rings' prefetch loads in stage A; the DMA copy stays at
the step top and its waits keep their immediates (the deferred atomics are
as many VMEM operations after the copy as the step's own). The split-dQ main
kernel (no G4) runs stages A, B and C, C closing the step (no fence after
it); ``stage_table_atomics`` is refused there (no atomics to defer).

Edge and interior steps: an edge step applies the mask policy's per-cell keep
through per-row key limits (one compare per cell; the dead-row term
``lse2 < +inf`` is folded into the row, so the same compare gates P and dS);
an interior step (every cell kept, every row and key in range) has no per-cell
select and keeps only the dead-row select of dS, which gives identical values.

Head packing (:class:`PackedStep`): the M rows of a step hold ``(head, row)``
pairs of several query heads (short query spans); Q / dO staging, the stats,
the keep and the dQ sink use each row's own head, and a dead packed row is
dead in every term.

With LDS accumulators (WMMA geometries) G1 / G3 update the LDS images and G4
feeds the sink one fragment at a time, which bounds the live registers.

Under ``loop_sync = "counted"`` (CDNA) the step stores a Q / dO tile and stats
loaded by the previous step (or the loop's prologue) at its top, issues the
next step's loads before its dQ atomics (``counted_issue``) and waits
``vmcnt(N) lgkmcnt(0)`` at its in-loop barriers, ``N`` the atomics issued after
the last load (:mod:`._attention_bwd_counted`); the math is unchanged.

After the q loops the family shell reads dK / dV back either as rows of the
I/O dtype (:meth:`BwdTileStep.dkv_rows`, the LDS-staged direct epilogue), as
per-lane cells of the I/O dtype (:meth:`BwdTileStep.dkv_cells`, the direct
epilogue without LDS) or as fp32 slots (:meth:`BwdTileStep.dkv_row_slots`,
atomic epilogue into the dK / dV workspace).

Every LDS buffer of the CTA lives in one flat allocation at the byte offsets
of the phase planner (:func:`._attention_bwd_lds.plan_bwd_lds`), so the
planner's footprint is the kernel's group segment and its phase aliasing
(stage-0 K, stage-0 V, the q loop and the epilogue reuse one region) is what
the IR does; barriers separate the phases.

Build-time facts: every atom's A / B / C lane maps are evaluated for every
lane and slot and must be contiguous along K and affine; the relabel route is
proved per atom pair; the transpose read is proved against the catalog
operand map, or, for the K-permuted G1 / G3 B of a 16x16x32 consumer, against
the stacked K-permuted plan (two 16x16x16 B reads 16 K rows apart). An
illegal geometry raises :class:`ValueError` before any IR is emitted.

Hazard structure (checked by the IR tests): the q-loop trip count is
``readfirstlane``'d, the empty-band exit (trip count 0) is decided before any
accumulator is live, and no matrix instruction sits inside a divergent
region.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cached_property

from rocke.core.ir import F32, I32, IRBuilder, Value, VectorType

from ._attention_bwd_addr import (
    PackedRows,
    RowSlab,
    StridedTensorAddr,
    WorkspaceAddr,
    chunk_coords,
    deal_chunks,
)
from ._attention_bwd_band import MaskPolicy
from ._attention_bwd_caps import bwd_arch_caps, check_transpose_read
from ._attention_bwd_counted import (
    COUNTED_ISSUES,
    CountedWaits,
    StagedStep,
    issue_tile_loads,
    store_tile,
    tile_chunks,
    zero_like,
)
from ._attention_bwd_frag import (
    DmaSlabView,
    LdsView,
    XorLdsView,
    atom_facts,
    check_tr16_coords,
    ir_type,
    operand_coords,
    tr16_read_plan,
)
from ._attention_bwd_gemm import (
    BlockGemm,
    LdsOperand,
    check_relabel_grids,
    kperm_b_operand_coords,
    relabel_acc_to_a,
)
from ._attention_bwd_lds import DKV_EPILOGUES, LdsPlan, default_xt_writer
from ._attention_bwd_sched import _G4_SLICINGS as _SCHED_G4_SLICINGS
from ._attention_bwd_sched import (
    COUNTED_LATE_DEFERRED,
    FENCED_SCHEDS,
    SPLIT_DEFERRED,
    STAGE_TABLE_SCHEDS,
    StageTable,
)
from ._attention_bwd_sinks import DqSink, DqSlot, PendingDq
from ._attention_bwd_valu import (
    batched_stats_rows,
    cast_c_frag,
    check_stats_vec_layout,
    softmax_p,
    stats_rows_neg,
)

__all__ = [
    "DEFAULT_G4_SLICING",
    "DEFAULT_STAGE_ORDER",
    "G4_AFTER_G3_ORDERS",
    "G4_SLICINGS",
    "STAGE_ORDERS",
    "BwdAtomTable",
    "BwdLds",
    "BwdTileGeometry",
    "BwdTileStep",
    "COPY_ORDERS",
    "DkDvAcc",
    "PackedStep",
    "QStepInputs",
    "RingStep",
    "StatsView",
    "copy_packed_rows_to_lds",
    "copy_row_sets_to_lds",
    "copy_rows_to_lds",
    "packed_row_loader",
    "slab_row_loader",
    "stage_units",
    "write_xt",
]

# K-outer reader per transpose source: "plain" element reads of the row-major
# tile, "tr16" the gfx950 transpose read, "xt" K-inner vector reads of the
# transposed XT image that :func:`write_xt` builds from the row-major tile.
_TRANSPOSE_READERS = {
    "lds_plain": "plain",
    "wmma_lds": "plain",
    "tr_read": "tr16",
    "xt_lds": "xt",
}
_INF = float("inf")
# Issue order of the three GEMMs that consume P / dS inside one q step (G1
# dV, G3 dK, G4 dQ with its sink), as GEMM names in issue order. The default
# is first; every permutation is legal and changes no accumulator's math.
STAGE_ORDERS = ("g1g3g4", "g3g1g4", "g4g1g3", "g1g4g3", "g3g4g1", "g4g3g1")
DEFAULT_STAGE_ORDER = STAGE_ORDERS[0]
# G4 dS key slicing (``block_k4`` = kK4 keys per slice). "step": one atom-K
# step at a time over the whole kN0 dS tile (read the step's operands, then its
# MMAs; kK4 is inert). "slice": read every operand of a kK4 key slice, then the
# slice's MMAs. "slice_db": two register buffers, slice s + 1 read before the
# MMAs of slice s. "slice_db_g3": "slice_db" with slice 0 read before the G3
# MMAs (under the relabel route the dS barrier moves from before G4 to before
# G3). Every value issues each dQ fragment's MMAs in the same K order, so every
# dQ partial is bit for bit the "step" one.
G4_SLICINGS = ("step", "slice", "slice_db", "slice_db_g3")
DEFAULT_G4_SLICING = G4_SLICINGS[0]
if G4_SLICINGS != _SCHED_G4_SLICINGS:  # the stage-table model reads the same set
    raise AssertionError("G4_SLICINGS differs from the stage-table model's set")
# "slice_db_g3" reads slice 0 under G3, so G3 must issue right before G4.
G4_AFTER_G3_ORDERS = tuple(o for o in STAGE_ORDERS if "g3g4" in o)


def stage_units(order: str) -> tuple[str, str, str]:
    """``("g1", "g3", "g4")`` style GEMM names of a stage order, in issue order."""
    if order not in STAGE_ORDERS:
        raise ValueError(f"stage_order={order!r} not in {STAGE_ORDERS}")
    return order[0:2], order[2:4], order[4:6]


def _atom_k(atom: str) -> int:
    return int(str(atom).rsplit("x", 1)[1])


def _rounding_point(b: IRBuilder, v: Value) -> Value:
    """``v`` (fp32) through an empty asm: the fp32 value is materialised.

    The dS operands are rounded to fp32 here and to the I/O dtype after it.
    Without this point the backend may fold ``ds_mult * dS`` and the
    conversion into one mixed-precision FMA (a single rounding) in one step
    variant and not in another, so the interior and edge steps (and every
    other variant of the step) would round some dS values differently.
    """
    return b.inline_asm("", "=v,0", [v], F32, sideeffect=False)


def _agpr_fence(b: IRBuilder, frag: Value) -> Value:
    """``frag`` through an empty asm whose output is an AGPR: a loop-carried
    accumulator passed through it at the back-edge is allocated to AGPRs (the
    ``agpr_fence`` register route)."""
    return b.inline_asm("", "=a,0", [frag], frag.type, sideeffect=False)


def _vgpr_fence(b: IRBuilder, frag: Value) -> Value:
    """``frag`` back to an arch VGPR (after the loop, before the epilogue)."""
    return b.inline_asm("", "=v,0", [frag], frag.type, sideeffect=False)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BwdAtomTable:
    """Atom K extent per GEMM pair (16 or 32); the 16x16 shape is implied."""

    g02: int
    g13: int
    g4: int


@dataclass(frozen=True)
class BwdTileGeometry:
    """Resolved, validated, hashable tile geometry of one backward instance.

    Built by :meth:`from_spec` from a family spec that has been resolved and
    validated by its family's policy (every knob set). The five block GEMMs and
    the operand readers are derived from it; :meth:`check` proves the lane-map
    facts and raises :class:`ValueError` for an illegal combination.
    """

    arch: str
    dtype: str
    waves: int
    k_m0: int
    k_n0: int
    k_k4: int
    head_size: int
    grid_g0: tuple[int, int]
    grid_g1: tuple[int, int]
    grid_g4: tuple[int, int]
    atoms: BwdAtomTable
    pt_route: str
    kv_residency: str
    kt_source: str
    transpose_source: str
    ring_depth: int = 1
    global_path: str = "vgpr"
    lds_swizzle: str | None = None
    sched: str = "none"
    setprio: bool = False
    edge_tiles: bool = False
    head_pack: bool = False
    agpr_alloc: tuple[int, int] | None = None
    s_in_agpr: bool = False
    acc_in_lds: bool = False
    dq_mode: str = "atomic"
    agpr_route: str | None = None
    stage_vec: int = 8
    asm_tail_nop: int | None = None
    asm_cluster: str = "k_step"
    asm_load_hoist: bool = False
    stage_order: str = DEFAULT_STAGE_ORDER
    loop_sync: str = "full"
    dkv_epilogue: str = "lds_staged"
    dkv_store_vec: int = 8
    qdo_stage: str = "split"
    stats_load: str = "at_use"
    dq_issue: str = "block"
    interior_clamp: bool = False
    barrier_merge: str = "off"
    ring_issue: str = "early"
    counted_issue: str = "early"
    # softmax-gradient VALU levers (see _attention_bwd_valu)
    bf16_cast: str = "generic"
    bwd_exp2: str = "exact"
    stats_fma: bool = False
    stats_vec: int = 1
    g4_slicing: str = DEFAULT_G4_SLICING

    @classmethod
    def from_spec(cls, spec, arch: str) -> BwdTileGeometry:
        """Geometry of a resolved family spec (``waves`` etc. already set)."""
        if spec.waves is None or spec.block_m is None:
            raise ValueError("BwdTileGeometry needs a resolved spec")
        epilogue = getattr(spec, "dkv_epilogue", "lds_staged")
        if getattr(spec, "dkv_mode", "direct") == "atomic" and epilogue != "direct":
            # atomic dK / dV stage nothing; the plan keeps the default buffer
            epilogue = "lds_staged"
        geom = cls(
            arch=arch,
            dtype=spec.dtype,
            waves=spec.waves,
            k_m0=spec.block_m,
            k_n0=spec.block_n,
            k_k4=spec.block_k4,
            head_size=spec.head_size,
            grid_g0=tuple(spec.warp_grid_g02),
            grid_g1=tuple(spec.warp_grid_g13),
            grid_g4=tuple(spec.warp_grid_g4),
            atoms=BwdAtomTable(
                g02=_atom_k(spec.atom_g02),
                g13=_atom_k(spec.atom_g13),
                g4=_atom_k(spec.atom_g4),
            ),
            pt_route=spec.pt_route,
            kv_residency=spec.kv_residency,
            kt_source=spec.kt_source,
            transpose_source=spec.transpose_source,
            ring_depth=spec.ring_depth,
            global_path=spec.global_path,
            lds_swizzle=spec.lds_swizzle,
            sched=spec.sched,
            setprio=bool(spec.setprio),
            edge_tiles=bool(spec.edge_tiles),
            head_pack=bool(spec.head_pack),
            agpr_alloc=spec.agpr_alloc,
            s_in_agpr=bool(spec.s_in_agpr),
            acc_in_lds=bool(spec.acc_in_lds),
            dq_mode=spec.dq_mode,
            agpr_route=getattr(spec, "agpr_route", None),
            stage_vec=spec.stage_vec,
            asm_tail_nop=getattr(spec, "asm_tail_nop", None),
            asm_cluster=getattr(spec, "asm_cluster", "k_step"),
            asm_load_hoist=bool(getattr(spec, "asm_load_hoist", False)),
            stage_order=getattr(spec, "stage_order", DEFAULT_STAGE_ORDER),
            loop_sync=getattr(spec, "loop_sync", "full"),
            dkv_epilogue=epilogue,
            dkv_store_vec=getattr(spec, "dkv_store_vec", 8),
            qdo_stage=getattr(spec, "qdo_stage", "split"),
            stats_load=getattr(spec, "stats_load", "at_use"),
            dq_issue=getattr(spec, "dq_issue", "block"),
            interior_clamp=bool(getattr(spec, "interior_clamp", False)),
            barrier_merge=getattr(spec, "barrier_merge", "off"),
            ring_issue=getattr(spec, "ring_issue", "early"),
            counted_issue=getattr(spec, "counted_issue", "early"),
            bf16_cast=getattr(spec, "bf16_cast", "generic"),
            bwd_exp2=getattr(spec, "bwd_exp2", "exact"),
            stats_fma=bool(getattr(spec, "stats_fma", False)),
            stats_vec=getattr(spec, "stats_vec", 1),
            g4_slicing=getattr(spec, "g4_slicing", DEFAULT_G4_SLICING),
        )
        geom.check()
        return geom

    # --- block GEMMs ------------------------------------------------------
    @cached_property
    def g0(self) -> BlockGemm:
        """G0 ``S = Q K^T`` (also G2 ``dP = dO V^T``): kM0 x kN0 over D."""
        return BlockGemm(
            self.arch, self.dtype, self.k_m0, self.k_n0, self.head_size,
            self.grid_g0, self.atoms.g02,
        )  # fmt: skip

    @cached_property
    def g1(self) -> BlockGemm:
        """G1 ``dV += P^T dO`` (also G3 ``dK += dS^T Q``): kN0 x D over kM0."""
        return BlockGemm(
            self.arch, self.dtype, self.k_n0, self.head_size, self.k_m0,
            self.grid_g1, self.atoms.g13,
        )  # fmt: skip

    @cached_property
    def g4(self) -> BlockGemm:
        """G4 ``dQ = dS K``: kM0 x D over kN0."""
        return BlockGemm(
            self.arch, self.dtype, self.k_m0, self.head_size, self.k_n0,
            self.grid_g4, self.atoms.g4,
        )  # fmt: skip

    @property
    def wave_size(self) -> int:
        return bwd_arch_caps(self.arch).wave_size

    @property
    def threads(self) -> int:
        return self.waves * self.wave_size

    @property
    def reader(self) -> str:
        """LDS reader of the K-outer (transposed) operands."""
        return _TRANSPOSE_READERS[self.transpose_source]

    @cached_property
    def relabel_plan(self):
        """The proved C-to-A relabel plan (``pt_route = "relabel"`` only)."""
        if self.pt_route != "relabel":
            return None
        return check_relabel_grids(self.g0, self.g1)

    @cached_property
    def b13_coords(self):
        """B coordinates of G1 / G3 (K-permuted when the relabel needs it)."""
        plan = self.relabel_plan
        if plan is None:
            return operand_coords(self.g1.op, "b")
        return kperm_b_operand_coords(plan, self.g0, self.g1)

    @property
    def g4_slice_buffers(self) -> int:
        """Register buffers of the G4 dS slices: 2 under the double-buffered
        values (slice ``s + 1`` read before the MMAs of slice ``s``), else 1."""
        return 2 if self.g4_slicing in ("slice_db", "slice_db_g3") else 1

    @property
    def g4_prefetch_before_g3(self) -> bool:
        """Whether G4 slice 0 is read before the G3 MMAs."""
        return self.g4_slicing == "slice_db_g3"

    def stage_table(self) -> StageTable:
        return StageTable.from_geometry(
            self, self.arch, dtype=self.dtype, stage_vec=self.stage_vec,
            dq_mode=self.dq_mode, acc_in_lds=self.acc_in_lds,
        )  # fmt: skip

    def check(self) -> None:
        """Lane-map facts, relabel proof and transpose-read proof.

        Raises :class:`ValueError` for an illegal geometry; no IR is emitted.
        """
        if self.transpose_source not in _TRANSPOSE_READERS:
            raise ValueError(
                f"transpose_source={self.transpose_source!r} has no operand reader "
                f"in this body (built: {tuple(_TRANSPOSE_READERS)})"
            )
        if self.grid_g0 != (1, self.waves) or self.grid_g1 != (self.waves, 1):
            if self.pt_route == "relabel":
                raise ValueError("the relabel route needs G0 grid (1, W), G1 (W, 1)")
        w0 = self.grid_g0[0] * self.grid_g0[1]
        if w0 > self.waves or self.waves % w0:
            raise ValueError(
                f"G0 grid {self.grid_g0} must hold W={self.waves} waves or a divisor "
                "of W (a replicated grid)"
            )
        for name, grid in (("G1", self.grid_g1), ("G4", self.grid_g4)):
            if grid[0] * grid[1] != self.waves:
                raise ValueError(f"{name} grid {grid} must hold W={self.waves} waves")
        for g in (self.g0, self.g1, self.g4):
            facts = atom_facts(g.op)
            if not (facts.a_contig and facts.b_contig and facts.affine):
                raise ValueError(
                    f"{g.op.op_id}: lane maps are not K-contiguous and affine "
                    f"({facts}); this geometry cannot be built on {self.arch}"
                )
        plan = self.relabel_plan  # proves the relabel (raises when illegal)
        if plan is not None and plan.needs_k_perm and not self.b13_coords.kperm:
            raise ValueError(
                "the relabel permutes the G1 / G3 K order; their B operand must "
                "be loaded with the same permutation"
            )
        if self.reader == "tr16":
            check_transpose_read(self.arch, 64)
            # the G1 / G3 B may be K-permuted (16x16x32 consumer of the
            # relabel): the stacked plan; every other operand is a catalog map
            tr16_read_plan(self.b13_coords)
            check_tr16_coords(operand_coords(self.g4.op, "b"))
            if self.pt_route == "lds":
                check_tr16_coords(operand_coords(self.g1.op, "a"))
        if self.dq_mode not in ("atomic", "split"):
            raise ValueError(f"dq_mode={self.dq_mode!r} not in ('atomic', 'split')")
        stage_units(self.stage_order)  # raises for an unknown order
        if self.stage_order != DEFAULT_STAGE_ORDER and self.sched in STAGE_TABLE_SCHEDS:
            raise ValueError(
                f"stage_order={self.stage_order!r} with sched={self.sched!r}: the "
                "stage-table count model assumes the default G1, G3, G4 order"
            )
        if self.dkv_epilogue not in DKV_EPILOGUES:
            raise ValueError(f"dkv_epilogue must be one of {DKV_EPILOGUES}")
        if self.dkv_store_vec not in (8, 4, 2, 1):
            raise ValueError("dkv_store_vec must be 8, 4, 2 or 1")
        if self.acc_in_lds and self.dkv_epilogue == "lds_staged_pair":
            raise ValueError("acc_in_lds has no dK / dV staging buffer to pair")
        if self.dq_issue not in ("block", "fragment", "fragment_regs"):
            raise ValueError(f"dq_issue={self.dq_issue!r} is not built in this body")
        if self.asm_load_hoist and (self.agpr_route != "asm_g13" or self.acc_in_lds):
            raise ValueError(
                "asm_load_hoist needs the asm_g13 route with register accumulators"
            )
        if self.bf16_cast != "generic" and self.dtype != "bf16":
            raise ValueError(f"bf16_cast={self.bf16_cast!r} needs dtype='bf16'")
        if self.stats_vec != 1:
            check_stats_vec_layout(self.g0.op, self.stats_vec)
        if self.barrier_merge not in ("off", "on"):
            raise ValueError(f"barrier_merge={self.barrier_merge!r} is not built")
        if self.ring_issue not in ("early", "late"):
            raise ValueError(f"ring_issue={self.ring_issue!r} is not built")
        if self.barrier_merge == "on" and (
            self.ring_depth != 2
            or self.global_path != "vgpr"
            or self.pt_route != "relabel"
            or self.acc_in_lds
            or stage_units(self.stage_order)[2] != "g4"
        ):
            raise ValueError(
                "barrier_merge='on' needs ring_depth=2, global_path='vgpr', the "
                "relabel route, register dK / dV and G4 last in stage_order"
            )
        self._check_counted()
        self._check_sched()
        self._check_g4_slicing()

    def _check_sched(self) -> None:
        """A schedule value (``sched`` other than ``none``, ``setprio``) runs
        on the bodies its stage table models: the synchronous step, the Q /
        dO prefetch ring (register or DMA path), the merged-barrier ring and
        the counted waits (``StageTable.pipeline``), with G4 and its dQ
        atomics in the step or, for the split-dQ main kernel (no G4), as the
        stages A..C. ``stage_table_atomics`` defers the dQ atomics: the
        split-dQ main kernel has none (``SPLIT_DEFERRED``); the counted late
        issue is refused by :meth:`_check_counted`."""
        if self.sched == "none" and not self.setprio:
            return
        if self.dq_mode != "atomic" and self.sched == "stage_table_atomics":
            raise ValueError(SPLIT_DEFERRED)

    @property
    def counted_atomics(self) -> int:
        """dQ atomics one wave issues per q step (one per G4 accumulator slot)."""
        g4 = self.g4
        return g4.frags_m * g4.frags_n * g4.op.c_frag_len

    def _check_counted(self) -> None:
        """Couplings of ``loop_sync = "counted"`` (see ``_attention_bwd_counted``)."""
        if self.counted_issue not in COUNTED_ISSUES:
            raise ValueError(f"counted_issue must be one of {COUNTED_ISSUES}")
        if self.loop_sync != "counted":
            if self.counted_issue != COUNTED_ISSUES[0]:
                raise ValueError("counted_issue applies to loop_sync='counted' only")
            return
        if bwd_arch_caps(self.arch).matrix_path != "mfma":
            raise ValueError(
                "loop_sync='counted' is CDNA-only (RDNA counts no-return atomics in "
                "vscnt, not vmcnt)"
            )
        if (self.global_path, self.ring_depth) != ("vgpr", 1):
            raise ValueError(
                "loop_sync='counted' owns the next-step loads: it needs "
                "global_path='vgpr' and ring_depth=1"
            )
        if self.sched == "stage_table_atomics" and self.counted_issue == "late":
            raise ValueError(COUNTED_LATE_DEFERRED)
        if (self.qdo_stage, self.stats_load) != ("split", "at_use"):
            raise ValueError(
                "loop_sync='counted' owns the Q / dO and stats issue order: it needs "
                "qdo_stage='split' and stats_load='at_use'"
            )
        CountedWaits(self.counted_atomics)  # raises past the vmcnt field

    def _check_g4_slicing(self) -> None:
        """Couplings of the G4 dS key slicing (``g4_slicing`` other than step)."""
        if self.g4_slicing not in G4_SLICINGS:
            raise ValueError(f"g4_slicing={self.g4_slicing!r} not in {G4_SLICINGS}")
        if self.g4_slicing == DEFAULT_G4_SLICING:
            return
        why = None
        if self.acc_in_lds or not bwd_arch_caps(self.arch).has_agprs:
            # the LDS-accumulator G4 feeds the sink one fragment at a time;
            # slicing would keep every dQ fragment of the wave live
            why = "MFMA register accumulators (acc_in_lds=False)"
        elif self.dq_issue != "block":
            why = "dq_issue='block' (the per-fragment issue reads dS its own way)"
        elif self.k_k4 % self.atoms.g4 or self.k_n0 % self.k_k4:
            why = "kK4 a multiple of the G4 atom K that divides kN0"
        elif self.g4_prefetch_before_g3 and self.stage_order not in G4_AFTER_G3_ORDERS:
            why = f"a stage_order with G3 right before G4 {G4_AFTER_G3_ORDERS}"
        elif self.dq_mode != "atomic":
            why = "dq_mode='atomic' (the split-dQ main kernel has no G4)"
        elif (
            self.barrier_merge != "off"
            or self.ring_depth != 1
            or self.global_path != "vgpr"
            or self.loop_sync == "counted"
        ):
            why = (
                "the synchronous body: barrier_merge='off', ring_depth=1, "
                "global_path='vgpr' and loop_sync != 'counted' (the slice reads "
                "and the moved dS barrier are built for its three barriers)"
            )
        elif self.sched != "none" or self.setprio:
            why = "sched='none' (the stage table does not model the dS key slices)"
        elif self.reader == "xt" and self.kt_source == "lds":
            why = "K^T from registers under xt_lds (no K-inner K^T image slices)"
        if why:
            raise ValueError(f"g4_slicing={self.g4_slicing!r} needs {why}")


# ---------------------------------------------------------------------------
# LDS realisation: one flat allocation, views at the planner's offsets
# ---------------------------------------------------------------------------


def quad_transpose4(b: IRBuilder, vals: Sequence[Value], lane: Value) -> list[Value]:
    """4 x 4 transpose of fp32 values inside each lane quad.

    Lane ``j`` (``lane mod 4``) holds ``a_j[0..3]``; it returns ``b_j[i] =
    a_i[j]``. Two exchange stages over the quad (partners ``j ^ 1``, then
    ``j ^ 2``, each one DPP ``quad_perm`` per slot pair): in stage ``m`` a lane
    keeps slot ``i`` when bit ``m`` of ``i`` equals bit ``m`` of ``j`` and
    takes its partner's slot ``i ^ (1 << m)`` otherwise. Pure data movement
    (bitcasts, selects, lane permutes): the values are bit for bit kept.
    """
    if len(vals) != 4:
        raise ValueError("quad_transpose4 takes 4 values per lane")
    x = [b.bitcast(v, I32) for v in vals]
    j = b.mod(lane, b.const_i32(4))
    zero = b.const_i32(0)
    for bit, perm, pairs in ((1, [1, 0, 3, 2], ((0, 1), (2, 3))),
                             (2, [2, 3, 0, 1], ((0, 2), (1, 3)))):  # fmt: skip
        hi = b.cmp_ne(b.land(j, b.const_i32(bit)), zero)
        for lo_s, hi_s in pairs:
            # a lane with the bit set sends its low slot, else its high slot
            send = b.select(hi, x[lo_s], x[hi_s])
            recv = b.quad_perm(send, perm)
            x[lo_s], x[hi_s] = b.select(hi, recv, x[lo_s]), b.select(hi, x[hi_s], recv)
    return [b.bitcast(v, F32) for v in x]


def _pad_elems(desc: str, elem_bytes: int) -> int:
    """Row pad in elements of a ``rowmajor:<swizzle>`` planner buffer."""
    sw = desc.split(":", 1)[1] if ":" in desc else "none"
    if sw.startswith("pad"):
        return int(sw[3:]) // elem_bytes
    if sw in ("none", "f32"):
        return 0
    raise ValueError(f"LDS buffer layout {desc!r} has no view in this body")


class BwdLds:
    """The CTA's LDS: one flat allocation of ``plan.total_bytes``.

    :meth:`view` returns a 2-D :class:`LdsView` of a planner buffer
    (``[rows][cols]`` 16-bit elements, or fp32 for the LDS accumulators): a
    row-major buffer (``rowmajor:*``) is ``[rows][D or kN0]``, an XT image
    (``xt:*``) is ``[D][kM0 or kN0]``; a padded layout widens the pitch and
    the ``xor`` layouts are :class:`XorLdsView` s of the planner's layout.
    """

    def __init__(self, b: IRBuilder, plan: LdsPlan, dtype: str) -> None:
        if plan.total_bytes % 16:
            raise ValueError("the LDS plan total must be a multiple of 16 bytes")
        self.plan = plan
        self.elem = ir_type(dtype)
        self.smem = b.smem_alloc(self.elem, (plan.total_bytes // 2,), "bwd_lds")

    def has(self, name: str) -> bool:
        return self.plan.has(name)

    def view(self, name: str, cols: int, slot: int = 0) -> LdsView:
        buf = self.plan.buffer(name, slot)
        if buf.offset % 16:
            raise ValueError(f"LDS buffer {name} is not 16-byte aligned")
        if buf.layout in ("rowmajor:f32", "vector:f32"):
            pitch, scale = cols, 2
            rows = buf.nbytes // (4 * cols)
        elif buf.layout in ("rowmajor:xor", "xt:xor"):
            lay = self.plan.layout(name)
            if lay.cols != cols or lay.elem_bytes != 2:
                raise ValueError(
                    f"LDS buffer {name}: XOR layout {lay.rows}x{lay.cols} viewed "
                    f"with {cols} columns"
                )
            return XorLdsView(
                self.smem, buf.offset // 2, cols, lay.rows, chunk=lay.chunk_bytes // 2
            )
        elif buf.layout.startswith(("rowmajor:", "xt:")):
            pitch = cols + _pad_elems(buf.layout, 2)
            scale = 1
            rows = buf.nbytes // (2 * pitch)
        elif buf.layout == "dma_slab_xor":
            return DmaSlabView(
                self.smem, buf.offset // 2, cols, buf.nbytes // (2 * cols)
            )
        else:
            raise ValueError(f"LDS buffer {name} layout {buf.layout!r} has no view")
        return LdsView(self.smem, buf.offset // 2, pitch, rows, scale)


def copy_rows_to_lds(
    b: IRBuilder,
    slab: RowSlab,
    view: LdsView,
    *,
    row0: Value,
    rows: int,
    cols: int,
    tid: Value,
    threads: int,
    vec: int = 8,
) -> None:
    """Copy rows ``row0 .. row0 + rows`` of a slab into ``view[0 .. rows)``.

    Each thread moves ``vec``-element row segments (the slab's ``stage_vec``
    decides the global access width; the LDS store is always ``vec`` wide).
    Rows are clamped by the slab (MMA operands come from valid rows).
    """
    load = slab_row_loader(b, slab, row0, vec)
    _copy_rows(b, load, view, rows=rows, cols=cols, tid=tid, threads=threads, vec=vec)


def slab_row_loader(b: IRBuilder, slab: RowSlab, row0: Value, vec: int = 8):
    """``load(r, c)``: ``vec`` elements of slab row ``row0 + r`` at column ``c``."""

    def load(r, c):
        return slab.load_vec(b.add(row0, r), c, vec)

    return load


def packed_row_loader(
    b: IRBuilder,
    addr: StridedTensorAddr,
    packed: PackedRows,
    vec: int = 8,
    *,
    clamp: bool = True,
):
    """``load(r, c)`` of M row ``r`` of a head-packed step (its own head).

    ``clamp=False`` (a step whose every row is inside the sequence) drops the
    row clamp of the head slab.
    """

    def load(r, c):
        h, q, _ok = packed.locate(b, r)
        return addr.slab(h, clamp=clamp).load_vec(q, c, vec)

    return load


def copy_packed_rows_to_lds(
    b: IRBuilder,
    addr: StridedTensorAddr,
    view: LdsView,
    packed: PackedRows,
    *,
    rows: int,
    cols: int,
    tid: Value,
    threads: int,
    vec: int = 8,
    clamp: bool = True,
) -> None:
    """Copy the M rows of a head-packed q step into ``view[0 .. rows)``.

    M row ``r`` holds row ``packed.locate(r)`` of its query head, read through
    a per-row i64 head rebase with the row clamped to the sequence (a dead
    packed row reads a valid row of the group's last head). ``clamp=False``
    (a step whose every row is inside the sequence) drops the row clamp.
    """
    load = packed_row_loader(b, addr, packed, vec, clamp=clamp)
    _copy_rows(b, load, view, rows=rows, cols=cols, tid=tid, threads=threads, vec=vec)


# Issue orders of :func:`copy_row_sets_to_lds` (the ``qdo_stage`` knob values).
COPY_ORDERS = ("split", "loads_first", "joint")


def copy_row_sets_to_lds(
    b: IRBuilder,
    sets: Sequence[tuple],
    *,
    order: str,
    rows: int,
    cols: int,
    tid: Value,
    threads: int,
    wave: Value | None = None,
    wave_size: int | None = None,
    vec: int = 8,
) -> None:
    """Copy several equally shaped row tiles ``(load, view)`` into LDS.

    ``order`` (the ``qdo_stage`` knob): ``"split"`` copies each tile in turn
    exactly as :func:`copy_rows_to_lds`; ``"loads_first"`` issues every global
    load of every tile before the first LDS store (a tail chunk past the tile
    loads the tile's last chunk and only its store is predicated);
    ``"joint"`` gives tile ``i`` the threads ``[i * n, (i + 1) * n)`` of one
    pass, ``n = rows * cols / vec`` chunks per tile, as whole waves (``n`` must
    be a multiple of ``wave_size`` and ``len(sets) * n <= threads``), so each
    tile's branch is wave-uniform. Every order moves the same bytes to the
    same LDS cells.
    """
    if order == "split":
        for load, view in sets:
            _copy_rows(
                b, load, view, rows=rows, cols=cols, tid=tid, threads=threads, vec=vec
            )
        return
    per_row = cols // vec
    total = rows * per_row
    if order == "loads_first":
        pending = []
        for load, view in sets:
            for chunk, tail in deal_chunks(b, tid, total, threads):
                src = b.smin(chunk, b.const_i32(total - 1)) if tail else chunk
                r, c = chunk_coords(b, src, per_row, vec)
                pending.append((view, r, c, load(r, c), chunk if tail else None))
        for view, r, c, val, guard in pending:
            if guard is None:
                view.store(b, r, c, val, vec)
            else:
                with b.scf_if(b.cmp_lt(guard, b.const_i32(total))):
                    view.store(b, r, c, val, vec)
        return
    if order != "joint":
        raise ValueError(f"copy order {order!r} not in {COPY_ORDERS}")
    if (
        wave is None
        or not wave_size
        or total % wave_size
        or len(sets) * total > threads
    ):
        raise ValueError(
            f"joint copy needs whole waves per tile: {len(sets)} x {total} chunks, "
            f"{threads} threads, wave size {wave_size}"
        )
    per_tile = total // wave_size  # waves per tile
    for i, (load, view) in enumerate(sets):
        hi = b.cmp_lt(wave, b.const_i32((i + 1) * per_tile))
        cond = hi if i == 0 else b.land(b.cmp_ge(wave, b.const_i32(i * per_tile)), hi)
        with b.scf_if(cond):
            chunk = tid if i == 0 else b.sub(tid, b.const_i32(i * total))
            r, c = chunk_coords(b, chunk, per_row, vec)
            view.store(b, r, c, load(r, c), vec)


def _row_loads(b, sets, *, rows, cols, tid, threads, vec=8) -> list:
    """The global loads of :func:`copy_row_sets_to_lds` ``"loads_first"``,
    returned as pending ``(view, row, col, value, vec, guard)`` stores (``guard``
    ``None`` or the predicate of a tail chunk past the tile)."""
    per_row = cols // vec
    total = rows * per_row
    pending = []
    for load, view in sets:
        for chunk, tail in deal_chunks(b, tid, total, threads):
            src = b.smin(chunk, b.const_i32(total - 1)) if tail else chunk
            r, c = chunk_coords(b, src, per_row, vec)
            guard = b.cmp_lt(chunk, b.const_i32(total)) if tail else None
            pending.append((view, r, c, load(r, c), vec, guard))
    return pending


def _copy_rows(b, load, view, *, rows, cols, tid, threads, vec) -> None:
    per_row = cols // vec
    total = rows * per_row
    for chunk, tail in deal_chunks(b, tid, total, threads):

        def body(chunk=chunk) -> None:
            r, c = chunk_coords(b, chunk, per_row, vec)
            view.store(b, r, c, load(r, c), vec)

        if not tail:
            body()
        else:
            with b.scf_if(b.cmp_lt(chunk, b.const_i32(total))):
                body()


def write_xt(
    b: IRBuilder,
    src: LdsView,
    dst: LdsView,
    *,
    seq: int,
    depth: int,
    tid: Value,
    threads: int,
    elem,
) -> None:
    """Transpose a published row-major tile ``src[seq][depth]`` into the XT
    image ``dst[depth][seq]``.

    The work distribution is the planner's writer
    (:func:`._attention_bwd_lds.default_xt_writer`, the one the conflict
    predictor models): unit ``u = iteration * threads + tid`` covers sequence
    rows ``s0 .. s0 + k_per_thread`` and head-dim columns ``d0 .. d0 + vec_d``,
    read as ``k_per_thread`` vector loads of ``vec_d`` elements and written as
    ``vec_d`` stores of ``k_per_thread`` elements (row ``d0 + j``). Units past
    the tile (a partial last iteration) are predicated off. The caller places
    the barriers: ``src`` must be visible before, ``dst`` is visible after the
    next barrier.
    """
    w = default_xt_writer(seq, depth, threads, elem_bytes=2)
    dl, n_sg, kpt, vec = w.d_lanes_eff, w.n_sg, w.k_per_thread, w.vec_d
    c = b.const_i32
    for it in range(w.iterations):
        u = tid if it == 0 else b.add(tid, c(it * threads))

        def body(u=u) -> None:
            rest = b.div(u, c(dl))
            sg = b.mod(rest, c(n_sg)) if w.n_dg > dl else rest
            dg = b.mod(u, c(dl))
            if w.n_dg > dl:
                dg = b.add(b.mul(b.div(rest, c(n_sg)), c(dl)), dg)
            s0 = b.mul(sg, c(kpt))
            d0 = b.mul(dg, c(vec))
            rows = [
                src.load(b, s0 if r == 0 else b.add(s0, c(r)), d0, dtype=elem, n=vec)
                for r in range(kpt)
            ]
            for j in range(vec):
                parts = [b.vec_extract(x, j) for x in rows]
                val = parts[0] if kpt == 1 else b.vec_pack(parts, elem)
                dst.store(b, d0 if j == 0 else b.add(d0, c(j)), s0, val, kpt)

        if (it + 1) * threads <= w.units:
            body()
        else:
            with b.scf_if(b.cmp_lt(u, c(w.units))):
                body()


# ---------------------------------------------------------------------------
# stats view
# ---------------------------------------------------------------------------


class StatsView:
    """Per-row ``lse2`` / ``Dsum`` of one q step, straight from the workspace.

    Each lane loads the values of its own C rows (per-wave redundant scalar
    loads; never a multi-dword load across the workspace end). Rows at or past
    ``ws_rows`` read the neutral values ``+inf`` (lse2) and ``0`` (Dsum).
    """

    def __init__(
        self,
        lse2: WorkspaceAddr,
        dsum: WorkspaceAddr,
        n_heads: Value | None = None,
        *,
        bounds: str = "clamp",
        span: int = 0,
    ) -> None:
        if lse2.width != 1 or dsum.width != 1:
            raise ValueError("lse2 / Dsum workspaces are one fp32 per row")
        if bounds not in ("clamp", "buffer"):
            raise ValueError(f"stats bounds={bounds!r} not in ('clamp', 'buffer')")
        if bounds != "clamp" and span < 1:
            raise ValueError("buffer-bounded stats need the q-step span (kM0)")
        self.lse2 = lse2
        self.dsum = dsum
        self.n_heads = n_heads  # query heads of the arrays (stats_vec > 1 only)
        self.bounds = bounds  # ws_bounds of the scalar row loads (rows)
        self.span = span

    def rows(
        self, b: IRBuilder, h: Value, q0: Value, rows: Sequence[Value]
    ) -> list[tuple[Value, Value]]:
        """``(lse2, Dsum)`` of q-step rows ``rows`` (offsets from ``q0``).

        Default: one clamped scalar load per row. With ``bounds = "buffer"``
        (``ws_bounds``) each row is one buffer load without the address clamp:
        the descriptor of a slab covers its valid rows (at most ``span``, the
        step reads rows ``[0, kM0)``), a row past the range reads 0, and every
        row then takes the default's ``row_ok`` select, so the values are bit
        for bit the default's.
        """
        sl = self.lse2.slab(h, q0)
        sd = self.dsum.slab(h, q0)
        if self.bounds == "clamp":
            return [(sl.load(r, 0, _INF), sd.load(r, 0, 0.0)) for r in rows]
        out = []
        for slab, neutral in ((sl, _INF), (sd, 0.0)):
            n = b.smin(slab.rows_left, b.const_i32(self.span))
            rsrc = b.buffer_rsrc(slab.ptr, b.mul(n, b.const_i32(4)))
            zero = b.const_i32(0)
            fill = b.const_f32(neutral)
            vals = []
            for r in rows:
                x = b.buffer_load(rsrc, b.mul(r, b.const_i32(4)), zero, F32)
                vals.append(b.select(slab.row_ok(r), x, fill))
            out.append(vals)
        return list(zip(out[0], out[1]))

    def batched_rows(
        self, b: IRBuilder, h: Value, q0: Value, rows: Sequence[Value], *, width: int
    ) -> list[tuple[Value, Value]]:
        """:meth:`rows` with one ``<width x f32>`` load per run of ``width``
        consecutive rows (``stats_vec``; bit for bit the same values)."""
        if self.n_heads is None:
            raise ValueError("stats_vec > 1 needs the StatsView head count")
        return batched_stats_rows(
            b, self.lse2, self.dsum, h, q0, rows, width=width, n_heads=self.n_heads
        )

    def packed_rows(
        self, b: IRBuilder, packed: PackedRows, rows: Sequence[Value]
    ) -> list[tuple[Value, Value]]:
        """``(lse2, Dsum)`` of M rows ``rows`` of a head-packed q step.

        Each row is rebased to its own head; a dead packed row reads the
        neutral ``+inf`` / ``0`` (selects, so a dead row is a dead stats row).
        """
        inf, zero = b.const_f32(_INF), b.const_f32(0.0)
        out = []
        for r in rows:
            h, q, ok = packed.locate(b, r)
            lse2 = self.lse2.slab(h).load(q, 0, _INF)
            dsum = self.dsum.slab(h).load(q, 0, 0.0)
            out.append((b.select(ok, lse2, inf), b.select(ok, dsum, zero)))
        return out

    # Loads and neutral values in two parts (``loop_sync = "counted"``: the
    # loads of the next step are carried in registers, and a neutral select
    # at the load would make the backend wait for the load right there).
    def raw(
        self, b: IRBuilder, h: Value, q0: Value, rows: Sequence[Value], packed=None
    ) -> list[tuple[Value, Value]]:
        """The ``(lse2, Dsum)`` loads of :meth:`rows` (``packed``: of
        :meth:`packed_rows`) without their neutral selects."""
        out = []
        if packed is None:
            sl, sd = self.lse2.slab(h, q0), self.dsum.slab(h, q0)
        for r in rows:
            q = r
            if packed is not None:
                hh, q, _ok = packed.locate(b, r)
                sl, sd = self.lse2.slab(hh), self.dsum.slab(hh)
            out.append((sl.load_raw(q, 0), sd.load_raw(q, 0)))
        return out

    def finish(
        self,
        b: IRBuilder,
        h: Value,
        q0: Value,
        rows: Sequence[Value],
        raw: Sequence[tuple[Value, Value]],
        packed=None,
    ) -> list[tuple[Value, Value]]:
        """The neutral selects of :meth:`raw` values: the values :meth:`rows`
        (or :meth:`packed_rows`) return for the same rows."""
        inf, zero = b.const_f32(_INF), b.const_f32(0.0)
        out = []
        if packed is None:
            sl, sd = self.lse2.slab(h, q0), self.dsum.slab(h, q0)
        for r, (lse2, dsum) in zip(rows, raw, strict=True):
            q, ok = r, None
            if packed is not None:
                hh, q, ok = packed.locate(b, r)
                sl, sd = self.lse2.slab(hh), self.dsum.slab(hh)
            lse2, dsum = sl.neutral(q, lse2, _INF), sd.neutral(q, dsum, 0.0)
            if ok is not None:
                lse2, dsum = b.select(ok, lse2, inf), b.select(ok, dsum, zero)
            out.append((lse2, dsum))
        return out


@dataclass(frozen=True, eq=False)
class PackedStep:
    """Inputs of a head-packed q step: the row map and the Q / dO tensors."""

    rows: PackedRows
    q_addr: StridedTensorAddr
    do_addr: StridedTensorAddr


@dataclass(frozen=True, eq=False)
class QStepInputs:
    """Where one q step reads: its first row ``q0``, head ``h`` and either the
    Q / dO slabs or (head packing) the packed row map."""

    q0: Value
    h: Value
    q_slab: RowSlab | None
    do_slab: RowSlab | None
    packed: PackedStep | None


@dataclass(frozen=True, eq=False)
class RingStep:
    """``barrier_merge = "on"`` inputs of a q step: its ring slot (uniform i32,
    0 or 1) and the next step of the loop, whose Q / dO / stats it stages into
    the other slot (the step itself when it is the last one of its loop)."""

    slot: Value
    nxt: QStepInputs


# ---------------------------------------------------------------------------
# accumulators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DkDvAcc:
    """dK / dV accumulators of one wave as ``[fm][fn]`` fragment lists.

    Empty with ``acc_in_lds`` (the accumulators live in LDS images).
    """

    dk: tuple[tuple[Value, ...], ...]
    dv: tuple[tuple[Value, ...], ...]

    def flat(self) -> list[Value]:
        return [v for row in self.dk for v in row] + [v for row in self.dv for v in row]

    @classmethod
    def from_flat(cls, values: Sequence[Value], frags_m: int, frags_n: int) -> DkDvAcc:
        n = frags_m * frags_n
        if len(values) != 2 * n:
            raise ValueError("accumulator count does not match the geometry")

        def grid(vals):
            return tuple(
                tuple(vals[fm * frags_n : (fm + 1) * frags_n]) for fm in range(frags_m)
            )

        return cls(dk=grid(list(values[:n])), dv=grid(list(values[n:])))


# ---------------------------------------------------------------------------
# tile step
# ---------------------------------------------------------------------------


class BwdTileStep:
    """Emitter of the per-CTA staging and the per-q-step G0..G4 program.

    One instance per CTA. ``lane`` / ``wave`` / ``tid`` come from
    :func:`._attention_bwd_gemm.lane_and_wave` (``wave`` is wave-uniform).
    """

    def __init__(
        self,
        b: IRBuilder,
        geom: BwdTileGeometry,
        *,
        mask: MaskPolicy,
        stats: StatsView,
        dq_sink: DqSink,
        lds: BwdLds,
        tid: Value,
        lane: Value,
        wave: Value,
        scale_log2: Value,
        ds_mult: Value | None,
        sentinel: bool = True,
    ) -> None:
        self.geom = geom
        self.mask = mask
        self.stats = stats
        self.sink = dq_sink
        self.lds = lds
        self.tid, self.lane, self.wave = tid, lane, wave
        self.scale_log2 = scale_log2
        self.ds_mult = ds_mult
        self.sentinel = sentinel
        self.sched = geom.stage_table()
        self.elem = ir_type(geom.dtype)
        d, m0, n0 = geom.head_size, geom.k_m0, geom.k_n0
        # A replicated G0 / G2 grid (fewer waves than the CTA): wave w computes
        # the sub-block of wave w mod k; only waves < k store P / dS.
        self.g0_replicated = geom.g0.waves != geom.waves
        wave0 = wave
        if self.g0_replicated:
            wave0 = b.mod(wave, b.const_i32(geom.g0.waves))
        self.o0 = geom.g0.wave_origin(b, wave0)
        self.o1 = geom.g1.wave_origin(b, wave)
        # G4 exists only with a sink that takes dQ (not under split dQ).
        self.o4 = geom.g4.wave_origin(b, wave) if dq_sink.emits_g4 else None
        self.k_regs = self.v_regs = self.kt_regs = None
        # Views of the planner buffers (no IR is emitted by a view). The dS
        # tile is planned for G4 and for the LDS route of dS^T only.
        self.v_q = lds.view("q", d)
        self.v_do = lds.view("do", d)
        self.v_ds = lds.view("ds", n0) if lds.has("ds") else None
        self.v_pt = lds.view("pt", n0) if geom.pt_route == "lds" else None
        if geom.barrier_merge == "on":  # fp32 stats rows of the ring slots
            self.v_lse2 = lds.view("lse2", 1)
            self.v_dsum = lds.view("dsum", 1)
        if lds.has("k_res"):
            self.v_k = lds.view("k_res", d)
        else:
            self.v_k = lds.view("k_stage", d)
        self.v_v = lds.view("v_res", d) if lds.has("v_res") else lds.view("v_stage", d)
        # XT images (xt_lds): Q^T / dO^T per step (ring slot 0), K^T once.
        self.v_qt = self.v_dot = self.v_kt = None
        if geom.reader == "xt":
            self.v_qt = lds.view("qt", m0)
            self.v_dot = lds.view("dot", m0)
            # (no K^T image without G4: the split-dQ main kernel)
            for kt in ("kt_res", "kt_stage"):
                if lds.has(kt):
                    self.v_kt = lds.view(kt, n0)
                    break
        if geom.acc_in_lds:
            self.v_dk_acc = lds.view("dk_acc", d)
            self.v_dv_acc = lds.view("dv_acc", d)
        else:
            self.v_dk_acc = self.v_dv_acc = None
        if m0 % 16 or n0 % 16:
            raise ValueError("tile extents must be multiples of 16")
        # loop_sync = "counted": the vmcnt model of the in-loop barriers
        self.waits = (
            CountedWaits(geom.counted_atomics) if geom.loop_sync == "counted" else None
        )

    # ----- one-time K / V staging ---------------------------------------
    def stage_kv(
        self, b: IRBuilder, k_slab: RowSlab, v_slab: RowSlab, k0: Value, live: Value
    ) -> None:
        """Stage K and V of the key tile (guarded by ``live``), then residency.

        Global reads happen only inside ``scf_if(live)``. With register
        residency the K (G0), V (G2) and K^T (G4; only when the sink takes
        dQ) fragments are read from LDS
        after a barrier; a barrier also separates the last stage-0 read from
        the first loop write, because the stage-0 phases alias the loop phase.
        Under ``xt_lds`` the published K tile is transposed into its XT image
        (``kt_stage`` or the resident ``kt_res``) behind one more barrier, and
        K^T is read from the image.
        """
        g = self.geom
        d, n0 = g.head_size, g.k_n0
        lane = self.lane
        kw = dict(rows=n0, cols=d, tid=self.tid, threads=g.threads)
        reg = g.kv_residency == "reg"
        with b.scf_if(live):
            copy_rows_to_lds(b, k_slab, self.v_k, row0=k0, **kw)
            if not reg:
                copy_rows_to_lds(b, v_slab, self.v_v, row0=k0, **kw)
        b.sync()
        if reg:
            self.k_regs = g.g0.load_all_b(
                b, LdsOperand(self.v_k, "k_inner"), lane, self.o0[1]
            )
        if self.v_kt is not None:
            write_xt(
                b, self.v_k, self.v_kt, seq=n0, depth=d, tid=self.tid,
                threads=g.threads, elem=self.elem,
            )  # fmt: skip
            b.sync()  # the K^T image is visible to every wave
        if g.kt_source == "reg" and self.sink.emits_g4:
            self.kt_regs = g.g4.load_all_b(b, self._kt_operand(), lane, self.o4[1])
        if reg:
            b.sync()  # stage-0 K reads complete before V overwrites the phase
            with b.scf_if(live):
                copy_rows_to_lds(b, v_slab, self.v_v, row0=k0, **kw)
            b.sync()
            self.v_regs = g.g0.load_all_b(
                b, LdsOperand(self.v_v, "k_inner"), lane, self.o0[1]
            )
            b.sync()  # stage-0 reads complete before the loop phase is written

    def _kt_operand(self) -> LdsOperand:
        """G4 B operand (K, K = key) in LDS: the K^T XT image read K-inner, or
        the row-major K tile read K-outer through the transpose source."""
        if self.v_kt is not None:
            return LdsOperand(self.v_kt, "k_inner")
        return LdsOperand(self.v_k, "k_outer", reader=self.geom.reader)

    def _k_outer_operand(self, view: LdsView) -> LdsOperand:
        """A K-outer tile read through the transpose source (P^T / dS^T on the
        LDS route; the XT reader has no image of those, so it reads them with
        the plain element reader)."""
        reader = self.geom.reader
        return LdsOperand(view, "k_outer", reader="plain" if reader == "xt" else reader)

    # ----- accumulators -------------------------------------------------
    def zero_acc(self, b: IRBuilder) -> DkDvAcc:
        """Zero the dK / dV accumulators (registers, or the LDS images)."""
        g = self.geom
        if g.acc_in_lds:
            self._zero_lds_acc(b)
            return DkDvAcc(dk=(), dv=())
        z = g.g1.zero_acc(b)
        z2 = g.g1.zero_acc(b)
        return DkDvAcc(dk=tuple(tuple(r) for r in z), dv=tuple(tuple(r) for r in z2))

    def _zero_lds_acc(self, b: IRBuilder) -> None:
        g = self.geom
        d, n0 = g.head_size, g.k_n0
        zero4 = b.zero_vec_f32(4)
        per_row = d // 4
        total = n0 * per_row
        for view in (self.v_dk_acc, self.v_dv_acc):
            for chunk, tail in deal_chunks(b, self.tid, total, g.threads):

                def body(chunk=chunk, view=view) -> None:
                    r, c = chunk_coords(b, chunk, per_row, 4)
                    view.store(b, r, c, zero4, 4)

                if not tail:
                    body()
                else:
                    with b.scf_if(b.cmp_lt(chunk, b.const_i32(total))):
                        body()
        b.sync()

    def _acc_lds_coords(self, b, fm, fn):
        return self.geom.g1.acc_coords(b, self.lane, self.o1, fm, fn)

    def _load_lds_acc(self, b, view: LdsView, fm: int, fn: int) -> Value:
        """Accumulator fragment ``[fm][fn]`` from its fp32 LDS image."""
        vals = [
            b.vec_extract(view.load(b, r, c, dtype=F32, n=1), 0)
            for r, c in self._acc_lds_coords(b, fm, fn)
        ]
        return b.vec_pack(vals, F32)

    def _store_lds_acc(self, b, view: LdsView, frag: Value, fm: int, fn: int) -> None:
        for s, (r, c) in enumerate(self._acc_lds_coords(b, fm, fn)):
            view.store(b, r, c, b.vec_extract(frag, s), 1)

    def _update_lds_acc(self, b, view: LdsView, a, bop) -> None:
        """``acc += A B`` on the LDS image, one fragment at a time (each lane
        reads back exactly the cells it writes)."""
        g1 = self.geom.g1
        for fm in range(g1.frags_m):
            for fn in range(g1.frags_n):
                frag = self._load_lds_acc(b, view, fm, fn)
                frag = g1.mma_frag(
                    b, frag, a, bop, fm, fn, lane=self.lane, origin=self.o1
                )
                self._store_lds_acc(b, view, frag, fm, fn)

    # ----- one q step ---------------------------------------------------
    def _loop_barrier(self, b: IRBuilder) -> None:
        """A barrier inside the q step: it publishes LDS writes (Q / dO, P,
        dS) or retires LDS reads before the next step overwrites them.

        No VMEM result crosses it from one wave to another: the global loads
        that feed the Q / dO ``ds_write`` are waited by data dependency in the
        wave that issued them, the lse2 / Dsum loads are used by that wave
        only, and the dQ atomics are never read in the kernel. ``loop_sync =
        "lds_only"`` therefore drains LDS only (the atomics stay in flight);
        ``"full"`` drains VMEM too; ``"counted"`` waits until only the step's
        dQ atomics are in flight (``vmcnt(N)``, see ``_attention_bwd_counted``).
        The staging and epilogue barriers are always full.
        """
        if self.geom.loop_sync == "lds_only":
            b.sync_lds_only()
        elif self.geom.loop_sync == "counted":
            self.waits.barrier(b)
        else:
            b.sync()

    # ----- loop_sync = "counted": next-step loads carried in registers ------
    def _qdo_chunks(self, b: IRBuilder):
        g = self.geom
        return tile_chunks(
            b, rows=g.k_m0, cols=g.head_size, tid=self.tid, threads=g.threads
        )

    @property
    def counted_n_stats(self) -> int:
        """Staged stats values per lane: (lse2, Dsum) per stats key."""
        g0 = self.geom.g0
        return 2 * g0.frags_m * g0.op.c_frag_len

    def counted_zero(self, b: IRBuilder) -> list[Value]:
        """Zeros of the :meth:`counted_loads` values (a guard loop's inits)."""
        g = self.geom
        n_chunks = -(-(g.k_m0 * g.head_size // 8) // g.threads)
        types = [F32] * self.counted_n_stats + [VectorType(self.elem, 8)] * (
            2 * n_chunks
        )
        return zero_like(b, types)

    def counted_loads(
        self,
        b: IRBuilder,
        *,
        h: Value,
        q0: Value,
        q_slab: RowSlab | None,
        do_slab: RowSlab | None,
        packed: PackedStep | None,
        edge: bool,
    ) -> StagedStep:
        """Issue every global load of a q step (lse2 / Dsum rows, then the Q
        and dO tiles) into registers: the values :meth:`run_q_step` stores at
        the top of that step under ``loop_sync = "counted"``. The rows, clamps
        and neutral values are those of the default step's loads."""
        g = self.geom
        row_of, keys = self._stats_rows(b)
        raw = self.stats.raw(
            b, h, q0, [row_of[k] for k in keys],
            packed=None if packed is None else packed.rows,
        )  # fmt: skip
        flat = tuple(v for pair in raw for v in pair)
        if packed is None:
            loads = (slab_row_loader(b, q_slab, q0), slab_row_loader(b, do_slab, q0))
        else:
            clamp = edge or g.interior_clamp
            loads = (
                packed_row_loader(b, packed.q_addr, packed.rows, clamp=clamp),
                packed_row_loader(b, packed.do_addr, packed.rows, clamp=clamp),
            )
        qdo = issue_tile_loads(b, loads, self._qdo_chunks(b))
        return StagedStep(stats=flat, qdo=tuple(qdo))

    def _issue_next(self, b: IRBuilder, next_loads) -> StagedStep:
        """The next step's loads at this point of the step: fenced by two
        ``sched_barrier(0)`` so the machine scheduler neither sinks them
        toward their next-step use nor moves this step's work across them
        (the issue order is the point of the counted waits)."""
        b.sched_barrier(0)
        nxt = next_loads()
        b.sched_barrier(0)
        self.waits.loads()
        return nxt

    def _store_staged(
        self, b: IRBuilder, staged: StagedStep, h: Value, q0: Value, packed
    ):
        """Write a staged step's Q / dO tile to LDS; its stats (the neutral
        selects applied here, at the step that uses them) as
        :meth:`_load_stats` returns them."""
        store_tile(b, (self.v_q, self.v_do), self._qdo_chunks(b), staged.qdo)
        row_of, keys = self._stats_rows(b)
        if len(staged.stats) != 2 * len(keys):
            raise ValueError("staged stats do not match the stats rows")
        raw = [tuple(staged.stats[2 * i : 2 * i + 2]) for i in range(len(keys))]
        fin = self.stats.finish(
            b, h, q0, [row_of[k] for k in keys], raw,
            packed=None if packed is None else packed.rows,
        )  # fmt: skip
        return row_of, keys, dict(zip(keys, fin))

    def run_q_step(
        self,
        b: IRBuilder,
        acc: DkDvAcc,
        q_slab: RowSlab,
        do_slab: RowSlab,
        qt: Value,
        h: Value,
        k0: Value,
        *,
        edge: bool = True,
        packed: PackedStep | None = None,
        ring: RingStep | None = None,
        prefetch=None,
        staged: StagedStep | None = None,
        next_loads: Callable[[], StagedStep] | None = None,
        pending: PendingDq | None = None,
    ) -> DkDvAcc | tuple[DkDvAcc, StagedStep] | tuple[DkDvAcc, PendingDq]:
        """G0..G4 and the dQ sink for q tile ``qt`` of head ``h``.

        ``edge`` selects the per-cell keep selects (edge tiles); an interior
        tile (every cell kept, every row in range) skips them and keeps only
        the dead-row sentinel select, which gives identical values.

        ``packed`` (head packing): the M rows hold ``(head, row)`` pairs of
        ``packed.rows``; Q / dO are staged per row from ``packed.q_addr`` /
        ``packed.do_addr`` (``q_slab`` / ``do_slab`` and ``h`` are unused), the
        stats, the keep and the dQ sink use each row's own head and row, and a
        dead packed row is dead in every term. A packed step with packing on
        must be an edge step (the caller guarantees it).

        Issue order: G0, G2, the softmax gradient and the P / dS hand-off,
        then G1, G3 and G4 (with its sink) in the geometry's ``stage_order``
        (default G1, G3, G4; G4 and its dS tile only when the sink takes dQ,
        see :attr:`DqSink.emits_g4`). Under the relabel route the barrier
        that makes the dS tile visible sits right before G4 in every order
        (right before G3 under ``g4_slicing = "slice_db_g3"``, which reads
        G4's first dS slice ahead of the G3 MMAs); under the LDS route the
        P / dS barrier precedes all three. G4 reads dS one atom-K step at a
        time (``"step"``) or in kK4 key slices
        (:meth:`BlockGemm.mma_k_sliced`). Under ``asm_g13`` with
        ``s_in_agpr`` G0 / G2 are AGPR-form asm clusters too.

        ``ring`` (``barrier_merge = "on"``): Q / dO / stats were staged into
        ring slot ``ring.slot`` before the previous barrier, the step stages
        ``ring.nxt`` into the other slot before the pre-G4 barrier, which is
        then its only barrier (see :meth:`_ring_views` for the hazard proof).

        ``prefetch`` (a ring step of ``_attention_bwd_ring``, ``ring_depth >
        1`` or ``global_path = "dma"`` without the merge): Q / dO (and, under
        DMA, lse2 / Dsum) were staged into a ring slot ahead of this step. The
        step reads that slot, issues the prefetch of a later step at its top
        and ends with the ring's wait and barrier; every in-loop barrier is
        LDS-only.

        ``loop_sync = "counted"``: ``staged`` holds this step's loaded Q / dO
        tile and stats (stored to LDS at the top; ``q_slab`` / ``do_slab``
        are unused), ``next_loads`` issues the next step's loads before the
        dQ atomics (``counted_issue``), and the result is ``(acc, next
        staged)``.

        Schedule (``geom.sched``): ``iglp*`` put one ``iglp_opt`` right after
        the staging barrier. A fenced sched (``FENCED_SCHEDS``) loads the
        lse2 / Dsum stats in stage A (after G0; unless ``stats_load =
        "step_top"``) and reads the register-fed operands one stage ahead: at
        the end of the stage before G1 its dO^T fragments, before G3 its Q^T
        fragments and, with K^T in LDS, before G4 its K^T fragments; the
        stage hooks of :class:`._attention_bwd_sched.StageTable` add the
        groups, priorities and fences. Every value, and every accumulator's
        MMA chain, is the default's.

        ``pending`` (``stage_table_atomics``, the deferred dQ sink): the dQ
        partial of the previous step (zeros before the first step), whose
        atomics are issued in stage A of this step; this step's partial is
        not handed to the sink but returned as ``(acc, PendingDq)`` for the
        next step (the caller flushes the last one with :meth:`flush_dq`).
        With a ring (``ring`` or ``prefetch``) the fenced stats come from the
        ring's slot, and a stage table issues the register rings' prefetch
        loads (``prefetch`` on the register path, ``ring`` with
        ``ring_issue = "early"``) in stage A instead of ahead of the step
        (the count model's :meth:`StageTable.ring_prefetch`). A schedule
        value does not run with the counted waits (``staged``; the validator
        refuses that composition).
        """
        g = self.geom
        g0, g1 = g.g0, g.g1
        m0 = g.k_m0
        lane = self.lane
        fenced = g.sched in FENCED_SCHEDS
        if (pending is not None) != (g.sched == "stage_table_atomics"):
            raise ValueError("a pending dQ partial is carried by stage_table_atomics")
        q0 = b.mul(qt, b.const_i32(m0))
        counted = g.loop_sync == "counted"
        stats = ring_loads = stats_src = nxt_staged = None
        barrier = self._loop_barrier
        if g.sched == "stage_table_atomics" and counted and g.counted_issue == "late":
            raise ValueError(COUNTED_LATE_DEFERRED)
        # A stage table issues the register rings' prefetch loads in stage A
        # (grouped with G0's MFMAs) instead of ahead of the step.
        loads_in_a = self.sched.ring_prefetch().get("where") == "A"
        if ring is not None:
            cur = self._ring_views(b, ring.slot)
            nxt = self._ring_views(b, b.sub(b.const_i32(1), ring.slot))
            v_q, v_do, v_ds = cur["q"], cur["do"], cur["ds"]
            if g.ring_issue == "early" and not loads_in_a:
                # the next step's global loads ahead of G0..G3 (and of this
                # step's dQ atomics, so their wait never covers those)
                ring_loads = self._ring_loads(b, ring.nxt, nxt)
        elif prefetch is not None:
            v_q, v_do, stats, stats_src = prefetch.begin(
                b, self, h, q0, packed, defer_loads=loads_in_a
            )
            v_ds = self.v_ds
            barrier = prefetch.barrier
        elif counted:
            if staged is None or next_loads is None:
                raise ValueError("loop_sync='counted' needs staged and next_loads")
            v_q, v_do, v_ds = self.v_q, self.v_do, self.v_ds
            self.waits.step_entry()
            stats = self._store_staged(b, staged, h, q0, packed)
            self._loop_barrier(b)
            if g.counted_issue == "early":
                nxt_staged = self._issue_next(b, next_loads)
        else:
            v_q, v_do, v_ds = self.v_q, self.v_do, self.v_ds
            stats = self._stage_qdo(b, q_slab, do_slab, h, q0, packed, edge=edge)
        # iglp_opt sits in the region of the step's matrix ops (after the
        # staging, whose tail guard can end a block).
        self.sched.emit_loop_head(b)

        # xt_lds: Q^T / dO^T images from the published tiles; a barrier before
        # the first G1 / G3 publishes them (see the unit loop below).
        xt_unpublished = self.v_qt is not None
        if xt_unpublished:
            if ring is not None or prefetch is not None or counted:
                raise ValueError("xt_lds runs on the synchronous body only")
            xw = dict(
                seq=m0, depth=g.head_size, tid=self.tid, threads=g.threads,
                elem=self.elem,
            )  # fmt: skip
            write_xt(b, self.v_q, self.v_qt, **xw)
            write_xt(b, self.v_do, self.v_dot, **xw)

        # G1 / G3 B operands (dO^T, Q^T) through the transpose reader; a
        # fenced sched reads them (and an LDS K^T) one stage ahead.
        coords = g.b13_coords
        if self.v_qt is not None:
            # the XT images hold dO / Q with K (the q rows) contiguous
            b_dot = LdsOperand(self.v_dot, "k_inner", coords=coords)
            b_qt = LdsOperand(self.v_qt, "k_inner", coords=coords)
        else:
            b_dot = LdsOperand(v_do, "k_outer", coords=coords, reader=g.reader)
            b_qt = LdsOperand(v_q, "k_outer", coords=coords, reader=g.reader)
        ahead = {}

        def read_ahead(name: str) -> None:
            if name == "g1":
                ahead["g1"] = g1.load_all_b(b, b_dot, lane, self.o1[1])
            elif name == "g3":
                ahead["g3"] = g1.load_all_b(b, b_qt, lane, self.o1[1])
            elif self.kt_regs is None:  # G4: K^T from LDS (resident)
                ahead["g4"] = g.g4.load_all_b(b, self._kt_operand(), lane, self.o4[1])

        # Without G4 (split dQ) G1 and G3 keep their relative order.
        g4_on = self.sink.emits_g4
        order = [u for u in stage_units(g.stage_order) if g4_on or u != "g4"]

        # Stage A: G0 S = Q K^T (+ G2 dP = dO V^T in stage B). Under asm_g13
        # with s_in_agpr both are AGPR-form asm clusters (S / dP in AGPRs, the
        # first K step reading srcC 0); otherwise intrinsics from zero.
        self.sched.emit_stage_begin(b, "A")
        if loads_in_a:
            # the ring's prefetch loads (the next tile's Q / dO and stats)
            # ahead of this step's dQ atomics, spread over G0's MFMAs
            if ring is not None:
                ring_loads = self._ring_loads(b, ring.nxt, nxt)
            else:
                prefetch.issue_loads(b)
        k_b = self.k_regs if self.k_regs is not None else LdsOperand(self.v_k)
        if self._g02_asm:
            s_acc = g0.mma(
                b, None, LdsOperand(v_q), k_b, lane=lane, origin=self.o0,
                route="agpr_asm", asm_agpr_base=self._g02_agpr_base(0),
                **self._asm_knobs,
            )  # fmt: skip
        else:
            s_acc = g0.mma(
                b, g0.zero_acc(b), LdsOperand(v_q), k_b, lane=lane, origin=self.o0
            )
        if fenced:
            if stats is None and ring is not None:
                stats = self._ring_stats(b, cur)  # the merged ring's slot
            elif stats is None:
                # the workspace, or the prefetch ring's stats slot
                stats = self._load_stats(
                    b, h, q0, None if packed is None else packed.rows, stats_src
                )
            read_ahead(order[0])
            if pending is not None:
                self.flush_dq(b, pending)
        self.sched.emit_stage_end(b, "A")
        self.sched.emit_stage_begin(b, "B")
        v_b = self.v_regs if self.v_regs is not None else LdsOperand(self.v_v)
        if self._g02_asm:
            dp_acc = g0.mma(
                b, None, LdsOperand(v_do), v_b, lane=lane, origin=self.o0,
                route="agpr_asm", asm_agpr_base=self._g02_agpr_base(1),
                **self._asm_knobs,
            )  # fmt: skip
        else:
            dp_acc = g0.mma(
                b, g0.zero_acc(b), LdsOperand(v_do), v_b, lane=lane, origin=self.o0
            )

        # P and dS per C slot (selects, never multiplies).
        if ring is not None and stats is None:
            stats = self._ring_stats(b, cur)
        p_f32, ds_f32 = self._softmax_grad(
            b, s_acc, dp_acc, h, q0, k0, edge=edge,
            rows=None if packed is None else packed.rows, stats=stats,
            stats_src=stats_src,
        )  # fmt: skip

        # Hand-off of P^T / dS^T to G1 / G3; dS to LDS for G4 (a sink that
        # takes no dQ, the split-dQ main kernel, has no G4 and no dS tile).
        if g.pt_route == "relabel":
            cast = g.bf16_cast
            a_p, _ = relabel_acc_to_a(b, g0, g1, p_f32, lane, bf16_cast=cast)
            a_ds, _ = relabel_acc_to_a(b, g0, g1, ds_f32, lane, bf16_cast=cast)
            if g4_on:
                self._write_c_tiles(b, v_ds, ds_f32)
        else:
            self._write_c_tiles(b, self.v_pt, p_f32)
            self._write_c_tiles(b, self.v_ds, ds_f32)
            barrier(b)  # also publishes the Q^T / dO^T images
            xt_unpublished = False
            a_p = self._k_outer_operand(self.v_pt)
            a_ds = self._k_outer_operand(self.v_ds)

        # G1 dV += P^T dO, G3 dK += dS^T Q and G4 dQ_part = dS K (then the dQ
        # sink), issued in ``stage_order``. Each accumulator's MMA chain is
        # the same in every order. Stage B holds the first of the three (after
        # G2 and the hand-off), C the second, D the third.
        out = {}
        record = [] if pending is not None else None
        # asm_load_hoist: the LDS operands of the G1 / G3 GEMM issued second
        # are read once the first GEMM's first-step operands are read, before
        # its asm cluster (an asm block is a scheduling barrier for memory
        # operations); the second GEMM then issues from those registers (same
        # MMAs on the same values). Every tile it reads is published before
        # the first G1 / G3 unit (Q / dO before G0 / G2; the Q^T / dO^T images
        # and the LDS P^T / dS^T by the barrier ahead of the first unit).
        operands = {"g1": [a_p, b_dot], "g3": [a_ds, b_qt]}
        pre_issue = {}
        if g.asm_load_hoist:
            lead_unit, hoisted_unit = (u for u in order if u != "g4")

            def hoist(name: str = hoisted_unit) -> None:
                a_op, b_op = operands[name]
                operands[name] = list(
                    g1.load_operands(
                        b, a_op, ahead.get(name, b_op), lane=lane, origin=self.o1
                    )
                )

            pre_issue[lead_unit] = hoist

        def g1_dv() -> None:
            a_op, b_op = operands["g1"]
            bop = ahead.get("g1", b_op)
            if g.acc_in_lds:
                self._update_lds_acc(b, self.v_dv_acc, a_op, bop)
                out["dv"] = None
            else:
                out["dv"] = g1.mma(
                    b, [list(r) for r in acc.dv], a_op, bop, lane=lane,
                    origin=self.o1, route=self._g13_route, **self._g13_asm,
                    pre_issue=pre_issue.get("g1"),
                )  # fmt: skip

        def g3_dk() -> None:
            a_op, b_op = operands["g3"]
            bop = ahead.get("g3", b_op)
            if g.acc_in_lds:
                self._update_lds_acc(b, self.v_dk_acc, a_op, bop)
                out["dk"] = None
            else:
                out["dk"] = g1.mma(
                    b, [list(r) for r in acc.dk], a_op, bop, lane=lane,
                    origin=self.o1, route=self._g13_route, **self._g13_asm,
                    pre_issue=pre_issue.get("g3"),
                )  # fmt: skip

        units = {
            "g1": g1_dv,
            "g3": g3_dk,
            "g4": lambda: self._g4_dq(
                b,
                h,
                q0,
                packed,
                edge=edge,
                v_ds=v_ds,
                kt_b=ahead.get("g4"),
                record=record,
                first=first,
            ),
        }
        ds_reader = self._ds_reader
        first = None
        for pos, (name, stage) in enumerate(zip(order, "BCD")):
            if name == ds_reader and g.pt_route == "relabel":
                if ring is not None:
                    if ring_loads is None:  # ring_issue = "late"
                        ring_loads = self._ring_loads(b, ring.nxt, nxt)
                    self._ring_stores(b, ring_loads)  # the next step's slot
                barrier(b)  # the dS tile above is visible to every wave
                xt_unpublished = False  # (and the Q^T / dO^T images)
            elif xt_unpublished:
                barrier(b)  # the Q^T / dO^T images are visible
                xt_unpublished = False
            if name == "g4" and counted and g.counted_issue == "late":
                # the next step's loads ahead of G4 and its atomics
                nxt_staged = self._issue_next(b, next_loads)
            if pos:
                self.sched.emit_stage_begin(b, stage)
            if name == "g3" and g.g4_prefetch_before_g3:
                first = self._g4_first_slice(b)
            units[name]()
            if fenced and pos + 1 < len(order):
                read_ahead(order[pos + 1])
            self.sched.emit_stage_end(b, stage)
        if prefetch is not None:
            prefetch.end(b)  # the ring's wait and barrier (publishes the next slot)
        elif ring is None:
            self._loop_barrier(b)  # q / do / ds / pt reads done before the next step
        if counted:
            self.waits.end_step()
        carry = nxt_staged if counted else None
        if record is not None:
            rows = None if packed is None else packed.rows
            pend = PendingDq(
                values=tuple(record),
                q0=q0 if rows is None else rows.row0,
                h=h if rows is None else rows.head0,
                n_heads=None if rows is None else rows.n_heads,
                rows_per_head=None if rows is None else rows.rows_per_head,
            )
            # (the counted step carries both: (next staged, pending dQ))
            carry = pend if carry is None else (carry, pend)
        if g.acc_in_lds:
            return acc if carry is None else (acc, carry)
        dk, dv = out["dk"], out["dv"]
        if g.agpr_route == "agpr_fence":
            dk = [[_agpr_fence(b, x) for x in row] for row in dk]
            dv = [[_agpr_fence(b, x) for x in row] for row in dv]
        res = DkDvAcc(dk=tuple(tuple(r) for r in dk), dv=tuple(tuple(r) for r in dv))
        return res if carry is None else (res, carry)

    def _stage_qdo(self, b, q_slab, do_slab, h, q0, packed, *, edge: bool = True):
        """Synchronous staging of the step's Q and dO (no ring), then the
        barrier that publishes them; returns the step-top stats (or None)."""
        g = self.geom
        stats = None
        if g.stats_load == "step_top":
            # lse2 / Dsum issued ahead of the staging and G0 / G2 (same loads).
            stats = self._load_stats(b, h, q0, None if packed is None else packed.rows)

        # Stage Q and dO (clamped rows) for this step.
        if packed is None:
            loads = (
                slab_row_loader(b, q_slab, q0),
                slab_row_loader(b, do_slab, q0),
            )
        else:
            # interior steps (``edge`` False): every row is < len_q, no clamp
            # (unless interior_clamp keeps it)
            clamp = edge or g.interior_clamp
            loads = (
                packed_row_loader(b, packed.q_addr, packed.rows, clamp=clamp),
                packed_row_loader(b, packed.do_addr, packed.rows, clamp=clamp),
            )
        copy_row_sets_to_lds(
            b, ((loads[0], self.v_q), (loads[1], self.v_do)), order=g.qdo_stage,
            rows=g.k_m0, cols=g.head_size, tid=self.tid, threads=g.threads,
            wave=self.wave, wave_size=g.wave_size,
        )  # fmt: skip
        self._loop_barrier(b)
        return stats

    def zero_pending_dq(
        self, b: IRBuilder, h: Value, n_heads: Value | None = None, rows_per_head=None
    ) -> PendingDq:
        """The deferred dQ partial before the first q step: every value
        ``+0.0`` at q row 0 of head ``h`` (a packed step: ``n_heads`` /
        ``rows_per_head`` of the packing). Its atomics add ``+0.0`` to valid
        (clamped) workspace cells, which leaves every sum bit for bit as it
        is: an fp32 sum that starts at ``+0.0`` is never ``-0.0``."""
        zero = b.const_f32(0.0)
        n = self.sched.dq_atomics_per_lane
        packed = n_heads is not None
        return PendingDq(
            values=(zero,) * n,
            q0=b.const_i32(0),
            h=h,
            n_heads=n_heads if packed else None,
            rows_per_head=rows_per_head if packed else None,
        )

    def flush_dq(self, b: IRBuilder, pending: PendingDq) -> None:
        """Issue the dQ atomics of a deferred partial through the sink.

        The slots are the G4 accumulator slots in the order the step recorded
        them (fragment by fragment, slot inner), with the rows clamped (a
        clamp is an identity on the rows of an interior step). Branch-free.
        """
        g4 = self.geom.g4
        if len(pending.values) != self.sched.dq_atomics_per_lane:
            raise ValueError("the pending dQ partial does not match the G4 tile")
        vals = iter(pending.values)
        slots = []
        for fm in range(g4.frags_m):
            for fn in range(g4.frags_n):
                for r, c in g4.acc_coords(b, self.lane, self.o4, fm, fn):
                    slots.append(DqSlot(row=r, col=c, value=next(vals)))
        if pending.n_heads is None:
            rows = pending.q0
        else:
            rows = PackedRows(
                head0=pending.h, n_heads=pending.n_heads,
                rows_per_head=pending.rows_per_head, row0=pending.q0,
            )  # fmt: skip
        self.sink.begin_head(b, pending.h)
        self.sink.consume(b, slots, rows, pending.h, clamp=True)
        if self.waits is not None:  # (the counted waits count them in A)
            self.waits.atomics(len(slots))

    @property
    def _ds_reader(self) -> str:
        """The first unit of the step that reads the dS tile written by the
        hand-off: G4, or G3 under ``slice_db_g3`` (it reads G4's first dS slice
        ahead of its own MMAs). Under the relabel route the barrier that
        publishes the tile sits right before that unit."""
        return "g3" if self.geom.g4_prefetch_before_g3 else "g4"

    def _g4_operands(self):
        """``(dS A operand, K^T B operand)`` of G4: dS from its LDS tile, K^T
        from registers (``kt_source = "reg"``) or from LDS."""
        kt_b = self.kt_regs if self.kt_regs is not None else self._kt_operand()
        return LdsOperand(self.v_ds), kt_b

    def _g4_first_slice(self, b):
        """G4's first dS key slice (``slice_db_g3``: read before the G3 MMAs)."""
        a_ds, kt_b = self._g4_operands()
        return self.geom.g4.load_k_slice(
            b, a_ds, kt_b, 0, self.geom.k_k4, lane=self.lane, origin=self.o4
        )

    def _g4_dq(
        self,
        b,
        h,
        q0,
        packed: PackedStep | None,
        *,
        edge: bool = True,
        v_ds=None,
        kt_b=None,
        record: list | None = None,
        first=None,
    ) -> None:
        """G4 ``dQ_part = dS K`` from the dS tile in LDS, then the dQ sink.

        The dS tile must be visible to every wave (the caller's barrier).
        ``edge=False`` (an interior step: every row is < len_q) drops the
        sink's row clamps, unless the geometry's ``interior_clamp`` keeps them.
        ``v_ds``: the step's dS slot (``None``: the single dS buffer).
        ``kt_b``: K^T fragments already in registers (read ahead), else the
        resident registers or the LDS copy. ``record`` (the deferred sink):
        append the partial's fp32 values (fragment by fragment, slot inner)
        instead of handing them to the sink.
        ``first``: G4's first dS key slice, already read (``slice_db_g3``).
        """
        g = self.geom
        g4, lane = g.g4, self.lane
        clamp = edge or g.interior_clamp
        if kt_b is None:
            kt_b = self.kt_regs if self.kt_regs is not None else self._kt_operand()
        sink_rows = q0 if packed is None else packed.rows

        def slots_of(fm, fn, frag):
            coords = g4.acc_coords(b, lane, self.o4, fm, fn)
            return [
                DqSlot(row=r, col=c, value=b.vec_extract(frag, s))
                for s, (r, c) in enumerate(coords)
            ]

        def hand_off(frags) -> None:
            """``[(fm, fn, frag), ...]`` to the sink in one call, or recorded."""
            if record is not None:
                n = g4.op.c_frag_len
                record.extend(
                    b.vec_extract(f, s) for _, _, f in frags for s in range(n)
                )
                return
            slots = []
            for fm, fn, frag in frags:
                slots += slots_of(fm, fn, frag)
            self.sink.consume(b, slots, sink_rows, h, clamp=clamp)
            if self.waits is not None:
                self.waits.atomics(len(slots))

        a_ds = LdsOperand(self.v_ds if v_ds is None else v_ds)
        if g.acc_in_lds or g.dq_issue != "block":
            # One fragment at a time, each handed to the sink at once, so a
            # single dQ fragment is live and its atomics can overlap the next
            # fragment's MMAs (LDS-accumulator geometries always; otherwise
            # dq_issue = "fragment" / "fragment_regs"). Each fragment's K order
            # is the block order, so every partial is bit-identical.
            # "fragment" reads the dS row of each fragment from LDS again;
            # "fragment_regs" reads all of dS into registers once.
            if g.dq_issue == "fragment_regs" and not g.acc_in_lds:
                a_ds = g4.load_all_a(b, a_ds, lane, self.o4[0])
            for fm in range(g4.frags_m):
                for fn in range(g4.frags_n):
                    frag = g4.mma_frag(
                        b, b.zero_vec_f32(g4.op.c_frag_len), a_ds, kt_b, fm, fn,
                        lane=lane, origin=self.o4,
                    )  # fmt: skip
                    hand_off([(fm, fn, frag)])
            return
        if g.g4_slicing == DEFAULT_G4_SLICING:
            dq = g4.mma(b, g4.zero_acc(b), a_ds, kt_b, lane=lane, origin=self.o4)
        else:
            # dS (and LDS K^T) read in kK4 key slices; same K order per fragment
            dq = g4.mma_k_sliced(
                b, g4.zero_acc(b), a_ds, kt_b, k_slice=g.k_k4,
                buffers=g.g4_slice_buffers, lane=lane, origin=self.o4, first=first,
            )  # fmt: skip
        hand_off(
            [
                (fm, fn, dq[fm][fn])
                for fm in range(g4.frags_m)
                for fn in range(g4.frags_n)
            ]
        )

    def _stats_rows(self, b):
        """``(row_of, keys)``: the M row of each ``(fm, slot)`` C row of the
        lane (it depends on ``(fm, slot)`` only) and the sorted keys."""
        g0 = self.geom.g0
        row_of = {}
        for fm in range(g0.frags_m):
            for s, (r, _c) in enumerate(g0.acc_coords(b, self.lane, self.o0, fm, 0)):
                row_of[(fm, s)] = r
        return row_of, sorted(row_of)

    # ----- barrier merge: 2-slot Q / dO / stats ring and 2-slot dS ----------
    def _ring_views(self, b, slot) -> dict[str, LdsView]:
        """Views of ring slot ``slot`` (the int 0, or a uniform i32 0 / 1).

        Slot 1 of a buffer is its slot-0 view shifted at run time by the
        planner's offset difference, so one step body serves both slots.

        Hazard proof of ``barrier_merge = "on"`` (step ``i`` uses slot ``s = i
        % 2`` of q, do, lse2, dsum and ds; ``X_i`` is its single barrier,
        before G4; a loop starts with slot 0 staged before a prologue barrier
        ``P``, every wave executes every barrier):

        * q / do / lse2 / dsum slot ``s``: written in step ``i - 1`` before
          ``X_{i-1}`` (or before ``P``), read in step ``i`` by G0 / G2, the
          stats reads and G1 / G3, all before ``X_i`` (G4 is last in
          ``stage_order``). The next write of slot ``s`` is in step ``i + 1``,
          after ``X_i``. Read-after-write and write-after-read are both
          ordered by one barrier.
        * ds slot ``s``: written in step ``i`` before ``X_i``, read by G4 of
          step ``i`` after ``X_i``. The next write is in step ``i + 2``, after
          ``X_{i+1}``, which no wave passes before finishing G4 of step ``i``.
        * A loop's prologue writes slot 0 after the previous loop's last
          barrier, past every read of the q / do / stats slots there; its
          first dS write follows ``P``, past the previous loop's last G4.
          :meth:`finish` adds a barrier after the last loop (the epilogue
          reuses the loop phase).
        * The global loads feeding the slot stores are waited by data
          dependency in their own wave, as in the single-slot staging, so
          ``loop_sync`` keeps its meaning for ``X_i`` and ``P``.
        """
        plan = self.lds.plan
        out = {}
        for name, view in (
            ("q", self.v_q),
            ("do", self.v_do),
            ("ds", self.v_ds),
            ("lse2", self.v_lse2),
            ("dsum", self.v_dsum),
        ):
            if isinstance(slot, int):
                if slot != 0:
                    raise ValueError("a static ring slot must be 0")
                out[name] = view
                continue
            delta = plan.offset(name, 1) - plan.offset(name, 0)
            if delta % 16:
                raise ValueError(f"LDS ring buffer {name} slots are not 16-byte apart")
            shift = b.mul(slot, b.const_i32(delta // 2))
            out[name] = replace(view, shift=shift)
        return out

    def _ring_loads(self, b, step: QStepInputs, views) -> list:
        """Issue the global loads of ``step``'s Q / dO tiles and lse2 / Dsum
        rows (the same loads as the single-slot staging and
        :meth:`_load_stats`); :meth:`_ring_stores` writes them to ``views``.

        Thread ``t < kM0`` moves stats row ``t`` (the lanes past ``kM0`` load
        row ``kM0 - 1`` and store nothing).
        """
        g = self.geom
        m0 = g.k_m0
        if step.packed is None:
            loads = (
                slab_row_loader(b, step.q_slab, step.q0),
                slab_row_loader(b, step.do_slab, step.q0),
            )
        else:
            loads = (
                packed_row_loader(b, step.packed.q_addr, step.packed.rows),
                packed_row_loader(b, step.packed.do_addr, step.packed.rows),
            )
        pending = _row_loads(
            b, ((loads[0], views["q"]), (loads[1], views["do"])), rows=m0,
            cols=g.head_size, tid=self.tid, threads=g.threads,
        )  # fmt: skip
        row = b.smin(self.tid, b.const_i32(m0 - 1))
        if step.packed is None:
            ((lse2, dsum),) = self.stats.rows(b, step.h, step.q0, [row])
        else:
            ((lse2, dsum),) = self.stats.packed_rows(b, step.packed.rows, [row])
        guard = b.cmp_lt(self.tid, b.const_i32(m0))
        pending.append((views["lse2"], row, 0, lse2, 1, guard))
        pending.append((views["dsum"], row, 0, dsum, 1, guard))
        return pending

    @staticmethod
    def _ring_stores(b, pending) -> None:
        """The LDS stores of :meth:`_ring_loads` (predicated where guarded)."""
        for view, r, c, val, n, guard in pending:
            if guard is None:
                view.store(b, r, c, val, n)
            else:
                with b.scf_if(guard):
                    view.store(b, r, c, val, n)

    def _ring_stats(self, b, views):
        """:meth:`_load_stats` from the step's ring slot (the staged rows hold
        the workspace values, neutral rows included, bit for bit)."""
        row_of, keys = self._stats_rows(b)
        out = {}
        for key in keys:
            r = row_of[key]
            out[key] = tuple(
                b.vec_extract(views[name].load(b, r, 0, dtype=F32, n=1), 0)
                for name in ("lse2", "dsum")
            )
        return row_of, keys, out

    def ring_prologue(self, b, step: QStepInputs) -> None:
        """Stage a q loop's first step into ring slot 0 (the caller skips an
        empty loop), then :meth:`ring_publish`."""
        views = self._ring_views(b, 0)
        self._ring_stores(b, self._ring_loads(b, step, views))

    def ring_publish(self, b) -> None:
        """The prologue barrier ``P`` of a q loop under the barrier merge."""
        self._loop_barrier(b)

    def _load_stats(self, b, h, q0, rows, source=None):
        """``(row_of, keys, {(fm, slot): (lse2, Dsum)})`` of the lane's C rows.

        Row coordinates depend on ``(fm, slot)`` only, so the stats are loaded
        once per step; ``rows`` as in :meth:`_softmax_grad`. ``source``: an
        object with the :class:`StatsView` interface (the DMA ring's LDS stats
        slot); ``None`` reads the workspace.
        """
        src = self.stats if source is None else source
        row_of, keys = self._stats_rows(b)
        if rows is None and self.geom.stats_vec != 1 and source is None:
            # the workspace rows as <n x f32> runs (stats_vec)
            stats = src.batched_rows(
                b, h, q0, [row_of[k] for k in keys], width=self.geom.stats_vec
            )
        elif rows is None:
            stats = src.rows(b, h, q0, [row_of[k] for k in keys])
        else:
            stats = src.packed_rows(b, rows, [row_of[k] for k in keys])
        return row_of, keys, dict(zip(keys, stats))

    def _softmax_grad(
        self, b, s_acc, dp_acc, h, q0, k0, *, edge: bool, rows=None, stats=None,
        stats_src=None,
    ):  # fmt: skip
        """``(P, ds_mult * dS)`` as fp32 accumulator-shaped fragment lists.

        ``rows`` (a :class:`PackedRows`, head packing) gives each M row its own
        head and sequence row; a dead packed row fails the keep. ``stats``: the
        :meth:`_load_stats` result when the step loaded them earlier
        (``stats_load = "step_top"``, or the register ring); ``None`` loads
        them here, from ``stats_src`` when given (the DMA ring's stats slot).
        """
        g0 = self.geom.g0
        lane = self.lane
        zero = b.const_f32(0.0)
        inf = b.const_f32(_INF)
        if stats is None:
            row_of, keys, stats = self._load_stats(b, h, q0, rows, stats_src)
        else:
            row_of, keys, stats = stats
        live = {}
        if self.sentinel:
            for key, (lse2, _dsum) in stats.items():
                live[key] = b.fcmp("olt", lse2, inf)
        # Edge steps: per-row key limits with the dead-row term folded in, so
        # one compare per cell gives keep (P) and keep && row_live (dS); a dead
        # row's P is +0 either way (exp2(x - inf) = +0).
        limits = {}
        if edge:
            for key in keys:
                if rows is None:
                    q, ok = b.add(q0, row_of[key]), None
                else:
                    _h, q, ok = rows.locate(b, row_of[key])
                if self.sentinel:
                    ok = live[key] if ok is None else b.land(ok, live[key])
                limits[key] = self.mask.row_limits(b, q, ok)
        neg = stats_rows_neg(b, stats) if self.geom.stats_fma else {}
        p_out, ds_out = [], []
        for fm in range(g0.frags_m):
            p_row, ds_row = [], []
            for fn in range(g0.frags_n):
                coords = g0.acc_coords(b, lane, self.o0, fm, fn)
                ps, dss = [], []
                for s, (r, c) in enumerate(coords):
                    lse2, dsum = stats[(fm, s)]
                    sv = b.vec_extract(s_acc[fm][fn], s)
                    dpv = b.vec_extract(dp_acc[fm][fn], s)
                    p = softmax_p(
                        b, sv, self.scale_log2, lse2, neg_lse2=neg.get((fm, s)),
                        bwd_exp2=self.geom.bwd_exp2,
                    )  # fmt: skip
                    if edge:
                        cond = self.mask.keep_in(b, limits[(fm, s)], b.add(k0, c))
                        p = b.select(cond, p, zero)
                    else:
                        cond = live.get((fm, s))
                    ds = b.fmul(p, b.fsub(dpv, dsum))
                    if cond is not None:
                        ds = b.select(cond, ds, zero)
                    ps.append(p)
                    if self.ds_mult is not None:  # None: 1.0, not applied
                        ds = b.fmul(self.ds_mult, ds)
                    dss.append(_rounding_point(b, ds))
                p_row.append(b.vec_pack(ps, F32))
                ds_row.append(b.vec_pack(dss, F32))
            p_out.append(p_row)
            ds_out.append(ds_row)
        return p_out, ds_out

    def _write_c_tiles(self, b, view: LdsView, frags) -> None:
        """Round fp32 C-layout fragments to the I/O dtype and store ``[q][key]``.

        On a replicated G0 / G2 grid only the waves below the grid's wave
        count store (the others hold copies of the same fragments).
        """
        if self.g0_replicated:
            first = b.cmp_lt(self.wave, b.const_i32(self.geom.g0.waves))
            with b.scf_if(first):
                self._store_c_tiles(b, view, frags)
            return
        self._store_c_tiles(b, view, frags)

    def _store_c_tiles(self, b, view: LdsView, frags) -> None:
        g0 = self.geom.g0
        for fm in range(g0.frags_m):
            for fn in range(g0.frags_n):
                half = cast_c_frag(b, frags[fm][fn], self.elem, self.geom.bf16_cast)
                for s, (r, c) in enumerate(
                    g0.acc_coords(b, self.lane, self.o0, fm, fn)
                ):
                    view.store(b, r, c, b.vec_extract(half, s), 1)

    @property
    def _g13_route(self) -> str:
        """MMA route of G1 / G3: AGPR-form inline asm under ``asm_g13``."""
        return "agpr_asm" if self.geom.agpr_route == "asm_g13" else "intrinsic"

    @property
    def _g13_asm(self) -> dict:
        """Cluster knobs of the G1 / G3 asm route (none on the intrinsic route)."""
        if self._g13_route != "agpr_asm":
            return {}
        return self._asm_knobs

    @property
    def _asm_knobs(self) -> dict:
        """The asm cluster knobs (shared by every asm-route GEMM of a step)."""
        return {
            "asm_tail_nop": self.geom.asm_tail_nop,
            "asm_cluster": self.geom.asm_cluster,
        }

    def _g02_agpr_base(self, which: int) -> int:
        """First physical AGPR of S (``which = 0``) or dP (``1``) under G0 / G2
        as asm: the top ``2 * R`` AGPRs of the reservation ``(n, n)``, S below
        dP (``R`` = the lane's S registers). The validator keeps ``n`` at or
        above the AGPR estimate (dK / dV plus S / dP), so dK / dV fit below."""
        g = self.geom
        g0 = g.g0
        r = g0.frags_m * g0.frags_n * g0.op.c_frag_len
        n = int(g.agpr_alloc[0])
        return n - (2 - which) * r

    @property
    def _g02_asm(self) -> bool:
        """G0 / G2 as AGPR-form asm clusters: ``asm_g13`` with ``s_in_agpr``
        (S and dP in AGPRs; the softmax gradient reads them with
        ``v_accvgpr_read``)."""
        g = self.geom
        return g.agpr_route == "asm_g13" and g.s_in_agpr and not g.acc_in_lds

    def finish(self, b: IRBuilder, acc: DkDvAcc) -> DkDvAcc:
        """Hook after the q loop (``dk_mult`` is applied by the epilogue).

        Under the ``agpr_fence`` route the accumulators come back from AGPRs
        once here, so the epilogue reads arch VGPRs. Under the barrier merge
        the last step's G4 reads dS after the last barrier, so a full barrier
        precedes the epilogue (which reuses the loop phase).
        """
        if self.geom.barrier_merge == "on":
            b.sync()
        if self.geom.agpr_route == "agpr_fence" and not self.geom.acc_in_lds:
            return DkDvAcc(
                dk=tuple(tuple(_vgpr_fence(b, x) for x in r) for r in acc.dk),
                dv=tuple(tuple(_vgpr_fence(b, x) for x in r) for r in acc.dv),
            )
        return acc

    # ----- epilogue helpers (used by the family shells) ------------------
    def dkv_row_slots(
        self, b: IRBuilder, which: str, acc: DkDvAcc
    ) -> list[tuple[Value, list[tuple[Value, Value]]]]:
        """dK (``"dk"``) or dV (``"dv"``) of the key tile as fp32 values by row.

        Returns ``[(row, [(col, value), ...]), ...]``: one entry per key row
        this lane holds (``row`` inside the tile), with every ``(column,
        fp32 sum)`` of that row (no multiplier, no rounding). The row of an
        accumulator slot depends on its fragment row and slot only, so the
        fragments along D share it. With LDS accumulators each lane reads back
        exactly the cells of the image it updated in the q loop, so no
        barrier is needed before this call. Used by the atomic epilogue.
        """
        g1 = self.geom.g1
        if which not in ("dk", "dv"):
            raise ValueError(f"which={which!r} must be 'dk' or 'dv'")
        out = []
        for fm in range(g1.frags_m):
            frags, coords = [], []
            for fn in range(g1.frags_n):
                if self.geom.acc_in_lds:
                    view = self.v_dk_acc if which == "dk" else self.v_dv_acc
                    frags.append(self._load_lds_acc(b, view, fm, fn))
                else:
                    frags.append((acc.dk if which == "dk" else acc.dv)[fm][fn])
                coords.append(self._acc_lds_coords(b, fm, fn))
            for s in range(g1.op.c_frag_len):
                row = coords[0][s][0]
                items = [
                    (coords[fn][s][1], b.vec_extract(frags[fn], s))
                    for fn in range(g1.frags_n)
                ]
                out.append((row, items))
        return out

    def _rounded_frag(self, b: IRBuilder, frag: Value, mult: Value | None) -> Value:
        """An accumulator fragment times ``mult``, rounded once to the I/O dtype."""
        if mult is not None:
            n = self.geom.g1.op.c_frag_len
            frag = b.vec_pack(
                [b.fmul(mult, b.vec_extract(frag, i)) for i in range(n)], F32
            )
        return b.vec_cast_f32_to(frag, self.elem)

    def dkv_rows(
        self,
        b: IRBuilder,
        which: str,
        acc: DkDvAcc,
        mult: Value | None,
        *,
        vec: int = 8,
        slot: int = 0,
        sync: bool = True,
    ):
        """Make dK (``"dk"``) or dV (``"dv"``) readable as rows of ``vec`` values.

        Returns a function ``read(r, c) -> <vec x dtype>`` (``mult`` applied,
        rounded once to the I/O dtype) for row ``r`` and column ``c`` (a
        multiple of ``vec``) of the ``kN0 x D`` tile.
        Register accumulators are transposed through epilogue buffer ``slot``
        (``dkv_out``, I/O dtype), with a barrier before the reads unless
        ``sync`` is False (the caller then places it after staging every
        slot); the caller places a barrier after its reads before a buffer is
        written again.
        """
        g = self.geom
        if vec not in (8, 4, 2, 1):
            raise ValueError(f"vec={vec} must be 8, 4, 2 or 1")
        if g.acc_in_lds:
            view = self.v_dk_acc if which == "dk" else self.v_dv_acc
            step = min(vec, 4)

            def read_acc(r, c):
                parts = [
                    view.load(
                        b, r, c if k == 0 else b.add(c, b.const_i32(k)),
                        dtype=F32, n=step,
                    )
                    for k in range(0, vec, step)
                ]  # fmt: skip
                vals = [b.vec_extract(p, i) for p in parts for i in range(step)]
                if mult is not None:
                    vals = [b.fmul(mult, v) for v in vals]
                return b.vec_cast_f32_to(b.vec_pack(vals, F32), self.elem)

            return read_acc
        frags = acc.dk if which == "dk" else acc.dv
        out = self.lds.view("dkv_out", g.head_size, slot)
        g1 = g.g1
        for fm in range(g1.frags_m):
            for fn in range(g1.frags_n):
                half = self._rounded_frag(b, frags[fm][fn], mult)
                for s, (r, c) in enumerate(self._acc_lds_coords(b, fm, fn)):
                    out.store(b, r, c, b.vec_extract(half, s), 1)
        if sync:
            b.sync()

        def read_out(r, c):
            return out.load(b, r, c, dtype=self.elem, n=vec)

        return read_out

    def dkv_quad_rows(
        self, b: IRBuilder, which: str, acc: DkDvAcc, mult: Value | None
    ) -> list[tuple[Value, Value, list[Value]]]:
        """dK / dV (``mult`` applied, fp32) as ``(row, col0, [4 values])``:
        one entry per accumulator fragment, the lane's 4 consecutive columns
        ``col0 .. col0 + 3`` of one row after a 4 x 4 transpose inside each
        lane quad (``c_quad_transposable`` of the G1 atom: before it lane
        ``j`` of a quad holds rows ``r0 .. r0 + 3`` of column ``c0 + j``;
        after it, row ``r0 + j`` of columns ``c0 .. c0 + 3``). Pure data
        movement: the same fp32 values reach the same elements."""
        g1 = self.geom.g1
        if which not in ("dk", "dv"):
            raise ValueError(f"which={which!r} must be 'dk' or 'dv'")
        if self.geom.acc_in_lds:
            raise ValueError("dkv_quad_rows needs register accumulators")
        frags = acc.dk if which == "dk" else acc.dv
        j = b.mod(self.lane, b.const_i32(4))
        out = []
        for fm in range(g1.frags_m):
            for fn in range(g1.frags_n):
                frag = frags[fm][fn]
                vals = [b.vec_extract(frag, s) for s in range(4)]
                if mult is not None:
                    vals = [b.fmul(mult, v) for v in vals]
                (r0, c), *_ = self._acc_lds_coords(b, fm, fn)
                out.append(
                    (b.add(r0, j), b.sub(c, j), quad_transpose4(b, vals, self.lane))
                )
        return out

    def dkv_f32_image(self, which: str) -> LdsView:
        """The fp32 LDS image of dK (``"dk"``) or dV (``"dv"``) under
        ``acc_in_lds`` (``kN0 x D``); a reader of other lanes' cells needs a
        barrier after the q loop first."""
        if not self.geom.acc_in_lds:
            raise ValueError("dkv_f32_image needs LDS accumulators")
        if which not in ("dk", "dv"):
            raise ValueError(f"which={which!r} must be 'dk' or 'dv'")
        return self.v_dk_acc if which == "dk" else self.v_dv_acc

    def dkv_f32_part(
        self,
        b: IRBuilder,
        which: str,
        acc: DkDvAcc,
        mult: Value | None,
        *,
        half: int,
        slot: int = 0,
    ) -> LdsView:
        """Stage one column half of the fp32 dK / dV through ``dkv_out``.

        The register accumulators of columns ``[half * D/2, (half + 1) *
        D/2)``, times ``mult`` (fp32, not rounded), are written to epilogue
        buffer ``slot`` viewed as an fp32 ``kN0 x D/2`` image (the I/O-dtype
        buffer holds exactly half the columns in fp32; a row pad is kept).
        Returns that view; its columns are relative to the half. The caller
        places the barriers: after staging (before any read) and after its
        reads (before the buffer is written again). A fragment covers 16
        aligned columns and ``D/2`` is a multiple of 16, so each fragment lies
        in one half: with the ``(W, 1)`` G1 grid the half of fragment ``fn``
        is static, otherwise each fragment is staged under a branch on its
        column.
        """
        g = self.geom
        if g.acc_in_lds:
            raise ValueError("LDS accumulators are read in place (dkv_f32_image)")
        if which not in ("dk", "dv") or half not in (0, 1):
            raise ValueError(f"which={which!r} / half={half!r}")
        hw = g.head_size // 2
        buf = self.lds.plan.buffer("dkv_out", slot)
        if buf.offset % 16 or buf.nbytes % (4 * g.k_n0):
            raise ValueError("dkv_out cannot hold an fp32 half image")
        pitch = buf.nbytes // (4 * g.k_n0)  # fp32 elements per row
        if pitch < hw:
            raise ValueError("dkv_out rows are narrower than half a row in fp32")
        view = LdsView(self.lds.smem, buf.offset // 2, pitch, g.k_n0, 2)
        g1 = g.g1
        if hw % g1.op.n:
            raise ValueError("D/2 must be a multiple of the G1 atom N")
        frags = acc.dk if which == "dk" else acc.dv
        static = g.grid_g1[1] == 1
        shift = b.const_i32(hw) if half else None
        for fm in range(g1.frags_m):
            for fn in range(g1.frags_n):
                if static and (fn * g1.op.n) // hw != half:
                    continue
                frag = frags[fm][fn]
                vals = [b.vec_extract(frag, i) for i in range(g1.op.c_frag_len)]
                if mult is not None:
                    vals = [b.fmul(mult, v) for v in vals]
                coords = self._acc_lds_coords(b, fm, fn)

                def put(coords=coords, vals=vals) -> None:
                    for (r, c), v in zip(coords, vals):
                        view.store(b, r, c if shift is None else b.sub(c, shift), v, 1)

                if static:
                    put()
                    continue
                c0 = coords[0][1]
                h = b.const_i32(hw)
                inside = b.cmp_lt(c0, h) if half == 0 else b.cmp_ge(c0, h)
                with b.scf_if(inside):
                    put()
        return view

    def dkv_cells(
        self, b: IRBuilder, which: str, acc: DkDvAcc, mult: Value | None
    ) -> list[tuple[Value, list[tuple[Value, Value]]]]:
        """dK (``"dk"``) or dV (``"dv"``) in the I/O dtype, by key row.

        The values :meth:`dkv_rows` stages (``mult`` applied, rounded once,
        the same operations), straight from the register accumulators:
        ``[(row, [(col, value), ...]), ...]``, one entry per key row this lane
        holds, with every column of that row (the row of an accumulator slot
        depends on its fragment row and slot only). No LDS, no barrier.
        """
        g1 = self.geom.g1
        if which not in ("dk", "dv"):
            raise ValueError(f"which={which!r} must be 'dk' or 'dv'")
        if self.geom.acc_in_lds:
            raise ValueError("dkv_cells needs register accumulators")
        frags = acc.dk if which == "dk" else acc.dv
        out = []
        for fm in range(g1.frags_m):
            halves, coords = [], []
            for fn in range(g1.frags_n):
                halves.append(self._rounded_frag(b, frags[fm][fn], mult))
                coords.append(self._acc_lds_coords(b, fm, fn))
            for s in range(g1.op.c_frag_len):
                items = [
                    (coords[fn][s][1], b.vec_extract(halves[fn], s))
                    for fn in range(g1.frags_n)
                ]
                out.append((coords[0][s][0], items))
        return out
