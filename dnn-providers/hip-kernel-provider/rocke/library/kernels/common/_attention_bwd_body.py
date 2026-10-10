# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared inner body of the attention backward kernels (dQ, dK, dV).

Family glue for the backward kernels (Python-only, no C++ twin; packs pin
``ROCKE_BACKEND=python``). One :class:`BwdTileStep` emits the work of one q
step (``kM0`` query rows of one query head) against the CTA's resident key
tile (``kN0`` keys), on a multi-wave CTA:

* **G0** ``S = Q K^T`` and **G2** ``dP = dO V^T`` on the warp grid ``(1, W)``
  (waves split the keys);
* ``P = select(keep, exp2(S * scale_log2 - lse2), 0)`` and
  ``dS = select(keep && lse2 < +inf, P * (dP - Dsum), 0)``: selects, never
  multiplies, so a dead row (``lse2 = +inf``) or a masked cell contributes an
  exact 0 whatever its inputs;
* **G1** ``dV += P^T dO`` and **G3** ``dK += dS^T Q`` on ``(W, 1)``: the
  ``P^T`` / ``dS^T`` A operands are a checked register relabel of the G0 / G2
  accumulators (``pt_route = "relabel"``) or are read back from LDS
  (``"lds"``); ``dO^T`` and ``Q^T`` are K-outer reads of the staged tiles
  through the transpose source (element reads for ``lds_plain`` /
  ``wmma_lds``, ``ds_read_tr16_b64`` for ``tr_read``);
* **G4** ``dQ_part = dS K`` with dS written to LDS once per step and K^T from
  registers (``kt_source = "reg"``, read once per CTA) or LDS; the partial is
  handed to the dQ sink (fp32 workspace atomics, branch-free).

G1, G3 and G4 issue in the order of ``stage_order`` (default ``"g1g3g4"``;
for example ``"g4g1g3"`` issues G4 and its atomics right after dS is
published, so they overlap the G1 / G3 matrix ops). Each accumulator's MMA
chain is unchanged by the order, so dK, dV and every dQ partial are bit for
bit those of the default order.

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
operand map. An illegal geometry raises :class:`ValueError` before any IR is
emitted.

Hazard structure (checked by the IR tests): the q-loop trip count is
``readfirstlane``'d, the empty-band exit (trip count 0) is decided before any
accumulator is live, and no matrix instruction sits inside a divergent
region.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from rocke.core.ir import F32, IRBuilder, Value

from ._attention_bwd_addr import (
    PackedRows,
    RowSlab,
    StridedTensorAddr,
    WorkspaceAddr,
)
from ._attention_bwd_band import MaskPolicy
from ._attention_bwd_caps import bwd_arch_caps, check_transpose_read
from ._attention_bwd_frag import (
    LdsView,
    atom_facts,
    check_tr16_coords,
    ir_type,
    operand_coords,
)
from ._attention_bwd_gemm import (
    BlockGemm,
    LdsOperand,
    check_relabel_grids,
    kperm_b_operand_coords,
    relabel_acc_to_a,
)
from ._attention_bwd_lds import DKV_EPILOGUES, LdsPlan
from ._attention_bwd_sched import STAGE_TABLE_SCHEDS, StageTable
from ._attention_bwd_sinks import DqSink, DqSlot

__all__ = [
    "DEFAULT_STAGE_ORDER",
    "STAGE_ORDERS",
    "BwdAtomTable",
    "BwdLds",
    "BwdTileGeometry",
    "BwdTileStep",
    "COPY_ORDERS",
    "DkDvAcc",
    "PackedStep",
    "StatsView",
    "copy_packed_rows_to_lds",
    "copy_row_sets_to_lds",
    "copy_rows_to_lds",
    "packed_row_loader",
    "slab_row_loader",
    "stage_units",
]

_TRANSPOSE_READERS = {
    "lds_plain": "plain",
    "wmma_lds": "plain",
    "tr_read": "tr16",
}
_INF = float("inf")
# Issue order of the three GEMMs that consume P / dS inside one q step (G1
# dV, G3 dK, G4 dQ with its sink), as GEMM names in issue order. The default
# is first; every permutation is legal and changes no accumulator's math.
STAGE_ORDERS = ("g1g3g4", "g3g1g4", "g4g1g3", "g1g4g3", "g3g4g1", "g4g3g1")
DEFAULT_STAGE_ORDER = STAGE_ORDERS[0]


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
    stage_order: str = DEFAULT_STAGE_ORDER
    loop_sync: str = "full"
    dkv_epilogue: str = "lds_staged"
    dkv_store_vec: int = 8
    qdo_stage: str = "split"
    stats_load: str = "at_use"
    dq_issue: str = "block"
    interior_clamp: bool = False

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
            stage_order=getattr(spec, "stage_order", DEFAULT_STAGE_ORDER),
            loop_sync=getattr(spec, "loop_sync", "full"),
            dkv_epilogue=epilogue,
            dkv_store_vec=getattr(spec, "dkv_store_vec", 8),
            qdo_stage=getattr(spec, "qdo_stage", "split"),
            stats_load=getattr(spec, "stats_load", "at_use"),
            dq_issue=getattr(spec, "dq_issue", "block"),
            interior_clamp=bool(getattr(spec, "interior_clamp", False)),
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
        for g in (self.g0, self.g1, self.g4):
            facts = atom_facts(g.op)
            if not (facts.a_contig and facts.b_contig and facts.affine):
                raise ValueError(
                    f"{g.op.op_id}: lane maps are not K-contiguous and affine "
                    f"({facts}); this geometry cannot be built on {self.arch}"
                )
        _ = self.relabel_plan  # proves the relabel (raises when illegal)
        if self.reader == "tr16":
            check_transpose_read(self.arch, 64)
            check_tr16_coords(self.b13_coords)
            check_tr16_coords(operand_coords(self.g4.op, "b"))
            if self.pt_route == "lds":
                check_tr16_coords(operand_coords(self.g1.op, "a"))
        if self.dq_mode != "atomic":
            raise ValueError("this body emits G4 with an atomic dQ sink only")
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


# ---------------------------------------------------------------------------
# LDS realisation: one flat allocation, views at the planner's offsets
# ---------------------------------------------------------------------------


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
    (``[rows][cols]`` 16-bit elements, or fp32 for the LDS accumulators).
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
        if buf.layout == "rowmajor:f32":
            pitch, scale = cols, 2
            rows = buf.nbytes // (4 * cols)
        elif buf.layout.startswith("rowmajor:"):
            pitch = cols + _pad_elems(buf.layout, 2)
            scale = 1
            rows = buf.nbytes // (2 * pitch)
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
            for it in range(-(-total // threads)):
                chunk = tid if it == 0 else b.add(tid, b.const_i32(it * threads))
                tail = (it + 1) * threads > total
                src = b.smin(chunk, b.const_i32(total - 1)) if tail else chunk
                r, c = _chunk_coords(b, src, per_row, vec)
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
            r, c = _chunk_coords(b, chunk, per_row, vec)
            view.store(b, r, c, load(r, c), vec)


def _chunk_coords(b, chunk, per_row, vec):
    """``(row, column)`` of a ``vec``-element chunk of a ``per_row``-chunk row."""
    if per_row == 1:
        return chunk, b.const_i32(0)
    r = b.div(chunk, b.const_i32(per_row))
    c = b.mul(b.mod(chunk, b.const_i32(per_row)), b.const_i32(vec))
    return r, c


def _copy_rows(b, load, view, *, rows, cols, tid, threads, vec) -> None:
    per_row = cols // vec
    total = rows * per_row
    for it in range(-(-total // threads)):
        chunk = tid if it == 0 else b.add(tid, b.const_i32(it * threads))

        def body(chunk=chunk) -> None:
            r = b.div(chunk, b.const_i32(per_row)) if per_row > 1 else chunk
            c = (
                b.mul(b.mod(chunk, b.const_i32(per_row)), b.const_i32(vec))
                if per_row > 1
                else b.const_i32(0)
            )
            view.store(b, r, c, load(r, c), vec)

        if (it + 1) * threads <= total:
            body()
        else:
            with b.scf_if(b.cmp_lt(chunk, b.const_i32(total))):
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

    def __init__(self, lse2: WorkspaceAddr, dsum: WorkspaceAddr) -> None:
        if lse2.width != 1 or dsum.width != 1:
            raise ValueError("lse2 / Dsum workspaces are one fp32 per row")
        self.lse2 = lse2
        self.dsum = dsum

    def rows(
        self, b: IRBuilder, h: Value, q0: Value, rows: Sequence[Value]
    ) -> list[tuple[Value, Value]]:
        """``(lse2, Dsum)`` of q-step rows ``rows`` (offsets from ``q0``)."""
        sl = self.lse2.slab(h, q0)
        sd = self.dsum.slab(h, q0)
        return [(sl.load(r, 0, _INF), sd.load(r, 0, 0.0)) for r in rows]

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


@dataclass(frozen=True, eq=False)
class PackedStep:
    """Inputs of a head-packed q step: the row map and the Q / dO tensors."""

    rows: PackedRows
    q_addr: StridedTensorAddr
    do_addr: StridedTensorAddr


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
        self.o0 = geom.g0.wave_origin(b, wave)
        self.o1 = geom.g1.wave_origin(b, wave)
        self.o4 = geom.g4.wave_origin(b, wave)
        self.k_regs = self.v_regs = self.kt_regs = None
        # Views of the planner buffers (no IR is emitted by a view).
        self.v_q = lds.view("q", d)
        self.v_do = lds.view("do", d)
        self.v_ds = lds.view("ds", n0)
        self.v_pt = lds.view("pt", n0) if geom.pt_route == "lds" else None
        if lds.has("k_res"):
            self.v_k = lds.view("k_res", d)
        else:
            self.v_k = lds.view("k_stage", d)
        self.v_v = lds.view("v_res", d) if lds.has("v_res") else lds.view("v_stage", d)
        if geom.acc_in_lds:
            self.v_dk_acc = lds.view("dk_acc", d)
            self.v_dv_acc = lds.view("dv_acc", d)
        else:
            self.v_dk_acc = self.v_dv_acc = None
        if m0 % 16 or n0 % 16:
            raise ValueError("tile extents must be multiples of 16")

    # ----- one-time K / V staging ---------------------------------------
    def stage_kv(
        self, b: IRBuilder, k_slab: RowSlab, v_slab: RowSlab, k0: Value, live: Value
    ) -> None:
        """Stage K and V of the key tile (guarded by ``live``), then residency.

        Global reads happen only inside ``scf_if(live)``. With register
        residency the K (G0), V (G2) and K^T (G4) fragments are read from LDS
        after a barrier; a barrier also separates the last stage-0 read from
        the first loop write, because the stage-0 phases alias the loop phase.
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
        if g.kt_source == "reg":
            self.kt_regs = g.g4.load_all_b(
                b, LdsOperand(self.v_k, "k_outer", reader=g.reader), lane, self.o4[1]
            )
        if reg:
            b.sync()  # stage-0 K reads complete before V overwrites the phase
            with b.scf_if(live):
                copy_rows_to_lds(b, v_slab, self.v_v, row0=k0, **kw)
            b.sync()
            self.v_regs = g.g0.load_all_b(
                b, LdsOperand(self.v_v, "k_inner"), lane, self.o0[1]
            )
            b.sync()  # stage-0 reads complete before the loop phase is written

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
            for it in range(-(-total // g.threads)):
                chunk = (
                    self.tid
                    if it == 0
                    else b.add(self.tid, b.const_i32(it * g.threads))
                )

                def body(chunk=chunk, view=view) -> None:
                    r = b.div(chunk, b.const_i32(per_row))
                    c = b.mul(b.mod(chunk, b.const_i32(per_row)), b.const_i32(4))
                    view.store(b, r, c, zero4, 4)

                if (it + 1) * g.threads <= total:
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
        ``"full"`` drains VMEM too. The staging and epilogue barriers are
        always full.
        """
        if self.geom.loop_sync == "lds_only":
            b.sync_lds_only()
        else:
            b.sync()

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
    ) -> DkDvAcc:
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
        (default G1, G3, G4). Under the relabel route the barrier that makes
        the dS tile visible sits right before G4 in every order; under the
        LDS route the P / dS barrier precedes all three.
        """
        g = self.geom
        g0, g1 = g.g0, g.g1
        d, m0 = g.head_size, g.k_m0
        lane = self.lane
        q0 = b.mul(qt, b.const_i32(m0))
        self.sched.emit_loop_head(b)
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
            rows=m0, cols=d, tid=self.tid, threads=g.threads, wave=self.wave,
            wave_size=g.wave_size,
        )  # fmt: skip
        self._loop_barrier(b)

        # Stage A: G0 S = Q K^T (+ G2 dP = dO V^T in stage B).
        self.sched.emit_stage_begin(b, "A")
        k_b = self.k_regs if self.k_regs is not None else LdsOperand(self.v_k)
        s_acc = g0.mma(
            b, g0.zero_acc(b), LdsOperand(self.v_q), k_b, lane=lane, origin=self.o0
        )
        self.sched.emit_stage_end(b, "A")
        self.sched.emit_stage_begin(b, "B")
        v_b = self.v_regs if self.v_regs is not None else LdsOperand(self.v_v)
        dp_acc = g0.mma(
            b, g0.zero_acc(b), LdsOperand(self.v_do), v_b, lane=lane, origin=self.o0
        )

        # P and dS per C slot (selects, never multiplies).
        p_f32, ds_f32 = self._softmax_grad(
            b, s_acc, dp_acc, h, q0, k0, edge=edge,
            rows=None if packed is None else packed.rows, stats=stats,
        )  # fmt: skip

        # Hand-off of P^T / dS^T to G1 / G3; dS to LDS for G4.
        if g.pt_route == "relabel":
            a_p, _ = relabel_acc_to_a(b, g0, g1, p_f32, lane)
            a_ds, _ = relabel_acc_to_a(b, g0, g1, ds_f32, lane)
            self._write_c_tiles(b, self.v_ds, ds_f32)
        else:
            self._write_c_tiles(b, self.v_pt, p_f32)
            self._write_c_tiles(b, self.v_ds, ds_f32)
            self._loop_barrier(b)
            a_p = LdsOperand(self.v_pt, "k_outer", reader=g.reader)
            a_ds = LdsOperand(self.v_ds, "k_outer", reader=g.reader)

        # G1 dV += P^T dO, G3 dK += dS^T Q and G4 dQ_part = dS K (then the dQ
        # sink), issued in ``stage_order``. Each accumulator's MMA chain is
        # the same in every order. Stage B holds the first of the three (after
        # G2 and the hand-off), C the second, D the third.
        coords = g.b13_coords
        b_dot = LdsOperand(self.v_do, "k_outer", coords=coords, reader=g.reader)
        b_qt = LdsOperand(self.v_q, "k_outer", coords=coords, reader=g.reader)
        out = {}

        def g1_dv() -> None:
            if g.acc_in_lds:
                self._update_lds_acc(b, self.v_dv_acc, a_p, b_dot)
                out["dv"] = None
            else:
                out["dv"] = g1.mma(
                    b, [list(r) for r in acc.dv], a_p, b_dot, lane=lane,
                    origin=self.o1, route=self._g13_route, **self._g13_asm,
                )  # fmt: skip

        def g3_dk() -> None:
            if g.acc_in_lds:
                self._update_lds_acc(b, self.v_dk_acc, a_ds, b_qt)
                out["dk"] = None
            else:
                out["dk"] = g1.mma(
                    b, [list(r) for r in acc.dk], a_ds, b_qt, lane=lane,
                    origin=self.o1, route=self._g13_route, **self._g13_asm,
                )  # fmt: skip

        units = {
            "g1": g1_dv,
            "g3": g3_dk,
            "g4": lambda: self._g4_dq(b, h, q0, packed, edge=edge),
        }
        for pos, (name, stage) in enumerate(zip(stage_units(g.stage_order), "BCD")):
            if name == "g4" and g.pt_route == "relabel":
                self._loop_barrier(b)  # the dS tile above is visible to every wave
            if pos:
                self.sched.emit_stage_begin(b, stage)
            units[name]()
            self.sched.emit_stage_end(b, stage)
        self._loop_barrier(b)  # q / do / ds / pt reads done before the next step
        if g.acc_in_lds:
            return acc
        dk, dv = out["dk"], out["dv"]
        if g.agpr_route == "agpr_fence":
            dk = [[_agpr_fence(b, x) for x in row] for row in dk]
            dv = [[_agpr_fence(b, x) for x in row] for row in dv]
        return DkDvAcc(dk=tuple(tuple(r) for r in dk), dv=tuple(tuple(r) for r in dv))

    def _g4_dq(self, b, h, q0, packed: PackedStep | None, *, edge: bool = True) -> None:
        """G4 ``dQ_part = dS K`` from the dS tile in LDS, then the dQ sink.

        The dS tile must be visible to every wave (the caller's barrier).
        ``edge=False`` (an interior step: every row is < len_q) drops the
        sink's row clamps, unless the geometry's ``interior_clamp`` keeps them.
        """
        g = self.geom
        g4, lane = g.g4, self.lane
        clamp = edge or g.interior_clamp
        if self.kt_regs is not None:
            kt_b = self.kt_regs
        else:
            kt_b = LdsOperand(self.v_k, "k_outer", reader=g.reader)
        sink_rows = q0 if packed is None else packed.rows

        def slots_of(fm, fn, frag):
            coords = g4.acc_coords(b, lane, self.o4, fm, fn)
            return [
                DqSlot(row=r, col=c, value=b.vec_extract(frag, s))
                for s, (r, c) in enumerate(coords)
            ]

        a_ds = LdsOperand(self.v_ds)
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
                    self.sink.consume(
                        b, slots_of(fm, fn, frag), sink_rows, h, clamp=clamp
                    )
        else:
            dq = g4.mma(b, g4.zero_acc(b), a_ds, kt_b, lane=lane, origin=self.o4)
            slots = []
            for fm in range(g4.frags_m):
                for fn in range(g4.frags_n):
                    slots += slots_of(fm, fn, dq[fm][fn])
            self.sink.consume(b, slots, sink_rows, h, clamp=clamp)

    def _load_stats(self, b, h, q0, rows):
        """``(row_of, keys, {(fm, slot): (lse2, Dsum)})`` of the lane's C rows.

        Row coordinates depend on ``(fm, slot)`` only, so the stats are loaded
        once per step; ``rows`` as in :meth:`_softmax_grad`.
        """
        g0 = self.geom.g0
        row_of = {}
        for fm in range(g0.frags_m):
            for s, (r, _c) in enumerate(g0.acc_coords(b, self.lane, self.o0, fm, 0)):
                row_of[(fm, s)] = r
        keys = sorted(row_of)
        if rows is None:
            stats = self.stats.rows(b, h, q0, [row_of[k] for k in keys])
        else:
            stats = self.stats.packed_rows(b, rows, [row_of[k] for k in keys])
        return row_of, keys, dict(zip(keys, stats))

    def _softmax_grad(
        self, b, s_acc, dp_acc, h, q0, k0, *, edge: bool, rows=None, stats=None
    ):
        """``(P, ds_mult * dS)`` as fp32 accumulator-shaped fragment lists.

        ``rows`` (a :class:`PackedRows`, head packing) gives each M row its own
        head and sequence row; a dead packed row fails the keep. ``stats``: the
        :meth:`_load_stats` result when the step loaded them earlier
        (``stats_load = "step_top"``); ``None`` loads them here.
        """
        g0 = self.geom.g0
        lane = self.lane
        zero = b.const_f32(0.0)
        inf = b.const_f32(_INF)
        if stats is None:
            row_of, keys, stats = self._load_stats(b, h, q0, rows)
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
                    p = b.exp2(b.fsub(b.fmul(sv, self.scale_log2), lse2))
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
        """Round fp32 C-layout fragments to the I/O dtype and store ``[q][key]``."""
        g0 = self.geom.g0
        for fm in range(g0.frags_m):
            for fn in range(g0.frags_n):
                half = b.vec_cast_f32_to(frags[fm][fn], self.elem)
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
        return {
            "asm_tail_nop": self.geom.asm_tail_nop,
            "asm_cluster": self.geom.asm_cluster,
        }

    def finish(self, b: IRBuilder, acc: DkDvAcc) -> DkDvAcc:
        """Hook after the q loop (``dk_mult`` is applied by the epilogue).

        Under the ``agpr_fence`` route the accumulators come back from AGPRs
        once here, so the epilogue reads arch VGPRs.
        """
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
