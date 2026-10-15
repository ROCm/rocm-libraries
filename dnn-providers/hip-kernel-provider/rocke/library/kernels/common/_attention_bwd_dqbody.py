# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Body of the split dQ kernel of the attention backward (Q-parallel, no atomics).

Family glue for the backward kernels (Python-only, no C++ twin; packs pin
``ROCKE_BACKEND=python``). The second orientation of the backward tile step:
one CTA owns ``kM0`` query rows (of one query head, or of ``G`` query heads
that share one K head and one V head, packed onto the rows) and walks the
key tiles their band reaches:

* stage 0, once per CTA: the Q and dO rows are staged through LDS and read
  into registers as the **B** operands of G0' / G2' (they stay resident over
  the key loop); every lane's ``lse2`` / ``Dsum`` (one query column per C
  fragment), the dead-row flag and the per-row key limits are formed once;
* per key tile: K and V (``kN0`` rows) are staged to LDS (or, with
  ``kv_prefetch``, the step issues the next tile's global loads into
  registers first and stores them to LDS after its matrix work), then
  **G0'** ``S^T = K Q^T`` and **G2'** ``dP^T = V dO^T`` (``kN0 x kM0`` over
  D, warp grid ``(1, W)``: the waves split the query columns),
  ``P = select(keep, exp2(S * scale_log2 - lse2), 0)`` and
  ``dS = select(keep && lse2 < +inf, P * (dP - Dsum), 0)`` in the transposed
  C layout (under ``stats_prescale``: ``P = exp2(fma(S, scale_log2,
  -lse2))`` and ``dS = P * fma(dP, ds_mult, -ds_mult * Dsum)`` with the
  negated stats formed once per CTA, one fma per cell each; otherwise the main
  kernel's ``exp2(S * scale_log2 - lse2)`` and ``ds_mult * dS``), rounded at
  the main kernel's
  rounding point, and **G4'** ``dQ += dS K`` (``kM0 x D`` over ``kN0``, warp
  grid ``(W, 1)``): the dS^T accumulator is the G4' A operand by the checked
  register relabel ``C(M, N) == A(N, M)`` (the same proof as the main
  kernel's G0 to G1 hand-off), so dS never goes through LDS; K is read as the
  K-outer B operand through the transpose source (``tr_read`` on gfx950,
  ``lds_plain``);
* after the loop: ``dQ * dq_mult`` rounded once to the I/O dtype, staged
  through LDS and stored as rows (``stage_vec`` access width): rows ``q <
  len_q`` get values, batched rows ``[len_q, S_q_max)`` exact zeros, THD rows
  outside the sequence are not touched.

Scheduling (``dq_sched``, the spec's ``sched``;
:class:`._attention_bwd_sched.DqStageTable`): a key step runs as stages A
(G0'), B (G2' and the softmax gradient) and C (G4'); the values add an
``iglp_opt``, the stage fences, the per-stage ``sched_group_barrier`` groups
and priorities. They are hints only: no op moves, dQ is the default's.

dQ is accumulated in fp32 registers in a fixed key order and rounded once:
the result is deterministic (bit for bit repeatable), with no workspace and
no atomics. A dQ row depends on its own dS row only, so a row that is never
stored (past ``len_q``, or a dead packed head) needs no mask.

The key range of a CTA is exact (:func:`kv_tile_range`: the key tiles holding
a kept cell of one of its live rows); with ``edge_tiles`` it runs as edge /
interior / edge loops over :func:`interior_kv_range` (interior tiles keep
every cell, so they skip the per-cell selects and keep only the dead-row
select) in the single loop's order, which gives bit-identical dQ. Both are
pure integer emitters, evaluated on the host by the exhaustive tests.

Hazard structure: the key-loop trip counts are ``readfirstlane``'d, the
accumulators start after the range is decided, the Q / dO staging is guarded
by the CTA-uniform ``q0 < len_q`` and no matrix op sits inside a divergent
region.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from rocke.core.ir import F32, I64, IRBuilder, Value

from ._attention_bwd_addr import (
    PackedRows,
    RowSlab,
    StridedTensorAddr,
    as_i32,
    chunk_coords,
    deal_chunks,
)
from ._attention_bwd_band import (
    MaskPolicy,
    _bound_or_none,
    _floor_div,
    dkv_store_rule,
)
from ._attention_bwd_body import (
    StatsView,
    _rounding_point,
    copy_packed_rows_to_lds,
    copy_rows_to_lds,
)
from ._attention_bwd_caps import bwd_arch_caps, check_transpose_read
from ._attention_bwd_frag import (
    LdsView,
    atom_facts,
    check_tr16_coords,
    eval_layout,
    ir_type,
)
from ._attention_bwd_gemm import (
    BlockGemm,
    LdsOperand,
    check_relabel_grids,
    kperm_b_operand_coords,
    relabel_acc_to_a,
)
from ._attention_bwd_sched import DQ_SCHEDS, DqStageTable

__all__ = [
    "DQ_TRANSPOSE_READERS",
    "BwdDqGeometry",
    "BwdDqStep",
    "DqLdsViews",
    "interior_kv_range",
    "kv_tile_range",
]

# Transpose sources of the G4' K-outer B operand (K as ``[key][d]`` in LDS).
DQ_TRANSPOSE_READERS = {"tr_read": "tr16", "lds_plain": "plain"}
_INF = float("inf")


# ---------------------------------------------------------------------------
# key-tile range of a block of query rows (the dual of q_tile_range)
# ---------------------------------------------------------------------------


def kv_tile_range(
    b: IRBuilder,
    band,
    q_first: Value,
    n_rows,
    *,
    k_n0: int,
    uniform: bool = True,
) -> tuple[Value, Value]:
    """``(kt_start, n_kt)``: key tiles of ``k_n0`` keys that rows ``[q_first,
    q_first + n_rows)`` attend (rows at or past ``len_q`` attend nothing).

    The live rows ``[q_first, q_last]`` (``q_last = min(q_first + n_rows,
    len_q) - 1``) keep keys ``[max(lo(q_first), 0), min(hi(q_last), len_kv -
    1)]``, ``lo(q) = q + off - left``, ``hi(q) = q + off + right`` (an
    unbounded side gives ``0`` / ``len_kv - 1``): both limits grow with the
    row and a bounded row interval holds ``left + right + 1 >= 1`` keys, so
    the union of the rows' intervals is that one interval and every tile in
    the range holds a kept cell. ``n_kt`` is 0 when no row is live, the band
    misses every key or ``len_kv == 0``. ``n_rows`` is an int or an i32
    value; ``q_first >= 0``. With ``uniform`` both results are
    ``readfirstlane``'d (the key-loop trip count lives in an SGPR).
    """
    if k_n0 < 1:
        raise ValueError("tile sizes must be positive")
    zero = b.const_i32(0)
    cn = b.const_i32(k_n0)
    rows = as_i32(b, n_rows)
    q_last = b.smin(
        b.sub(b.add(q_first, rows), b.const_i32(1)),
        b.sub(band.sq, b.const_i32(1)),
    )
    off = band.off
    c_first = q_first if off is None else b.add(q_first, off)
    c_last = q_last if off is None else b.add(q_last, off)
    left = _bound_or_none(band, "left")
    right = _bound_or_none(band, "right")
    if left is None:
        k_lo = zero
    else:
        lft = as_i32(b, left)
        k_lo = b.sub(c_first, lft)
        if not isinstance(left, int):
            k_lo = b.select(b.cmp_lt(lft, zero), zero, k_lo)
        k_lo = b.smax(k_lo, zero)
    if right is None:
        k_hi = band.sk_minus1
    else:
        r = as_i32(b, right)
        k_hi = b.add(c_last, r)
        if not isinstance(right, int):
            k_hi = b.select(b.cmp_lt(r, zero), band.sk_minus1, k_hi)
        k_hi = b.smin(k_hi, band.sk_minus1)
    # Empty when no row is live or the rows' band misses every key.
    dead = b.lor(b.cmp_lt(q_last, q_first), b.cmp_lt(k_hi, k_lo))
    k_hi = b.select(dead, b.const_i32(-1), k_hi)
    kt_start = b.div(k_lo, cn)
    kt_stop = b.div(b.add(b.smax(k_hi, b.const_i32(-1)), cn), cn)
    n_kt = b.smax(b.sub(kt_stop, kt_start), zero)
    if uniform:
        kt_start = b.readfirstlane(kt_start)
        n_kt = b.readfirstlane(n_kt)
    return kt_start, n_kt


def interior_kv_range(
    b: IRBuilder,
    band,
    q_first: Value,
    n_rows,
    kt_start: Value,
    n_kt: Value,
    *,
    k_n0: int,
    masked: bool = True,
    uniform: bool = True,
) -> tuple[Value, Value]:
    """``(i0, i1)``: the interior key tiles of rows ``[q_first, q_first +
    n_rows)`` are ``[i0, i1)``.

    ``kt_start <= i0 <= i1 <= kt_start + n_kt``. A key tile is interior when
    every row is ``< len_q``, every key is ``< len_kv`` and (``masked``) every
    cell is inside the band: the right edge binds at ``(q_first, k_last)``,
    the left edge at ``(q_last, k_first)``. Each condition bounds the tile
    index from one side, so the interior set is contiguous and every visited
    tile outside it is an edge tile.
    """
    zero = b.const_i32(0)
    rows = as_i32(b, n_rows)
    kt_end = b.add(kt_start, n_kt)
    q_last = b.sub(b.add(q_first, rows), b.const_i32(1))
    lo = kt_start
    # k0 + k_n0 - 1 < len_kv  <->  kt < floor(len_kv / k_n0)
    hi = b.smin(kt_end, b.div(band.sk, b.const_i32(k_n0)))
    off = band.off
    right = _bound_or_none(band, "right") if masked else None
    left = _bound_or_none(band, "left") if masked else None
    if right is not None:
        # k_last <= q_first + off + right  ->  kt <= floor((... - k_n0 + 1) / k_n0)
        r = as_i32(b, right)
        top = b.add(q_first, r) if off is None else b.add(b.add(q_first, off), r)
        last = _floor_div(b, b.sub(top, b.const_i32(k_n0 - 1)), k_n0)
        stop = b.add(last, b.const_i32(1))
        if not isinstance(right, int):
            stop = b.select(b.cmp_lt(r, zero), hi, stop)
        hi = b.smin(hi, stop)
    if left is not None:
        # k_first >= q_last + off - left  ->  kt >= ceil((...) / k_n0)
        lft = as_i32(b, left)
        need = b.sub(q_last, lft) if off is None else b.sub(b.add(q_last, off), lft)
        first = b.sub(zero, _floor_div(b, b.sub(zero, need), k_n0))
        if not isinstance(left, int):
            first = b.select(b.cmp_lt(lft, zero), lo, first)
        lo = b.smax(lo, first)
    hi = b.select(b.cmp_lt(q_last, band.sq), hi, lo)  # a row past len_q: none
    i0 = b.smin(lo, kt_end)
    i1 = b.smax(b.smin(hi, kt_end), i0)
    if uniform:
        i0 = b.readfirstlane(i0)
        i1 = b.readfirstlane(i1)
    return i0, i1


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _atom_k(atom: str) -> int:
    return int(str(atom).rsplit("x", 1)[1])


@dataclass(frozen=True)
class BwdDqGeometry:
    """Resolved, validated tile geometry of one split dQ kernel.

    :meth:`check` proves the lane-map facts of both GEMMs, the G0' C to G4' A
    register relabel, the transpose read of the G4' B operand and that a C
    fragment holds one query column per lane; it raises :class:`ValueError`
    before any IR is emitted.
    """

    arch: str
    dtype: str
    waves: int
    k_m0: int
    k_n0: int
    head_size: int
    atom_g02: int
    atom_g4: int
    transpose_source: str
    # the key step's schedule value and the knobs its stage table reads
    sched: str = "none"
    kv_pad: int = 0
    kv_prefetch: bool = False
    stage_vec: int = 8

    @classmethod
    def from_spec(cls, spec, arch: str) -> BwdDqGeometry:
        """Geometry of a resolved :class:`AttnBwdDqSpec`."""
        if spec.waves is None or spec.block_m is None or spec.block_n is None:
            raise ValueError("BwdDqGeometry needs a resolved spec")
        geom = cls(
            arch=arch,
            dtype=spec.dtype,
            waves=spec.waves,
            k_m0=spec.block_m,
            k_n0=spec.block_n,
            head_size=spec.head_size,
            atom_g02=_atom_k(spec.atom_g02),
            atom_g4=_atom_k(spec.atom_g4),
            transpose_source=spec.transpose_source,
            sched=getattr(spec, "sched", "none"),
            kv_pad=int(getattr(spec, "kv_pad", 0)),
            kv_prefetch=bool(getattr(spec, "kv_prefetch", False)),
            stage_vec=int(getattr(spec, "stage_vec", 8)),
        )
        geom.check()
        return geom

    def stage_table(self) -> DqStageTable:
        """The key step's stage table (``dq_sched``; the hooks of
        :meth:`BwdDqStep.run_kv_step`)."""
        return DqStageTable.from_geometry(self)

    @cached_property
    def g0t(self) -> BlockGemm:
        """G0' ``S^T = K Q^T`` (also G2' ``dP^T = V dO^T``): kN0 x kM0 over D."""
        return BlockGemm(
            self.arch, self.dtype, self.k_n0, self.k_m0, self.head_size,
            (1, self.waves), self.atom_g02,
        )  # fmt: skip

    @cached_property
    def g4t(self) -> BlockGemm:
        """G4' ``dQ += dS K``: kM0 x D over kN0."""
        return BlockGemm(
            self.arch, self.dtype, self.k_m0, self.head_size, self.k_n0,
            (self.waves, 1), self.atom_g4,
        )  # fmt: skip

    @property
    def wave_size(self) -> int:
        return bwd_arch_caps(self.arch).wave_size

    @property
    def threads(self) -> int:
        return self.waves * self.wave_size

    @property
    def reader(self) -> str:
        return DQ_TRANSPOSE_READERS[self.transpose_source]

    @cached_property
    def relabel_plan(self):
        """The proved G0' C to G4' A relabel plan (raises when illegal)."""
        return check_relabel_grids(self.g0t, self.g4t)

    @cached_property
    def b4_coords(self):
        """B coordinates of G4' (K-permuted when the relabel needs it)."""
        return kperm_b_operand_coords(self.relabel_plan, self.g0t, self.g4t)

    def check(self) -> None:
        if self.transpose_source not in DQ_TRANSPOSE_READERS:
            raise ValueError(
                f"transpose_source={self.transpose_source!r} has no reader in the "
                f"split dQ kernel (built: {tuple(DQ_TRANSPOSE_READERS)})"
            )
        if self.sched not in DQ_SCHEDS:
            raise ValueError(f"dq sched {self.sched!r} not in {DQ_SCHEDS}")
        for g in (self.g0t, self.g4t):
            facts = atom_facts(g.op)
            if not (facts.a_contig and facts.b_contig and facts.affine):
                raise ValueError(
                    f"{g.op.op_id}: lane maps are not K-contiguous and affine "
                    f"({facts}); this dQ geometry cannot be built on {self.arch}"
                )
        plan = self.relabel_plan
        if self.reader == "tr16":
            check_transpose_read(self.arch, 64)
            if plan.needs_k_perm:
                raise ValueError(
                    "transpose_source='tr_read' with a K-permuted G4' B operand "
                    f"(atom_g4 K = {self.atom_g4}) is not built; use 'lds_plain'"
                )
            check_tr16_coords(self.b4_coords)
        # one query column per lane in every C fragment (per-column stats)
        for lane in eval_layout(self.g0t.op.c_layout()):
            if len({c for _r, c in lane}) != 1:
                raise ValueError(
                    f"{self.g0t.op.op_id}: a C fragment spans several columns per "
                    "lane; the per-column stats of the dQ kernel need one"
                )


# ---------------------------------------------------------------------------
# LDS views and the per-CTA step
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DqLdsViews:
    """The CTA's LDS windows: stage 0 (Q, dO), the key loop (K, V) and the
    epilogue (dQ rows); the three phases alias one region, separated by
    barriers."""

    q: LdsView
    do: LdsView
    k: LdsView
    v: LdsView
    out: LdsView


class BwdDqStep:
    """Emitter of the split dQ kernel's stage 0, key steps and epilogue.

    One instance per CTA; ``lane`` / ``wave`` / ``tid`` come from
    :func:`._attention_bwd_gemm.lane_and_wave`. ``rows`` (head packing): a
    :class:`PackedRows` map of the M rows; ``None``: the rows are ``q0 + r``
    of head ``h``.
    """

    def __init__(
        self,
        b: IRBuilder,
        geom: BwdDqGeometry,
        *,
        mask: MaskPolicy,
        stats: StatsView,
        lds: DqLdsViews,
        tid: Value,
        lane: Value,
        wave: Value,
        scale_log2: Value,
        ds_mult: Value | None,
        prescale: bool,
        sentinel: bool = True,
    ) -> None:
        self.geom = geom
        self.mask = mask
        self.stats = stats
        self.lds = lds
        self.tid, self.lane, self.wave = tid, lane, wave
        self.scale_log2 = scale_log2
        self.ds_mult = ds_mult
        self.prescale = prescale
        self.sentinel = sentinel
        self.elem = ir_type(geom.dtype)
        self.o0 = geom.g0t.wave_origin(b, wave)
        self.o4 = geom.g4t.wave_origin(b, wave)
        self.qt_regs = self.dot_regs = None
        self.cols: dict = {}
        self.sched = geom.stage_table()  # the key step's hints (dq_sched)

    # ----- stage 0 -------------------------------------------------------
    def stage_q(
        self,
        b: IRBuilder,
        *,
        live: Value,
        h: Value,
        q0: Value,
        q_slab: RowSlab | None,
        do_slab: RowSlab | None,
        q_addr: StridedTensorAddr,
        do_addr: StridedTensorAddr,
        rows: PackedRows | None,
        vec: int = 8,
    ) -> None:
        """Q / dO rows to the resident B fragments, then the column stats.

        The global reads sit inside ``scf_if(live)`` (``live``: the CTA has a
        row ``< len_q``; a CTA without one reads nothing and runs no key
        step). A barrier publishes the copies, a second one retires the
        stage-0 reads before the key loop overwrites the aliased region.
        """
        g = self.geom
        d, m0 = g.head_size, g.k_m0
        kw = dict(rows=m0, cols=d, tid=self.tid, threads=g.threads, vec=vec)
        with b.scf_if(live):
            if rows is None:
                copy_rows_to_lds(b, q_slab, self.lds.q, row0=q0, **kw)
                copy_rows_to_lds(b, do_slab, self.lds.do, row0=q0, **kw)
            else:
                copy_packed_rows_to_lds(b, q_addr, self.lds.q, rows, **kw)
                copy_packed_rows_to_lds(b, do_addr, self.lds.do, rows, **kw)
        b.sync()
        g0t = g.g0t
        self.qt_regs = g0t.load_all_b(
            b, LdsOperand(self.lds.q, "k_inner"), self.lane, self.o0[1]
        )
        self.dot_regs = g0t.load_all_b(
            b, LdsOperand(self.lds.do, "k_inner"), self.lane, self.o0[1]
        )
        self._column_stats(b, h, q0, rows)
        b.sync()  # stage-0 reads complete before the key loop writes K / V

    def _column_stats(self, b, h, q0, rows) -> None:
        """Per q fragment: the lane's column, ``(lse2, Dsum)`` (under
        ``prescale``: ``-lse2`` and ``-ds_mult * Dsum``), the dead-row flag and
        the row's key limits (dead rows keep nothing)."""
        g0t = self.geom.g0t
        cols = [
            g0t.acc_coords(b, self.lane, self.o0, 0, fn)[0][1]
            for fn in range(g0t.frags_n)
        ]
        if rows is None:
            st = self.stats.rows(b, h, q0, cols)
        else:
            st = self.stats.packed_rows(b, rows, cols)
        inf = b.const_f32(_INF)
        self.cols = {}
        for fn, (col, (lse2, dsum)) in enumerate(zip(cols, st)):
            live = b.fcmp("olt", lse2, inf) if self.sentinel else None
            if rows is None:
                q, ok = b.add(q0, col), None
            else:
                _h, q, ok = rows.locate(b, col)
            if live is not None:
                ok = live if ok is None else b.land(ok, live)
            limits = self.mask.row_limits(b, q, ok)
            if self.prescale:
                # one fma per cell for the exp2 argument and one for dS
                lse2 = b.fneg(lse2)
                if self.ds_mult is not None:
                    dsum = b.fneg(b.fmul(dsum, self.ds_mult))
            self.cols[fn] = (lse2, dsum, live, limits)

    # ----- one key step ----------------------------------------------------
    def zero_dq(self, b: IRBuilder) -> list[list[Value]]:
        return self.geom.g4t.zero_acc(b)

    def _tile_loads(self, b, slab: RowSlab, row0: Value, vec: int = 8) -> list:
        """Global loads of a ``kN0 x D`` tile for this thread: ``[(r, c, value,
        guard)]``; a tail chunk past the tile loads the tile's last chunk and
        carries its guard (only its LDS store is predicated)."""
        g = self.geom
        per_row = g.head_size // vec
        total = g.k_n0 * per_row
        out = []
        for chunk, tail in deal_chunks(b, self.tid, total, g.threads):
            src = b.smin(chunk, b.const_i32(total - 1)) if tail else chunk
            r, c = chunk_coords(b, src, per_row, vec)
            guard = b.cmp_lt(chunk, b.const_i32(total)) if tail else None
            out.append((r, c, slab.load_vec(b.add(row0, r), c, vec), guard))
        return out

    @staticmethod
    def _tile_stores(b, view: LdsView, pending: list, vec: int = 8) -> None:
        for r, c, val, guard in pending:
            if guard is None:
                view.store(b, r, c, val, vec)
            else:
                with b.scf_if(guard):
                    view.store(b, r, c, val, vec)

    def prologue(
        self, b: IRBuilder, k_slab: RowSlab, v_slab: RowSlab, k0: Value, live: Value
    ) -> None:
        """``kv_prefetch``: the first key tile of the range to LDS (its global
        reads inside ``scf_if(live)``, ``live``: the range is not empty), then
        a barrier; every later tile is prefetched by the step before it."""
        g = self.geom
        kw = dict(rows=g.k_n0, cols=g.head_size, tid=self.tid, threads=g.threads)
        with b.scf_if(live):
            copy_rows_to_lds(b, k_slab, self.lds.k, row0=k0, **kw)
            copy_rows_to_lds(b, v_slab, self.lds.v, row0=k0, **kw)
        b.sync()

    def run_kv_step(
        self,
        b: IRBuilder,
        dq: Sequence[Sequence[Value]],
        k_slab: RowSlab,
        v_slab: RowSlab,
        k0: Value,
        *,
        edge: bool = True,
        prefetch: bool = False,
        vec: int = 8,
    ) -> list[list[Value]]:
        """K / V of key tile ``k0`` to LDS, G0' / G2', dS^T, G4'.

        ``prefetch`` (``kv_prefetch``): tile ``k0`` is already in LDS (the
        prologue or the previous step); the step issues the global loads of
        tile ``k0 + kN0`` into registers first, computes, and after the barrier
        that retires its LDS reads stores them to LDS and publishes them with
        a second barrier (rows are clamped to the sequence, so the prefetch
        past the last tile reads valid rows and is never used). Same values
        as the plain step, two barriers per step either way.

        Schedule (``geom.sched``, :class:`._attention_bwd_sched.DqStageTable`):
        the step runs as stages A (G0'), B (G2' and the softmax gradient) and
        C (G4'); ``iglp*`` put one ``iglp_opt`` ahead of A, the fenced values
        a ``sched_barrier(0)`` after A and B, the stage tables their groups
        (and priorities). Stage A opens before the prefetch loads (so they
        sit in A's scheduling region); no op moves, so every value is the
        default's.
        """
        g = self.geom
        g0t, g4t = g.g0t, g.g4t
        d, n0 = g.head_size, g.k_n0
        lane = self.lane
        sched = self.sched
        if not prefetch:
            kw = dict(rows=n0, cols=d, tid=self.tid, threads=g.threads, vec=vec)
            copy_rows_to_lds(b, k_slab, self.lds.k, row0=k0, **kw)
            copy_rows_to_lds(b, v_slab, self.lds.v, row0=k0, **kw)
            b.sync()
        sched.emit_loop_head(b)
        sched.emit_stage_begin(b, "A")
        if prefetch:
            k_next = b.add(k0, b.const_i32(n0))
            pend_k = self._tile_loads(b, k_slab, k_next, vec)
            pend_v = self._tile_loads(b, v_slab, k_next, vec)
        s_acc = g0t.mma(
            b, g0t.zero_acc(b), LdsOperand(self.lds.k), self.qt_regs, lane=lane,
            origin=self.o0,
        )  # fmt: skip
        sched.emit_stage_end(b, "A")
        sched.emit_stage_begin(b, "B")
        dp_acc = g0t.mma(
            b, g0t.zero_acc(b), LdsOperand(self.lds.v), self.dot_regs, lane=lane,
            origin=self.o0,
        )  # fmt: skip
        ds = self._softmax_grad_t(b, s_acc, dp_acc, k0, edge=edge)
        a_ds, _ = relabel_acc_to_a(b, g0t, g4t, ds, lane)
        sched.emit_stage_end(b, "B")
        sched.emit_stage_begin(b, "C")
        kt = LdsOperand(self.lds.k, "k_outer", coords=g.b4_coords, reader=g.reader)
        dq = g4t.mma(b, dq, a_ds, kt, lane=lane, origin=self.o4)
        sched.emit_stage_end(b, "C")
        b.sync()  # K / V reads complete before the next step's copies
        if prefetch:
            self._tile_stores(b, self.lds.k, pend_k, vec)
            self._tile_stores(b, self.lds.v, pend_v, vec)
            b.sync()  # the next tile is visible to every wave
        return dq

    def _softmax_grad_t(self, b, s_acc, dp_acc, k0, *, edge: bool):
        """``ds_mult * dS^T`` as fp32 C fragments of G0' (rows keys, columns
        queries). Edge tiles: the per-cell keep (the dead-row term folded into
        the column's limits) gates P and dS; interior tiles keep only the
        dead-row select of dS (identical values)."""
        g0t = self.geom.g0t
        zero = b.const_f32(0.0)
        m = self.ds_mult
        out = []
        for fm in range(g0t.frags_m):
            row = []
            for fn in range(g0t.frags_n):
                lse2, dsum, live, limits = self.cols[fn]
                coords = g0t.acc_coords(b, self.lane, self.o0, fm, fn)
                vals = []
                for s, (r, _c) in enumerate(coords):
                    sv = b.vec_extract(s_acc[fm][fn], s)
                    dpv = b.vec_extract(dp_acc[fm][fn], s)
                    if self.prescale:  # lse2 holds -lse2
                        p = b.exp2(b.fma(sv, self.scale_log2, lse2))
                    else:
                        p = b.exp2(b.fsub(b.fmul(sv, self.scale_log2), lse2))
                    if edge:
                        cond = self.mask.keep_in(b, limits, b.add(k0, r))
                        p = b.select(cond, p, zero)
                    else:
                        cond = live
                    if self.prescale and m is not None:
                        # dsum holds -ds_mult * Dsum: one fma per cell
                        ds = b.fmul(p, b.fma(dpv, m, dsum))
                        if cond is not None:
                            ds = b.select(cond, ds, zero)
                    else:
                        ds = b.fmul(p, b.fsub(dpv, dsum))
                        if cond is not None:
                            ds = b.select(cond, ds, zero)
                        if m is not None:  # None: 1.0, not applied
                            ds = b.fmul(m, ds)
                    vals.append(_rounding_point(b, ds))
                row.append(b.vec_pack(vals, F32))
            out.append(row)
        return out

    # ----- epilogue --------------------------------------------------------
    def store_dq(
        self,
        b: IRBuilder,
        dq: Sequence[Sequence[Value]],
        *,
        mult: Value | None,
        dq_slab: RowSlab,
        head_step: Value | None,
        h: Value,
        q0: Value,
        rows: PackedRows | None,
        len_q: Value,
        s_q_max: Value,
        seq_mode: str,
    ) -> None:
        """``dQ * mult`` rounded once to the I/O dtype, transposed through the
        LDS row buffer (aliasing the key-loop region; every earlier LDS read
        is retired by the last barrier) and stored as rows of 8 elements.

        ``dq_slab`` is dQ rebased to head ``h`` at row 0, unclamped (formed
        before the key loops); a packed row of head ``hh`` moves its pointer
        by ``(hh - h) * head_step`` (``head_step``: the head stride in bytes,
        i64; ``None`` without packing).

        Rows ``q < len_q`` get values; batched rows ``[len_q, S_q_max)`` get
        exact zeros; rows at or past ``S_q_max`` (batched) or ``len_q`` (THD)
        and rows of dead packed heads are not touched.
        """
        g = self.geom
        g4t = g.g4t
        d, m0 = g.head_size, g.k_m0
        out = self.lds.out
        n = g4t.op.c_frag_len
        for fm in range(g4t.frags_m):
            for fn in range(g4t.frags_n):
                frag = dq[fm][fn]
                if mult is not None:
                    frag = b.vec_pack(
                        [b.fmul(mult, b.vec_extract(frag, i)) for i in range(n)], F32
                    )
                half = b.vec_cast_f32_to(frag, self.elem)
                coords = g4t.acc_coords(b, self.lane, self.o4, fm, fn)
                for s, (r, c) in enumerate(coords):
                    out.store(b, r, c, b.vec_extract(half, s), 1)
        b.sync()
        vec = 8
        per_row = d // vec
        total = m0 * per_row
        zero = b.zero_vec(self.elem, vec)
        assert dq_slab.last is None and (rows is None) == (head_step is None)
        slab = dq_slab
        threads = g.threads
        for chunk, tail in deal_chunks(b, self.tid, total, threads):
            r, c = chunk_coords(b, chunk, per_row, vec)
            if rows is None:
                q, ok, row_slab = b.add(q0, r), None, slab
            else:
                hh, q, ok = rows.locate(b, r)
                # hh >= h: the packed heads are h .. h + G - 1
                shift = b.mul(b.zext(b.sub(hh, h), I64), head_step)
                row_slab = dataclasses.replace(
                    slab, ptr=b.global_ptr_add(slab.ptr, shift)
                )
            store, row_live = dkv_store_rule(
                b, q, len_q, seq_mode=seq_mode, s_kv_max=s_q_max
            )
            if ok is not None:
                store = b.land(store, ok)
            if tail:
                store = b.land(store, b.cmp_lt(chunk, b.const_i32(total)))
            with b.scf_if(store):
                v = b.select(row_live, out.load(b, r, c, dtype=self.elem, n=vec), zero)
                row_slab.store_vec(q, c, v)
