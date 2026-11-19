# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Q / dO prefetch ring of the attention backward q loop.

Family glue for the backward main kernel (Python-only, no C++ twin; packs pin
``ROCKE_BACKEND=python``). Two knobs of the spec select it:

* ``ring_depth`` ``R`` in ``{1, 2, 3}``: Q, dO and the lse2 / Dsum rows of a
  q step are staged into one of ``R`` LDS ring slots ``R - 1`` steps ahead of
  the step that reads them (the softmax gradient reads its stats from the
  slot);
* ``global_path``: ``"vgpr"`` stages through registers (global load, then
  ``ds_write``), ``"dma"`` (gfx950) copies global memory straight into LDS
  with ``global_load_lds_dwordx4`` (Q, dO) and ``global_load_lds_dword`` (lse2,
  Dsum), issued as inline asm so the compiler's wait insertion does not see
  them (it would otherwise drain an in-flight copy before the first unrelated
  LDS read).

``(vgpr, 1)`` is the synchronous staging of the tile step and never uses this
module. Every other value runs one schedule per q loop (step ``i`` reads slot
``i mod R``; ``it`` is the loop counter):

* prologue (before the loop, only when it has a trip): tiles ``0 .. R - 2``
  into slots ``0 .. R - 2``, then an LDS-only barrier. The register ring
  loads the tile ``R - 2`` but stores only tile 0 (``R = 3`` carries tile 1
  in registers into the loop);
* top of step ``i``: the register ring issues the global loads of tile ``i +
  R - 1`` (its Q / dO chunks and, in the threads of wave 0, its stats rows):
  the step's only global loads, so the compiler's wait for them falls at
  their slot stores and leaves the step's dQ atomics in flight; the DMA ring
  issues the copy of tile ``i + R - 1`` into slot ``(i + R - 1) mod R``
  (``R = 1``: tile ``i`` into slot 0, then ``s_waitcnt vmcnt(0)`` and an
  LDS-only barrier);
* every in-step barrier is LDS-only (the in-flight prefetch stays in flight);
* end of step ``i``: the register ring stores the registers of tile ``i + 1``
  into slot ``(i + 1) mod R`` (``R = 3``: the set loaded one step earlier,
  carried through the back edge); the DMA ring waits ``s_waitcnt vmcnt(N)``
  until its own part of tile ``i + 1`` has landed; then an LDS-only barrier
  publishes tile ``i + 1`` and retires every read of slot ``i mod R``.

Steps past the loop's last trip fetch the current step again (a valid
address; the slot it lands in is never read), so every step issues the same
instructions and the wait immediates hold on every trip. After a DMA loop a
``s_waitcnt vmcnt(0)`` and an LDS-only barrier drain the copies still in
flight (``R = 3``) before anything else writes the aliased LDS phases.

Wait model (DMA path, :func:`dma_wait_counts`): VM_CNT completes in issue
order and counts loads, stores and no-return atomics. At the end of step ``i``
the operations a wave issued after its part of tile ``i + 1`` are at least
the q step's dQ atomics (``n_a`` per lane, branch-free; under
``stage_table_atomics`` the previous step's, issued in stage A after the
copy: as many, zeros on the first step, :meth:`QRing.check_step`) and, for
``R = 3``,
its part of tile ``i + 2`` (at least ``n_qd`` instructions, the smallest
per-wave share of a batch); the immediate is their sum (``R = 2``: ``n_a``),
clamped to the 6-bit counter. An under-count only waits longer; an
over-count would be a race. No other VMEM operation is in the DMA loop: Q,
dO and the stats arrive by DMA, K / V / K^T come from registers or LDS.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property

from rocke.core.ir import F32, I32, IRBuilder, Value

from ._attention_bwd_addr import RowSlab, WsSlab, as_i32
from ._attention_bwd_band import HostInt
from ._attention_bwd_body import (
    BwdLds,
    PackedStep,
    StatsView,
    packed_row_loader,
    slab_row_loader,
)
from ._attention_bwd_caps import check_lds_dma_width
from ._attention_bwd_frag import LdsView
from ._attention_bwd_lds import LDS_LINE_BYTES, DmaSlabLayout

__all__ = [
    "DMA_PIECE_BYTES",
    "RING_VMCNT_MAX",
    "DmaBatch",
    "LdsStatsView",
    "QRing",
    "PrefetchStep",
    "PrefetchTile",
    "dma_lane_source",
    "dma_wait_counts",
    "ring_slot_schedule",
]

# Bytes one wave instruction of ``global_load_lds_dwordx4`` writes (64 lanes x
# 16 bytes, lane ``l`` at ``M0 + 16 l``).
DMA_PIECE_BYTES = 1024
# gfx9 VM_CNT is a 6-bit counter.
RING_VMCNT_MAX = 63
_STATS = ("lse2", "dsum")
_INF = float("inf")
# M0 holds the LDS destination of an LDS DMA; one wait state between the SALU
# write of M0 and the DMA (the CK form). The compiler does not track inline
# asm, so it inserts no ``vmcnt`` wait for these copies anywhere. The 64-bit
# global base (``s``) is the slab pointer formed with a readfirstlane'd offset
# (``RowSlab.sgpr_ptr``): the production compiler can otherwise hand a uniform
# pointer it keeps in a VGPR pair to the ``s`` operand.
_ASM_X4 = "s_mov_b32 m0, $1\n\ts_nop 0\n\tglobal_load_lds_dwordx4 $0, $2"
_ASM_X1 = "s_mov_b32 m0, $1\n\ts_nop 0\n\tglobal_load_lds_dword $0, $2"
_ASM_CONSTRAINTS = "v,s,s,~{m0},~{memory}"


# ---------------------------------------------------------------------------
# host model (no IR)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DmaBatch:
    """The DMA instructions that stage one q tile, and their wave split.

    Q and dO are ``block_m x head_size`` 16-bit tiles of ``pieces`` 1 KiB
    instructions each (``("q", j)``, ``("do", j)``); lse2 and Dsum are one
    ``dword`` instruction each (64 lanes, one row per lane). Instruction ``k``
    of :attr:`instructions` is issued by wave ``k mod waves``.
    """

    block_m: int
    head_size: int
    waves: int
    wave_size: int = 64
    elem_bytes: int = 2

    def __post_init__(self) -> None:
        nbytes = self.block_m * self.head_size * self.elem_bytes
        if self.wave_size != 64:
            raise ValueError("the LDS DMA ring is wave64 only")
        if nbytes % DMA_PIECE_BYTES:
            raise ValueError(
                f"a {self.block_m} x {self.head_size} tile is not whole 1 KiB DMA pieces"
            )
        if self.block_m > self.wave_size or self.block_m % 16:
            raise ValueError("the stats DMA needs block_m <= 64, a multiple of 16")
        if self.waves < 1:
            raise ValueError("waves must be positive")

    @property
    def pieces(self) -> int:
        """1 KiB instructions per tensor (Q or dO)."""
        return self.block_m * self.head_size * self.elem_bytes // DMA_PIECE_BYTES

    @cached_property
    def instructions(self) -> tuple[tuple[str, int], ...]:
        q = tuple(("q", j) for j in range(self.pieces))
        do = tuple(("do", j) for j in range(self.pieces))
        return q + do + tuple((s, 0) for s in _STATS)

    def of_wave(self, wave: int) -> tuple[tuple[str, int], ...]:
        return tuple(
            ins for k, ins in enumerate(self.instructions) if k % self.waves == wave
        )

    @property
    def min_per_wave(self) -> int:
        """The smallest number of instructions any wave issues per tile."""
        return min(len(self.of_wave(w)) for w in range(self.waves))


@dataclass(frozen=True)
class RingWaits:
    """``s_waitcnt vmcnt`` immediates of the DMA ring (None: no wait there)."""

    prologue: int | None
    step_top: int | None
    step_end: int | None


def dma_wait_counts(ring_depth: int, n_qd: int, n_a: int) -> RingWaits:
    """Wait immediates of the DMA ring (see the module docstring).

    ``n_qd``: the guaranteed DMA instructions per wave of one tile; ``n_a``:
    the dQ atomics per lane of one q step.
    """
    if ring_depth not in (1, 2, 3) or n_qd < 0 or n_a < 0:
        raise ValueError(f"bad ring wait inputs {(ring_depth, n_qd, n_a)}")

    def clamp(n: int) -> int:
        return min(n, RING_VMCNT_MAX)

    if ring_depth == 1:
        return RingWaits(prologue=None, step_top=0, step_end=None)
    if ring_depth == 2:
        return RingWaits(prologue=0, step_top=None, step_end=clamp(n_a))
    return RingWaits(prologue=clamp(n_qd), step_top=None, step_end=clamp(n_qd + n_a))


def ring_slot_schedule(ring_depth: int, path: str) -> dict:
    """Slot arithmetic of the ring, as offsets from the loop counter ``it``.

    ``read``: the slot step ``it`` reads; ``fill``: the slot the step's
    prefetch writes (DMA: issued at the top; registers: stored at the end);
    ``ahead``: how many steps ahead of ``it`` the prefetched tile is.
    """
    if ring_depth == 1:
        if path != "dma":
            raise ValueError("ring_depth 1 on the register path is not a ring")
        return {"read": 0, "fill": 0, "ahead": 0}
    ahead = ring_depth - 1
    fill = ahead if path == "dma" else 1
    return {"read": 0, "fill": fill, "ahead": ahead}


def dma_lane_source(b, layout: DmaSlabLayout, piece: int, lane):
    """``(row, col)`` of the 16-byte vector lane ``lane`` of DMA piece
    ``piece`` must fetch, so that it lands at slot ``64 piece + lane`` of the
    slab layout (equal to ``layout.dma_source(64 * piece + lane)``).

    ``b`` is an ``IRBuilder`` or a duck-typed integer builder; ``lane`` an
    i32 value (or an int). The result is ``row = r_j + r_l``, ``col = c_j +
    c_l``: a static part per piece plus a lane-only part (rows are a multiple
    of 8, so the line of every lane is ``l >> 3`` modulo 8).
    """
    eb = layout.elem_bytes
    chunk_elems = DmaSlabLayout.CHUNK // eb
    lanes_per_line = LDS_LINE_BYTES // DmaSlabLayout.CHUNK  # 8
    lines = DMA_PIECE_BYTES // LDS_LINE_BYTES  # 8 lines per piece
    if layout.rows % lines:
        raise ValueError("DMA slab tiles need a multiple of 8 rows")
    l_line = b.lshr(lane, b.const_i32(3))
    chunk = b.xor(b.land(lane, b.const_i32(lanes_per_line - 1)), l_line)
    if layout.row_bytes >= LDS_LINE_BYTES:
        line_elems = LDS_LINE_BYTES // eb
        r_j = (lines * piece) % layout.rows
        c_j = (lines * piece) // layout.rows * line_elems
        row = b.add(l_line, b.const_i32(r_j)) if r_j else l_line
        col = b.mul(chunk, b.const_i32(chunk_elems))
        return row, (b.add(col, b.const_i32(c_j)) if c_j else col)
    fold = layout.fold
    cpr = layout.row_bytes // DmaSlabLayout.CHUNK  # chunks per (short) row
    # line = 8 piece + l >> 3; row = fold * line + chunk / cpr; col = chunk % cpr
    r_j = fold * lines * piece
    row = b.add(
        b.mul(l_line, b.const_i32(fold)),
        b.lshr(chunk, b.const_i32(cpr.bit_length() - 1)),
    )
    row = b.add(row, b.const_i32(r_j)) if r_j else row
    col = b.mul(b.land(chunk, b.const_i32(cpr - 1)), b.const_i32(chunk_elems))
    return row, col


# ---------------------------------------------------------------------------
# emission
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PrefetchTile:
    """Where one q step's Q / dO (and stats rows) come from."""

    q0: Value
    h: Value
    q_slab: RowSlab | None
    do_slab: RowSlab | None
    packed: PackedStep | None = None


class LdsStatsView:
    """lse2 / Dsum of one q step from its ring slot (the
    :class:`._attention_bwd_body.StatsView` interface).

    DMA path (``rows_left`` given): lane ``l`` of the stats copy fetched
    workspace row ``min(l, rows_left - 1)`` (a copy cannot select), so a row at
    or past ``rows_left`` reads the neutral value here (``+inf`` / ``0``), the
    select the workspace read applies. Register path (``rows_left`` None):
    the slot already holds the selected values (the workspace read, or the
    packed-row read, ran before the store).
    """

    def __init__(self, lse2: LdsView, dsum: LdsView, rows_left: Value | None) -> None:
        self.lse2, self.dsum, self.rows_left = lse2, dsum, rows_left

    def _row(self, b, r):
        lse2 = b.vec_extract(self.lse2.load(b, r, 0, dtype=F32, n=1), 0)
        dsum = b.vec_extract(self.dsum.load(b, r, 0, dtype=F32, n=1), 0)
        return lse2, dsum

    def rows(self, b: IRBuilder, h, q0, rows) -> list[tuple[Value, Value]]:
        inf, zero = b.const_f32(_INF), b.const_f32(0.0)
        out = []
        for r in rows:
            lse2, dsum = self._row(b, r)
            if self.rows_left is not None:
                ok = b.cmp_lt(r, self.rows_left)
                lse2, dsum = b.select(ok, lse2, inf), b.select(ok, dsum, zero)
            out.append((lse2, dsum))
        return out

    def packed_rows(self, b, packed, rows):
        if self.rows_left is not None:
            raise ValueError("the DMA ring does not stage head-packed steps")
        return [self._row(b, r) for r in rows]


class QRing:
    """Per-CTA state of the ring: slot views, lane terms, the wait model.

    Constructed only for ``(global_path, ring_depth) != ("vgpr", 1)``; the
    constructor emits the CTA-invariant lane terms (DMA) at its call site.
    """

    def __init__(
        self,
        b: IRBuilder,
        geom,
        lds: BwdLds,
        *,
        tid: Value,
        lane: Value,
        wave: Value,
        stats: StatsView,
        n_atomics: int,
    ) -> None:
        self.geom = geom
        self.path = geom.global_path
        self.depth = geom.ring_depth
        if (self.path, self.depth) == ("vgpr", 1):
            raise ValueError("(vgpr, 1) is the synchronous staging, not a ring")
        self.lds = lds
        self.tid, self.lane, self.wave = tid, lane, wave
        self.stats = stats
        self.n_atomics = n_atomics
        plan = lds.plan
        d, m0 = geom.head_size, geom.k_m0
        if m0 > geom.wave_size:
            raise ValueError("a ring stats slot holds one row per lane of wave 0")
        names = ("q", "do") + _STATS
        strides = {plan.ring_stride(n) for n in names}
        if len(strides) != 1:
            raise ValueError(f"ring slots are not evenly spaced: {strides}")
        self.stride_bytes = strides.pop()
        self.v_q = lds.view("q", d)
        self.v_do = lds.view("do", d)
        self.schedule = ring_slot_schedule(self.depth, self.path)
        self.offsets = {n: plan.offset(n, 0) for n in names}
        if min(plan.buffer(n, 0).nbytes for n in _STATS) < geom.wave_size * 4:
            raise ValueError("a stats slot must hold one fp32 per lane")
        if self.path == "dma":
            check_lds_dma_width(geom.arch, 4)
            check_lds_dma_width(geom.arch, 1)
            if geom.head_pack or geom.stage_vec != 8:
                raise ValueError("the DMA ring needs stage_vec = 8 and no head packing")
            self.batch = DmaBatch(m0, d, geom.waves, geom.wave_size)
            self.waits = dma_wait_counts(self.depth, self.batch.min_per_wave, n_atomics)
            self.layout = plan.layout("q")
            if not isinstance(self.layout, DmaSlabLayout):
                raise ValueError("DMA ring buffers must use the slab layout")
            addr = b.trunc(b.smem_addr_of(lds.smem), I32)
            self.lds_base = b.to_sgpr_u32(addr)
            self.lane_src = dma_lane_source(b, self.layout, 0, lane)
        else:
            self.batch = self.waits = None

    def check_step(self, table) -> None:
        """The step's count model (a ``StageTable``) keeps the DMA wait
        immediates sound: a wave issues at least ``n_atomics`` VMEM
        operations after its copy and before the step's wait (the step's own
        dQ atomics in D, or under ``stage_table_atomics`` the previous
        step's in A: as many, zeros on the first step). Raises
        ``ValueError`` otherwise."""
        if self.path != "dma" or self.depth == 1:
            return  # R = 1 waits vmcnt(0); the register ring's waits are the compiler's
        after = table.vm_after_prefetch()
        if after < self.n_atomics:
            raise ValueError(
                f"the step issues {after} VMEM operations after its DMA copy; the "
                f"ring's wait immediates assume at least {self.n_atomics}"
            )

    # ----- slots ----------------------------------------------------------
    def slot_of(self, b: IRBuilder, it: Value, offset: int) -> Value | None:
        """Slot index ``(it + offset) mod R`` (``None`` for one slot)."""
        if self.depth == 1:
            return None
        x = b.add(it, b.const_i32(offset)) if offset else it
        if self.depth == 2:
            return b.land(x, b.const_i32(1))
        return b.mod(x, b.const_i32(self.depth))

    def _slot_elems(self, b, slot):
        if slot is None:
            return None
        if isinstance(slot, int):
            return b.const_i32(slot * self.stride_bytes // 2) if slot else None
        return b.mul(slot, b.const_i32(self.stride_bytes // 2))

    def views(self, b: IRBuilder, slot) -> tuple[LdsView, LdsView]:
        """Q and dO views of ring slot ``slot`` (int, i32 value or None)."""
        se = self._slot_elems(b, slot)
        if se is None:
            return self.v_q, self.v_do
        return (
            dataclasses.replace(self.v_q, shift=se),
            dataclasses.replace(self.v_do, shift=se),
        )

    def _stats_views(self, b, slot) -> list[LdsView]:
        plan = self.lds.plan
        se = self._slot_elems(b, slot)
        out = []
        for n in _STATS:
            buf = plan.buffer(n, 0)
            out.append(
                LdsView(self.lds.smem, buf.offset // 2, 1, buf.nbytes // 4, 2, shift=se)
            )
        return out

    def lds_stats(self, b: IRBuilder, slot, tile: PrefetchTile) -> LdsStatsView:
        """The stats of ``tile`` from ring slot ``slot``."""
        lse2, dsum = self._stats_views(b, slot)
        rows_left = None
        if self.path == "dma":
            rows_left = self.stats.lse2.slab(tile.h, tile.q0).rows_left
        return LdsStatsView(lse2, dsum, rows_left)

    # ----- DMA ------------------------------------------------------------
    def issue_dma(self, b: IRBuilder, tile: PrefetchTile, slot) -> None:
        """Issue the DMA batch of ``tile`` into ring slot ``slot``."""
        g = self.geom
        slot_bytes = None
        if slot is not None and not (isinstance(slot, int) and slot == 0):
            s = b.const_i32(slot) if isinstance(slot, int) else slot
            slot_bytes = b.mul(s, b.const_i32(self.stride_bytes))
        ws = {
            "lse2": self.stats.lse2.slab(tile.h, tile.q0),
            "dsum": self.stats.dsum.slab(tile.h, tile.q0),
        }
        slabs = {"q": tile.q_slab, "do": tile.do_slab}
        for w in range(min(g.waves, len(self.batch.instructions))):
            mine = self.batch.of_wave(w)
            if g.waves == 1:
                self._emit_pieces(b, mine, tile, slabs, ws, slot_bytes)
                continue
            with b.scf_if(b.cmp_eq(self.wave, b.const_i32(w))):
                self._emit_pieces(b, mine, tile, slabs, ws, slot_bytes)

    def _m0(self, b, name: str, piece: int, slot_bytes):
        off = self.offsets[name] + piece * DMA_PIECE_BYTES
        m0 = b.add(self.lds_base, b.const_i32(off)) if off else self.lds_base
        return b.add(m0, slot_bytes) if slot_bytes is not None else m0

    def _emit_pieces(self, b, pieces, tile, slabs, ws, slot_bytes) -> None:
        row_l, col_l = self.lane_src
        for name, j in pieces:
            m0 = self._m0(b, name, j, slot_bytes)
            if name in _STATS:
                slab: WsSlab = ws[name]
                voff = b.shl(slab.index(self.lane, 0), b.const_i32(2))
                ptr = slab.sgpr_ptr()
                b.inline_asm(_ASM_X1, _ASM_CONSTRAINTS, [voff, m0, ptr], None)
                continue
            rslab: RowSlab = slabs[name]
            r_j, c_j = dma_lane_source(HostInt(), self.layout, j, 0)
            row = b.add(tile.q0, b.add(row_l, b.const_i32(r_j)) if r_j else row_l)
            col = b.add(col_l, b.const_i32(c_j)) if c_j else col_l
            voff = b.shl(rslab.index(row, col), b.const_i32(1))
            ptr = rslab.sgpr_ptr()
            b.inline_asm(_ASM_X4, _ASM_CONSTRAINTS, [voff, m0, ptr], None)

    # ----- prologue / drain ----------------------------------------------
    def prologue(self, b: IRBuilder, trip: Value, tiles, *, name: str) -> list[Value]:
        """Stage tiles ``0 .. R - 2`` of a q loop (only when ``trip > 0``),
        then an LDS-only barrier; returns the registers the loop carries in
        (register ring, ``R = 3``: tile 1), else ``[]``. ``tiles`` holds
        ``R - 1`` tiles (a missing step is the first step again)."""
        if len(tiles) != max(self.depth - 1, 0):
            raise ValueError(
                f"the prologue of ring depth {self.depth} takes R - 1 tiles"
            )
        zero = b.const_i32(0)
        has = b.cmp_lt(zero, trip)
        out: list[Value] = []
        if self.path == "dma":
            if self.depth > 1:
                with b.scf_if(has):
                    for k, t in enumerate(tiles):
                        self.issue_dma(b, t, k)
                    b.s_waitcnt(vmcnt=self.waits.prologue)
                b.sync_lds_only()
            return out
        if self.depth == 2:
            with b.scf_if(has):
                self.store_regs(b, self.load_regs(b, tiles[0]), 0)
            b.sync_lds_only()
            return out
        # R = 3: tile 0 into slot 0, tile 1 carried out of a 0 / 1-trip loop
        one = b.const_i32(1)
        init = [b.zero_vec(self.lds.elem, 8) for _ in range(self.n_reg_chunks)]
        init += [b.const_f32(0.0) for _ in _STATS]
        loop = b.scf_for_iter(
            zero, b.select(has, one, zero), one,
            [(f"{name}pr{i}", v) for i, v in enumerate(init)],
            iv_name=f"{name}pit", elide_trailing_barrier=False,
        )  # fmt: skip
        with loop as (_iv, _cur):
            self.store_regs(b, self.load_regs(b, tiles[0]), 0)
            b.scf_yield(*self.load_regs(b, tiles[1]))
        b.sync_lds_only()
        return list(loop.results)

    def drain(self, b: IRBuilder) -> None:
        """After a DMA loop: no copy stays in flight into the aliased LDS."""
        if self.path == "dma":
            b.s_waitcnt(vmcnt=0)
            b.sync_lds_only()

    @property
    def n_reg_chunks(self) -> int:
        """Register vectors (8 elements) per thread of one tile (Q and dO);
        the register set adds one lse2 and one Dsum value."""
        g = self.geom
        total = g.k_m0 * g.head_size // 8
        if self._joint(total):
            return 1
        return 2 * -(-total // g.threads)

    def _joint(self, total: int) -> bool:
        g = self.geom
        return not self._packed and total < g.threads and total % g.wave_size == 0

    # ----- registers ------------------------------------------------------
    def load_regs(self, b: IRBuilder, tile: PrefetchTile) -> list[Value]:
        """Global loads of ``tile``'s Q / dO chunks owned by this thread,
        then its stats row (thread ``t`` holds row ``min(t, kM0 - 1)``)."""
        regs = [v for _t, _r, _c, v, _g in self._reg_chunks(b, tile, load=True)]
        return regs + self._load_stats_row(b, tile)

    def _load_stats_row(self, b, tile) -> list[Value]:
        """lse2 / Dsum of row ``min(tid, kM0 - 1)`` of ``tile``, with the
        neutral values of :class:`StatsView` (rows past the workspace) and,
        when packing, of a dead packed row."""
        r = b.smin(self.tid, b.const_i32(self.geom.k_m0 - 1))
        if tile.packed is None:
            row = self.stats.rows(b, tile.h, tile.q0, [r])[0]
        else:
            row = self.stats.packed_rows(b, tile.packed.rows, [r])[0]
        return list(row)

    def store_regs(self, b: IRBuilder, regs, slot) -> None:
        """``ds_write`` the registers of :meth:`load_regs` into slot ``slot``."""
        v_q, v_do = self.views(b, slot)
        views = {"q": v_q, "do": v_do}
        chunks = self._reg_chunks(b, None, load=False)
        if len(chunks) + len(_STATS) != len(regs):
            raise ValueError("register set does not match the chunk plan")
        stats_regs = regs[len(chunks) :]
        with b.scf_if(b.cmp_lt(self.tid, b.const_i32(self.geom.k_m0))):
            for view, val in zip(self._stats_views(b, slot), stats_regs):
                view.store(b, self.tid, 0, val, 1)
        for (t, r, c, _v, guard), val in zip(chunks, regs):
            if t == "joint":
                idx = b.select(
                    guard[1], v_do.index(b, r, c), v_q.index(b, r, c)
                )  # wave-uniform tensor choice
                cond = guard[0]
                if cond is None:
                    b.smem_store_vN(self.lds.smem, [idx], val, 8)
                else:
                    with b.scf_if(cond):
                        b.smem_store_vN(self.lds.smem, [idx], val, 8)
                continue
            if guard is None:
                views[t].store(b, r, c, val, 8)
            else:
                with b.scf_if(guard):
                    views[t].store(b, r, c, val, 8)

    def _reg_chunks(self, b, tile, *, load: bool):
        """``(tensor, row, col, value, guard)`` of every 8-element chunk this
        thread moves. Joint distribution (one pass over the Q and dO chunks)
        when a tile has fewer chunks than the CTA has threads and the tiles
        are slabs; else each tensor in turn (a tail chunk loads the tile's
        last chunk, only its store is predicated)."""
        g = self.geom
        d, m0, threads = g.head_size, g.k_m0, g.threads
        per_row = d // 8
        total = m0 * per_row
        packed = self._packed
        if tile is not None and bool(tile.packed is not None) != packed:
            raise ValueError("a packed tile on an unpacked ring (or the reverse)")
        out = []
        if self._joint(total):
            # threads [0, total) move Q, [total, 2 total) move dO
            is_do = b.cmp_ge(self.tid, b.const_i32(total))
            chunk = b.sub(self.tid, b.select(is_do, b.const_i32(total), b.const_i32(0)))
            guard = None
            if 2 * total < threads:
                guard = b.cmp_lt(self.tid, b.const_i32(2 * total))
                chunk = b.smin(chunk, b.const_i32(total - 1))
            r, c = _chunk_rc(b, chunk, per_row)
            val = None
            if load:
                q, do = tile.q_slab, tile.do_slab
                stride = b.select(is_do, as_i32(b, do.t_stride), as_i32(b, q.t_stride))
                joint = dataclasses.replace(
                    q, ptr=b.select(is_do, do.ptr, q.ptr), t_stride=stride
                )
                val = slab_row_loader(b, joint, tile.q0)(r, c)
            return [("joint", r, c, val, (guard, is_do))]
        for t in ("q", "do"):
            if load:
                if packed:
                    addr = tile.packed.q_addr if t == "q" else tile.packed.do_addr
                    loader = packed_row_loader(b, addr, tile.packed.rows)
                else:
                    loader = slab_row_loader(
                        b, tile.q_slab if t == "q" else tile.do_slab, tile.q0
                    )
            for it in range(-(-total // threads)):
                chunk = (
                    self.tid if it == 0 else b.add(self.tid, b.const_i32(it * threads))
                )
                tail = (it + 1) * threads > total
                src = b.smin(chunk, b.const_i32(total - 1)) if tail else chunk
                r, c = _chunk_rc(b, src, per_row)
                guard = b.cmp_lt(chunk, b.const_i32(total)) if tail else None
                out.append((t, r, c, loader(r, c) if load else None, guard))
        return out

    @property
    def _packed(self) -> bool:
        return bool(self.geom.head_pack)


def _chunk_rc(b, chunk, per_row):
    if per_row == 1:
        return chunk, b.const_i32(0)
    shift = per_row.bit_length() - 1
    if per_row != 1 << shift:
        raise ValueError("chunks per row must be a power of two")
    r = b.lshr(chunk, b.const_i32(shift))
    c = b.shl(b.land(chunk, b.const_i32(per_row - 1)), b.const_i32(3))
    return r, c


# ---------------------------------------------------------------------------
# one q step of the ring
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class PrefetchStep:
    """The ring's part of q step ``it`` (handed to ``BwdTileStep.run_q_step``).

    ``ahead``: the tile the step prefetches (step ``it + R - 1``, or the step
    itself past the loop's last trip); ``carried``: registers of tile ``it +
    1`` loaded one step earlier (register ring, ``R = 3``). After the step,
    :attr:`yields` holds the registers to carry into the next trip.
    """

    ring: QRing
    it: Value
    here: PrefetchTile
    ahead: PrefetchTile
    carried: list = dataclasses.field(default_factory=list)
    yields: list = dataclasses.field(default_factory=list)
    _regs: list = dataclasses.field(default_factory=list)

    def barrier(self, b: IRBuilder) -> None:
        b.sync_lds_only()

    def begin(self, b: IRBuilder, step, h, q0, packed, *, defer_loads: bool = False):
        """Issue the prefetch; return ``(v_q, v_do, stats, stats_src)``.

        ``defer_loads`` (register ring under a stage table): issue no load
        and no fence here; the step calls :meth:`issue_loads` in its stage A,
        where the stage table's groups spread the loads over G0's MFMAs.
        """
        ring = self.ring
        if defer_loads and ring.path == "dma":
            raise ValueError("the DMA copy is issued at the step top")
        read = ring.slot_of(b, self.it, ring.schedule["read"])
        v_q, v_do = ring.views(b, read)
        if ring.path == "dma":
            if ring.depth == 1:
                ring.issue_dma(b, self.here, None)
                b.s_waitcnt(vmcnt=ring.waits.step_top)
                b.sync_lds_only()
            else:
                ring.issue_dma(
                    b, self.ahead, ring.slot_of(b, self.it, ring.schedule["fill"])
                )
            return v_q, v_do, None, ring.lds_stats(b, read, self.here)
        # register ring: the prefetch loads (Q / dO and the stats rows of a
        # later step) are the only global loads of the step, so the
        # compiler's one wait for them sits at their slot stores, after this
        # step's dQ atomics (in-order VM_CNT: the atomics stay in flight).
        # The fence keeps the scheduler from sinking them toward the stores
        # (under a stage table the fence after stage A does).
        if not defer_loads:
            self.issue_loads(b)
            b.sched_barrier(0)
        return v_q, v_do, None, ring.lds_stats(b, read, self.here)

    def issue_loads(self, b: IRBuilder) -> None:
        """The register ring's global loads of the prefetched tile."""
        if self.ring.path == "dma" or self._regs:
            raise ValueError("the register prefetch is issued once per step")
        self._regs = self.ring.load_regs(b, self.ahead)

    def end(self, b: IRBuilder) -> None:
        ring = self.ring
        if ring.path == "dma":
            if ring.waits.step_end is not None:
                b.s_waitcnt(vmcnt=ring.waits.step_end)
            b.sync_lds_only()
            return
        fill = ring.slot_of(b, self.it, ring.schedule["fill"])
        if ring.depth == 2:
            ring.store_regs(b, self._regs, fill)
        else:
            ring.store_regs(b, self.carried, fill)
            self.yields = list(self._regs)
        b.sync_lds_only()
