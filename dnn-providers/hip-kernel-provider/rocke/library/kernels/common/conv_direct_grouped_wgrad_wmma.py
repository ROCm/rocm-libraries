# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grouped wgrad for 4 and 8 channels per group on wave32 WMMA (gfx1250).

The wave32 counterpart of :mod:`conv_direct_grouped_wgrad_mfma`. gfx1250 has
no 4x4x4 batched MFMA, so the groups are packed block-diagonally into one
16x16x32 WMMA instead: a wave owns 16 channels -- four groups of 4 or two of
8 -- and, per tap ``(r, s)`` and 32 output columns ``q0 ..``,

    A[o][k] = dY[y + PAD - r][q0 + k][o]          (M = 16 dY channels)
    B[k][c] = X[y][q0 + k + s - PAD][c]           (N = 16 X channels)
    C[o][c] += sum_k A[o][k] B[k][c]

The diagonal 4x4 / 8x8 blocks of ``C`` (``o`` and ``c`` in the same group) are
the group's ``dW[o][r][s][c]`` over those columns; the rest is cross-group
and dropped. That wastes 3/4 (cpg 4) or 1/2 (cpg 8) of the WMMA, which these
layers can afford: they are bound by moving X and dY, not by the math.

Both operands want a lane per channel and 16 consecutive pixels per lane,
the transpose of NHWC. The rows are staged in LDS as ``[column][channel]``
and ``ds_load_tr16_b128`` does the transpose on the read; the tap shift
``s`` is a whole-pixel offset of the read address.

A block of ``waves_c`` waves owns ``16 * waves_c`` channels and
``32 * q_steps`` output columns and walks (image, row chunk) units. Per input
row the block copies the X row and the dY row ``y + PAD`` into LDS
(``global_load_async_to_lds`` into a ring of ``stages`` buffers), and the dY
fragments of the ``KH - 1`` previous rows stay in a register ring. Every lane
issues every copy, so each wave counts the same copies per row and
``s_wait_asynccnt`` can leave the younger rows in flight: a lane whose chunk
is padding copies a valid address into a dummy slot past the buffer and
zero-fills its own slot with an LDS store. The waves own disjoint channels, so each stores its own
accumulators into one partial slot ``ws[part][tap][o]`` (``tap = (r * KW + s)
* cpg + c``) that ``DwWgradFoldSpec`` folds into dW.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import F32, IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    dilation_reason,
    emit_direct_params,
)

_WAVE = 32
_TILE = 16  # channels per wave (WMMA M = N)
_KSTEP = 32  # output columns per WMMA (WMMA K)
_PAD_CH = 8  # LDS row padding (channels) against bank conflicts
_DUMMY = 8  # elements past each buffer: the padding lanes' copy target


@dataclass(frozen=True)
class GroupedWgradWmmaSpec:
    problem: DirectConvProblem
    name: str = "grouped_wgrad_wmma"
    waves_c: int = 4  # waves along the channels, 16 each
    q_steps: int = 2  # 32-column WMMA steps per block
    rows_per_chunk: int = 32  # input rows per unit
    target_blocks: int = 1024
    stages: int = 2  # LDS buffers (rows in flight + 1)

    @property
    def threads(self) -> int:
        return self.waves_c * _WAVE

    @property
    def block_c(self) -> int:
        return _TILE * self.waves_c

    @property
    def block_q(self) -> int:
        return _KSTEP * self.q_steps

    def x_cols(self) -> int:
        return self.block_q + self.problem.KW - 1

    def pitch(self) -> int:
        return self.block_c + _PAD_CH

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        flags = {"bf16": True} if p.dtype == "bf16" else {}
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}p{p.PAD}s{p.stride}c{p.cpg}",
            f"wc{self.waves_c}",
            f"qs{self.q_steps}",
            f"rpc{self.rows_per_chunk}",
            f"tb{self.target_blocks}",
            f"st{self.stages}",
            flags=flags,
        )

    def n_channel_tiles(self) -> int:
        return ceil_div(self.problem.total_c, self.block_c)

    def n_q(self) -> int:
        return ceil_div(self.problem.Wo, self.block_q)

    def n_units(self) -> int:
        p = self.problem
        return p.N * ceil_div(p.H, self.rows_per_chunk)

    def n_z(self) -> int:
        planes = self.n_channel_tiles() * self.n_q()
        return min(self.n_units(), max(1, ceil_div(self.target_blocks, planes)))

    def n_partitions(self) -> int:
        return self.n_q() * self.n_z()

    def fold_taps(self) -> int:
        p = self.problem
        return p.KH * p.KW * p.cpg

    def fold_channels(self) -> int:
        return self.problem.total_k

    def workspace_bytes(self) -> int:
        return self.n_partitions() * self.fold_taps() * self.problem.total_k * 4

    def launch_geometry(self):
        return (self.n_channel_tiles(), self.n_q(), self.n_z()), (self.threads, 1, 1)

    def lds_bytes(self) -> int:
        return (
            self.stages
            * ((self.x_cols() + self.block_q) * self.pitch() + 2 * _DUMMY)
            * 2
        )


def is_valid_grouped_wgrad_wmma_spec(spec: GroupedWgradWmmaSpec, arch: str = "gfx1250"):
    p = spec.problem
    why = dilation_reason(p)
    if why is not None:
        return False, why
    if p.cpg != p.kpg or p.cpg not in (4, 8):
        return False, "cpg = kpg in (4, 8)"
    if p.dtype not in ("fp16", "bf16"):
        return False, "fp16/bf16 only"
    if p.stride != 1:
        return False, "stride 1 only"
    if p.PAD > p.KH - 1 or p.PAD > p.KW - 1:
        return False, "PAD <= K - 1"
    if p.total_c % _TILE:
        return False, "C must be a multiple of 16 (whole packed tiles)"
    if spec.waves_c not in (1, 2, 4, 8):
        return False, "waves_c in (1, 2, 4, 8)"
    if spec.q_steps not in (1, 2, 3, 4):
        return False, "q_steps in 1..4"
    if spec.rows_per_chunk < 1:
        return False, "rows_per_chunk >= 1"
    if spec.target_blocks < 1:
        return False, "target_blocks >= 1"
    if spec.stages not in (2, 3, 4):
        return False, "stages in 2..4"
    if spec.lds_bytes() > 160 * 1024:
        return False, f"LDS {spec.lds_bytes()} B > 160 KiB"
    if arch != "gfx1250":
        return False, "ds_load_tr16_b128 and wmma 16x16x32 need gfx1250"
    return True, "ok"


def build_grouped_wgrad_wmma(
    spec: GroupedWgradWmmaSpec, arch: str = "gfx1250"
) -> KernelDef:
    ok, why = is_valid_grouped_wgrad_wmma_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid GroupedWgradWmmaSpec: {why}")
    p = spec.problem
    KH, KW, PAD, CPG = p.KH, p.KW, p.PAD, p.cpg
    TAPS = spec.fold_taps()
    QS = spec.q_steps
    BC, BQ = spec.block_c, spec.block_q
    XC = spec.x_cols()
    PITCH = spec.pitch()
    T = spec.threads
    D = spec.stages
    PRE = KH - 1  # dY rows a unit needs before its first input row
    CH8 = BC // 8  # 16-byte chunks per staged pixel
    io_type = _io_type(p.dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = T
    b.kernel.attrs["lds_alias_scopes"] = True
    params = emit_direct_params(b, direction="wgrad", io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1 = ci(0), ci(1)
    p_N = params["p_N"]
    p_C = params["p_total_c"]
    p_K = params["p_total_k"]
    p_H, p_W = params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    sdy_n, sdy_h, sdy_w = (
        params["p_dY_stride_n"],
        params["p_dY_stride_ho"],
        params["p_dY_stride_wo"],
    )
    sx_n, sx_h, sx_w = (
        params["p_X_stride_n"],
        params["p_X_stride_hi"],
        params["p_X_stride_wi"],
    )
    dy_ptr, x_ptr = params["A"], params["B"]

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    ctile, qtile, zslot = b.block_id_x(), b.block_id_y(), b.block_id_z()
    c_blk0 = b.mul(ctile, ci(BC))
    q0 = b.mul(qtile, ci(BQ))
    n_q = b.div(b.add(p_Wo, ci(BQ - 1)), ci(BQ))
    n_chunks = b.div(b.add(p_H, ci(spec.rows_per_chunk - 1)), ci(spec.rows_per_chunk))
    n_units = b.mul(p_N, n_chunks)
    # Same formula as GroupedWgradWmmaSpec.n_z, which sizes the grid.
    n_ct = b.div(b.add(p_C, ci(BC - 1)), ci(BC))
    planes = b.mul(n_ct, n_q)
    n_z = b.smin(
        n_units,
        b.smax(b.div(b.add(ci(spec.target_blocks), b.sub(planes, c1)), planes), c1),
    )

    # ---- LDS: [column][channel] rows, one allocation per buffer ------------
    X_DUMMY, Y_DUMMY = XC * PITCH, BQ * PITCH
    x_stage = [
        b.smem_alloc(io_type, [1, X_DUMMY + _DUMMY], name_hint=f"xs{i}")
        for i in range(D)
    ]
    y_stage = [
        b.smem_alloc(io_type, [1, Y_DUMMY + _DUMMY], name_hint=f"ys{i}")
        for i in range(D)
    ]

    # ---- copy slots: slot = i * T + tid -> (column, 8-channel chunk) -------
    def copy_slots(cols: int):
        out = []
        for i in range(ceil_div(cols * CH8, T)):
            slot = b.add(ci(i * T), tid)
            col = b.div(slot, ci(CH8))
            ch = b.mul(b.mod(slot, ci(CH8)), ci(8))
            partial = (i + 1) * T > cols * CH8  # the last pass has idle lanes
            out.append(
                (
                    col,
                    b.add(b.mul(col, ci(PITCH)), ch),
                    b.cmp_lt(b.add(c_blk0, ch), p_C),
                    b.cmp_lt(col, ci(cols)) if partial else None,
                    b.add(c_blk0, ch),
                )
            )
        return out

    def slot_ok(ok: Value, in_tile):
        return ok if in_tile is None else b.land(ok, in_tile)

    x_slots = []
    for col, lds_i, ok_c, in_tile, cc in copy_slots(XC):
        wi = b.add(b.sub(q0, ci(PAD)), col)
        ok = b.land(ok_c, b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, p_W)))
        x_slots.append(
            (lds_i, slot_ok(ok, in_tile), in_tile, b.add(b.mul(wi, sx_w), cc))
        )
    y_slots = []
    for col, lds_i, ok_c, in_tile, cc in copy_slots(BQ):
        wo = b.add(q0, col)
        ok = b.land(ok_c, b.cmp_lt(wo, p_Wo))
        y_slots.append(
            (lds_i, slot_ok(ok, in_tile), in_tile, b.add(b.mul(wo, sdy_w), cc))
        )
    PER_STEP = len(x_slots) + len(y_slots)  # copies per row step, every lane

    zero8 = b.zero_vec(io_type, 8)

    def copy_row(src, stage, dummy: int, slots, row_base: Value, row_ok: Value):
        for lds_i, ok, in_tile, off in slots:
            live = b.land(ok, row_ok)
            b.global_load_async_to_lds(
                src,
                b.select(live, b.add(row_base, off), c0),
                stage,
                [c0, b.select(live, lds_i, ci(dummy))],
                width_bytes=16,
            )
            pad = b.lnot(live) if in_tile is None else b.land(in_tile, b.lnot(live))
            with b.scf_if(pad):
                b.smem_store_vN(stage, [c0, lds_i], zero8, 8)

    # ---- fragment reads: ds_load_tr16_b128 of a 16 (column) x 16 (channel)
    # block of [column][channel]. Lane l reads column (l // 16) * 16 + l % 8
    # (+ 8 for the high half) at channels (l % 16) // 8 * 8 ..; after the
    # transpose it holds channel l % 16 at columns (l // 16) * 16 + 0 .. 15,
    # the WMMA A / B layout with K along the columns.
    l16 = b.mod(lane, ci(16))
    lhalf = b.div(lane, ci(16))
    rd_base = b.add(
        b.mul(b.add(b.mul(lhalf, ci(16)), b.mod(lane, ci(8))), ci(PITCH)),
        b.add(b.mul(wave, ci(_TILE)), b.mul(b.div(l16, ci(8)), ci(8))),
    )

    def frag(stage, col: int) -> Value:
        addr = b.add(rd_base, ci(col * PITCH))
        lo = b.ds_read_tr16_b128(stage, c0, addr, dtype=io_type)
        hi = b.ds_read_tr16_b128(stage, c0, b.add(addr, ci(8 * PITCH)), dtype=io_type)
        return b.vec_concat(lo, hi)

    def wmma(a: Value, bb: Value, acc: Value) -> Value:
        if p.dtype == "bf16":
            return b.wmma_gfx1250_f32_16x16x32_bf16(a, bb, acc)
        return b.wmma_gfx1250_f32_16x16x32_f16(a, bb, acc)

    zero_acc = b.zero_vec_f32(8)
    zero_frag = b.zero_vec(io_type, 16)
    n_acc = KH * KW
    # The carried ring holds the dY fragments of the KH - 1 rows before the
    # current one, newest first: [row][q].
    n_ring = (KH - 1) * QS

    def pack(accs, ring):
        return list(accs) + [v for row in ring for v in row]

    def unpack(vals):
        accs = list(vals[:n_acc])
        flat = list(vals[n_acc:])
        return accs, [flat[j * QS : (j + 1) * QS] for j in range(KH - 1)]

    unit_loop = b.scf_for_iter(
        zslot,
        n_units,
        n_z,
        [(f"acc{i}", zero_acc) for i in range(n_acc)]
        + [(f"ring{i}", zero_frag) for i in range(n_ring)],
        iv_name="unit",
        elide_trailing_barrier=False,
    )
    with unit_loop as (unit, carried):
        accs, ring = unpack(list(carried))
        n_img = b.mod(unit, p_N)
        chunk = b.div(unit, p_N)
        y0 = b.mul(chunk, ci(spec.rows_per_chunk))
        n_rows = b.smin(ci(spec.rows_per_chunk), b.sub(p_H, y0))
        n_steps = b.add(n_rows, ci(PRE))
        x_img = b.mul(n_img, sx_n)
        y_img = b.mul(n_img, sdy_n)

        def issue(buf: int, t: Value):
            """Copy step t: X row y0 - PRE + t (steps >= PRE) and dY row y + PAD."""
            y = b.add(b.sub(y0, ci(PRE)), t)
            x_ok = b.land(b.cmp_ge(t, ci(PRE)), b.cmp_lt(t, n_steps))
            copy_row(
                x_ptr,
                x_stage[buf],
                X_DUMMY,
                x_slots,
                b.add(x_img, b.mul(y, sx_h)),
                x_ok,
            )
            pr = b.add(y, ci(PAD))
            y_ok = b.land(
                b.land(b.cmp_ge(pr, c0), b.cmp_lt(pr, p_Ho)), b.cmp_lt(t, n_steps)
            )
            copy_row(
                dy_ptr,
                y_stage[buf],
                Y_DUMMY,
                y_slots,
                b.add(y_img, b.mul(pr, sdy_h)),
                y_ok,
            )

        def step(t: Value, buf: int, accs, ring, do_compute: bool):
            """Publish step t, copy step t + D - 1, multiply step t."""
            # Steps t .. t + D - 2 are in flight: wait for step t's copies,
            # then the barrier publishes them (and the zero fills) and tells
            # every wave the buffer of step t - 1 is free.
            b.s_wait_asynccnt((D - 2) * PER_STEP)
            b.sync_lds_only()
            issue((buf + D - 1) % D, b.add(t, ci(D - 1)))
            newest = [frag(y_stage[buf], _KSTEP * q) for q in range(QS)]
            full = [newest] + list(ring)  # full[r] = the dY row of tap r
            if do_compute:
                accs = list(accs)
                for s in range(KW):
                    for q in range(QS):
                        xb = frag(x_stage[buf], _KSTEP * q + s)
                        for r in range(KH):
                            i = r * KW + s
                            accs[i] = wmma(full[r][q], xb, accs[i])
            return accs, full[: KH - 1]

        for t in range(D - 1):
            issue(t, ci(t))
        # Prologue: the PRE dY rows the first input row pairs with.
        for t in range(PRE):
            accs, ring = step(ci(t), t % D, accs, ring, False)
        # Main: one input row per step, unrolled by D so buffers are constant.
        trips = b.smax(b.div(b.add(n_rows, ci(D - 1)), ci(D)), c1)
        row_loop = b.scf_for_iter(
            c0,
            trips,
            c1,
            [(f"a{i}", v) for i, v in enumerate(pack(accs, ring))],
            iv_name="rb",
            elide_trailing_barrier=False,
        )
        with row_loop as (rb, la):
            la_acc, la_ring = unpack(list(la))
            for u in range(D):
                t = b.add(b.add(b.mul(rb, ci(D)), ci(PRE)), ci(u))
                la_acc, la_ring = step(t, (PRE + u) % D, la_acc, la_ring, True)
            b.scf_yield(*pack(la_acc, la_ring))
        # Drain the copy issued by the last step before the next unit
        # reuses the buffers.
        b.s_wait_asynccnt(0)
        b.sync_lds_only()
        b.scf_yield(*list(row_loop.results))
    accs, _ = unpack(list(unit_loop.results))

    # ---- epilogue: each wave stores the diagonal blocks of its tile --------
    # Lane l, item e: dY channel o = 8 * (l // 16) + e, X channel c = l % 16.
    # cpg 4: lane l keeps items 4 * (g % 2) .. + 3 when g // 2 == l // 16,
    # g = c // 4; cpg 8: all 8 items when c // 8 == l // 16.
    part = b.add(b.mul(zslot, n_q), qtile)
    gc = b.div(l16, ci(CPG))
    c_in = b.mod(l16, ci(CPG))
    if CPG == 4:
        keep = b.cmp_eq(b.div(gc, ci(2)), lhalf)
        hi = b.cmp_eq(b.mod(gc, ci(2)), c1)
        o_loc = b.mul(gc, ci(4))
    else:
        keep = b.cmp_eq(gc, lhalf)
        o_loc = b.mul(lhalf, ci(8))
    o_glb = b.add(b.add(c_blk0, b.mul(wave, ci(_TILE))), o_loc)
    with b.scf_if(b.land(keep, b.cmp_lt(o_glb, p_K))):
        for r in range(KH):
            for s in range(KW):
                acc = accs[r * KW + s]
                tap = b.add(ci((r * KW + s) * CPG), c_in)
                off = b.add(b.mul(b.add(b.mul(part, ci(TAPS)), tap), p_K), o_glb)
                if CPG == 4:
                    vals = [
                        b.select(hi, b.vec_extract(acc, 4 + e), b.vec_extract(acc, e))
                        for e in range(4)
                    ]
                    b.global_store_vN(
                        params["D"], off, b.vec_pack(vals, F32), 4, align=16
                    )
                else:
                    b.global_store_vN(params["D"], off, acc, 8, align=16)
    return b.kernel
