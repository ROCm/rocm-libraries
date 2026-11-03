# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Depthwise wgrad on the batched 4x4x4 MFMA (stride 1, any filter size).

A depthwise wgrad has no channel reduction, so the MFMA's batch axis is the
only place the channels fit: ``mfma_f32_4x4x4`` runs 16 independent 4x4x4
products per wave and each takes one channel. Per input row ``y`` and per
4-column group ``q0 .. q0 + 3`` of the output, one block computes

    A[s][k] = X[y][q0 + k + s - PAD]          (Hankel: shift down the rows)
    B[k][r] = dY[y + PAD - r][q0 + k]
    C[s][r] += sum_k A[s][k] B[k][r]          = dW[c][r][s] over those columns

with the filter tiled into 4x4 tap groups (one accumulator each, lanes past
the filter clamped and dropped by the epilogue). Lane ``l`` of a wave serves
channel ``l // 4`` of the wave's 16 and supplies row ``l % 4`` of A and column
``l % 4`` of B; its accumulator holds column ``r`` and items ``s``.

Both operands want consecutive columns of one channel while NHWC keeps the
channel contiguous, so rows reach them in two steps: an LDS DMA brings the
row in NHWC order, and ``ds_read_tr16_b64`` hands lane ``L`` four consecutive
columns of channel ``L``, which it stores channel-major (X into ``main_in``,
the newest dY row into ``dnew``).

* A: the four lanes of a channel each read a quarter of its ``main_in`` row
  as aligned 16-byte words and share them with quad broadcasts; every A
  fragment is then two ``v_perm_b32`` of that window (a lane's tap shift is
  its byte selector), so no LDS read is unaligned.
* B: the KH dY rows a row pairs with stay in registers. Lane ``(g, r)``
  holds dY row ``y + PAD - 4g - r``; each step rotates the quad by one
  (``quad_perm``), carries lane 3 of group ``g - 1`` into lane 0 of ``g``,
  and takes the newest row into lane 0 of group 0.

A block owns 64 channels and a ``4 * q_tiles`` column tile (or
``unfold_n`` small images packed side by side), and walks the (images, row
chunk) units ``z, z + gridDim.z, ...``; every unit accumulates into the same
registers and the block stores one partial slot ``ws[part][tap][c]`` that
:class:`DwWgradFoldSpec` sums into dW. When a unit is a whole image the
prologue stages only the PAD dY rows below its top edge.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, Value, VectorType
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    _mfma,
    dilation_reason,
    emit_direct_params,
)

_WAVE = 64
_NW = 4  # waves per block
_BC = 64  # channels per block: 16 per wave
_TAPS = 4  # M = N of the 4x4x4 atom
_CH = _BC // 8  # 16-byte chunks per staged pixel


@dataclass(frozen=True)
class DwWgradMfmaSpec:
    problem: DirectConvProblem
    name: str = "dw_wgrad_mfma"
    q_tiles: int = 8  # output columns per block / 4
    rows_per_chunk: int = 16  # input rows per unit
    prefetch: int = 2  # staging buffers
    target_blocks: int = 2048
    # Images packed side by side in one column tile (small images), at the
    # pitch of image_pitch(): neighbours share their zero padding.
    unfold_n: int = 1

    @property
    def block_q(self) -> int:
        return 4 * self.q_tiles

    @property
    def rg(self) -> int:
        return ceil_div(self.problem.KH, _TAPS)

    @property
    def sg(self) -> int:
        return ceil_div(self.problem.KW, _TAPS)

    @property
    def preshift(self) -> bool:
        # A lane's tap shift reaches past one v_perm (3 halves) from KW 4 on.
        return min(_TAPS, self.problem.KW) - 1 > 2

    def win_words(self) -> int:
        return 2 * (self.q_tiles - 1) + 2 * (self.sg - 1) + 3

    def win_reads(self) -> int:
        return ceil_div(self.win_words() + (1 if self.preshift else 0), 4)

    def in_pitch(self) -> int:
        """main_in elements per channel: the lane's whole aligned window."""
        return 8 * self.win_reads()

    def in_groups(self) -> int:
        """4-column groups of X staged and transposed per row."""
        cols = 2 * (self.win_words() + (1 if self.preshift else 0))
        return ceil_div(cols, 4)

    def d_pitch(self) -> int:
        return self.block_q + 4

    def dnew_elems(self) -> int:
        return _BC * self.d_pitch()

    def ix(self) -> int:
        """DMA instructions per wave per staged X row."""
        return ceil_div(ceil_div(4 * self.in_groups() * _CH, _WAVE), _NW)

    def iy(self) -> int:
        return ceil_div(ceil_div(self.block_q * _CH, _WAVE), _NW)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        flags = {"bf16": True} if p.dtype == "bf16" else {}
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}p{p.PAD}s{p.stride}",
            f"qt{self.q_tiles}",
            f"rpc{self.rows_per_chunk}",
            f"pf{self.prefetch}",
            f"un{self.unfold_n}",
            f"tb{self.target_blocks}",
            flags=flags,
        )

    def n_channel_tiles(self) -> int:
        return ceil_div(self.problem.groups, _BC)

    def n_q(self) -> int:
        return ceil_div(self.problem.Wo, self.block_q)

    def image_pitch(self) -> int:
        """Columns between packed images: an output's KW taps stay inside
        its own image and padding, and the dY rows do not overlap."""
        p = self.problem
        return max(p.Wo, p.Wo + p.KW - 1 - p.PAD, p.PAD + p.W)

    def n_units(self) -> int:
        p = self.problem
        return ceil_div(p.N, self.unfold_n) * ceil_div(p.H, self.rows_per_chunk)

    def n_z(self) -> int:
        planes = self.n_channel_tiles() * self.n_q()
        return min(self.n_units(), max(1, ceil_div(self.target_blocks, planes)))

    def n_partitions(self) -> int:
        return self.n_q() * self.n_z()

    def workspace_bytes(self) -> int:
        p = self.problem
        return self.n_partitions() * p.KH * p.KW * p.groups * 4

    def launch_geometry(self):
        return (self.n_channel_tiles(), self.n_q(), self.n_z()), (_NW * _WAVE, 1, 1)

    def lds_bytes(self) -> int:
        stage = self.prefetch * _NW * (self.ix() + self.iy()) * _WAVE * 16
        return stage + (_BC * self.in_pitch() + self.dnew_elems()) * 2


def is_valid_dw_wgrad_mfma_spec(spec: DwWgradMfmaSpec, arch: str = "gfx950"):
    p = spec.problem
    why = dilation_reason(p)
    if why is not None:
        return False, why
    if p.cpg != 1 or p.kpg != 1:
        return False, "depthwise only"
    if p.dtype not in ("fp16", "bf16"):
        return False, "fp16/bf16 only"
    if p.stride != 1:
        return False, "stride 1 only"
    if p.PAD > p.KH - 1 or p.PAD > p.KW - 1:
        return False, "PAD <= K - 1"
    if p.groups % 8:
        return False, "C must be a multiple of 8"
    if not 1 <= spec.q_tiles <= 8:
        return False, "q_tiles in 1..8"
    if spec.prefetch not in (1, 2, 3, 4):
        return False, "prefetch in 1..4"
    if spec.unfold_n not in (1, 2, 3, 4):
        return False, "unfold_n in 1..4"
    if spec.unfold_n > 1 and spec.unfold_n * spec.image_pitch() > spec.block_q:
        return False, (
            f"{spec.unfold_n} images at pitch {spec.image_pitch()} exceed the "
            f"{spec.block_q}-column tile"
        )
    if spec.rows_per_chunk < 1:
        return False, "rows_per_chunk >= 1"
    if spec.target_blocks < 1:
        return False, "target_blocks >= 1"
    if spec.lds_bytes() > 160 * 1024:
        return False, f"LDS {spec.lds_bytes()} B > 160 KiB"
    if arch not in ("gfx950",):
        return False, "ds_read_tr16_b64 needs gfx950"
    return True, "ok"


def build_dw_wgrad_mfma(spec: DwWgradMfmaSpec, arch: str = "gfx950") -> KernelDef:
    ok, why = is_valid_dw_wgrad_mfma_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DwWgradMfmaSpec: {why}")
    p = spec.problem
    KH, KW, PAD = p.KH, p.KW, p.PAD
    TAPS = KH * KW
    QT, BQ = spec.q_tiles, spec.block_q
    RG, SG = spec.rg, spec.sg
    D = spec.prefetch
    IX, IY = spec.ix(), spec.iy()
    GI = spec.in_groups()
    XC = 4 * GI  # staged X columns
    REAL_COLS = BQ + KW - 1
    HALF = spec.in_pitch()
    DP = spec.d_pitch()
    WW = spec.win_words()
    io_type = _io_type(p.dtype)
    T = _NW * _WAVE

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = T
    b.kernel.attrs["lds_alias_scopes"] = True
    params = emit_direct_params(b, direction="wgrad", io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci((1 << 31) - 1)
    p_N = params["p_N"]
    p_C = params["p_groups"]
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

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    ctile, qtile, zslot = b.block_id_x(), b.block_id_y(), b.block_id_z()
    c_blk0 = b.mul(ctile, ci(_BC))
    q0 = b.mul(qtile, ci(BQ))
    n_q = b.div(b.add(p_Wo, ci(BQ - 1)), ci(BQ))
    UN = spec.unfold_n
    n_chunks = b.div(b.add(p_H, ci(spec.rows_per_chunk - 1)), ci(spec.rows_per_chunk))
    n_ngrp = b.div(b.add(p_N, ci(UN - 1)), ci(UN))
    n_units = b.mul(n_ngrp, n_chunks)
    # Same formula as DwWgradMfmaSpec.n_z, which sizes the grid.
    n_ct = b.div(b.add(p_C, ci(_BC - 1)), ci(_BC))
    planes = b.mul(n_ct, n_q)
    n_z = b.smin(
        n_units,
        b.smax(b.div(b.add(ci(spec.target_blocks), b.sub(planes, c1)), planes), c1),
    )

    a_rsrc = b.buffer_rsrc(params["A"], params["A_bytes"])  # dY
    b_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])  # X

    # ---- LDS: one allocation per staging buffer (one alias scope each) -----
    X_ELEMS = _NW * IX * _WAVE * 8
    Y_ELEMS = _NW * IY * _WAVE * 8
    x_stage = [
        b.smem_alloc(io_type, [1, X_ELEMS], name_hint=f"xs{i}") for i in range(D)
    ]
    y_stage = [
        b.smem_alloc(io_type, [1, Y_ELEMS], name_hint=f"ys{i}") for i in range(D)
    ]
    x_addr = [b.smem_addr_of(s) for s in x_stage]
    y_addr = [b.smem_addr_of(s) for s in y_stage]
    MIN = _BC * HALF
    DNEW = spec.dnew_elems()
    main_in = b.smem_alloc(io_type, [1, MIN], name_hint="min")
    # The newest dY row, channel-major; older rows live in registers.
    dnew = b.smem_alloc(io_type, [1, DNEW], name_hint="dnew")

    # ---- DMA lane geometry: slot L = (wave * I + i) * 64 + lane -> (col, chunk)
    def dma_lanes(n_instr: int, cols: int):
        out = []
        for i in range(n_instr):
            slot = b.add(b.mul(b.add(b.mul(wave, ci(n_instr)), ci(i)), ci(_WAVE)), lane)
            col = b.div(slot, ci(_CH))
            cc = b.add(c_blk0, b.mul(b.mod(slot, ci(_CH)), ci(8)))
            out.append((col, b.land(b.cmp_lt(col, ci(cols)), b.cmp_lt(cc, p_C)), cc))
        return out

    # Lane offsets: (offset without the image and row, ok, packed image j).
    if UN > 1:
        # Image j owns tile columns [j * P, (j + 1) * P); with one column
        # tile (q0 = 0) X column x sits at j * P + PAD + x, dY column q at
        # j * P + q.
        pitch = b.smax(b.smax(p_Wo, b.add(p_Wo, ci(KW - 1 - PAD))), b.add(p_W, ci(PAD)))

        def split(col):
            j = b.div(col, pitch)
            return j, b.sub(col, b.mul(j, pitch))

    else:

        def split(col):
            return c0, b.add(q0, col)

    x_offs = []
    for col, okl, cc in dma_lanes(IX, min(XC, REAL_COLS)):
        j, u = split(col)
        wi = b.sub(u, ci(PAD))
        okc = b.land(okl, b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, p_W)))
        okc = b.land(okc, b.cmp_lt(j, ci(UN)))
        off = b.add(b.add(b.mul(wi, sx_w), cc), b.mul(j, sx_n))
        x_offs.append((off, okc, j))
    y_offs = []
    for col, okl, cc in dma_lanes(IY, BQ):
        j, wo = split(col)
        ok = b.land(b.land(okl, b.cmp_lt(wo, p_Wo)), b.cmp_lt(j, ci(UN)))
        off = b.add(b.add(b.mul(wo, sdy_w), cc), b.mul(j, sdy_n))
        y_offs.append((off, ok, j))

    def dma(rsrc, addr, n_instr, i, voff):
        lds = b.smem_ptr_add(
            addr,
            b.zext(b.mul(b.add(b.mul(wave, ci(n_instr)), ci(i)), ci(_WAVE * 16)), I64),
        )
        b.async_buffer_load_lds(rsrc, lds, voff, c0, 4)

    # ---- transpose geometry (lane L ends up with channel L) ---------------
    # ds_read_tr16_b64: out lane (b, i', j') item x = in lane (b, x, i') item
    # j' (lanes as 16b + 4i + j), so in lane (b, i, j) reads column 4g + i,
    # channels 4 * (4b + j) .. + 3.
    tr_row = b.mod(b.div(lane, ci(4)), ci(4))
    tr_cg = b.add(b.mul(b.div(lane, ci(16)), ci(4)), b.mod(lane, ci(4)))
    tr_base = b.add(b.mul(tr_row, ci(_BC)), b.mul(tr_cg, ci(4)))

    def transpose_rows(stage_x, stage_y, with_x: bool):
        """Staged NHWC rows -> main_in / dnew, channel-major."""
        reads = []
        if with_x:
            for g in range(ceil_div(GI, _NW)):
                gg = b.add(wave, ci(g * _NW))
                reads.append((g, gg, True))
        for g in range(ceil_div(QT, _NW)):
            gg = b.add(wave, ci(g * _NW))
            reads.append((g, gg, False))

        def one(gg, is_x):
            if is_x:
                v = b.ds_read_tr16_b64(
                    stage_x, c0, b.add(tr_base, b.mul(gg, ci(4 * _BC))), dtype=io_type
                )
                dst = b.add(b.mul(lane, ci(HALF)), b.mul(gg, ci(4)))
                b.smem_store_vN(main_in, [c0, dst], v, 4)
            else:
                v = b.ds_read_tr16_b64(
                    stage_y, c0, b.add(tr_base, b.mul(gg, ci(4 * _BC))), dtype=io_type
                )
                dst = b.add(b.mul(lane, ci(DP)), b.mul(gg, ci(4)))
                b.smem_store_vN(dnew, [c0, dst], v, 4)

        for g, gg, is_x in reads:
            n_groups = GI if is_x else QT
            if (g + 1) * _NW > n_groups:
                with b.scf_if(b.cmp_lt(gg, ci(n_groups))):
                    one(gg, is_x)
            else:
                one(gg, is_x)

    # ---- per-lane MFMA operand geometry ------------------------------------
    lane4 = b.mod(lane, ci(4))
    ch_w = b.add(b.mul(wave, ci(16)), b.div(lane, ci(4)))  # channel in the tile
    lane_s = b.smin(lane4, ci(min(_TAPS, KW) - 1))
    if spec.preshift:
        lane_word = b.div(lane_s, c2)
    else:
        lane_word = c0
    lane_res = b.sub(lane_s, b.mul(lane_word, c2))
    sel0 = ci(0x03020100)
    sel1 = ci(0x01010101)
    pre_sel = b.add(sel0, b.mul(sel1, b.mul(lane_word, ci(4))))
    res_sel = b.add(sel0, b.mul(sel1, b.mul(lane_res, c2)))
    frag_t = VectorType(io_type, 4)
    # The four lanes of a channel each read a quarter of its main_in window
    # (16 bytes per read) and share them with quad broadcasts.
    NWR = spec.win_reads()
    a_reads = [
        b.add(
            b.mul(ch_w, ci(HALF)),
            b.mul(b.smin(b.add(lane4, ci(4 * k)), ci(NWR - 1)), ci(8)),
        )
        for k in range(ceil_div(NWR, 4))
    ]
    # Likewise the newest dY row: lane j of a channel reads q tiles j, j + 4.
    d_reads = [
        b.add(
            b.mul(ch_w, ci(DP)),
            b.mul(b.smin(b.add(lane4, ci(4 * k)), ci(QT - 1)), ci(4)),
        )
        for k in range(ceil_div(QT, 4))
    ]
    head = b.cmp_eq(lane4, c0)

    def bcast(v: Value, src: int) -> Value:
        return b.quad_perm(v, [src] * 4)

    def words(v: Value, n: int):
        w = b.vec_bitcast(v, VectorType(I32, n))
        return [b.vec_extract(w, e) for e in range(n)]

    def shift_ring(ring):
        """Advance the dY ring one row: lane (g, r) takes (g, r - 1)'s row,
        lane (g, 0) takes (g - 1, 3)'s, and lane (0, 0) the newest row."""
        fresh = []
        for k, idx in enumerate(d_reads):
            v = b.smem_load_vN(dnew, c0, idx, dtype=io_type, n=4)
            fresh.append(words(v, 2))
        rot = [
            [[b.quad_perm(d, [3, 0, 1, 2]) for d in ring[g][qt]] for qt in range(QT)]
            for g in range(RG)
        ]
        out = []
        for g in range(RG):
            row = []
            for qt in range(QT):
                if g == 0:
                    src = [bcast(d, qt % 4) for d in fresh[qt // 4]]
                else:
                    src = rot[g - 1][qt]
                row.append([b.select(head, src[e], rot[g][qt][e]) for e in range(2)])
            out.append(row)
        return out

    def compute(ring, accs):
        """MFMA the X row in main_in against the dY rows in ``ring``."""
        part = [
            words(b.smem_load_vN(main_in, c0, idx, dtype=io_type, n=8), 4)
            for idx in a_reads
        ]
        win = [bcast(part[m // 4][e], m % 4) for m in range(NWR) for e in range(4)]
        if spec.preshift:
            win = [b.perm_b32(win[j + 1], win[j], pre_sel) for j in range(WW)]
        bfr = [
            [b.vec_bitcast(b.vec_pack(ring[g][qt], I32), frag_t) for qt in range(QT)]
            for g in range(RG)
        ]
        accs = list(accs)
        for sgi in range(SG):
            for qt in range(QT):
                k = 2 * qt + 2 * sgi
                a0 = b.perm_b32(win[k + 1], win[k], res_sel)
                a1 = b.perm_b32(win[k + 2], win[k + 1], res_sel)
                af = b.vec_bitcast(b.vec_pack([a0, a1], I32), frag_t)
                for rgi in range(RG):
                    i = rgi * SG + sgi
                    accs[i] = _mfma(b, p.dtype, "4x4x4", af, bfr[rgi][qt], accs[i])
        return accs

    zero_acc = b.zero_vec_f32(4)
    n_acc = RG * SG
    PER_STEP = IX + IY

    def pack(accs, ring):
        return list(accs) + [d for g in ring for qt in g for d in qt]

    def unpack(vals):
        accs = list(vals[:n_acc])
        flat = list(vals[n_acc:])
        ring = [
            [flat[(g * QT + qt) * 2 : (g * QT + qt) * 2 + 2] for qt in range(QT)]
            for g in range(RG)
        ]
        return accs, ring

    def emit_body(PRE: int):
        """Unit loop and epilogue, PRE dY rows staged before a unit's first
        X row: KH - 1 in general, PAD when a unit is a whole image (the
        rows above it are padding, which the zeroed ring already holds)."""
        unit_loop = b.scf_for_iter(
            zslot,
            n_units,
            n_z,
            [(f"acc{PRE}_{i}", zero_acc) for i in range(n_acc)],
            iv_name=f"unit{PRE}",
            elide_trailing_barrier=False,
        )
        with unit_loop as (unit, carried):
            # A fresh ring per unit: its zeros stand in for the dY rows above
            # the image, which the whole-image prologue does not stage.
            accs = list(carried)
            ring = [[[c0, c0] for _ in range(QT)] for _ in range(RG)]
            n_img = b.mul(b.mod(unit, n_ngrp), ci(UN))
            chunk = b.div(unit, n_ngrp)
            y0 = b.mul(chunk, ci(spec.rows_per_chunk))
            n_rows = b.smin(ci(spec.rows_per_chunk), b.sub(p_H, y0))
            n_steps = b.add(n_rows, ci(PRE))
            x_img = b.mul(n_img, sx_n)
            y_img = b.mul(n_img, sdy_n)

            def issue(buf: int, t: Value):
                """DMA step t: X row y0 - PRE + t (steps >= PRE) and dY row y + PAD."""
                y = b.add(b.sub(y0, ci(PRE)), t)
                x_ok = b.land(b.cmp_ge(t, ci(PRE)), b.cmp_lt(t, n_steps))
                x_row = b.add(x_img, b.mul(y, sx_h))
                for i, (off, okl, j) in enumerate(x_offs):
                    ok = b.land(okl, x_ok)
                    if UN > 1:
                        ok = b.land(ok, b.cmp_lt(b.add(n_img, j), p_N))
                    voff = b.select(ok, b.mul(b.add(off, x_row), c2), oob)
                    dma(b_rsrc, x_addr[buf], IX, i, voff)
                pr = b.add(y, ci(PAD))
                y_ok = b.land(
                    b.land(b.cmp_ge(pr, c0), b.cmp_lt(pr, p_Ho)), b.cmp_lt(t, n_steps)
                )
                y_row = b.add(y_img, b.mul(pr, sdy_h))
                for i, (off, okl, j) in enumerate(y_offs):
                    ok = b.land(okl, y_ok)
                    if UN > 1:
                        ok = b.land(ok, b.cmp_lt(b.add(n_img, j), p_N))
                    voff = b.select(ok, b.mul(b.add(off, y_row), c2), oob)
                    dma(a_rsrc, y_addr[buf], IY, i, voff)

            def step(t: Value, buf: int, accs, ring, do_compute: bool):
                # Issue the row D - 1 ahead into the buffer the last step drained.
                issue((buf + D - 1) % D, b.add(t, ci(D - 1)))
                b.s_waitcnt(vmcnt=(D - 1) * PER_STEP, lgkmcnt=0)
                b.s_barrier_bare()
                transpose_rows(x_stage[buf], y_stage[buf], do_compute)
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()
                ring = shift_ring(ring)
                if do_compute:
                    accs = compute(ring, accs)
                return accs, ring

            for t in range(D - 1):
                issue(t % D, ci(t))
            # Prologue: the PRE dY rows the first input row pairs with.
            for t in range(PRE):
                accs, ring = step(ci(t), t % D, accs, ring, False)
            # Main: one input row per step, unrolled by D so buffers are constant.
            trips = b.smax(b.div(b.add(n_rows, ci(D - 1)), ci(D)), c1)
            row_loop = b.scf_for_iter(
                c0,
                trips,
                c1,
                [(f"a{PRE}_{i}", v) for i, v in enumerate(pack(accs, ring))],
                iv_name=f"rb{PRE}",
                elide_trailing_barrier=False,
            )
            with row_loop as (rb, la):
                la_acc, la_ring = unpack(list(la))
                for u in range(D):
                    t = b.add(b.add(b.mul(rb, ci(D)), ci(PRE)), ci(u))
                    la_acc, la_ring = step(t, (PRE + u) % D, la_acc, la_ring, True)
                b.scf_yield(*pack(la_acc, la_ring))
            # Drain the tail DMAs before the next unit reuses the buffers.
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.s_barrier_bare()
            b.scf_yield(*unpack(list(row_loop.results))[0])
        accs = list(unit_loop.results)

        # ---- epilogue: ws[part][tap][c], part = zslot * n_q + qtile -------------
        part = b.add(b.mul(zslot, n_q), qtile)
        c_glb = b.add(c_blk0, ch_w)
        c_ok = b.cmp_lt(c_glb, p_C)
        for rgi in range(RG):
            r = b.add(lane4, ci(rgi * _TAPS))
            row0 = b.add(
                b.mul(b.add(b.mul(part, ci(TAPS)), b.mul(r, ci(KW))), p_C), c_glb
            )
            with b.scf_if(b.land(c_ok, b.cmp_lt(r, ci(KH)))):
                for sgi in range(SG):
                    acc = accs[rgi * SG + sgi]
                    for i in range(_TAPS):
                        s = sgi * _TAPS + i
                        if s < KW:
                            off = b.add(row0, b.mul(ci(s), p_C))
                            b.global_store(
                                params["D"], off, b.vec_extract(acc, i), align=4
                            )

    if PAD == KH - 1:
        emit_body(PAD)
    else:
        # Whole images per unit (one row chunk) skip the KH - 1 - PAD zero rows.
        with b.scf_if_else(b.cmp_eq(n_chunks, c1)) as (whole, chunked):
            with whole:
                emit_body(PAD)
            with chunked:
                emit_body(KH - 1)
    return b.kernel
