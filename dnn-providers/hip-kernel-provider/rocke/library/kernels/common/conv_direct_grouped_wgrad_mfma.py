# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grouped wgrad for 4 and 8 channels per group on the batched 4x4x4 MFMA.

With ``cpg = kpg = 4`` a group's weight gradient per tap is one 4x4 block,
which is exactly one batch of ``mfma_f32_4x4x4``: a wave runs 16 groups at
once. Per input row ``y``, tap ``(r, s)`` and 4-column group ``q0``:

    A[c][k] = X[y][q0 + k + s - PAD][c]      (lane = channel c, 4 columns)
    B[k][o] = dY[y + PAD - r][q0 + k][o]
    C[c][o] += sum_k A[c][k] B[k][o]          = dW[o][r][s][c] over those columns

Both operands come straight out of the NHWC row staged in LDS:
``ds_read_tr16_b64`` hands lane ``L`` four consecutive columns of channel
``L``, and the tap shift ``s`` is just a row offset of the per-lane read
addresses, so every read is aligned and nothing is re-laid out.

``cpg = kpg = 8``: a group is four 4x4 blocks. Lanes ``4b .. 4b + 3`` hold
channels ``4b ..`` of the A read; a second dY read with the two 4-channel
halves of each group swapped (``channel ^ 4``) gives the off-diagonal blocks,
so each tap carries two accumulators.

A block of ``waves_q`` waves owns 64 channels and ``4 * wave_q4 * waves_q``
output columns (each wave its own columns) and walks (image, row chunk)
units; dY rows stay in a register ring of KH rows. The waves reduce through
LDS at the end and the block stores one partial slot ``ws[part][tap][o]``
(``tap = (r * KW + s) * cpg + c``) that ``DwWgradFoldSpec`` folds into dW.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import F32, I64, IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    _mfma,
    dilation_reason,
    emit_direct_params,
)

_WAVE = 64
_BC = 64  # channels per block
_CH = _BC // 8  # 16-byte chunks per staged pixel


@dataclass(frozen=True)
class GroupedWgradMfmaSpec:
    problem: DirectConvProblem
    name: str = "grouped_wgrad_mfma"
    waves_q: int = 4  # waves along the output columns
    wave_q4: int = 1  # 4-column groups per wave
    rows_per_chunk: int = 56  # input rows per unit
    prefetch: int = 2  # staging buffers
    target_blocks: int = 1024

    @property
    def threads(self) -> int:
        return self.waves_q * _WAVE

    @property
    def block_q(self) -> int:
        return 4 * self.wave_q4 * self.waves_q

    @property
    def halves(self) -> int:
        """4x4 blocks along k per group: 1 for cpg 4, 2 for cpg 8."""
        return self.problem.cpg // 4

    def x_cols(self) -> int:
        return self.block_q + self.problem.KW - 1

    def ix(self) -> int:
        return ceil_div(ceil_div(self.x_cols() * _CH, _WAVE), self.waves_q)

    def iy(self) -> int:
        return ceil_div(ceil_div(self.block_q * _CH, _WAVE), self.waves_q)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        flags = {"bf16": True} if p.dtype == "bf16" else {}
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}p{p.PAD}s{p.stride}c{p.cpg}",
            f"wq{self.waves_q}",
            f"q4{self.wave_q4}",
            f"rpc{self.rows_per_chunk}",
            f"pf{self.prefetch}",
            f"tb{self.target_blocks}",
            flags=flags,
        )

    def n_channel_tiles(self) -> int:
        return ceil_div(self.problem.total_c, _BC)

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

    def n_acc(self) -> int:
        p = self.problem
        return p.KH * p.KW * self.halves

    def lds_bytes(self) -> int:
        stage = self.prefetch * self.waves_q * (self.ix() + self.iy()) * _WAVE * 16
        red = self.waves_q * _WAVE * 4 * 4
        return stage + red


def is_valid_grouped_wgrad_mfma_spec(spec: GroupedWgradMfmaSpec, arch: str = "gfx950"):
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
    if p.total_c % 8:
        return False, "C must be a multiple of 8"
    if spec.waves_q not in (1, 2, 4, 8):
        return False, "waves_q in (1, 2, 4, 8)"
    if spec.wave_q4 not in (1, 2, 4):
        return False, "wave_q4 in (1, 2, 4)"
    if spec.prefetch not in (2, 3, 4):
        return False, "prefetch in 2..4"
    if spec.rows_per_chunk < 1:
        return False, "rows_per_chunk >= 1"
    if spec.target_blocks < 1:
        return False, "target_blocks >= 1"
    if spec.n_acc() * 4 > 128:
        return False, "too many accumulators"
    if spec.lds_bytes() > 160 * 1024:
        return False, f"LDS {spec.lds_bytes()} B > 160 KiB"
    if arch not in ("gfx950",):
        return False, "ds_read_tr16_b64 needs gfx950"
    return True, "ok"


def build_grouped_wgrad_mfma(
    spec: GroupedWgradMfmaSpec, arch: str = "gfx950"
) -> KernelDef:
    ok, why = is_valid_grouped_wgrad_mfma_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid GroupedWgradMfmaSpec: {why}")
    p = spec.problem
    KH, KW, PAD, CPG = p.KH, p.KW, p.PAD, p.cpg
    HV = spec.halves
    TAPS = spec.fold_taps()
    WQ, Q4 = spec.waves_q, spec.wave_q4
    BQ = spec.block_q
    XC = spec.x_cols()
    D = spec.prefetch
    IX, IY = spec.ix(), spec.iy()
    T = spec.threads
    PRE = KH - 1  # dY rows a unit needs before its first input row
    io_type = _io_type(p.dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = T
    b.kernel.attrs["lds_alias_scopes"] = True
    params = emit_direct_params(b, direction="wgrad", io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci((1 << 31) - 1)
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

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    ctile, qtile, zslot = b.block_id_x(), b.block_id_y(), b.block_id_z()
    c_blk0 = b.mul(ctile, ci(_BC))
    q0 = b.mul(qtile, ci(BQ))
    n_q = b.div(b.add(p_Wo, ci(BQ - 1)), ci(BQ))
    n_chunks = b.div(b.add(p_H, ci(spec.rows_per_chunk - 1)), ci(spec.rows_per_chunk))
    n_units = b.mul(p_N, n_chunks)
    # Same formula as GroupedWgradMfmaSpec.n_z, which sizes the grid.
    n_ct = b.div(b.add(p_C, ci(_BC - 1)), ci(_BC))
    planes = b.mul(n_ct, n_q)
    n_z = b.smin(
        n_units,
        b.smax(b.div(b.add(ci(spec.target_blocks), b.sub(planes, c1)), planes), c1),
    )

    a_rsrc = b.buffer_rsrc(params["A"], params["A_bytes"])  # dY
    b_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])  # X

    # ---- LDS: one allocation per staging buffer (one alias scope each) -----
    X_ELEMS = WQ * IX * _WAVE * 8
    Y_ELEMS = WQ * IY * _WAVE * 8
    x_stage = [
        b.smem_alloc(io_type, [1, X_ELEMS], name_hint=f"xs{i}") for i in range(D)
    ]
    y_stage = [
        b.smem_alloc(io_type, [1, Y_ELEMS], name_hint=f"ys{i}") for i in range(D)
    ]
    x_addr = [b.smem_addr_of(s) for s in x_stage]
    y_addr = [b.smem_addr_of(s) for s in y_stage]

    # ---- DMA lane geometry: slot L = (wave * I + i) * 64 + lane -> (col, chunk)
    def dma_lanes(n_instr: int, cols: int):
        out = []
        for i in range(n_instr):
            slot = b.add(b.mul(b.add(b.mul(wave, ci(n_instr)), ci(i)), ci(_WAVE)), lane)
            col = b.div(slot, ci(_CH))
            cc = b.add(c_blk0, b.mul(b.mod(slot, ci(_CH)), ci(8)))
            out.append((col, b.land(b.cmp_lt(col, ci(cols)), b.cmp_lt(cc, p_C)), cc))
        return out

    x_offs = []
    for col, okl, cc in dma_lanes(IX, XC):
        wi = b.add(b.sub(q0, ci(PAD)), col)
        okc = b.land(okl, b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, p_W)))
        x_offs.append((b.add(b.mul(wi, sx_w), cc), okc))
    y_offs = []
    for col, okl, cc in dma_lanes(IY, BQ):
        wo = b.add(q0, col)
        y_offs.append((b.add(b.mul(wo, sdy_w), cc), b.land(okl, b.cmp_lt(wo, p_Wo))))

    def dma(rsrc, addr, n_instr, i, voff):
        lds = b.smem_ptr_add(
            addr,
            b.zext(b.mul(b.add(b.mul(wave, ci(n_instr)), ci(i)), ci(_WAVE * 16)), I64),
        )
        b.async_buffer_load_lds(rsrc, lds, voff, c0, 4)

    # ---- operand reads ------------------------------------------------------
    # ds_read_tr16_b64: out lane (b, i', j') item x = in lane (b, x, i') item
    # j' (lanes as 16b + 4i + j), so in lane (b, i, j) reads column col0 + i,
    # channels 4 * (4b + j) .. + 3, and out lane L gets channel L.
    tr_row = b.mod(b.div(lane, ci(4)), ci(4))
    tr_cg = b.add(b.mul(b.div(lane, ci(16)), ci(4)), b.mod(lane, ci(4)))
    tr_base = b.add(b.mul(tr_row, ci(_BC)), b.mul(tr_cg, ci(4)))
    # The partner half of an 8-channel group (channel ^ 4).
    tr_swap = b.add(b.mul(tr_row, ci(_BC)), b.mul(b.xor(tr_cg, c1), ci(4)))
    col_w = b.mul(wave, ci(4 * Q4))  # this wave's first column in the tile

    def read(stage, base: Value, col: int) -> Value:
        idx = b.add(base, b.mul(b.add(col_w, ci(col)), ci(_BC)))
        return b.ds_read_tr16_b64(stage, c0, idx, dtype=io_type)

    zero_acc = b.zero_vec_f32(4)
    zero_frag = b.zero_vec(io_type, 4)
    n_acc = KH * KW * HV
    # The carried ring holds the dY fragments of the KH - 1 rows before the
    # current one, newest first: [row][q4][half].
    n_ring = (KH - 1) * Q4 * HV

    def acc_i(r, s, h):
        return (r * KW + s) * HV + h

    def pack(accs, ring):
        return list(accs) + [v for row in ring for q in row for v in q]

    def unpack(vals):
        accs = list(vals[:n_acc])
        flat = list(vals[n_acc:])
        ring = []
        for j in range(KH - 1):
            ring.append(
                [flat[(j * Q4 + q) * HV : (j * Q4 + q) * HV + HV] for q in range(Q4)]
            )
        return accs, ring

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
        PER_STEP = IX + IY

        def issue(buf: int, t: Value):
            """DMA step t: X row y0 - PRE + t (steps >= PRE) and dY row y + PAD."""
            y = b.add(b.sub(y0, ci(PRE)), t)
            x_ok = b.land(b.cmp_ge(t, ci(PRE)), b.cmp_lt(t, n_steps))
            x_row = b.add(x_img, b.mul(y, sx_h))
            for i, (off, okl) in enumerate(x_offs):
                voff = b.select(b.land(okl, x_ok), b.mul(b.add(off, x_row), c2), oob)
                dma(b_rsrc, x_addr[buf], IX, i, voff)
            pr = b.add(y, ci(PAD))
            y_ok = b.land(
                b.land(b.cmp_ge(pr, c0), b.cmp_lt(pr, p_Ho)), b.cmp_lt(t, n_steps)
            )
            y_row = b.add(y_img, b.mul(pr, sdy_h))
            for i, (off, okl) in enumerate(y_offs):
                voff = b.select(b.land(okl, y_ok), b.mul(b.add(off, y_row), c2), oob)
                dma(a_rsrc, y_addr[buf], IY, i, voff)

        def step(t: Value, buf: int, accs, ring, do_compute: bool):
            """Barrier, DMA step t + D - 1, read and MFMA step t."""
            # Step t's DMA has landed (steps up to t + D - 2 may still fly)
            # and every wave is done reading the buffer refilled below.
            b.s_waitcnt(vmcnt=(D - 2) * PER_STEP, lgkmcnt=0)
            b.s_barrier_bare()
            issue((buf + D - 1) % D, b.add(t, ci(D - 1)))
            newest = []
            for q in range(Q4):
                h = [read(y_stage[buf], tr_base, 4 * q)]
                if HV == 2:
                    h.append(read(y_stage[buf], tr_swap, 4 * q))
                newest.append(h)
            full = [newest] + list(ring)  # full[r] = the dY row of tap r
            if do_compute:
                accs = list(accs)
                for s in range(KW):
                    for q in range(Q4):
                        a = read(x_stage[buf], tr_base, 4 * q + s)
                        for r in range(KH):
                            for h in range(HV):
                                i = acc_i(r, s, h)
                                accs[i] = _mfma(
                                    b, p.dtype, "4x4x4", a, full[r][q][h], accs[i]
                                )
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
        # Drain the tail DMAs before the next unit reuses the buffers.
        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.s_barrier_bare()
        b.scf_yield(*list(row_loop.results))
    accs, _ = unpack(list(unit_loop.results))

    # ---- epilogue: sum the waves in LDS, store ws[part][tap][o] -------------
    # Lane L, item i of an accumulator of half h: X channel c_L = 4 * (L // 4)
    # + i, dY channel o_L = (4 * (L // 4)) ^ (4 * h) + L % 4, both in the tile.
    red = b.smem_alloc(F32, [1, WQ * _WAVE * 4], name_hint="red")
    part = b.add(b.mul(zslot, n_q), qtile)
    lane4 = b.mod(lane, ci(4))
    quad = b.mul(b.div(lane, ci(4)), ci(4))
    for i_acc in range(n_acc):
        r, s, h = i_acc // (KW * HV), (i_acc // HV) % KW, i_acc % HV
        b.smem_store_vN_f32(red, [c0, b.mul(tid, ci(4))], accs[i_acc], 4)
        b.sync()
        with b.scf_if(b.cmp_lt(tid, ci(_WAVE))):
            tot = b.smem_load_vN_f32(red, c0, b.mul(lane, ci(4)), n=4)
            for w in range(1, WQ):
                v = b.smem_load_vN_f32(
                    red, c0, b.mul(b.add(lane, ci(w * _WAVE)), ci(4)), n=4
                )
                tot = b.vector_add(tot, v)
            o_loc = b.add(b.xor(quad, ci(4 * h)), lane4)
            o_glb = b.add(c_blk0, o_loc)
            # Channel c of the group: (4 * (L // 4) + i) % cpg.
            with b.scf_if(b.cmp_lt(o_glb, p_K)):
                for e in range(4):
                    c_in = b.mod(b.add(quad, ci(e)), ci(CPG))
                    tap = b.add(ci((r * KW + s) * CPG), c_in)
                    off = b.add(b.mul(b.add(b.mul(part, ci(TAPS)), tap), p_K), o_glb)
                    b.global_store(params["D"], off, b.vec_extract(tot, e), align=4)
        b.sync()
    return b.kernel
