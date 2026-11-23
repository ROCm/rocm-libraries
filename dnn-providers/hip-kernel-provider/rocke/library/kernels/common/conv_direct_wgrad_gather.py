# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct wgrad for dense layers with few input channels, any stride.

Computes, for ``groups == 1``::

    dW[k, r, s, c] = sum_{n, ho, wo} dY[n, ho, wo, k] * X[n, ho*S + r - PAD, wo*S + s - PAD, c]

as one GEMM per filter row ``r``: ``M = k``, ``N = (s, c)`` (``KW * C`` columns,
padded to 16), reduced over the output columns of a 32-wide strip by
``mfma_f32_16x16x32``. That is the shape for a stem layer (``C = 3``) or a
narrow one (``C = 24``), where the per-tap ``(k, c)`` tile of
``conv_direct_wgrad_tiled`` would leave most of the MFMA's ``N`` empty.

The block streams OUTPUT rows of one strip. Per step it DMAs ``S`` new X rows
and one dY row into LDS rings (``prefetch`` steps ahead). The dY fragment is a
transpose read of the ``[32 px][K]`` row, as in the tiled kernel. The X
operand is gathered: lane ``(n = (s, c), 8 columns)`` reads its 8 elements out
of the raw ``[column][C]`` X row at a stride of ``S * C``.

X rows are DMAed as whole 16-byte chunks from an 8-element-aligned start, so a
strip at the image edge stages neighbouring-row data in its padding columns;
those strips zero them in LDS before the row is read.

The epilogue stores one partial slot per block (``ws[part][(r, s, c)][k]``),
folded into ``dW`` by ``conv_direct_dw_wgrad.DwWgradFoldSpec`` with
``taps = KH * KW * C`` and ``channels = K``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rocke.core.ir import I64, IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    _mfma,
    dilation_reason,
    emit_direct_params,
)
from kernels.common.conv_direct_wgrad_tiled import pick_swizzle

_ATOM = 16
_WO = 32
_VEC = 8
_WAVE = 64


@dataclass(frozen=True)
class WgradGatherSpec:
    problem: DirectConvProblem
    name: str = "direct_wgrad_gather"
    wave_k16: int = 2  # k atoms per wave
    waves_k: int = 1
    prefetch: int = 2
    rows_per_seg: int = 16  # output rows per segment (rounded up to the unroll)
    target_blocks: int = 1024

    @property
    def block_k(self) -> int:
        return _ATOM * self.wave_k16 * self.waves_k

    @property
    def threads(self) -> int:
        return self.waves_k * _WAVE

    @property
    def n_atoms(self) -> int:
        p = self.problem
        return ceil_div(p.KW * p.cpg, _ATOM)

    @property
    def unroll(self) -> int:
        p = self.problem
        return self.prefetch + ceil_div(p.KH, p.stride)

    def rps(self) -> int:
        return ceil_div(self.rows_per_seg, self.unroll) * self.unroll

    def x_phase(self) -> int:
        p = self.problem
        return (-p.PAD * p.cpg) % _VEC

    def x_elems(self) -> int:
        """Elements of one staged X row: phase + the strip's columns."""
        p = self.problem
        return self.x_phase() + ((_WO - 1) * p.stride + p.KW) * p.cpg

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        flags = {"bf16": True} if p.dtype == "bf16" else {}
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}p{p.PAD}s{p.stride}c{p.cpg}",
            f"wk{self.wave_k16}x{self.waves_k}",
            f"pf{self.prefetch}",
            f"rps{self.rows_per_seg}",
            f"tb{self.target_blocks}",
            flags=flags,
        )

    def n_k_tiles(self) -> int:
        return ceil_div(self.problem.kpg, self.block_k)

    def n_units(self) -> int:
        p = self.problem
        return p.N * ceil_div(p.Wo, _WO) * ceil_div(p.Ho, self.rps())

    def n_partitions(self) -> int:
        return min(self.n_units(), ceil_div(self.target_blocks, self.n_k_tiles()))

    def fold_taps(self) -> int:
        p = self.problem
        return p.KH * p.KW * p.cpg

    def fold_channels(self) -> int:
        return self.problem.kpg

    def workspace_bytes(self) -> int:
        return self.n_partitions() * self.fold_taps() * self.fold_channels() * 4

    def launch_geometry(self):
        return (self.n_k_tiles(), self.n_partitions(), 1), (self.threads, 1, 1)

    def dma_instrs(self) -> Tuple[int, int]:
        """16-byte DMA instructions per wave and row: (X, dY)."""
        ix = ceil_div(ceil_div(ceil_div(self.x_elems(), _VEC), _WAVE), self.waves_k)
        iy = ceil_div(ceil_div(_WO * self.block_k // _VEC, _WAVE), self.waves_k)
        return ix, iy

    def x_slot_elems(self) -> int:
        """Elements of one X ring slot: whole DMA instructions of every wave,
        +16 because a transpose read of the last column atom runs past it."""
        return self.threads * self.dma_instrs()[0] * _VEC + 16

    def dy_slot_elems(self) -> int:
        return self.threads * self.dma_instrs()[1] * _VEC

    def lds_bytes(self) -> int:
        """The X and dY rings as the builder allocates them."""
        U = self.unroll
        return (
            self.problem.stride * U * self.x_slot_elems() + U * self.dy_slot_elems()
        ) * 2


def is_valid_wgrad_gather_spec(
    spec: WgradGatherSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    why = dilation_reason(p)
    if why is not None:
        return False, why
    if p.groups != 1:
        return False, "dense layers only"
    if p.dtype not in ("fp16", "bf16"):
        return False, "fp16/bf16 only"
    if not target.memory.has_ds_read_tr:
        return False, f"needs ds_read_b64_tr_b16 (absent on {arch})"
    if target.wave_size != 64 or target.async_lds_max_dwords < 4:
        return False, f"needs wave64 + 16-byte buffer_load_lds (absent on {arch})"
    if p.kpg % _VEC:
        return False, "K must be a multiple of 8"
    if p.KW * p.cpg > 96:
        return False, "KW * C must be at most 96"
    # Whole 16-byte chunks from an aligned start at a build-time phase.
    if (p.W * p.cpg) % _VEC or (_WO * p.stride * p.cpg) % _VEC:
        return False, "W * C and 32 * stride * C must be multiples of 8"
    acc = p.KH * spec.wave_k16 * spec.n_atoms * 4
    if acc > 192:
        return False, f"{acc} accumulator VGPRs (> 192)"
    if spec.wave_k16 < 1 or spec.waves_k < 1 or spec.threads > 1024:
        return False, "wave_k16 >= 1 and waves_k in 1..16"
    if spec.prefetch < 1:
        return False, "prefetch >= 1"
    if spec.lds_bytes() > 160 * 1024:
        return False, f"LDS {spec.lds_bytes()} B > 160 KiB"
    if spec.rows_per_seg < 1:
        return False, "rows_per_seg >= 1"
    if spec.target_blocks < 1:
        return False, "target_blocks >= 1"
    return True, "ok"


def build_wgrad_gather(spec: WgradGatherSpec, arch: str = "gfx950") -> KernelDef:
    ok, why = is_valid_wgrad_gather_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid WgradGatherSpec for {arch}: {why}")
    p = spec.problem
    KH, KW, PAD, S, C = p.KH, p.KW, p.PAD, p.stride, p.cpg
    io_dtype = p.dtype
    io_type = _io_type(io_dtype)
    WK16 = spec.wave_k16
    NA = spec.n_atoms
    BK = spec.block_k
    CHK = BK // _VEC
    PH = spec.x_phase()
    XCH = ceil_div(spec.x_elems(), _VEC)  # 16-byte chunks of a staged X row
    D = spec.prefetch
    U = spec.unroll
    NX, NY = S * U, U
    RPS = spec.rps()
    PRE = max(KH - S, 0)
    TAPS = KH * KW * C

    IX, IY = spec.dma_instrs()
    X_ELEMS = spec.x_slot_elems()
    # When a pixel row of the strip (S * C elements) and the phase keep every
    # lane's 8-byte read aligned, the raw row is itself a [pixel][(s, c)]
    # tile with contiguous columns, so the X operand is a transpose read too.
    X_TR = (S * C) % 4 == 0 and PH % 4 == 0
    Y_ELEMS = spec.dy_slot_elems()
    PER_STEP = S * IX + IY

    swz_k = pick_swizzle(CHK, (0,), _WO, _WO, tuple(range(0, BK, _ATOM)))

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    b.kernel.attrs["lds_alias_scopes"] = True
    params = emit_direct_params(b, direction="wgrad", io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci((1 << 31) - 1)
    p_N = params["p_N"]
    p_H, p_W = params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    p_K = params["p_total_k"]
    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    k_tile, part = b.block_id_x(), b.block_id_y()
    k_blk0 = b.mul(k_tile, ci(BK))
    n_q = b.div(b.add(p_Wo, ci(_WO - 1)), ci(_WO))
    n_hb = b.div(b.add(p_Ho, ci(RPS - 1)), ci(RPS))
    n_units = b.mul(b.mul(p_N, n_q), n_hb)
    n_kt = b.div(b.add(p_K, ci(BK - 1)), ci(BK))
    n_part = b.smin(
        n_units, b.div(b.add(ci(spec.target_blocks), b.sub(n_kt, c1)), n_kt)
    )

    a_rsrc = b.buffer_rsrc(params["A"], params["A_bytes"])  # dY
    b_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])  # X
    sdy_n, sdy_h, sdy_w = (
        params["p_dY_stride_n"],
        params["p_dY_stride_ho"],
        params["p_dY_stride_wo"],
    )
    sx_n, sx_h = params["p_X_stride_n"], params["p_X_stride_hi"]

    x_bufs = [
        b.smem_alloc(io_type, [1, X_ELEMS], name_hint=f"xg{i}") for i in range(NX)
    ]
    y_bufs = [
        b.smem_alloc(io_type, [1, Y_ELEMS], name_hint=f"yg{i}") for i in range(NY)
    ]
    x_addr = [b.smem_addr_of(s) for s in x_bufs]
    y_addr = [b.smem_addr_of(s) for s in y_bufs]

    def swz_val(row: Value, bits):
        v = None
        for j, bit in enumerate(bits):
            term = b.shl(b.land(b.lshr(row, ci(bit)), c1), ci(j + 1))
            v = term if v is None else b.lor(v, term)
        return v

    def dma(rsrc, addr, n_instr, i, voff):
        lds = b.smem_ptr_add(
            addr,
            b.zext(b.mul(b.add(b.mul(wave, ci(n_instr)), ci(i)), ci(_WAVE * 16)), I64),
        )
        b.async_buffer_load_lds(rsrc, lds, voff, c0, 4)

    # X: chunk L of a row is elements [8 L, 8 L + 8) from the aligned start.
    x_chunks = []
    for i in range(IX):
        slot = b.add(b.mul(b.add(b.mul(wave, ci(IX)), ci(i)), ci(_WAVE)), lane)
        x_chunks.append((b.mul(slot, ci(_VEC)), b.cmp_lt(slot, ci(XCH))))
    # dY: slot -> (pixel row, swizzled k chunk).
    y_chunks = []
    for i in range(IY):
        slot = b.add(b.mul(b.add(b.mul(wave, ci(IY)), ci(i)), ci(_WAVE)), lane)
        row = b.div(slot, ci(CHK))
        phys = b.mod(slot, ci(CHK))
        sw = swz_val(row, swz_k)
        log = phys if sw is None else b.xor(phys, sw)
        y_chunks.append((row, b.mul(log, ci(_VEC)), b.cmp_lt(row, ci(_WO))))

    # ---- fragment addresses -------------------------------------------------
    lane_hi = b.div(lane, ci(16))
    lane_q = b.mod(b.div(lane, ci(4)), ci(4))
    lane_c4 = b.mod(lane, ci(4))
    lane_n = b.mod(lane, ci(16))
    wave_k0 = b.mul(wave, ci(WK16 * _ATOM))

    def dy_index(rd: int, col0: Value) -> Value:
        row = b.add(b.add(b.mul(lane_hi, ci(8)), lane_q), ci(rd * 4))
        col = b.add(col0, b.mul(lane_c4, ci(4)))
        chunk = b.div(col, ci(_VEC))
        sw = swz_val(row, swz_k)
        phys = chunk if sw is None else b.xor(chunk, sw)
        return b.add(
            b.mul(b.add(b.mul(row, ci(CHK)), phys), ci(_VEC)), b.mod(col, ci(_VEC))
        )

    dy_idx = [
        [dy_index(rd, b.add(wave_k0, ci(mi * _ATOM))) for rd in (0, 1)]
        for mi in range(WK16)
    ]
    # X gather base per atom: element PH + (S * px0 + s) * C + c, px0 = 8 * (lane / 16).
    x_base = []
    for nj in range(NA):
        n = b.add(lane_n, ci(nj * _ATOM))
        s_i = b.smin(b.div(n, ci(C)), ci(KW - 1))  # n >= KW*C: any in-range column
        c_i = b.mod(n, ci(C))
        x_base.append(
            b.add(
                ci(PH), b.add(b.mul(b.add(b.mul(lane_hi, ci(8 * S)), s_i), ci(C)), c_i)
            )
        )

    def read_dy(buf: int) -> List[Value]:
        out = []
        for mi in range(WK16):
            lo = b.ds_read_tr16_b64(y_bufs[buf], c0, dy_idx[mi][0], dtype=io_type)
            hi = b.ds_read_tr16_b64(y_bufs[buf], c0, dy_idx[mi][1], dtype=io_type)
            out.append(b.vec_concat(lo, hi))
        return out

    x_tr_idx = [
        [
            b.add(
                ci(PH + nj * _ATOM),
                b.add(
                    b.mul(
                        b.add(b.add(b.mul(lane_hi, ci(8)), lane_q), ci(rd * 4)),
                        ci(S * C),
                    ),
                    b.mul(lane_c4, ci(4)),
                ),
            )
            for rd in (0, 1)
        ]
        for nj in range(NA)
    ]

    def read_x(buf: int, nj: int) -> Value:
        if X_TR:
            lo = b.ds_read_tr16_b64(x_bufs[buf], c0, x_tr_idx[nj][0], dtype=io_type)
            hi = b.ds_read_tr16_b64(x_bufs[buf], c0, x_tr_idx[nj][1], dtype=io_type)
            return b.vec_concat(lo, hi)
        vals = [
            b.vec_extract(
                b.smem_load_vN(
                    x_bufs[buf],
                    c0,
                    b.add(x_base[nj], ci(j * S * C)),
                    dtype=io_type,
                    n=1,
                ),
                0,
            )
            for j in range(8)
        ]
        return b.vec_pack(vals, io_type)

    zero_acc = b.zero_vec_f32(4)
    zero_elem = b.vec_extract(b.zero_vec(io_type, 2), 0)
    n_acc = KH * WK16 * NA

    def acc_index(r, mi, nj):
        return (r * WK16 + mi) * NA + nj

    seg_loop = b.scf_for_iter(
        part,
        n_units,
        n_part,
        [(f"acc{i}", zero_acc) for i in range(n_acc)],
        iv_name="unit",
        elide_trailing_barrier=False,
    )
    with seg_loop as (unit, carried):
        accs = list(carried)
        hb = b.mod(unit, n_hb)
        rest = b.div(unit, n_hb)
        q = b.mod(rest, n_q)
        n_img = b.div(rest, n_q)
        h0 = b.mul(hb, ci(RPS))
        q0 = b.mul(q, ci(_WO))
        col0 = b.sub(b.mul(q0, ci(S)), ci(PAD))  # first X column of the strip
        x_row0 = b.sub(b.mul(h0, ci(S)), ci(PAD))
        n_steps = b.smin(ci(RPS), b.sub(p_Ho, h0))
        # Strip X start, 8-aligned: element col0 * C sits at PH.
        x_start = b.add(b.mul(n_img, sx_n), b.sub(b.mul(col0, ci(C)), ci(PH)))
        y_offs = []
        for row, ch, okr in y_chunks:
            wo = b.add(q0, row)
            kk = b.add(k_blk0, ch)
            ok = b.land(okr, b.land(b.cmp_lt(wo, p_Wo), b.cmp_lt(kk, p_K)))
            y_offs.append((b.add(b.add(b.mul(n_img, sdy_n), b.mul(wo, sdy_w)), kk), ok))
        # Padding columns staged from neighbouring rows: [lo, hi) element
        # ranges of the row to zero, for the edge strips.
        pad_lo = b.smax(b.mul(b.sub(c0, col0), ci(C)), c0)  # cols < 0
        cols_in = b.sub(p_W, col0)  # cols >= W start here (relative)
        pad_hi0 = b.add(ci(PH), b.mul(cols_in, ci(C)))
        edge = b.lor(b.cmp_lt(col0, c0), b.cmp_lt(pad_hi0, ci(spec.x_elems())))

        def issue_x(xr: Value, slot: int):
            hi = b.add(x_row0, xr)
            ok_r = b.land(
                b.land(b.cmp_ge(hi, c0), b.cmp_lt(hi, p_H)),
                b.cmp_lt(xr, b.add(b.mul(n_steps, ci(S)), ci(PRE))),
            )
            row_off = b.add(x_start, b.mul(hi, sx_h))
            for i, (e, okl) in enumerate(x_chunks):
                voff = b.select(b.land(okl, ok_r), b.mul(b.add(row_off, e), c2), oob)
                dma(b_rsrc, x_addr[slot], IX, i, voff)

        def issue_y(t: Value, slot: int):
            ho = b.add(h0, t)
            ok_r = b.cmp_lt(t, n_steps)
            for i, (off, okl) in enumerate(y_offs):
                voff = b.select(
                    b.land(okl, ok_r), b.mul(b.add(off, b.mul(ho, sdy_h)), c2), oob
                )
                dma(a_rsrc, y_addr[slot], IY, i, voff)

        def issue_step(t: Value, tm: int):
            for j in range(S):
                issue_x(b.add(b.mul(t, ci(S)), ci(PRE + j)), (S * tm + PRE + j) % NX)
            issue_y(t, tm % NY)

        def fix_padding(slots: List[int]):
            """Zero the padding columns of freshly landed X rows (edge strips)."""
            with b.scf_if(edge):
                for slot in slots:
                    for k in range(ceil_div(spec.x_elems(), spec.threads)):
                        e = b.add(tid, ci(k * spec.threads))
                        rel = b.sub(e, ci(PH))
                        is_pad = b.land(
                            b.cmp_lt(e, ci(spec.x_elems())),
                            b.lor(b.cmp_lt(rel, pad_lo), b.cmp_ge(e, pad_hi0)),
                        )
                        with b.scf_if(is_pad):
                            b.smem_store_vN(x_bufs[slot], [c0, e], zero_elem, 1)
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()

        for j in range(PRE):
            issue_x(ci(j), j % NX)
        for t in range(D):
            issue_step(ci(t), t)

        def step(t: Value, tm: int, accs):
            b.s_waitcnt(vmcnt=(D - 1) * PER_STEP, lgkmcnt=0)
            b.s_barrier_bare()
            new_slots = [(S * tm + PRE + j) % NX for j in range(S)]
            if tm == 0:
                # Step 0 of a segment also consumes the PRE leading rows.
                new_slots = [j % NX for j in range(PRE)] + new_slots
            fix_padding(new_slots)
            issue_step(b.add(t, ci(D)), (tm + D) % U)
            dyf = read_dy(tm % NY)
            accs = list(accs)
            for r in range(KH):
                slot = (S * tm + r) % NX
                for nj in range(NA):
                    xf = read_x(slot, nj)
                    for mi in range(WK16):
                        i = acc_index(r, mi, nj)
                        accs[i] = _mfma(b, io_dtype, "16x16x32", dyf[mi], xf, accs[i])
            return accs

        trips = b.smax(b.div(b.add(n_steps, ci(U - 1)), ci(U)), c1)
        loop = b.scf_for_iter(
            c0,
            trips,
            c1,
            [(f"a{i}", v) for i, v in enumerate(accs)],
            iv_name="rb",
            elide_trailing_barrier=False,
        )
        with loop as (rb, la):
            la = list(la)
            for u in range(U):
                la = step(b.add(b.mul(rb, ci(U)), ci(u)), u, la)
            b.scf_yield(*la)
        accs_out = list(loop.results)
        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.s_barrier_bare()
        b.scf_yield(*accs_out)
    accs = list(seg_loop.results)

    # ---- epilogue: partial slot ws[part][(r, s, c)][k] ----------------------
    # acc lane layout: k = (lane / 16) * 4 + slot of atom mi, n = lane % 16 of atom nj.
    D_ptr = params["D"]
    k_lane0 = b.add(b.add(k_blk0, wave_k0), b.mul(lane_hi, ci(4)))
    for r in range(KH):
        for nj in range(NA):
            n = b.add(lane_n, ci(nj * _ATOM))
            n_ok = b.cmp_lt(n, ci(KW * C))
            tap = b.add(ci(r * KW * C), n)  # (r, s, c) flattened
            row_base = b.mul(b.add(b.mul(part, ci(TAPS)), tap), p_K)
            for mi in range(WK16):
                for slot in range(4):
                    kk = b.add(k_lane0, ci(mi * _ATOM + slot))
                    ok = b.land(n_ok, b.cmp_lt(kk, p_K))
                    val = b.vec_extract(accs[acc_index(r, mi, nj)], slot)
                    with b.scf_if(ok):
                        b.global_store(D_ptr, b.add(row_base, kk), val, align=4)
    return b.kernel
