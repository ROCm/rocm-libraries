# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Depthwise wgrad on the gfx1250 wave32 WMMA, one-hot (Toeplitz) form.

``dW[c][r][s] = sum_{n,y,q} dY[n, y, q, c] * X[n, y + r - PH, q + s - PW, c]``
(stride 1, cpg = kpg = 1, groups = C).

Every wave owns 8 channels. One ``16x16x32`` WMMA contracts 4 dY pixels of the
8 channels (K = 8 channels x 4 pixels, a lane half holding 2 pixels of all 8):

- A (16 x 32), row ``m = 8 t + c'``: the dY value of channel ``c'`` at dY row
  ``y + t`` and pixel ``q0 + p``, at the K slot of channel ``c'`` only (one-hot
  over the 8 channels; the other 7 slots of each pixel are zero);
- B (32 x 16), column ``n = 8 j + s``: the X values of all 8 channels at X row
  ``y + rb + j - PH`` and column ``q0 + p + s0 + s - PW`` (two 16-byte LDS
  reads of NHWC-staged rows);
- D: element ``(8 t + c', 8 j + s)`` adds into ``dW[c'][rb + j - t][s0 + s]``.

So one A (the dY pixels) feeds every tile ``(rb, s0)``, ``rb in {0, 2, ..,
2 floor(KH / 2)}`` and ``s0 in {0, 8, ..}``, and the two dY rows ``y`` and
``y + 1`` of a step land every filter row exactly once (``t = 0`` covers rows
``rb``, ``rb + 1``; ``t = 1`` rows ``rb - 1``, ``rb``). Out-of-range taps are
masked in the epilogue. Only 1/8 of each WMMA is useful (the one-hot), the
price of operands that come straight from 16-byte NHWC reads.

The column padding may differ from the row padding (``problem.PAD_W``), so
1xK filters with "same" column padding build too.

Work: grid x = channel groups of ``8 * waves`` x column tiles of ``tile_q`` dY
columns; grid y = ``n_split`` blocks that each walk work items
``(image, row tile of `rows` dY rows)``, ``split, split + n_split, ..``,
accumulating in registers. Per item the block stages the dY tile and the X
rows / columns it meets into LDS (zero-filled padding), syncs, and runs
``rows / 2 * tile_q / 4 * tiles`` WMMAs per wave.

Epilogue: fp32 atomics into replica ``linear block id % ws_replicas`` of a
``[ws_replicas][C][KH * KW]`` scratch that ``conv_wgrad_workspace_reduce``
(wg_M = C, wg_N = KH * KW) folds into dW, as the other depthwise wgrad kernels
do. The caller zeroes the scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from rocke.core.ir import (
    CPOL_ATOMIC_CASCADE_DEV,
    I16,
    I32,
    IRBuilder,
    KernelDef,
    Value,
    VectorType,
)
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _buf_load_vN,
    _io_type,
    dilation_reason,
    emit_direct_params,
    pad_w_reason,
)

DW_WGRAD_WMMA_CH = 8  # channels per wave (the 8-wide K group of a lane half)


@dataclass(frozen=True)
class DwWgradWmmaSpec:
    problem: DirectConvProblem
    name: str = "direct_depthwise_wgrad_wmma"
    wave_size: int = 32
    waves: int = 4  # channel groups of 8 per block
    tile_q: int = 32  # dY columns per work item (multiple of 4)
    rows: int = 8  # dY rows per work item (even)
    target_blocks: int = 1024  # grid fill: n_split = ceil(target / grid x)
    ws_replicas: int = 64
    prefetch: bool = True  # carry the next item's global loads in registers

    # ---- geometry -------------------------------------------------------
    @property
    def pad_h(self) -> int:
        return self.problem.PAD

    @property
    def pad_w(self) -> int:
        return self.problem.PAD_W

    @property
    def block_ch(self) -> int:
        return DW_WGRAD_WMMA_CH * self.waves

    @property
    def threads_per_block(self) -> int:
        return self.waves * self.wave_size

    @property
    def n_rb(self) -> int:
        return self.problem.KH // 2 + 1

    @property
    def n_s(self) -> int:
        return ceil_div(self.problem.KW, 8)

    @property
    def tiles(self) -> int:
        return self.n_rb * self.n_s

    @property
    def x_rows(self) -> int:
        return self.rows + 2 * (self.problem.KH // 2)

    @property
    def x_cols(self) -> int:
        return self.tile_q + 8 * self.n_s - 1

    @property
    def ch_pitch(self) -> int:
        return self.block_ch + 8  # 16-byte skew per pixel

    @property
    def lds_bytes(self) -> int:
        return (self.x_rows * self.x_cols + self.rows * self.tile_q) * self.ch_pitch * 2

    def _grid_x(self) -> int:
        p = self.problem
        return ceil_div(p.groups, self.block_ch) * ceil_div(p.Wo, self.tile_q)

    def _items(self) -> int:
        p = self.problem
        return p.N * ceil_div(p.Ho, self.rows)

    def n_split(self) -> int:
        return max(1, min(ceil_div(self.target_blocks, self._grid_x()), self._items()))

    def launch_geometry(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        return self.grid(), (self.threads_per_block, 1, 1)

    def grid(self) -> Tuple[int, int, int]:
        return (self._grid_x(), self.n_split(), 1)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"f{p.KH}x{p.KW}p{self.pad_h}x{self.pad_w}",
            f"wv{self.waves}",
            f"q{self.tile_q}",
            f"r{self.rows}",
            f"tb{self.target_blocks}",
            flags={
                f"wsr{self.ws_replicas}": self.ws_replicas > 1,
                "nopf": not self.prefetch,
                "bf16": p.dtype == "bf16",
            },
        )

    def validate(self) -> None:
        why = dw_wgrad_wmma_reason(self)
        if why:
            raise ValueError(f"DwWgradWmmaSpec: {why}")


def dw_wgrad_wmma_reason(spec: DwWgradWmmaSpec) -> str:
    """Why a spec cannot be built ("" when it can); arch-independent rules."""
    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {p.dtype!r}; expected fp16 or bf16"
    if p.cpg != 1 or p.kpg != 1:
        return f"requires cpg=kpg=1 (got {p.cpg}, {p.kpg})"
    if p.stride != 1:
        return f"requires stride 1 (got {p.stride})"
    if p.groups % DW_WGRAD_WMMA_CH:
        return f"channels {p.groups} must be a multiple of 8 (16-byte NHWC reads)"
    if p.KH < 1 or p.KW < 1:
        return f"filter {p.KH}x{p.KW} is not positive"
    why = pad_w_reason(p)
    if why is not None:
        return why
    if spec.pad_h < 0 or spec.pad_w < 0 or p.Ho < 1 or p.Wo < 1:
        return "degenerate geometry"
    if spec.wave_size != 32:
        return "the WMMA 16x16x32 lane maps are wave32"
    if spec.waves < 1 or spec.waves > 16:
        return f"waves must be 1..16 (got {spec.waves})"
    if spec.tile_q < 4 or spec.tile_q % 4:
        return f"tile_q must be a positive multiple of 4 (got {spec.tile_q})"
    if spec.rows < 2 or spec.rows % 2:
        return f"rows must be a positive even number (got {spec.rows})"
    if spec.target_blocks < 1 or spec.ws_replicas < 1:
        return "target_blocks and ws_replicas must be >= 1"
    steps = spec.rows // 2 * spec.tile_q // 4 * spec.tiles
    if steps > 1024:
        return f"{steps} unrolled WMMAs per item exceed 1024 (smaller rows / tile_q)"
    return ""


def is_valid_dw_wgrad_wmma_spec(
    spec: DwWgradWmmaSpec, arch: str = "gfx1250"
) -> Tuple[bool, str]:
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    why = dilation_reason(spec.problem)
    if why is not None:
        return False, why
    why = dw_wgrad_wmma_reason(spec)
    if why:
        return False, why
    if target.wave_size != 32:
        return False, f"wave32 kernel (got {arch}, wave{target.wave_size})"
    ab = "bf16" if spec.problem.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    ):
        return False, f"missing wmma 16x16x32 {ab} on {arch}"
    if spec.threads_per_block > target.max_threads_per_block:
        return False, f"{spec.threads_per_block} threads > hardware cap on {arch}"
    if spec.lds_bytes > target.lds_capacity_bytes:
        return False, f"LDS {spec.lds_bytes} B > {target.lds_capacity_bytes} on {arch}"
    return True, "ok"


def build_dw_wgrad_wmma(spec: DwWgradWmmaSpec, arch: str = "gfx1250") -> KernelDef:
    """Build the one-hot WMMA depthwise wgrad kernel (see the module docstring)."""
    from rocke.core.arch import ArchTarget

    # One validation: is_valid_dw_wgrad_wmma_spec runs dw_wgrad_wmma_reason.
    ok, why = is_valid_dw_wgrad_wmma_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid DwWgradWmmaSpec for {arch}: {why}")
    p = spec.problem
    KH, KW = p.KH, p.KW
    PH, PW = spec.pad_h, spec.pad_w
    CB = spec.block_ch
    CP = spec.ch_pitch
    CPX = CB // 8  # 16-byte chunks per pixel
    THREADS = spec.threads_per_block
    TQ, RT = spec.tile_q, spec.rows
    XR, XC = spec.x_rows, spec.x_cols
    NRB, NS = spec.n_rb, spec.n_s
    KK = KH * KW
    io_type = _io_type(p.dtype)
    ab = "bf16" if p.dtype == "bf16" else "f16"
    wmma_op = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="wmma", a_dtype=ab, b_dtype=ab, c_dtype="fp32", m=16, n=16, k=32
    )

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    params = emit_direct_params(b, direction="wgrad", io_type=io_type)
    dy_rsrc = b.buffer_rsrc(params["A"], params["A_bytes"])
    x_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])
    d_rsrc = b.buffer_rsrc(params["D"], params["D_bytes"])
    p_N, p_H, p_W = params["p_N"], params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo, p_C = params["p_Ho"], params["p_Wo"], params["p_groups"]

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)
    c8 = b.const_i32(8)
    c16 = b.const_i32(16)
    oob = b.const_i32((1 << 30) - 1)  # past every in-range byte offset

    tid = b.thread_id_x()
    c32 = b.const_i32(32)
    wave = b.readfirstlane(b.div(tid, c32))
    lane = b.mod(tid, c32)
    kh = b.div(lane, c16)  # K half: pixels 2 kh, 2 kh + 1; D row octet t
    col16 = b.mod(lane, c16)  # A row m / B, D column n
    a_t = b.div(col16, c8)  # A row m = 8 t + c'
    a_c = b.mod(col16, c8)
    n_j = b.div(col16, c8)  # B / D column n = 8 j + s
    n_s = b.mod(col16, c8)

    # ---- block coordinates ------------------------------------------------
    c_TQ = b.const_i32(TQ)
    c_RT = b.const_i32(RT)
    c_CB = b.const_i32(CB)
    n_q = b.div(b.add(p_Wo, b.const_i32(TQ - 1)), c_TQ)
    n_cg = b.div(b.add(p_C, b.const_i32(CB - 1)), c_CB)
    grid_x = b.mul(n_cg, n_q)
    n_yt = b.div(b.add(p_Ho, b.const_i32(RT - 1)), c_RT)
    items = b.mul(p_N, n_yt)
    # n_split = clamp(ceil(target / grid_x), 1, items): the host grid's formula.
    c_tb = b.const_i32(spec.target_blocks)
    n_split = b.div(b.add(c_tb, b.sub(grid_x, c1)), grid_x)
    n_split = b.select(b.cmp_lt(items, n_split), items, n_split)
    n_split = b.select(b.cmp_lt(n_split, c1), c1, n_split)
    bx = b.block_id_x()
    split = b.block_id_y()
    q_blk = b.mod(bx, n_q)
    c_base = b.mul(b.div(bx, n_q), c_CB)
    w0 = b.mul(q_blk, c_TQ)
    wave_ch = b.mul(wave, c8)

    sx_n, sx_h, sx_w = (
        params[k] for k in ("p_X_stride_n", "p_X_stride_hi", "p_X_stride_wi")
    )
    sy_n, sy_h, sy_w = (
        params[k] for k in ("p_dY_stride_n", "p_dY_stride_ho", "p_dY_stride_wo")
    )

    xs = b.smem_alloc(io_type, [XR * XC, CP], name_hint="dww_x")
    ys = b.smem_alloc(io_type, [RT * TQ, CP], name_hint="dww_dy")

    # ---- staging plan (thread-static parts) --------------------------------
    def plan(n_pix: int, ncols: int):
        out = []
        total = n_pix * CPX
        for i in range(-(-total // THREADS)):
            idx = b.add(tid, b.const_i32(i * THREADS))
            live = b.cmp_lt(idx, b.const_i32(total))
            pix = b.div(idx, b.const_i32(CPX))
            ch8 = b.mul(b.mod(idx, b.const_i32(CPX)), c8)
            rr = b.div(pix, b.const_i32(ncols))
            cc = b.mod(pix, b.const_i32(ncols))
            out.append((live, pix, ch8, rr, cc))
        return out

    x_plan = plan(XR * XC, XC)
    y_plan = plan(RT * TQ, TQ)

    def fetch(item: Value):
        n_img = b.div(item, n_yt)
        y0 = b.mul(b.sub(item, b.mul(n_img, n_yt)), c_RT)
        xv, yv = [], []
        for live, pix, ch8, rr, cc in x_plan:
            h = b.add(y0, b.sub(rr, b.const_i32(PH)))
            w = b.add(w0, b.sub(cc, b.const_i32(PW)))
            c = b.add(c_base, ch8)
            ok = b.land(live, b.land(b.cmp_ge(h, c0), b.cmp_lt(h, p_H)))
            ok = b.land(ok, b.land(b.cmp_ge(w, c0), b.cmp_lt(w, p_W)))
            ok = b.land(ok, b.cmp_lt(c, p_C))
            e = b.add(
                b.add(b.mul(n_img, sx_n), b.mul(h, sx_h)), b.add(b.mul(w, sx_w), c)
            )
            off = b.select(ok, b.mul(e, c2), oob)
            xv.append(_buf_load_vN(b, p.dtype, x_rsrc, off, c0, 4))
        for live, pix, ch8, rr, cc in y_plan:
            h = b.add(y0, rr)
            w = b.add(w0, cc)
            c = b.add(c_base, ch8)
            ok = b.land(live, b.land(b.cmp_lt(h, p_Ho), b.cmp_lt(w, p_Wo)))
            ok = b.land(ok, b.cmp_lt(c, p_C))
            e = b.add(
                b.add(b.mul(n_img, sy_n), b.mul(h, sy_h)), b.add(b.mul(w, sy_w), c)
            )
            off = b.select(ok, b.mul(e, c2), oob)
            yv.append(_buf_load_vN(b, p.dtype, dy_rsrc, off, c0, 4))
        return xv, yv

    def stage(xv, yv):
        for (live, pix, ch8, _, _), v in zip(x_plan, xv):
            with b.scf_if(live):
                b.smem_store_vN(xs, [pix, ch8], v, 8)
        for (live, pix, ch8, _, _), v in zip(y_plan, yv):
            with b.scf_if(live):
                b.smem_store_vN(ys, [pix, ch8], v, 8)

    # ---- per-lane constants for the fragments ------------------------------
    # One-hot A: dY of channel c' sits in dword c' / 2, half c' % 2.
    is_d = [b.cmp_eq(b.div(a_c, c2), b.const_i32(d)) for d in range(4)]
    a_shift = b.mul(b.mod(a_c, c2), b.const_i32(16))
    frag_t = VectorType(io_type, 16)
    a_ch = b.add(wave_ch, a_c)
    kh2 = b.mul(kh, c2)
    # A pixel index (row y + t, column p0 + 2 kh + jj): lane part.
    a_pix = [b.add(b.mul(a_t, c_TQ), b.add(kh2, b.const_i32(jj))) for jj in range(2)]
    # B pixel index (row y + rb + j, column p0 + 2 kh + jj + s0 + s): lane part.
    c_XC = b.const_i32(XC)
    b_pix = [
        b.add(b.mul(n_j, c_XC), b.add(b.add(kh2, b.const_i32(jj)), n_s))
        for jj in range(2)
    ]

    def compute(accs: List[Value]) -> List[Value]:
        """One staged item: per 4-column chunk, the A fragments of every dY row
        pair first, then every X fragment once (it depends on y + rb only) fed
        to each (y, rb) pair that shares it."""
        out = list(accs)
        for qc in range(TQ // 4):
            p0 = 4 * qc
            a_frags = []
            for yp in range(RT // 2):
                words = []
                for jj in range(2):
                    v = b.smem_load_vN(
                        ys,
                        b.add(a_pix[jj], b.const_i32(2 * yp * TQ + p0)),
                        a_ch,
                        dtype=io_type,
                        n=1,
                    )
                    w = b.shl(b.zext(b.bitcast(b.vec_extract(v, 0), I16), I32), a_shift)
                    words += [b.select(is_d[d], w, c0) for d in range(4)]
                a_frags.append(b.vec_bitcast(b.vec_pack(words, I32), frag_t))
            for rbb in range(0, RT - 2 + 2 * (NRB - 1) + 1, 2):
                pairs = [
                    (yp, rbi)
                    for yp in range(RT // 2)
                    for rbi in range(NRB)
                    if 2 * yp + 2 * rbi == rbb
                ]
                for si in range(NS):
                    base = rbb * XC + p0 + 8 * si
                    halves = [
                        b.smem_load_vN(
                            xs,
                            b.add(b_pix[jj], b.const_i32(base)),
                            wave_ch,
                            dtype=io_type,
                            n=8,
                        )
                        for jj in range(2)
                    ]
                    b_frag = b.vec_concat(*halves)
                    for yp, rbi in pairs:
                        t = rbi * NS + si
                        out[t] = b.mma(wmma_op, a_frags[yp], b_frag, out[t])
        return out

    # ---- item loop: the next item's global loads ride in registers while the
    # current one computes (prologue fetch, then store -> fetch next -> compute).
    zero_acc = b.zero_vec_f32(8)
    acc_carry = [(f"dww_acc{t}", zero_acc) for t in range(spec.tiles)]
    if spec.prefetch:
        xv0, yv0 = fetch(split)
        n_x = len(xv0)
        loop = b.scf_for_iter(
            split,
            items,
            n_split,
            acc_carry
            + [(f"dww_xv{i}", v) for i, v in enumerate(xv0)]
            + [(f"dww_yv{i}", v) for i, v in enumerate(yv0)],
            iv_name="dww_item",
        )
        with loop as (item, carried):
            carried = list(carried)
            accs_in = carried[: spec.tiles]
            xv = carried[spec.tiles : spec.tiles + n_x]
            yv = carried[spec.tiles + n_x :]
            b.sync_lds_only()  # the previous item's LDS reads are done
            stage(xv, yv)
            b.sync_lds_only()
            nxt = b.add(item, n_split)
            nxt = b.select(
                b.cmp_lt(nxt, items), nxt, item
            )  # past the end: refetch (unused)
            xn, yn = fetch(nxt)
            b.scf_yield(*compute(accs_in), *xn, *yn)
    else:
        loop = b.scf_for_iter(split, items, n_split, acc_carry, iv_name="dww_item")
        with loop as (item, accs_in):
            xv, yv = fetch(item)
            b.sync_lds_only()
            stage(xv, yv)
            b.sync_lds_only()
            b.scf_yield(*compute(list(accs_in)))
    accs = loop.results[: spec.tiles]

    # ---- epilogue: atomics into the replica slab -----------------------------
    lin = b.add(b.mul(split, grid_x), bx)
    rep = b.mod(lin, b.const_i32(spec.ws_replicas))
    rep_c = b.mul(rep, p_C)
    c4 = b.const_i32(4)
    c_KH = b.const_i32(KH)
    c_KW = b.const_i32(KW)
    c_KK = b.const_i32(KK)
    for rbi in range(NRB):
        for si in range(NS):
            # Row of D lane (kh, i): m = 8 kh + i -> t = kh, c' = i.
            r = b.sub(b.add(b.const_i32(2 * rbi), n_j), kh)
            s = b.add(b.const_i32(8 * si), n_s)
            tap_ok = b.land(
                b.land(b.cmp_ge(r, c0), b.cmp_lt(r, c_KH)), b.cmp_lt(s, c_KW)
            )
            tap = b.add(b.mul(r, c_KW), s)
            acc = accs[rbi * NS + si]
            for i in range(8):
                c = b.add(b.add(c_base, wave_ch), b.const_i32(i))
                ok = b.land(tap_ok, b.cmp_lt(c, p_C))
                byte = b.mul(b.add(b.mul(b.add(rep_c, c), c_KK), tap), c4)
                with b.scf_if(ok):
                    b.buffer_atomic_fadd_f32(
                        d_rsrc,
                        byte,
                        c0,
                        b.vec_extract(acc, i),
                        cachepolicy=CPOL_ATOMIC_CASCADE_DEV,
                    )
    return b.kernel
