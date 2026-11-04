# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Depthwise forward convolution as a Toeplitz WMMA product (gfx1250).

The forward twin of the windowed depthwise dgrad's Toeplitz form (the
``mfma=True`` form of :func:`build_direct_depthwise_dgrad_windowed` in
:mod:`conv_direct_grouped`): a stride-1 depthwise dgrad is a stride-1
depthwise convolution with a flipped filter, so this kernel is that one with
un-flipped taps and the forward tensor roles.

Each wave owns 8 channels. One ``16x16x32`` WMMA multiplies one-hot weights
(A) by 4 window columns x 8 channels of an input row (B) into 32 output
columns x 8 channels (D): tile rows ``8 q + k`` are channel ``k`` at column
parity ``q``, tile column ``n`` is the column pair ``2 (n % SUB)`` of image
``n // SUB`` (``w_fold`` images per tile, ``SUB = 16 / w_fold``). A filter row
of ``KW`` taps takes ``ceil((KW + 1) / 4)`` passes, each 4 window columns
further right.

A wave32 lane ``l`` (``t = l // 16``) carries 16 K elements, two 8-channel
chunks ``j``:

- A: the one-hot weight ``W[k, r, s]`` at slot ``8 j + k``,
  ``s = 4 ph + 2 t + j - q`` (zero outside ``[0, KW)``);
- B: the 8 channels of window columns ``2 (n % SUB) + 2 t + j + 4 ph``;
- D: the 8 channels at output column ``2 (n % SUB) + t``.

Rows stream through the block: input row ``y`` of the block's window
(``hi = h0 + y - PAD``) is staged in LDS (double buffered, one barrier per
row) and multiplied into the KH circular accumulator tiles of the output rows
``y - r`` it feeds; output row ``y - (KH - 1)`` is finished after row ``y``
and drained through LDS as 16-byte stores. Rows and columns outside the
tensors read zero / drop their store through sentinel offsets.

Non-finite semantics: an Inf / NaN in X reaches every output of its 8-channel
group within its rows of the receptive field, over ``4 * passes`` columns
(``0 * Inf = NaN`` on the zero-weight taps). Finite inputs give the same
results as a scalar kernel up to fp32 summation order.

AOT: N, H, W, Ho, Wo, groups and the strides are kernargs; KH, KW and PAD are
capabilities; block_waves, block_h, w_fold and prefetch_rows are knobs.

C++ twin: ``platform/cpp/instances/common/conv_direct_dw_fwd_wmma.cpp``
(``instance_conv_direct_dw_fwd_wmma.h``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rocke.core.ir import IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DW_DGRAD_MFMA_CH,
    DW_DGRAD_MFMA_LDS_BUDGET,
    DW_DGRAD_MFMA_MAX_PREFETCH_ROWS,
    DW_DGRAD_MFMA_W_FOLDS,
    DW_DGRAD_WIN_MAX_TENSOR_BYTES,
    DW_DGRAD_WIN_OOB_LANE,
    DW_DGRAD_WIN_OOB_UNIFORM,
    DirectConvProblem,
    _buf_load_half,
    _buf_load_vN,
    _buf_store_vN,
    _dw_mfma_in_passes,
    _dw_mfma_lds_bytes,
    _dw_mfma_passes,
    _dw_mfma_tile_w,
    _dw_win_byte_strides,
    _dw_win_tile,
    _io_type,
    _trunc_f32,
    _trunc_f32_elem,
    dilation_reason,
    emit_direct_params,
    one_padding_reason,
    problem_dims_reason,
)
from kernels.common.conv_fwd_wmma_gate import wmma_fwd_op

DW_FWD_WMMA_ARCHES = ("gfx1250",)
DW_FWD_WMMA_WAVE = 32
# Up to 8 waves per block; the register-resident fragments (A weights, KH
# accumulator tiles, a row of B fragments, the prefetched rows) stay under
# this many VGPRs so no form spills. 16-wave blocks (wider channel runs per
# pixel) are for the small filters whose fragments fit the tighter budget.
DW_FWD_WMMA_MAX_WAVES = 16
DW_FWD_WMMA_MAX_FRAG_VGPRS = 224
DW_FWD_WMMA_WIDE_WAVES = 8
DW_FWD_WMMA_WIDE_FRAG_VGPRS = 112
# Statically unrolled WMMAs per wave (rows x KH x passes): a whole 56..64-row
# map in one tile.
DW_FWD_WMMA_MAX_UNROLL = 1024


@dataclass(frozen=True)
class DwFwdWmmaSpec:
    """Toeplitz WMMA depthwise forward (stride 1, cpg = kpg = 1).

    ``block_ch = 8 * block_waves`` channels per block; a block covers
    ``tile_w = 32 / w_fold`` output columns of ``w_fold`` images and
    ``block_h`` output rows. Grid:
    ``(ceil(Wo / tile_w), ceil(C / block_ch), ceil(N / w_fold) * ceil(Ho / block_h))``.
    """

    problem: DirectConvProblem
    name: str = "rocke_direct_dw_fwd_wmma"
    block_waves: int = 4
    block_h: int = 8
    w_fold: int = 1
    prefetch_rows: int = 2

    @property
    def wave_size(self) -> int:
        return DW_FWD_WMMA_WAVE

    @property
    def threads_per_block(self) -> int:
        return self.block_waves * DW_FWD_WMMA_WAVE

    @property
    def block_ch(self) -> int:
        return self.block_waves * DW_DGRAD_MFMA_CH

    @property
    def tile_w(self) -> int:
        return _dw_mfma_tile_w(self.w_fold)

    def grid(self) -> Tuple[int, int, int]:
        p = self.problem
        return (
            ceil_div(p.Wo, self.tile_w),
            ceil_div(p.groups, self.block_ch),
            ceil_div(p.N, self.w_fold) * ceil_div(p.Ho, self.block_h),
        )

    def launch_geometry(self):
        return self.grid(), (self.threads_per_block, 1, 1)

    def in_passes(self) -> int:
        """16-byte input loads per thread and window row."""
        return _dw_mfma_in_passes(
            self.tile_w,
            self.problem.KW,
            self.w_fold,
            self.block_ch,
            self.threads_per_block,
        )

    def frag_vgprs(self) -> int:
        """VGPRs of the register-resident fragments: the one-hot weights
        (KH x passes) and KH accumulator tiles at 8 each, one row of B
        fragments, and the prefetched rows' 16-byte loads."""
        p = self.problem
        kwp = _dw_mfma_passes(p.KW)
        return 8 * (p.KH * kwp + p.KH + kwp) + 4 * self.prefetch_rows * self.in_passes()

    def lds_bytes(self) -> int:
        return _dw_mfma_lds_bytes(
            self.in_passes(),
            self.threads_per_block,
            self.block_ch,
            self.w_fold,
            self.tile_w,
        )

    def unrolled_wmmas(self) -> int:
        p = self.problem
        return self.block_h * p.KH * _dw_mfma_passes(p.KW)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"r{p.KH}s{p.KW}p{p.PAD}",
            f"wv{self.block_waves}",
            f"bh{self.block_h}",
            f"f{self.w_fold}",
            f"pf{self.prefetch_rows}",
            flags={"bf16": p.dtype == "bf16"},
        )


def dw_fwd_wmma_reason(spec: DwFwdWmmaSpec) -> Optional[str]:
    """Why ``spec`` cannot be built or run on its problem (``None`` if it can)."""
    p = spec.problem
    why = problem_dims_reason(p)
    if why is not None:
        return why
    if p.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {p.dtype!r}; expected fp16 or bf16"
    if p.cpg != 1 or p.kpg != 1:
        return f"requires cpg=kpg=1 (got {p.cpg}, {p.kpg})"
    if p.stride != 1:
        return f"requires stride=1 (got {p.stride})"
    why = dilation_reason(p) or one_padding_reason(p)
    if why is not None:
        return why
    if not 0 <= p.PAD < max(p.KH, p.KW) or p.Ho <= 0 or p.Wo <= 0:
        return f"degenerate geometry (PAD={p.PAD}, Ho={p.Ho}, Wo={p.Wo})"
    if spec.block_h < 1:
        return f"block_h must be >= 1 (got {spec.block_h})"
    if not 1 <= spec.block_waves <= DW_FWD_WMMA_MAX_WAVES:
        return (
            f"block_waves must be 1..{DW_FWD_WMMA_MAX_WAVES} (got {spec.block_waves})"
        )
    if spec.w_fold not in DW_DGRAD_MFMA_W_FOLDS:
        return f"w_fold must be one of {DW_DGRAD_MFMA_W_FOLDS} (got {spec.w_fold})"
    if not 1 <= spec.prefetch_rows <= DW_DGRAD_MFMA_MAX_PREFETCH_ROWS:
        return (
            f"prefetch_rows must be 1..{DW_DGRAD_MFMA_MAX_PREFETCH_ROWS} "
            f"(got {spec.prefetch_rows})"
        )
    # A plan-time rule: the IR does not depend on the group count.
    if p.groups % DW_DGRAD_MFMA_CH:
        return f"groups % {DW_DGRAD_MFMA_CH} != 0 (got {p.groups})"
    frag_vgprs = spec.frag_vgprs()
    if (
        spec.block_waves > DW_FWD_WMMA_WIDE_WAVES
        and frag_vgprs > DW_FWD_WMMA_WIDE_FRAG_VGPRS
    ):
        return (
            f"{spec.block_waves} waves need fragments <= "
            f"{DW_FWD_WMMA_WIDE_FRAG_VGPRS} VGPRs (got {frag_vgprs})"
        )
    if frag_vgprs > DW_FWD_WMMA_MAX_FRAG_VGPRS:
        return (
            f"fragments need {frag_vgprs} VGPRs (> "
            f"{DW_FWD_WMMA_MAX_FRAG_VGPRS}); lower prefetch_rows or the filter"
        )
    if spec.unrolled_wmmas() > DW_FWD_WMMA_MAX_UNROLL:
        return (
            f"unrolled body too large ({spec.unrolled_wmmas()} WMMAs > "
            f"{DW_FWD_WMMA_MAX_UNROLL}); lower block_h"
        )
    if spec.lds_bytes() > DW_DGRAD_MFMA_LDS_BUDGET:
        return (
            f"stages {spec.lds_bytes()} LDS bytes per workgroup (budget "
            f"{DW_DGRAD_MFMA_LDS_BUDGET}); lower block_waves"
        )
    # Every in-range byte offset plus a sentinel must stay a valid unsigned
    # 32-bit buffer offset.
    x_bytes = p.N * p.H * p.W * p.groups * 2
    y_bytes = p.N * p.Ho * p.Wo * p.groups * 2
    if max(x_bytes, y_bytes) > DW_DGRAD_WIN_MAX_TENSOR_BYTES:
        return (
            f"X/Y of {max(x_bytes, y_bytes)} bytes exceed the "
            f"{DW_DGRAD_WIN_MAX_TENSOR_BYTES}-byte sentinel addressing range"
        )
    return None


def is_valid_dw_fwd_wmma_spec(
    spec: DwFwdWmmaSpec, arch: str = "gfx1250"
) -> Tuple[bool, str]:
    if arch not in DW_FWD_WMMA_ARCHES:
        return False, f"needs one of {DW_FWD_WMMA_ARCHES} (got {arch})"
    why = dw_fwd_wmma_reason(spec)
    return (False, why) if why else (True, "ok")


def build_dw_fwd_wmma(spec: DwFwdWmmaSpec, arch: str = "gfx1250") -> KernelDef:
    """Build the Toeplitz WMMA depthwise forward kernel.

    Tensor roles (direct forward ABI):
      A param -- X: input,   [N, H, W, groups], NHWC
      B param -- W: weights, [groups, KH, KW, 1], KRSC
      D param -- Y: output,  [N, Ho, Wo, groups], NHWK

    Y[n, ho, wo, c] = sum_{r, s} W[c, r, s] * X[n, ho - PAD + r, wo - PAD + s, c].
    """
    ok, why = is_valid_dw_fwd_wmma_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DwFwdWmmaSpec for {arch}: {why}")

    p = spec.problem
    KH, KW, PAD = p.KH, p.KW, p.PAD
    F = spec.w_fold
    SUB = 16 // F
    TW = spec.tile_w
    KWP = _dw_mfma_passes(KW)
    U = TW + 4 * KWP - 2  # window columns per image
    ROWS = spec.block_h
    THREADS = spec.threads_per_block
    BLOCK_CH = spec.block_ch
    CH = DW_DGRAD_MFMA_CH
    PER_PX = BLOCK_CH // CH  # 16-byte channel chunks per pixel
    CH_PAD = BLOCK_CH + CH
    io_type = _io_type(p.dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS
    params = emit_direct_params(b, direction="fwd", io_type=io_type)
    p_N = params["p_N"]
    p_Hi = params["p_Hi"]
    p_Wi = params["p_Wi"]
    p_Ho = params["p_Ho"]
    p_Wo = params["p_Wo"]
    p_G = params["p_groups"]
    c2 = b.const_i32(2)
    x_img, x_row, x_px, y_img, y_row, y_px = _dw_win_byte_strides(
        b, params, fwd=True, c2=c2
    )

    c0 = b.const_i32(0)
    oob_lane = b.const_i32(DW_DGRAD_WIN_OOB_LANE)
    oob_uniform = b.const_i32(DW_DGRAD_WIN_OOB_UNIFORM)
    zero_acc = b.zero_vec_f32(8)
    wmma_op = wmma_fwd_op(p.dtype, arch)
    zero_f32 = b.const_f32(0.0)
    zero_half = _trunc_f32_elem(b, p.dtype, zero_f32)

    tid = b.thread_id_x()
    bx = b.block_id_x()
    by = b.block_id_y()
    n_tile, h0 = _dw_win_tile(b, p_Ho, ROWS)
    c_wave_size = b.const_i32(DW_FWD_WMMA_WAVE)
    lane = b.mod(tid, c_wave_size)
    wave = b.div(tid, c_wave_size)
    x0 = b.mul(bx, b.const_i32(TW))
    img0 = b.mul(n_tile, b.const_i32(F))
    ch_blk = b.mul(by, b.const_i32(BLOCK_CH))
    wave_ch = b.mul(wave, b.const_i32(CH))
    c_wave = b.add(ch_blk, wave_ch)
    ch_ok = b.cmp_lt(c_wave, p_G)
    c16 = b.const_i32(16)
    c8 = b.const_i32(8)
    col16 = b.mod(lane, c16)  # tile row m (A) / tile column n (B, D)
    t = b.div(lane, c16)  # K half of A/B, row octet (column parity) of D
    k = b.mod(col16, c8)
    q = b.div(col16, c8)

    a_rsrc = b.buffer_rsrc(params["A"], params["A_bytes"])
    b_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])
    d_rsrc = b.buffer_rsrc(params["D"], params["D_bytes"])

    # One-hot A fragments weights[r][ph]: W[k, r, s] at slot k.
    w_base = b.mul(b.add(c_wave, k), b.const_i32(KH * KW * 2))
    c_KW = b.const_i32(KW)
    is_k = [b.cmp_eq(k, b.const_i32(j)) for j in range(CH)]
    weights: List[List[Value]] = []
    for r in range(KH):
        row: List[Value] = []
        for ph in range(KWP):
            comps = []
            for j in range(2):
                s = b.sub(b.add(b.mul(t, c2), b.const_i32(4 * ph + j)), q)
                ok = b.land(ch_ok, b.land(b.cmp_ge(s, c0), b.cmp_lt(s, c_KW)))
                tap = b.add(b.const_i32(r * KW), s)
                off = b.select(ok, b.add(w_base, b.mul(tap, c2)), oob_lane)
                w = _buf_load_half(b, p.dtype, b_rsrc, off, c0)
                comps += [b.select(is_k[i], w, zero_half) for i in range(CH)]
            row.append(b.vec_pack(comps, io_type))
        weights.append(row)

    # Lane parts shared by the B reads and the D staging.
    c_sub = b.const_i32(SUB)
    f = b.div(col16, c_sub)
    pair2 = b.mul(b.mod(col16, c_sub), c2)
    c_per_px = b.const_i32(PER_PX)

    # Y staging [buffer, image column, channel]: lane (n, t) writes its 8
    # channels of column 2 (n % SUB) + t; the block then drains each pixel's
    # BLOCK_CH channels in 16-byte chunks.
    o_smem = b.smem_alloc(io_type, [2, F * TW, CH_PAD], name_hint="y_stage")
    o_col = b.add(b.mul(f, b.const_i32(TW)), b.add(pair2, t))
    n_out = F * TW * PER_PX
    c_TW = b.const_i32(TW)
    drain: List[Tuple[Value, Value, Value]] = []
    for i in range(ceil_div(n_out, THREADS)):
        idx = b.add(tid, b.const_i32(i * THREADS))
        px = b.div(idx, c_per_px)
        ch8 = b.mul(b.mod(idx, c_per_px), c8)
        img_o = b.add(img0, b.div(px, c_TW))
        wo = b.add(x0, b.mod(px, c_TW))
        ch_o = b.add(ch_blk, ch8)
        ok = b.land(
            b.land(b.cmp_lt(idx, b.const_i32(n_out)), b.cmp_lt(img_o, p_N)),
            b.land(b.cmp_lt(wo, p_Wo), b.cmp_lt(ch_o, p_G)),
        )
        g_off = b.add(b.mul(img_o, y_img), b.add(b.mul(wo, y_px), b.mul(ch_o, c2)))
        drain.append((px, ch8, b.select(ok, g_off, oob_lane)))

    # X staging [buffer, window column, channel]: the block loads the U
    # window columns of each image with 16-byte chunks; padding columns,
    # images past p_N and channels past p_groups read zero through the lane
    # sentinel.
    n_in = F * U * PER_PX
    in_passes = ceil_div(n_in, THREADS)
    in_px = ceil_div(in_passes * THREADS, PER_PX)
    i_smem = b.smem_alloc(io_type, [2, in_px, CH_PAD], name_hint="x_stage")
    c_U = b.const_i32(U)
    in_plan: List[Tuple[Value, Value, Value]] = []
    for i in range(in_passes):
        idx = b.add(tid, b.const_i32(i * THREADS))
        px = b.div(idx, c_per_px)
        ch8 = b.mul(b.mod(idx, c_per_px), c8)
        img_i = b.add(img0, b.div(px, c_U))
        wi = b.add(x0, b.sub(b.mod(px, c_U), b.const_i32(PAD)))
        ch_i = b.add(ch_blk, ch8)
        ok = b.land(
            b.land(b.cmp_lt(idx, b.const_i32(n_in)), b.cmp_lt(img_i, p_N)),
            b.land(b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, p_Wi)), b.cmp_lt(ch_i, p_G)),
        )
        g_off = b.add(b.mul(img_i, x_img), b.add(b.mul(wi, x_px), b.mul(ch_i, c2)))
        in_plan.append((px, ch8, b.select(ok, g_off, oob_lane)))
    f_u = b.mul(f, c_U)
    t2 = b.mul(t, c2)
    b_px = [
        [b.add(f_u, b.add(pair2, b.add(t2, b.const_i32(4 * ph + j)))) for j in range(2)]
        for ph in range(KWP)
    ]

    row_ok: Dict[int, Value] = {}

    def load_row(y: int) -> List[Value]:
        """Issue the 16-byte X window loads of local input row ``y``."""
        rel = y - PAD
        hi = b.add(h0, b.const_i32(rel))
        ok = b.cmp_lt(hi, p_Hi)
        if rel < 0:
            ok = b.land(b.cmp_ge(hi, c0), ok)
        row_ok[y] = ok
        x_r = b.select(ok, b.mul(hi, x_row), oob_uniform)
        vals = [
            _buf_load_vN(b, p.dtype, a_rsrc, b.add(off, x_r), c0, 4)
            for _, _, off in in_plan
        ]
        b.sched_barrier(0)
        return vals

    def y_row_part(o_local: int) -> Value:
        ho = b.add(h0, b.const_i32(o_local))
        return b.select(b.cmp_lt(ho, p_Ho), b.mul(ho, y_row), oob_uniform)

    def drain_out(buf: Value, y_r: Value) -> None:
        for px, ch8, g_off in drain:
            v = b.smem_load_vN(o_smem, buf, px, ch8, dtype=io_type, n=8)
            _buf_store_vN(b, p.dtype, d_rsrc, b.add(g_off, y_r), c0, v, 4)

    n_rows = ROWS + KH - 1
    PF = spec.prefetch_rows
    pending: Dict[int, List[Value]] = {}
    for y in range(min(PF, n_rows)):
        pending[y] = load_row(y)

    def stage_row(raw: List[Value], buf: Value) -> None:
        for (px, ch8, _), v in zip(in_plan, raw):
            b.smem_store_vN(i_smem, [buf, px, ch8], v, 8)

    def multiply(accs: List[Value], buf: Value, rs: List[int]) -> List[Value]:
        xs = [
            b.vec_concat(
                *(
                    b.smem_load_vN(
                        i_smem, buf, b_px[ph][j], wave_ch, dtype=io_type, n=8
                    )
                    for j in range(2)
                )
            )
            for ph in range(KWP)
        ]
        out = list(accs)
        for ph in range(KWP):
            for i, r in enumerate(rs):
                out[i] = b.mma(wmma_op, weights[r][ph], xs[ph], out[i])
        return out

    # Input rows outside the tile's own output rows (the KH - 1 halo rows)
    # are the ones that fall outside the image when a tile covers it: such a
    # row adds nothing, so its staging and WMMAs run under a block-uniform
    # guard (an scf.if, and a one-trip loop carrying the accumulators). Every
    # phase still syncs and drains its finished row, so the double buffers
    # keep alternating.
    c1 = b.const_i32(1)
    pend_out: Optional[Tuple[Value, Value]] = None
    acc: List[Value] = [zero_acc] * KH
    for y in range(n_rows):
        # Filter rows r whose output row y - r this block owns.
        rs = [r for r in range(KH) if 0 <= y - r < ROWS]
        slots = [(y - r) % KH for r in rs]
        raw = pending.pop(y)
        buf = b.const_i32(y % 2)
        rel = y - PAD
        halo = rel < 0 or rel >= ROWS

        if halo:
            with b.scf_if(row_ok[y]):
                stage_row(raw, buf)
        else:
            stage_row(raw, buf)
        if pend_out is not None:
            b.smem_store_vN(o_smem, [buf, o_col, wave_ch], pend_out[0], 8)
        b.sync_lds_only()
        if pend_out is not None:
            drain_out(buf, pend_out[1])
            pend_out = None
        if y + PF < n_rows:
            pending[y + PF] = load_row(y + PF)
        if halo:
            trip = b.select(row_ok[y], c1, c0)
            loop = b.scf_for_iter(
                c0,
                trip,
                c1,
                [(f"dfw_y{y}_i{i}", acc[sl]) for i, sl in enumerate(slots)],
                iv_name=f"dfw_y{y}",
            )
            with loop as (_, accs):
                b.scf_yield(*multiply(accs, buf, rs))
            new = loop.results
        else:
            new = multiply([acc[sl] for sl in slots], buf, rs)
        for sl, v in zip(slots, new):
            acc[sl] = v
        o_local = y - (KH - 1)
        if not 0 <= o_local < ROWS:
            continue
        slot = o_local % KH
        pend_out = (_trunc_f32(b, p.dtype, acc[slot]), y_row_part(o_local))
        acc[slot] = zero_acc
    if pend_out is not None:
        buf = b.const_i32(n_rows % 2)
        b.smem_store_vN(o_smem, [buf, o_col, wave_ch], pend_out[0], 8)
        b.sync_lds_only()
        drain_out(buf, pend_out[1])
    return b.kernel
