# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Depthwise wgrad kernels on the VALU, and the partial-slot fold.

A depthwise wgrad does ``2 * KH * KW`` FLOPs per dY element, so it is bound by
the bytes of X and dY: the kernels here read both once and keep the MFMA out
of it. Every block reduces over the spatial work it owns and stores one
partial slot ``ws[part][tap][c]`` with a plain store; :class:`DwWgradFoldSpec`
sums the partitions into ``dW``. (fp32 atomics on a depthwise ``dW`` -- a few
thousand addresses -- serialize: same-address atomics cost ~100 ns each.)

* :class:`DwWgradLdsSpec` -- ``C % 8 == 0``: a block owns 8 channels per lane
  and a column tile, streams output rows, and DMAs the S new X rows and one
  dY row per step into LDS rings.
* :class:`DwWgradColsSpec` -- small ``C`` (stem layers, ``C = 3``): lanes own a
  filter row and a strip of output columns instead of channels.

dW is ``[C, KH, KW, 1]``; the fold writes it whole, so it needs no zeroing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rocke.core.ir import F32, IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    _io_type,
    dilation_reason,
    emit_direct_params,
)

_WAVE = 64


# ---------------------------------------------------------------------------
# Fold: dW[c, tap] = sum_part ws[part][tap][c]
# ---------------------------------------------------------------------------

_FOLD_SLICES = 16
_FOLD_COLS = 64
# Partition loads each slice keeps in flight.
_FOLD_UNROLL = 4


@dataclass(frozen=True)
class DwWgradFoldSpec:
    """Fold of ``taps`` partial gradients per output channel into an fp32 dW."""

    taps: int
    name: str = "dw_wgrad_fold"

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(self.name, f"t{self.taps}")

    def launch_geometry(self, channels: int):
        return (ceil_div(channels * self.taps, _FOLD_COLS), 1, 1), (
            _FOLD_SLICES * _WAVE,
            1,
            1,
        )


# Kernargs after the A/D pointers and byte sizes, in order.
DW_WGRAD_FOLD_ARGS = ("parts", "channels")


def dw_wgrad_fold_signature(spec: DwWgradFoldSpec) -> list:
    """A = the fp32 partial slots, D = dW, as the AOT plans bind helper steps."""
    from rocke.helpers.spec import SignatureBuilder

    sb = (
        SignatureBuilder()
        .ptr("A", "fp32")
        .ptr("D", "fp32")
        .scalar("A_bytes", "i32")
        .scalar("D_bytes", "i32")
    )
    for name in DW_WGRAD_FOLD_ARGS:
        sb = sb.scalar(name, "i32")
    return sb.build()


def build_dw_wgrad_fold(spec: DwWgradFoldSpec, arch: str = "gfx950") -> KernelDef:
    """Each block owns 64 outputs; 16 waves split the partition axis, meet in LDS."""
    from rocke.core.ir import I32, PtrType

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _FOLD_SLICES * _WAVE
    ws = b.param("A", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    dw = b.param(
        "D",
        PtrType(F32, "global"),
        noalias=True,
        writeonly=True,
        align=4,
    )
    b.param("A_bytes", I32)
    b.param("D_bytes", I32)
    parts = b.param("parts", I32)
    chans = b.param("channels", I32)

    def ci(v):
        return b.const_i32(v)

    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    slc = b.div(tid, ci(_WAVE))
    n_out = b.mul(chans, ci(spec.taps))
    o = b.add(b.mul(b.block_id_x(), ci(_FOLD_COLS)), lane)
    o_ok = b.cmp_lt(o, n_out)
    o_safe = b.select(o_ok, o, ci(0))
    zero = b.const_f32(0.0)
    U = _FOLD_UNROLL
    # Partition p = slc + SLICES * (U * it + j).
    n_it = b.div(b.add(parts, ci(_FOLD_SLICES * U - 1)), ci(_FOLD_SLICES * U))
    loop = b.scf_for_iter(
        ci(0),
        n_it,
        ci(1),
        [(f"s{j}", zero) for j in range(U)],
        iv_name="it",
        elide_trailing_barrier=False,
    )
    with loop as (it, acc):
        acc = list(acc)
        for j in range(U):
            pidx = b.add(slc, b.mul(b.add(b.mul(it, ci(U)), ci(j)), ci(_FOLD_SLICES)))
            ok = b.cmp_lt(pidx, parts)
            v = b.global_load_f32(
                ws, b.add(b.mul(b.select(ok, pidx, ci(0)), n_out), o_safe)
            )
            acc[j] = b.fadd(acc[j], b.select(ok, v, zero))
        b.scf_yield(*acc)
    tot = loop.results[0]
    for v in loop.results[1:]:
        tot = b.fadd(tot, v)
    red = b.smem_alloc(F32, [1, _FOLD_SLICES * _WAVE], name_hint="fold")
    b.smem_store_vN_f32(red, [ci(0), tid], tot, 1)
    b.sync()
    with b.scf_if(b.land(b.cmp_eq(slc, ci(0)), o_ok)):
        t = None
        for q in range(_FOLD_SLICES):
            v = b.vec_extract(
                b.smem_load_vN_f32(red, ci(0), b.add(lane, ci(q * _WAVE)), n=1), 0
            )
            t = v if t is None else b.fadd(t, v)
        tap = b.div(o, chans)
        c = b.mod(o, chans)
        off = b.add(b.mul(c, ci(spec.taps)), tap)
        b.global_store(dw, off, t, align=4)
    return b.kernel


# ---------------------------------------------------------------------------
# Small-C depthwise wgrad: lanes along output columns
# ---------------------------------------------------------------------------
#
# At C = 3 a channel-vectorised lane would idle most of its width, so here a
# lane owns one filter row r and a strip of `strip` consecutive output
# columns, keeps KW x C accumulators, and per output row loads the dY strip
# and the X row segment it touches as whole 16-byte vectors (the pixels are
# C * 2 bytes, so the vectors straddle pixels; edge columns are masked with
# per-lane masks that do not change along the rows). Block x is the filter
# row; block y is a partition that walks a contiguous run of (image, output
# row, column group) units and stores one partial slot ws[part][r][s][c],
# which DwWgradFoldSpec folds.


@dataclass(frozen=True)
class DwWgradColsSpec:
    problem: DirectConvProblem
    name: str = "dw_wgrad_cols"
    strip: int = 8
    threads: int = 64
    target_blocks: int = 2048

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        flags = {"bf16": True} if p.dtype == "bf16" else {}
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}p{p.PAD}s{p.stride}c{p.groups}",
            f"st{self.strip}",
            f"t{self.threads}",
            f"tb{self.target_blocks}",
            flags=flags,
        )

    def n_units(self) -> int:
        p = self.problem
        return ceil_div(p.N * p.Ho * (p.Wo // self.strip), self.threads)

    def n_partitions(self) -> int:
        return min(self.n_units(), ceil_div(self.target_blocks, self.problem.KH))

    def workspace_bytes(self) -> int:
        p = self.problem
        return self.n_partitions() * p.KH * p.KW * p.groups * 4

    def launch_geometry(self):
        return (self.problem.KH, self.n_partitions(), 1), (self.threads, 1, 1)

    def lds_bytes(self) -> int:
        """The block reduction buffer: KW x C fp32 accumulators per thread."""
        p = self.problem
        return self.threads * p.KW * p.groups * 4


def is_valid_dw_wgrad_cols_spec(spec: DwWgradColsSpec, arch: str = "gfx950"):
    p = spec.problem
    why = dilation_reason(p)
    if why is not None:
        return False, why
    if p.cpg != 1 or p.kpg != 1:
        return False, "depthwise only"
    if p.dtype not in ("fp16", "bf16"):
        return False, "fp16/bf16 only"
    if p.groups > 8:
        return False, "small-C kernel: C <= 8"
    if p.KW * p.groups > 96:
        return False, "too many accumulators"
    if spec.threads < _WAVE or spec.threads % _WAVE or spec.threads > 1024:
        return False, "threads must be a multiple of 64"
    C = p.groups
    # Whole 16-byte vectors at a build-time phase: the strip and the rows
    # must keep every vector start on an 8-element boundary.
    if (spec.strip * C) % 8 or (spec.strip * p.stride * C) % 8:
        return False, "strip * C (and strip * stride * C) must be multiples of 8"
    if (p.W * C) % 8 or (p.Wo * C) % 8 or spec.strip < 1 or p.Wo % spec.strip:
        return False, "W*C and Wo*C must be multiples of 8 and Wo of the strip"
    if spec.target_blocks < 1:
        return False, "target_blocks >= 1"
    if spec.lds_bytes() > 160 * 1024:
        return False, f"LDS {spec.lds_bytes()} B > 160 KiB"
    return True, "ok"


def build_dw_wgrad_cols(spec: DwWgradColsSpec, arch: str = "gfx950") -> KernelDef:
    ok, why = is_valid_dw_wgrad_cols_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DwWgradColsSpec: {why}")
    p = spec.problem
    KH, KW, PAD, S, C = p.KH, p.KW, p.PAD, p.stride, p.groups
    WS = spec.strip
    T = spec.threads
    io_type = _io_type(p.dtype)
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = T
    params = emit_direct_params(b, direction="wgrad", io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci((1 << 31) - 1)
    p_H, p_W = params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    tid = b.thread_id_x()
    r_tap = b.block_id_x()  # filter row
    part = b.block_id_y()
    strips_per_row = b.div(p_Wo, ci(WS))
    n_strips = b.mul(b.mul(params["p_N"], p_Ho), strips_per_row)
    n_units = b.div(b.add(n_strips, ci(T - 1)), ci(T))
    n_part = b.smin(n_units, ci(ceil_div(spec.target_blocks, KH)))
    u_lo = b.div(b.mul(part, n_units), n_part)
    u_hi = b.div(b.mul(b.add(part, c1), n_units), n_part)

    a_rsrc = b.buffer_rsrc(params["A"], params["A_bytes"])  # dY
    b_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])  # X
    X_COLS = (WS - 1) * S + KW
    # Window element e (column e // C of the window) sits at vector element
    # e + PH: the window starts PAD columns left of an 8-aligned strip.
    PH = (-PAD * C) % 8
    n_dy_vec = WS * C // 8
    n_x_vec = ceil_div(PH + X_COLS * C, 8)

    def load8(rsrc, elem_off: Value, ok: Value) -> Value:
        voff = b.select(ok, b.mul(elem_off, c2), oob)
        if p.dtype == "bf16":
            return b.buffer_load_vN_bf16(rsrc, voff, c0, 4)
        return b.buffer_load_vN_f16(rsrc, voff, c0, 4)

    def elem(vecs, e):
        return b.cast_to_f32(b.vec_extract(vecs[e // 8], e % 8))

    zero = b.const_f32(0.0)
    n_acc = KW * C
    loop = b.scf_for_iter(
        u_lo,
        u_hi,
        c1,
        [(f"acc{i}", zero) for i in range(n_acc)],
        iv_name="unit",
        elide_trailing_barrier=False,
    )
    with loop as (unit, carried):
        accs = list(carried)
        strip = b.add(b.mul(unit, ci(T)), tid)
        live = b.cmp_lt(strip, n_strips)
        cs = b.mod(strip, strips_per_row)
        rest = b.div(strip, strips_per_row)
        ho = b.mod(rest, p_Ho)
        n_img = b.div(rest, p_Ho)
        wo0 = b.mul(cs, ci(WS))
        hi = b.add(b.mul(ho, ci(S)), b.sub(r_tap, ci(PAD)))
        hi_ok = b.land(live, b.land(b.cmp_ge(hi, c0), b.cmp_lt(hi, p_H)))
        wi0 = b.sub(b.mul(wo0, ci(S)), ci(PAD))
        dy_base = b.add(
            b.add(
                b.mul(n_img, params["p_dY_stride_n"]),
                b.mul(ho, params["p_dY_stride_ho"]),
            ),
            b.mul(wo0, params["p_dY_stride_wo"]),
        )
        x_base = b.add(
            b.add(
                b.mul(n_img, params["p_X_stride_n"]), b.mul(hi, params["p_X_stride_hi"])
            ),
            b.mul(wi0, params["p_X_stride_wi"]),
        )
        dy_vecs = [
            load8(a_rsrc, b.add(dy_base, ci(8 * i)), live) for i in range(n_dy_vec)
        ]
        x_vecs = [
            load8(b_rsrc, b.add(x_base, ci(8 * i - PH)), hi_ok) for i in range(n_x_vec)
        ]
        dy = [[elem(dy_vecs, j * C + c) for c in range(C)] for j in range(WS)]
        xs = []
        for col in range(X_COLS):
            wi = b.add(wi0, ci(col))
            okc = b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, p_W))
            xs.append(
                [b.select(okc, elem(x_vecs, PH + col * C + c), zero) for c in range(C)]
            )
        for j in range(WS):
            for s in range(KW):
                for c in range(C):
                    i = s * C + c
                    accs[i] = b.fma(dy[j][c], xs[j * S + s][c], accs[i])
        b.scf_yield(*accs)
    accs = list(loop.results)

    # Block reduction: every lane holds this block's r row; sum over lanes.
    red = b.smem_alloc(F32, [1, T * n_acc], name_hint="red")
    for i in range(n_acc):
        b.smem_store_vN_f32(red, [c0, b.add(b.mul(ci(i), ci(T)), tid)], accs[i], 1)
    b.sync()
    # Thread tid sums outputs tid, tid + T, ... (KW * C may exceed T).
    for k in range(ceil_div(n_acc, T)):
        o = tid if k == 0 else b.add(tid, ci(k * T))
        with b.scf_if(b.cmp_lt(o, ci(n_acc))):
            tot = None
            for q in range(T):
                v = b.vec_extract(
                    b.smem_load_vN_f32(red, c0, b.add(b.mul(o, ci(T)), ci(q)), n=1), 0
                )
                tot = v if tot is None else b.fadd(tot, v)
            # ws[part][tap][c], tap = r*KW + s; output o is (s, c) = divmod(o, C).
            s_i = b.div(o, ci(C))
            c_i = b.mod(o, ci(C))
            tap = b.add(b.mul(r_tap, ci(KW)), s_i)
            off = b.add(b.mul(b.add(b.mul(part, ci(KH * KW)), tap), ci(C)), c_i)
            b.global_store(params["D"], off, tot, align=4)
    return b.kernel


# ---------------------------------------------------------------------------
# Depthwise wgrad, LDS row rings
# ---------------------------------------------------------------------------
#
# DwWgradSpec re-reads its KH input rows per output row from L2, and with a
# few MB of rows in flight per XCD the L2 thrashes (HBM traffic ~1.75x the
# tensors). Here the block owns a channel tile and a column tile and streams
# output rows: per step it DMAs S new X rows and one dY row into LDS rings
# (prefetch steps ahead), and every lane reads its window out of LDS. Each
# byte of X and dY leaves HBM once per (channel tile, column tile).
#
# Ring slots fold to constants by unrolling U = prefetch + ceil(KH / S) steps:
# the X ring has S*U row slots and the dY ring U.


@dataclass(frozen=True)
class DwWgradLdsSpec:
    problem: DirectConvProblem
    name: str = "dw_wgrad_lds"
    lanes_c: int = 8  # lanes along channels (8 channels each)
    lanes_w: int = 32  # lanes along columns
    cols_per_lane: int = 1
    prefetch: int = 2
    rows_per_seg: int = 32  # output rows per segment (rounded up to the unroll)
    target_blocks: int = 1024

    @property
    def block_c(self) -> int:
        return 8 * self.lanes_c

    @property
    def block_wo(self) -> int:
        return self.lanes_w * self.cols_per_lane

    @property
    def threads(self) -> int:
        return self.lanes_c * self.lanes_w

    @property
    def unroll(self) -> int:
        return self.prefetch + ceil_div(self.problem.KH, self.problem.stride)

    def x_cols(self) -> int:
        p = self.problem
        return (self.block_wo - 1) * p.stride + p.KW

    def rps(self) -> int:
        return ceil_div(self.rows_per_seg, self.unroll) * self.unroll

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        flags = {"bf16": True} if p.dtype == "bf16" else {}
        return kernel_name_join(
            self.name,
            f"f{p.KH}x{p.KW}p{p.PAD}s{p.stride}",
            f"lc{self.lanes_c}",
            f"lw{self.lanes_w}",
            f"cpl{self.cols_per_lane}",
            f"pf{self.prefetch}",
            f"rps{self.rows_per_seg}",
            f"tb{self.target_blocks}",
            flags=flags,
        )

    def n_channel_tiles(self) -> int:
        return ceil_div(self.problem.groups, self.block_c)

    def n_units(self) -> int:
        p = self.problem
        return p.N * ceil_div(p.Wo, self.block_wo) * ceil_div(p.Ho, self.rps())

    def n_partitions(self) -> int:
        return min(self.n_units(), ceil_div(self.target_blocks, self.n_channel_tiles()))

    def workspace_bytes(self) -> int:
        p = self.problem
        return self.n_partitions() * p.KH * p.KW * p.groups * 4

    def launch_geometry(self):
        return (self.n_channel_tiles(), self.n_partitions(), 1), (self.threads, 1, 1)

    def dma_instrs(self, cols: int) -> int:
        """16-byte DMA instructions per wave for one LDS row of ``cols`` pixels."""
        n_waves = self.threads // _WAVE
        return ceil_div(ceil_div(cols * (self.block_c // 8), _WAVE), n_waves)

    def x_row_elems(self) -> int:
        """Elements of one X ring slot: whole DMA instructions of every wave."""
        return self.threads * self.dma_instrs(self.x_cols()) * 8

    def dy_row_elems(self) -> int:
        return self.threads * self.dma_instrs(self.block_wo) * 8

    def lds_bytes(self) -> int:
        """The X and dY rings as the builder allocates them, plus the reduction."""
        U = self.unroll
        x = self.problem.stride * U * self.x_row_elems() * 2
        dy = U * self.dy_row_elems() * 2
        red = self.lanes_w * self.block_c * 4
        return x + dy + red


def is_valid_dw_wgrad_lds_spec(spec: DwWgradLdsSpec, arch: str = "gfx950"):
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    why = dilation_reason(p)
    if why is not None:
        return False, why
    if target.wave_size != 64 or target.async_lds_max_dwords < 4:
        return False, f"needs wave64 + 16-byte buffer_load_lds (absent on {arch})"
    if p.cpg != 1 or p.kpg != 1:
        return False, "depthwise only"
    if p.dtype not in ("fp16", "bf16"):
        return False, "fp16/bf16 only"
    if p.groups % 8:
        return False, "C must be a multiple of 8"
    if spec.threads < _WAVE or spec.threads % _WAVE or spec.threads > 1024:
        return False, "threads must be a multiple of 64, at most 1024"
    if p.KH * p.KW * 8 > 128:
        return False, "too many accumulators"
    if spec.prefetch < 1:
        return False, "prefetch >= 1"
    if spec.lds_bytes() > 160 * 1024:
        return False, f"LDS {spec.lds_bytes()} B > 160 KiB"
    if spec.rows_per_seg < 1:
        return False, "rows_per_seg >= 1"
    if spec.target_blocks < 1:
        return False, "target_blocks >= 1"
    return True, "ok"


def build_dw_wgrad_lds(spec: DwWgradLdsSpec, arch: str = "gfx950") -> KernelDef:
    from rocke.core.ir import I64

    ok, why = is_valid_dw_wgrad_lds_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DwWgradLdsSpec: {why}")
    p = spec.problem
    KH, KW, PAD, S = p.KH, p.KW, p.PAD, p.stride
    TAPS = KH * KW
    LC, LW, WL = spec.lanes_c, spec.lanes_w, spec.cols_per_lane
    BC, BWO = spec.block_c, spec.block_wo
    CH = BC // 8  # 16-byte chunks per pixel
    XC = spec.x_cols()
    T = spec.threads
    D = spec.prefetch
    U = spec.unroll
    NX, NY = S * U, U
    RPS = spec.rps()
    PRE = max(KH - S, 0)  # X rows a segment needs before its first step's own
    io_type = _io_type(p.dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = T
    b.kernel.attrs["lds_alias_scopes"] = True
    params = emit_direct_params(b, direction="wgrad", io_type=io_type)

    def ci(v: int) -> Value:
        return b.const_i32(v)

    c0, c1, c2 = ci(0), ci(1), ci(2)
    oob = ci((1 << 31) - 1)
    p_C = params["p_groups"]
    p_H, p_W = params["p_Hi"], params["p_Wi"]
    p_Ho, p_Wo = params["p_Ho"], params["p_Wo"]
    tid = b.thread_id_x()
    lane = b.mod(tid, ci(_WAVE))
    wave = b.readfirstlane(b.div(tid, ci(_WAVE)))
    lc = b.mod(tid, ci(LC))
    lw = b.div(tid, ci(LC))
    ctile, part = b.block_id_x(), b.block_id_y()
    c_blk0 = b.mul(ctile, ci(BC))
    n_ct = b.div(b.add(p_Wo, ci(BWO - 1)), ci(BWO))
    n_hb = b.div(b.add(p_Ho, ci(RPS - 1)), ci(RPS))
    n_units = b.mul(b.mul(params["p_N"], n_ct), n_hb)
    n_tiles = b.div(b.add(p_C, ci(BC - 1)), ci(BC))
    n_part = b.smin(
        n_units, b.div(b.add(ci(spec.target_blocks), b.sub(n_tiles, c1)), n_tiles)
    )

    a_rsrc = b.buffer_rsrc(params["A"], params["A_bytes"])  # dY
    b_rsrc = b.buffer_rsrc(params["B"], params["B_bytes"])  # X
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

    # ---- LDS rings: one allocation per slot so the waitcnt pass can tell them apart
    IX = spec.dma_instrs(XC)  # DMA instrs per wave per X row
    IY = spec.dma_instrs(BWO)
    X_ELEMS = spec.x_row_elems()
    Y_ELEMS = spec.dy_row_elems()
    x_bufs = [
        b.smem_alloc(io_type, [1, X_ELEMS], name_hint=f"xr{i}") for i in range(NX)
    ]
    y_bufs = [
        b.smem_alloc(io_type, [1, Y_ELEMS], name_hint=f"yr{i}") for i in range(NY)
    ]
    x_addr = [b.smem_addr_of(s) for s in x_bufs]
    y_addr = [b.smem_addr_of(s) for s in y_bufs]

    # DMA lane geometry: slot L = (wave * I + i) * 64 + lane -> (col, chunk).
    def dma_lanes(n_instr: int, cols: int):
        out = []
        for i in range(n_instr):
            slot = b.add(b.mul(b.add(b.mul(wave, ci(n_instr)), ci(i)), ci(_WAVE)), lane)
            col = b.div(slot, ci(CH))
            ch = b.mul(b.mod(slot, ci(CH)), ci(8))
            out.append((col, ch, b.cmp_lt(col, ci(cols))))
        return out

    x_lanes = dma_lanes(IX, XC)
    y_lanes = dma_lanes(IY, BWO)

    def dma(rsrc, addr, n_instr, i, voff):
        lds = b.smem_ptr_add(
            addr,
            b.zext(b.mul(b.add(b.mul(wave, ci(n_instr)), ci(i)), ci(_WAVE * 16)), I64),
        )
        b.async_buffer_load_lds(rsrc, lds, voff, c0, 4)

    # Fragment read element indices (lane: chunk lc of a column).
    def col_index(col: Value) -> Value:
        return b.add(b.mul(col, ci(BC)), b.mul(lc, ci(8)))

    def cvt8(v: Value) -> List[Value]:
        return [b.cast_to_f32(b.vec_extract(v, e)) for e in range(8)]

    zero = b.const_f32(0.0)
    n_acc = TAPS * 8
    zero_accs = [(f"acc{i}", zero) for i in range(n_acc)]

    seg_loop = b.scf_for_iter(
        part, n_units, n_part, zero_accs, iv_name="unit", elide_trailing_barrier=False
    )
    with seg_loop as (unit, carried):
        accs = list(carried)
        hb = b.mod(unit, n_hb)
        rest = b.div(unit, n_hb)
        ct = b.mod(rest, n_ct)
        n_img = b.div(rest, n_ct)
        h0 = b.mul(hb, ci(RPS))
        q0 = b.mul(ct, ci(BWO))
        xi0 = b.sub(b.mul(q0, ci(S)), ci(PAD))  # first X column of the tile
        x_row0 = b.sub(b.mul(h0, ci(S)), ci(PAD))  # X row of relative index 0
        n_steps = b.smin(ci(RPS), b.sub(p_Ho, h0))

        # Per-lane DMA offsets of this unit (rows added per issue).
        x_offs = []
        for col, ch, okl in x_lanes:
            wi = b.add(xi0, col)
            okc = b.land(okl, b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, p_W)))
            cc = b.add(c_blk0, ch)
            okc = b.land(okc, b.cmp_lt(cc, p_C))
            x_offs.append((b.add(b.add(b.mul(n_img, sx_n), b.mul(wi, sx_w)), cc), okc))
        y_offs = []
        for col, ch, okl in y_lanes:
            wo = b.add(q0, col)
            cc = b.add(c_blk0, ch)
            oky = b.land(okl, b.land(b.cmp_lt(wo, p_Wo), b.cmp_lt(cc, p_C)))
            y_offs.append(
                (b.add(b.add(b.mul(n_img, sdy_n), b.mul(wo, sdy_w)), cc), oky)
            )

        def issue_x(xr: Value, slot: int):
            hi = b.add(x_row0, xr)
            # Rows of later steps than the segment's own stay masked.
            ok_r = b.land(
                b.land(b.cmp_ge(hi, c0), b.cmp_lt(hi, p_H)),
                b.cmp_lt(xr, b.add(b.mul(n_steps, ci(S)), ci(PRE))),
            )
            for i, (off, okl) in enumerate(x_offs):
                voff = b.select(
                    b.land(okl, ok_r), b.mul(b.add(off, b.mul(hi, sx_h)), c2), oob
                )
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
            """DMA the S X rows that step t adds and dY row t (tm = t mod U)."""
            for j in range(S):
                xr = b.add(b.mul(t, ci(S)), ci(PRE + j))
                issue_x(xr, (S * tm + PRE + j) % NX)
            issue_y(t, tm % NY)

        PER_STEP = S * IX + IY
        # Prologue: the PRE leading X rows (waited for in full), then steps 0..D-1.
        for j in range(PRE):
            issue_x(ci(j), j % NX)
        for t in range(D):
            issue_step(ci(t), t)

        def compute(tm: int, accs):
            dyv = []
            for jw in range(WL):
                col = b.add(b.mul(lw, ci(WL)), ci(jw))
                dyv.append(
                    cvt8(
                        b.smem_load_vN(
                            y_bufs[tm % NY], c0, col_index(col), dtype=io_type, n=8
                        )
                    )
                )
            accs = list(accs)
            for r in range(KH):
                slot = (S * tm + r) % NX
                win = {}
                for jw in range(WL):
                    for s in range(KW):
                        xc = jw * S + s
                        if xc not in win:
                            col = b.add(b.mul(lw, ci(WL * S)), ci(xc))
                            win[xc] = cvt8(
                                b.smem_load_vN(
                                    x_bufs[slot], c0, col_index(col), dtype=io_type, n=8
                                )
                            )
                        xv = win[xc]
                        for e in range(8):
                            i = (r * KW + s) * 8 + e
                            accs[i] = b.fma(dyv[jw][e], xv[e], accs[i])
            return accs

        def step(t: Value, tm: int, accs):
            b.s_waitcnt(vmcnt=(D - 1) * PER_STEP, lgkmcnt=0)
            b.s_barrier_bare()
            issue_step(b.add(t, ci(D)), (tm + D) % U)
            return compute(tm, accs)

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

    # ---- reduce over the LW column lanes, store ws[part][tap][c] ----
    red = b.smem_alloc(F32, [1, LW * BC], name_hint="red")
    for t in range(TAPS):
        base = b.add(b.mul(lw, ci(BC)), b.mul(lc, ci(8)))
        for e0 in (0, 4):
            vec = b.vec_pack([accs[t * 8 + e0 + e] for e in range(4)], F32)
            b.smem_store_vN_f32(red, [c0, b.add(base, ci(e0))], vec, 4)
        b.sync()
        for k in range(ceil_div(BC, T)):
            o = b.add(tid, ci(k * T))
            o_ok = b.cmp_lt(o, ci(BC))
            tot = None
            for q in range(LW):
                idx = b.select(o_ok, b.add(ci(q * BC), o), c0)
                v = b.vec_extract(b.smem_load_vN_f32(red, c0, idx, n=1), 0)
                tot = v if tot is None else b.fadd(tot, v)
            c_glb = b.add(c_blk0, o)
            okw = b.land(o_ok, b.cmp_lt(c_glb, p_C))
            off = b.add(b.mul(b.add(b.mul(part, ci(TAPS)), ci(t)), p_C), c_glb)
            with b.scf_if(okw):
                b.global_store(params["D"], off, tot, align=4)
        b.sync()
    return b.kernel
