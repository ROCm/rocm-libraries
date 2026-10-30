# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shape-generic Winograd ``F(mh x mw, 3 x 3)`` forward transform kernels (AOT).

The kernels of :mod:`kernels.common.conv_winograd_fwd` bake the problem
extents in, so every conv shape is its own set of binaries. The kernels here
compute the same K1 / K2a / K2 / K4 stages (same workspace layouts, same
arithmetic, same OOB-select addressing) but take every extent as a kernel
argument, so one binary serves every shape its knobs fit: they can be compiled
ahead of time like the direct and implicit-GEMM conv instances. K3 needs no
counterpart -- the WMMA batched GEMM
(:mod:`kernels.common.conv_winograd_gemm_wmma`) already takes its extents at
run time.

What stays compile time (the instance's knobs): the tile, the I/O and operand
dtypes, the channel vector width (the channel extent of the stage must be a
multiple of it), the workgroup size, the K2 run length or LDS region, and
whether the prescale is on. Divisions by runtime extents use a multiply-high
by a host-computed magic number (:func:`magic_div`), exact for the 31-bit
indices the kernels use.

Differences from the shape-specialised kernels:

* the fused prescale (``RtInputSpec.fused``, register path) reduces the
  tile maxima over a runtime number of channel lanes (<= ``block_size``, so
  ``C <= block_size * vec``), through xor shuffles and LDS; the split form
  (K2a per-pixel maxima, then K2) serves any C;
* K2's register path has no ``wg_tile_rows`` stacking;
* the LDS path uses a fixed number of channel lanes (``lds_lanes``) and loads
  ``vec``-channel pieces;
* the prescale limit (which depends on ``C / groups``) is a kernel argument.

Each stage's spec has a ``kernel_name()`` that is its AOT identity, and
:func:`stage_launch` turns (spec, problem) into the grid, block and kernarg
values of one launch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rocke.core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fwd import (
    _CPOL_NT,
    _MAX_BUFFER_BYTES,
    _OOB,
    _PRESCALE_LOG2_LIMIT,
    PRESCALE_BETA,
    _abs_bits_vec,
    _bits_to_f32,
    _hmax_bits,
    _IROps,
    _ir_dtype,
    _lincomb,
    _load_raw,
    _max_bits,
    _pack_vec_f32,
    _pow2_ceil,
    _pow2_f32,
    _raw_to_f32,
    _store_packed,
    _store_vec_f32,
    at_1d,
    bt_1d,
    filter_matrix_int,
)

_R = 3
_FILTER_BLOCK = 128
_PRESCALE_BLOCK = 256
_IO = ("fp16", "bf16")
_TILES = (2, 4, 6)
_PRESCALE_MAX_EXP = 100


# ---------------------------------------------------------------------------
# Runtime division
# ---------------------------------------------------------------------------


def magic_div(d: int) -> Tuple[int, int]:
    """``(magic, shift)`` with ``n // d == umulhi(n, magic) >> shift`` for
    ``0 <= n < 2**31``; ``d == 1`` is ``(0, 0)`` (the kernel selects ``n``).

    ``l = ceil(log2 d)``, ``magic = ceil(2**(31 + l) / d)`` (< 2**32 for
    ``d >= 2``): the rounding error ``magic * d - 2**(31 + l) < d <= 2**l``
    times ``n < 2**31`` stays below ``2**(31 + l)``, so the floor is exact.
    """
    if d < 1:
        raise ValueError(f"divisor must be >= 1 (got {d})")
    if d == 1:
        return 0, 0
    lg = (d - 1).bit_length()
    magic = -(-(1 << (31 + lg)) // d)
    assert magic < (1 << 32)
    return magic, lg - 1


def _i32(v: int) -> int:
    """An unsigned 32-bit kernarg as the signed value the launcher packs."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


class _Div:
    """A runtime divisor: kernargs ``<name>``, ``<name>_mg``, ``<name>_sh``."""

    def __init__(self, b: IRBuilder, name: str):
        self.b = b
        self.d = b.param(name, I32)
        self.mg = b.param(f"{name}_mg", I32)
        self.sh = b.param(f"{name}_sh", I32)

    def div(self, n: Value) -> Value:
        b = self.b
        q = b.lshr(b.umul_hi_i32(n, self.mg), self.sh)
        return b.select(b.cmp_eq(self.mg, b.const_i32(0)), n, q)

    def divmod(self, n: Value) -> Tuple[Value, Value]:
        q = self.div(n)
        return q, self.b.sub(n, self.b.mul(q, self.d))


def _div_values(name: str, d: int) -> dict:
    mg, sh = magic_div(d)
    return {name: d, f"{name}_mg": _i32(mg), f"{name}_sh": sh}


def _div_sig(sb: SignatureBuilder, name: str) -> SignatureBuilder:
    return (
        sb.scalar(name, "i32").scalar(f"{name}_mg", "i32").scalar(f"{name}_sh", "i32")
    )


def _oob(b: IRBuilder, ok: Value, off: Value) -> Value:
    return b.select(ok, off, b.const_i32(_OOB))


def _c(b: IRBuilder, v: int) -> Value:
    return b.const_i32(int(v))


def _mul_c(b: IRBuilder, v: Value, k: int) -> Value:
    """``v * k`` for a small constant ``k`` (0 and 1 folded)."""
    if k == 0:
        return b.const_i32(0)
    if k == 1:
        return v
    return b.mul(v, b.const_i32(k))


# ---------------------------------------------------------------------------
# Geometry (host side)
# ---------------------------------------------------------------------------


def _split_tile(m) -> Tuple[int, int]:
    if isinstance(m, (tuple, list)):
        return int(m[0]), int(m[1])
    return int(m), int(m)


@dataclass(frozen=True)
class RtGeometry:
    """Host-side extents of one problem (one batch chunk) for tile ``(mh, mw)``."""

    problem: ConvProblem
    mh: int
    mw: int

    @property
    def nh(self) -> int:
        return self.mh + _R - 1

    @property
    def nw(self) -> int:
        return self.mw + _R - 1

    @property
    def n2(self) -> int:
        return self.nh * self.nw

    @property
    def tiles_h(self) -> int:
        p = self.problem
        return ceil_div(ceil_div(p.Ho, p.dH), self.mh)

    @property
    def tiles_w(self) -> int:
        p = self.problem
        return ceil_div(ceil_div(p.Wo, p.dW), self.mw)

    @property
    def num_tiles(self) -> int:
        p = self.problem
        return p.N * p.dH * p.dW * self.tiles_h * self.tiles_w

    @property
    def prescale_scale(self) -> float:
        """``beta / 2**L`` of :func:`conv_winograd_fwd._prescale_exp`."""
        beta = math.sqrt(PRESCALE_BETA[self.mh] * PRESCALE_BETA[self.mw])
        lim = _PRESCALE_LOG2_LIMIT - math.ceil(math.log2(self.problem.cpg) / 2)
        return beta / float(2**lim)

    def buffer_bytes(self, prescale: bool) -> dict:
        p = self.problem
        t = self.num_tiles
        return {
            "x": p.N * p.Hi * p.Wi * p.C * 2,
            "w": p.K * 9 * p.cpg * 2,
            "y": p.N * p.Ho * p.Wo * p.K * 2,
            "U": self.n2 * p.K * p.cpg * 2,
            "V": t * self.n2 * p.C * 2,
            "M": t * self.n2 * p.K * 2,
            "e": (t * 4 + p.N * p.Hi * p.Wi * 4) if prescale else 4,
        }

    def check(self, prescale: bool) -> Optional[str]:
        p = self.problem
        if p.is_3d or (p.Y, p.X) != (3, 3) or (p.sH, p.sW) != (1, 1):
            return "Winograd F(m,3) needs a 2-D 3x3 stride-1 conv"
        if p.Ho <= 0 or p.Wo <= 0:
            return "empty output"
        for nm, nbytes in self.buffer_bytes(prescale).items():
            if nbytes > _MAX_BUFFER_BYTES:
                return f"{nm} needs {nbytes} bytes (32-bit buffer offsets)"
        return None


def _ps_bits(geo: RtGeometry) -> int:
    import struct

    return struct.unpack("<i", struct.pack("<f", geo.prescale_scale))[0]


def _prescale_exp_rt(b: IRBuilder, amax: Value, ps_scale: Value) -> Value:
    """``clamp(ceil(log2(amax * ps_scale)), 0, 100)``; ``ps_scale = beta / 2**L``."""
    bits = b.bitcast(b.fmul(amax, ps_scale), I32)
    expo = b.sub(b.land(b.lshr(bits, _c(b, 23)), _c(b, 0xFF)), _c(b, 127))
    frac = b.land(bits, _c(b, 0x7FFFFF))
    e_val = b.add(expo, b.select(b.cmp_ne(frac, _c(b, 0)), _c(b, 1), _c(b, 0)))
    return b.smin(b.smax(e_val, _c(b, 0)), _c(b, _PRESCALE_MAX_EXP))


# ---------------------------------------------------------------------------
# Stage specs
# ---------------------------------------------------------------------------


def _check_common(mh, mw, io, op, vec):
    if mh not in _TILES or mw not in _TILES:
        raise ValueError(f"tile must be in {_TILES} (got {mh}x{mw})")
    if io not in _IO or op not in _IO:
        raise ValueError("dtypes must be fp16 / bf16")
    if op == "bf16" and ((mh, mw) != (2, 2) or io != "bf16"):
        raise ValueError("bf16 operands only for F(2,3) with bf16 I/O")
    if vec not in (1, 2, 4, 8):
        raise ValueError(f"vec must be 1/2/4/8 (got {vec})")


def _tile_tag(mh, mw) -> str:
    return f"f{mh}" if mh == mw else f"f{mh}x{mw}"


@dataclass(frozen=True)
class RtFilterSpec:
    """K1: ``w -> U``; ``vec`` divides ``C / groups``."""

    mh: int = 4
    mw: int = 4
    io_dtype: str = "fp16"
    operand_dtype: str = "fp16"
    vec: int = 8
    filter_rows: int = 1

    def __post_init__(self):
        _check_common(self.mh, self.mw, self.io_dtype, self.operand_dtype, self.vec)
        if (self.mh + 2) % self.filter_rows:
            raise ValueError("filter_rows must divide n")

    stage = "K1"

    def kernel_name(self) -> str:
        return kernel_name_join(
            "wino_rt_filter",
            _tile_tag(self.mh, self.mw),
            self.io_dtype,
            f"op{self.operand_dtype}",
            f"v{self.vec}",
            f"fr{self.filter_rows}",
        )

    def reason(self, geo: RtGeometry) -> Optional[str]:
        if (geo.mh, geo.mw) != (self.mh, self.mw):
            return "tile mismatch"
        if geo.problem.cpg % self.vec:
            return f"C/groups={geo.problem.cpg} not a multiple of vec {self.vec}"
        return None


@dataclass(frozen=True)
class RtPrescaleSpec:
    """K2a: per-pixel ``max |x|`` over C; ``vec`` divides C."""

    io_dtype: str = "fp16"
    vec: int = 8

    def __post_init__(self):
        if self.io_dtype not in _IO or self.vec not in (1, 2, 4, 8):
            raise ValueError("bad K2a spec")

    stage = "K2a"

    def kernel_name(self) -> str:
        return kernel_name_join("wino_rt_pmax", self.io_dtype, f"v{self.vec}")

    def reason(self, geo: RtGeometry) -> Optional[str]:
        if geo.problem.C % self.vec:
            return f"C={geo.problem.C} not a multiple of vec {self.vec}"
        return None


@dataclass(frozen=True)
class RtInputSpec:
    """K2: ``x -> V`` (and ``e``); ``vec`` divides C.

    ``lds = (bh, bw)``: stage ``bh x bw`` tiles x ``lds_lanes * vec`` channels
    through LDS, ``tiles_per_thread`` of them per thread; ``(0, 0)``: the
    register path, a run of ``tiles_per_thread`` tiles along W per thread.
    """

    mh: int = 4
    mw: int = 4
    io_dtype: str = "fp16"
    operand_dtype: str = "fp16"
    vec: int = 8
    block_size: int = 256
    prescale: bool = False
    tiles_per_thread: int = 1
    lds: Tuple[int, int] = (0, 0)
    lds_lanes: int = 16
    nt_stores: bool = True
    # Fused prescale: the workgroup holds all channels of its runs and reduces
    # the tile maxima itself (no K2a); register path only.
    fused: bool = False

    def __post_init__(self):
        _check_common(self.mh, self.mw, self.io_dtype, self.operand_dtype, self.vec)
        if self.block_size < 32 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two >= 32")
        if self.fused and (not self.prescale or self.input_lds):
            raise ValueError("fused needs the prescale and the register path")
        if self.tiles_per_thread < 1:
            raise ValueError("tiles_per_thread must be >= 1")
        if self.input_lds:
            bh, bw = self.lds
            if (bh * bw) % self.tiles_per_thread:
                raise ValueError("tiles_per_thread must divide bh * bw")
            if self.lds_block > 1024:
                raise ValueError("LDS region too large for one workgroup")

    stage = "K2"

    @property
    def input_lds(self) -> bool:
        return self.lds[0] > 0

    @property
    def nh(self) -> int:
        return self.mh + 2

    @property
    def nw(self) -> int:
        return self.mw + 2

    @property
    def lds_block(self) -> int:
        bh, bw = self.lds
        return bh * bw * self.lds_lanes // self.tiles_per_thread

    def kernel_name(self) -> str:
        tags = [
            _tile_tag(self.mh, self.mw),
            self.io_dtype,
            f"op{self.operand_dtype}",
            f"v{self.vec}",
        ]
        if self.input_lds:
            tags += [
                "lds{}x{}".format(*self.lds),
                f"ll{self.lds_lanes}",
                f"t{self.tiles_per_thread}",
            ]
        else:
            tags += [f"b{self.block_size}", f"r{self.tiles_per_thread}"]
        if self.prescale:
            tags.append("psf" if self.fused else "ps")
        if not self.nt_stores:
            tags.append("rt")
        return kernel_name_join("wino_rt_input", *tags)

    def reason(self, geo: RtGeometry) -> Optional[str]:
        if (geo.mh, geo.mw) != (self.mh, self.mw):
            return "tile mismatch"
        if geo.problem.C % self.vec:
            return f"C={geo.problem.C} not a multiple of vec {self.vec}"
        if self.fused and _pow2_ceil(geo.problem.C // self.vec) > self.block_size:
            return f"fused prescale needs C/vec <= {self.block_size}"
        return None


@dataclass(frozen=True)
class RtOutputSpec:
    """K4: ``M -> y`` (``x 2**e``); ``vec`` divides K."""

    mh: int = 4
    mw: int = 4
    io_dtype: str = "fp16"
    operand_dtype: str = "fp16"
    vec: int = 8
    block_size: int = 256
    prescale: bool = False
    nt_stores: bool = True

    def __post_init__(self):
        _check_common(self.mh, self.mw, self.io_dtype, self.operand_dtype, self.vec)
        if self.block_size < 32 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two >= 32")

    stage = "K4"

    def kernel_name(self) -> str:
        tags = [
            _tile_tag(self.mh, self.mw),
            self.io_dtype,
            f"op{self.operand_dtype}",
            f"v{self.vec}",
            f"b{self.block_size}",
        ]
        if self.prescale:
            tags.append("ps")
        if not self.nt_stores:
            tags.append("rt")
        return kernel_name_join("wino_rt_output", *tags)

    def reason(self, geo: RtGeometry) -> Optional[str]:
        if (geo.mh, geo.mw) != (self.mh, self.mw):
            return "tile mismatch"
        if geo.problem.K % self.vec:
            return f"K={geo.problem.K} not a multiple of vec {self.vec}"
        return None


# ---------------------------------------------------------------------------
# K1: filter transform
# ---------------------------------------------------------------------------


def filter_signature(spec: RtFilterSpec) -> list:
    sb = (
        SignatureBuilder()
        .ptr("w", spec.io_dtype)
        .ptr("u", spec.operand_dtype)
        .scalar("w_bytes", "i32")
        .scalar("u_bytes", "i32")
        .scalar("K", "i32")
        .scalar("cg", "i32")
        .scalar("items", "i32")
    )
    return _div_sig(sb, "cgv").build()


def build_filter(
    spec: RtFilterSpec, *, arch: str = "gfx1250"
) -> KernelDef:  # noqa: ARG001
    """K1 of :func:`conv_winograd_fwd.build_winograd_filter_transform`, extents at run time."""
    vec = spec.vec
    n, nw = spec.mh + 2, spec.mw + 2
    gint, den = filter_matrix_int(spec.mh)
    gint_w, den_w = filter_matrix_int(spec.mw)
    scale = 1.0 / float(den * den_w)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _FILTER_BLOCK
    w = b.param(
        "w", PtrType(_ir_dtype(spec.io_dtype), "global"), noalias=True, readonly=True
    )
    u = b.param(
        "u",
        PtrType(_ir_dtype(spec.operand_dtype), "global"),
        noalias=True,
        writeonly=True,
    )
    w_bytes = b.param("w_bytes", I32)
    u_bytes = b.param("u_bytes", I32)
    K = b.param("K", I32)
    cg = b.param("cg", I32)
    items = b.param("items", I32)
    cgv = _Div(b, "cgv")
    w_rsrc = b.buffer_rsrc(w, w_bytes)
    u_rsrc = b.buffer_rsrc(u, u_bytes)

    gid = b.add(b.mul(b.block_id_x(), _c(b, _FILTER_BLOCK)), b.thread_id_x())
    row_y = b.block_id_y()
    ok = b.cmp_lt(gid, items)
    k, cm = cgv.divmod(gid)
    c = _mul_c(b, cm, vec)
    cg2 = b.mul(cg, _c(b, 2))
    wbase = _oob(b, ok, b.mul(b.add(b.mul(k, b.mul(cg, _c(b, 9))), c), _c(b, 2)))
    taps = []
    off = wbase
    for ky in range(3):
        row = []
        for kx in range(3):
            if ky or kx:
                off = b.add(off, cg2)
            row.append(
                _raw_to_f32(b, _load_raw(b, w_rsrc, off, spec.io_dtype, vec), vec)
            )
        taps.append(row)
    sc = b.const_f32(scale)
    nr = spec.filter_rows
    kcg2 = b.mul(K, cg2)  # bytes of one U[bin]
    for rr in range(nr):
        if nr == n:
            row_i = _c(b, rr)
            coef = [b.const_f32(float(gint[rr][ky])) for ky in range(3)]
        else:
            row_i = b.add(_mul_c(b, row_y, nr), _c(b, rr))
            coef = []
            for ky in range(3):
                cval = b.const_f32(float(gint[0][ky]))
                for i in range(1, n):
                    cval = b.select(
                        b.cmp_eq(row_i, _c(b, i)), b.const_f32(float(gint[i][ky])), cval
                    )
                coef.append(cval)
        tmp = [
            [
                b.fma(
                    coef[2],
                    taps[2][kx][ch],
                    b.fma(coef[1], taps[1][kx][ch], b.fmul(coef[0], taps[0][kx][ch])),
                )
                for ch in range(vec)
            ]
            for kx in range(3)
        ]
        # U[(row_i * nw + j)][k][c]
        ubase = _oob(
            b,
            ok,
            b.add(
                b.mul(_mul_c(b, row_i, nw), kcg2),
                b.mul(b.add(b.mul(k, cg), c), _c(b, 2)),
            ),
        )
        uoff = ubase
        for j in range(nw):
            if j:
                uoff = b.add(uoff, kcg2)
            vals = [
                b.fmul(
                    _lincomb(b, [(gint_w[j][kx], tmp[kx][ch]) for kx in range(3)]), sc
                )
                for ch in range(vec)
            ]
            _store_vec_f32(b, u_rsrc, uoff, vals, spec.operand_dtype)
    return b.kernel


def filter_launch(spec: RtFilterSpec, geo: RtGeometry):
    p = geo.problem
    cgv = p.cpg // spec.vec
    items = p.K * cgv
    vals = dict(K=p.K, cg=p.cpg, items=items, **_div_values("cgv", cgv))
    grid = (ceil_div(items, _FILTER_BLOCK), (spec.mh + 2) // spec.filter_rows, 1)
    return grid, _FILTER_BLOCK, vals


# ---------------------------------------------------------------------------
# K2a: per-pixel max |x| over C
# ---------------------------------------------------------------------------


def prescale_signature(spec: RtPrescaleSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("x", spec.io_dtype)
        .ptr("e", "i32")
        .scalar("x_bytes", "i32")
        .scalar("e_bytes", "i32")
        .scalar("C", "i32")
        .scalar("nvec", "i32")
        .scalar("tpc_lg", "i32")
        .scalar("loads", "i32")
        .scalar("npix", "i32")
        .scalar("T", "i32")
        .build()
    )


def build_prescale(
    spec: RtPrescaleSpec, *, arch: str = "gfx1250"
) -> KernelDef:  # noqa: ARG001
    """K2a of :func:`conv_winograd_fwd.build_winograd_prescale`, extents at run time.

    ``tpc = 2**tpc_lg <= 32`` lanes per pixel, each reduces ``loads``
    channel vectors (a runtime loop), then xor shuffles inside the lane group.
    """
    vec = spec.vec
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _PRESCALE_BLOCK
    x = b.param(
        "x", PtrType(_ir_dtype(spec.io_dtype), "global"), noalias=True, readonly=True
    )
    e_ptr = b.param("e", PtrType(I32, "global"), noalias=True, writeonly=True)
    x_bytes = b.param("x_bytes", I32)
    e_bytes = b.param("e_bytes", I32)
    C = b.param("C", I32)
    nvec = b.param("nvec", I32)
    tpc_lg = b.param("tpc_lg", I32)
    loads = b.param("loads", I32)
    npix = b.param("npix", I32)
    T = b.param("T", I32)
    x_rsrc = b.buffer_rsrc(x, x_bytes)
    e_rsrc = b.buffer_rsrc(e_ptr, e_bytes)

    tid = b.thread_id_x()
    tpc = b.shl(_c(b, 1), tpc_lg)
    lc = b.land(tid, b.sub(tpc, _c(b, 1)))
    lp = b.lshr(tid, tpc_lg)
    ppw = b.lshr(_c(b, _PRESCALE_BLOCK), tpc_lg)
    pix = b.add(b.mul(b.block_id_x(), ppw), lp)
    pok = b.cmp_lt(pix, npix)
    base = b.mul(b.add(b.mul(pix, C), _mul_c(b, lc, vec)), _c(b, 2))
    step_b = b.mul(tpc, _c(b, vec * 2))
    zero = _c(b, 0)
    loop = b.scf_for_iter(zero, loads, _c(b, 1), [("pm_acc", zero)], iv_name="pm_q")
    with loop as (q, (acc,)):
        idx = b.add(lc, b.mul(q, tpc))
        ok = b.land(pok, b.cmp_lt(idx, nvec))
        off = _oob(b, ok, b.add(base, b.mul(q, step_b)))
        a_ = _hmax_bits(
            b, _abs_bits_vec(b, _load_raw(b, x_rsrc, off, spec.io_dtype, vec), vec), vec
        )
        b.scf_yield(b.smax(acc, a_))
    mx = loop.results[0]
    s = 1
    while s < 32:
        sh = b.warp_shuffle_xor(mx, s)
        mx = b.select(b.cmp_lt(_c(b, s), tpc), b.smax(mx, sh), mx)
        s *= 2
    ok_s = b.land(pok, b.cmp_eq(lc, _c(b, 0)))
    b.buffer_store_f32(
        e_rsrc,
        _oob(b, ok_s, b.mul(b.add(pix, T), _c(b, 4))),
        _c(b, 0),
        b.bitcast(mx, F32),
    )
    return b.kernel


def prescale_launch(spec: RtPrescaleSpec, geo: RtGeometry):
    p = geo.problem
    nv = p.C // spec.vec
    tpc = min(_pow2_ceil(nv), 32)
    npix = p.N * p.Hi * p.Wi
    vals = dict(
        C=p.C,
        nvec=nv,
        tpc_lg=tpc.bit_length() - 1,
        loads=ceil_div(nv, tpc),
        npix=npix,
        T=geo.num_tiles,
    )
    return (ceil_div(npix, _PRESCALE_BLOCK // tpc), 1, 1), _PRESCALE_BLOCK, vals


# ---------------------------------------------------------------------------
# K2 / K4 thread mapping (register path)
# ---------------------------------------------------------------------------


@dataclass
class _Coords:
    ok: Value
    cv: Value
    c_ok: Value
    n: Value
    a: Value
    bsub: Value
    ti: Value
    tj0: Value
    t0: Value
    tiles_w: Value
    tpc: Value


_RUN_SCALARS = ("nvec", "tpc_lg", "rpw", "num_runs", "tiles_w")
_RUN_DIVS = ("cx", "rpr", "th", "ddw", "ddh")


def _run_sig(sb: SignatureBuilder) -> SignatureBuilder:
    for nm in _RUN_SCALARS:
        sb = sb.scalar(nm, "i32")
    for nm in _RUN_DIVS:
        sb = _div_sig(sb, nm)
    return sb


def _run_coords(b: IRBuilder, tiles: int) -> _Coords:
    """Thread -> (run of ``tiles`` tiles along W, channel vector), all runtime."""
    nvec = b.param("nvec", I32)
    tpc_lg = b.param("tpc_lg", I32)
    rpw = b.param("rpw", I32)
    num_runs = b.param("num_runs", I32)
    tiles_w = b.param("tiles_w", I32)
    cx = _Div(b, "cx")
    rpr = _Div(b, "rpr")
    th = _Div(b, "th")
    ddw = _Div(b, "ddw")
    ddh = _Div(b, "ddh")
    tid = b.thread_id_x()
    bid = b.block_id_x()
    tpc = b.shl(_c(b, 1), tpc_lg)
    lc = b.land(tid, b.sub(tpc, _c(b, 1)))
    lr = b.lshr(tid, tpc_lg)
    rb, cxi = cx.divmod(bid)
    cv = b.add(b.mul(cxi, tpc), lc)
    c_ok = b.cmp_lt(cv, nvec)
    rho = b.add(b.mul(rb, rpw), lr)
    ok = b.cmp_lt(rho, num_runs)
    row, jr = rpr.divmod(rho)
    tj0 = _mul_c(b, jr, tiles)
    t0 = b.add(b.mul(row, tiles_w), tj0)
    r, ti = th.divmod(row)
    r, bsub = ddw.divmod(r)
    r, a = ddh.divmod(r)
    return _Coords(ok, cv, c_ok, r, a, bsub, ti, tj0, t0, tiles_w, tpc)


def _run_values(
    geo: RtGeometry, nvec: int, tiles: int, block: int
) -> Tuple[dict, tuple]:
    p = geo.problem
    tpc = min(_pow2_ceil(nvec), block)
    cx = ceil_div(nvec, tpc)
    rpw = block // tpc
    rpr = ceil_div(geo.tiles_w, tiles)
    rows = geo.num_tiles // geo.tiles_w
    num_runs = rows * rpr
    vals = dict(
        nvec=nvec,
        tpc_lg=tpc.bit_length() - 1,
        rpw=rpw,
        num_runs=num_runs,
        tiles_w=geo.tiles_w,
        **_div_values("cx", cx),
        **_div_values("rpr", rpr),
        **_div_values("th", geo.tiles_h),
        **_div_values("ddw", p.dW),
        **_div_values("ddh", p.dH),
    )
    return vals, (ceil_div(num_runs, rpw) * cx, 1, 1)


def _group_max_rt(
    b: IRBuilder, vals: List[Value], tpc: Value, block: int
) -> List[Value]:
    """Max of each i32 in ``vals`` over the ``tpc`` (runtime power of two,
    ``<= block``) threads of a run: xor shuffles inside a 32-lane row, then
    across the row maxima through LDS (``conv_winograd_fwd._group_max_bits``
    with a runtime lane count)."""
    out = []
    for v in vals:
        s = 1
        while s < 32:
            sh = b.warp_shuffle_xor(v, s)
            v = b.select(b.cmp_lt(_c(b, s), tpc), b.smax(v, sh), v)
            s *= 2
        out.append(v)
    rows = block // 32
    if rows <= 1:
        return out
    nv = len(out)
    red = b.smem_alloc_f32([rows * nv], name_hint="wino_rt_amax")
    tid = b.thread_id_x()
    row_id = b.lshr(tid, _c(b, 5))
    with b.scf_if(b.cmp_eq(b.land(tid, _c(b, 31)), _c(b, 0))):
        for i, v in enumerate(out):
            idx = b.add(_mul_c(b, row_id, nv), _c(b, i))
            b.smem_store_vN_f32(red, [idx], b.vec_pack([b.bitcast(v, F32)], F32), 1)
    b.sync()
    # The rows of one run: ``per = tpc / 32`` rows from ``first`` (tpc > 32);
    # with tpc <= 32 a run lives in one row and the shuffles were enough.
    per = b.lshr(tpc, _c(b, 5))
    big = b.cmp_lt(_c(b, 32), tpc)
    first = b.mul(b.select(big, b.div(row_id, b.select(big, per, _c(b, 1))), row_id),
                  b.select(big, per, _c(b, 1)))  # fmt: skip
    res = []
    for i in range(nv):
        acc = out[i]
        for r in range(rows):
            idx = b.add(_mul_c(b, b.add(first, _c(b, r)), nv), _c(b, i))
            ok = b.land(big, b.cmp_lt(_c(b, r), per))
            idx = b.select(ok, idx, _c(b, 0))
            x = b.bitcast(b.vec_extract(b.smem_load_vN_f32(red, idx, n=1), 0), I32)
            acc = b.select(ok, b.smax(acc, x), acc)
        res.append(acc)
    return res


# ---------------------------------------------------------------------------
# K2: input transform
# ---------------------------------------------------------------------------

_INPUT_GEOM = ("C", "Hi", "Wi", "pH", "pW", "dH", "dW", "dHm", "dWm", "T", "ps_bits")


def input_signature(spec: RtInputSpec) -> list:
    sb = (
        SignatureBuilder()
        .ptr("x", spec.io_dtype)
        .ptr("v", spec.operand_dtype)
        .ptr("e", "i32")
        .scalar("x_bytes", "i32")
        .scalar("v_bytes", "i32")
        .scalar("e_bytes", "i32")
    )
    for nm in _INPUT_GEOM:
        sb = sb.scalar(nm, "i32")
    if spec.input_lds:
        sb = sb.scalar("nvec", "i32").scalar("tiles_h", "i32").scalar("tiles_w", "i32")
        for nm in ("cx", "nbw", "nbh", "ddw", "ddh"):
            sb = _div_sig(sb, nm)
    else:
        sb = _run_sig(sb)
    return sb.build()


def _input_tiles_transform(b, spec, raw, R, tile_ok, scales, vbase0, v_rsrc, cb2):
    """``V = B^T d B`` of ``R`` tiles adjacent along W; ``cb2 = C * 2`` (runtime)."""
    n, m = spec.nh, spec.mh
    nw, mw = spec.nw, spec.mw
    vec = spec.vec
    ops = _IROps(b)
    vstride = _mul_c(b, cb2, n * nw)
    colt = {}
    pending = []
    b.sched_barrier(0)

    def col_transform(vv):
        if vv not in colt:
            data = [_raw_to_f32(b, raw[uu][vv], vec) for uu in range(n)]
            colt[vv] = [
                bt_1d(ops, m, [data[uu][ch] for uu in range(n)]) for ch in range(vec)
            ]
        return colt[vv]

    for r in range(R):
        cols = [col_transform(mw * r + vv) for vv in range(nw)]
        base = vbase0 if r == 0 else b.add(vbase0, _mul_c(b, vstride, r))
        off = _oob(b, tile_ok[r], base)
        first = True
        for i in range(n):
            outs = [
                bt_1d(ops, mw, [cols[vv][ch][i] for vv in range(nw)])
                for ch in range(vec)
            ]
            for j in range(nw):
                if not first:
                    off = b.add(off, cb2)
                first = False
                vals = [outs[ch][j] for ch in range(vec)]
                if scales[r] is not None:
                    vals = [b.fmul(x_, scales[r]) for x_ in vals]
                pending.append((off, _pack_vec_f32(b, vals, spec.operand_dtype)))
    b.sched_barrier(0)
    cpol = _CPOL_NT if spec.nt_stores else 0
    for off, packed in pending:
        _store_packed(b, v_rsrc, off, packed, spec.operand_dtype, cpol)


def build_input(
    spec: RtInputSpec, *, arch: str = "gfx1250"
) -> KernelDef:  # noqa: ARG001
    """K2 of :func:`conv_winograd_fwd.build_winograd_input_transform`, extents at run time."""
    if spec.input_lds:
        return _build_input_lds(spec)
    n, m = spec.nh, spec.mh
    nw, mw = spec.nw, spec.mw
    vec = spec.vec
    R = spec.tiles_per_thread
    ncols = mw * R + (nw - mw)
    ps = spec.prescale

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    x = b.param(
        "x", PtrType(_ir_dtype(spec.io_dtype), "global"), noalias=True, readonly=True
    )
    v = b.param(
        "v",
        PtrType(_ir_dtype(spec.operand_dtype), "global"),
        noalias=True,
        writeonly=True,
    )
    e_ptr = b.param("e", PtrType(I32, "global"), noalias=True)
    x_bytes = b.param("x_bytes", I32)
    v_bytes = b.param("v_bytes", I32)
    e_bytes = b.param("e_bytes", I32)
    g = {nm: b.param(nm, I32) for nm in _INPUT_GEOM}
    x_rsrc = b.buffer_rsrc(x, x_bytes)
    v_rsrc = b.buffer_rsrc(v, v_bytes)
    e_rsrc = b.buffer_rsrc(e_ptr, e_bytes) if ps else None
    rc = _run_coords(b, R)

    C, Hi, Wi = g["C"], g["Hi"], g["Wi"]
    cb2 = b.mul(C, _c(b, 2))
    c = _mul_c(b, rc.cv, vec)
    h0 = b.add(b.sub(rc.a, g["pH"]), b.mul(rc.ti, g["dHm"]))
    w0 = b.add(b.sub(rc.bsub, g["pW"]), b.mul(rc.tj0, g["dWm"]))
    img_row0 = b.mul(rc.n, Hi)
    row_ok, row_pix = [], []
    hi = h0
    for uu in range(n):
        if uu:
            hi = b.add(hi, g["dH"])
        row_ok.append(b.land(rc.ok, b.cmp_ult(hi, Hi)))
        row_pix.append(b.mul(b.add(img_row0, hi), Wi))
    col_ok, col_w = [], []
    wi = w0
    for vv in range(ncols):
        if vv:
            wi = b.add(wi, g["dW"])
        col_ok.append(b.cmp_ult(wi, Wi))
        col_w.append(wi)
    c2 = _mul_c(b, c, 2)
    xrow = [
        _oob(b, b.land(row_ok[uu], rc.c_ok), b.add(b.mul(row_pix[uu], cb2), c2))
        for uu in range(n)
    ]
    xcol = [_oob(b, col_ok[vv], b.mul(col_w[vv], cb2)) for vv in range(ncols)]
    raw = [
        [
            _load_raw(b, x_rsrc, b.add(xrow[uu], xcol[vv]), spec.io_dtype, vec)
            for vv in range(ncols)
        ]
        for uu in range(n)
    ]
    base_ok = b.land(rc.ok, rc.c_ok)
    tile_ok = []
    for r in range(R):
        if r == 0:
            tile_ok.append(base_ok)
        else:
            tile_ok.append(
                b.land(base_ok, b.cmp_lt(b.add(rc.tj0, _c(b, r)), rc.tiles_w))
            )

    scales: List[Optional[Value]] = [None] * R
    if ps and spec.fused:
        colmax = []
        for vv in range(ncols):
            acc = None
            for uu in range(n):
                a_ = _abs_bits_vec(b, raw[uu][vv], vec)
                acc = a_ if acc is None else _max_bits(b, acc, a_, vec)
            colmax.append(acc)
        tmax = []
        for r in range(R):
            acc = colmax[mw * r]
            for vv in range(1, nw):
                acc = _max_bits(b, acc, colmax[mw * r + vv], vec)
            tmax.append(_hmax_bits(b, acc, vec))
        tmax = _group_max_rt(b, tmax, rc.tpc, spec.block_size)
        ps_scale = b.bitcast(g["ps_bits"], F32)
        lane0 = b.cmp_eq(rc.cv, _c(b, 0))
        for r in range(R):
            e_r = _prescale_exp_rt(b, _bits_to_f32(b, tmax[r], spec.io_dtype), ps_scale)
            t_r = b.add(rc.t0, _c(b, r)) if r else rc.t0
            b.buffer_store_f32(
                e_rsrc,
                _oob(b, b.land(lane0, tile_ok[r]), b.mul(t_r, _c(b, 4))),
                _c(b, 0),
                b.bitcast(e_r, F32),
            )
            scales[r] = _pow2_f32(b, b.sub(_c(b, 0), e_r))
    elif ps:
        T = g["T"]
        prow = [
            _oob(b, row_ok[uu], b.mul(b.add(row_pix[uu], T), _c(b, 4)))
            for uu in range(n)
        ]
        pcol = [_oob(b, col_ok[vv], b.mul(col_w[vv], _c(b, 4))) for vv in range(ncols)]
        colmax = []
        for vv in range(ncols):
            acc = None
            for uu in range(n):
                pm = b.buffer_load(e_rsrc, b.add(prow[uu], pcol[vv]), _c(b, 0), I32)
                acc = pm if acc is None else b.smax(acc, pm)
            colmax.append(acc)
        ps_scale = b.bitcast(g["ps_bits"], F32)
        lane0 = b.cmp_eq(rc.cv, _c(b, 0))
        for r in range(R):
            acc = colmax[mw * r]
            for vv in range(1, nw):
                acc = b.smax(acc, colmax[mw * r + vv])
            e_r = _prescale_exp_rt(b, _bits_to_f32(b, acc, spec.io_dtype), ps_scale)
            t_r = b.add(rc.t0, _c(b, r)) if r else rc.t0
            b.buffer_store_f32(
                e_rsrc,
                _oob(b, b.land(lane0, tile_ok[r]), b.mul(t_r, _c(b, 4))),
                _c(b, 0),
                b.bitcast(e_r, F32),
            )
            scales[r] = _pow2_f32(b, b.sub(_c(b, 0), e_r))

    vbase0 = b.add(b.mul(rc.t0, _mul_c(b, cb2, spec.nh * spec.nw)), c2)
    _input_tiles_transform(b, spec, raw, R, tile_ok, scales, vbase0, v_rsrc, cb2)
    return b.kernel


def input_launch(spec: RtInputSpec, geo: RtGeometry):
    p = geo.problem
    vals = dict(
        C=p.C,
        Hi=p.Hi,
        Wi=p.Wi,
        pH=p.pH,
        pW=p.pW,
        dH=p.dH,
        dW=p.dW,
        dHm=p.dH * spec.mh,
        dWm=p.dW * spec.mw,
        T=geo.num_tiles,
        ps_bits=_ps_bits(geo),
    )
    nv = p.C // spec.vec
    if spec.input_lds:
        bh, bw = spec.lds
        tpc = spec.lds_lanes
        cx = ceil_div(nv, tpc)
        nbh = ceil_div(geo.tiles_h, bh)
        nbw = ceil_div(geo.tiles_w, bw)
        vals.update(
            nvec=nv,
            tiles_h=geo.tiles_h,
            tiles_w=geo.tiles_w,
            **_div_values("cx", cx),
            **_div_values("nbw", nbw),
            **_div_values("nbh", nbh),
            **_div_values("ddw", p.dW),
            **_div_values("ddh", p.dH),
        )
        grid = (p.N * p.dH * p.dW * nbh * nbw * cx, 1, 1)
        return grid, spec.lds_block, vals
    rv, grid = _run_values(geo, nv, spec.tiles_per_thread, spec.block_size)
    vals.update(rv)
    return grid, spec.block_size, vals


def _build_input_lds(spec: RtInputSpec) -> KernelDef:
    """K2 through LDS (``conv_winograd_fwd._build_input_lds``), extents at run time.

    A workgroup stages ``bh x bw`` tiles x ``lds_lanes * vec`` channels; every
    channel lane past C is masked (a small-C problem leaves lanes idle).
    """
    n, m = spec.nh, spec.mh
    nw, mw = spec.nw, spec.mw
    vec = spec.vec
    bh, bw = spec.lds
    tpc = spec.lds_lanes
    cc = tpc * vec
    lv = vec
    rh = m * bh + n - m
    rw = mw * bw + nw - mw
    block = spec.lds_block
    tpt = spec.tiles_per_thread
    ppx = cc // lv
    npx = rh * rw
    loads = ceil_div(npx * ppx, block)
    ps = spec.prescale
    dt = _ir_dtype(spec.io_dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = block
    x = b.param("x", PtrType(dt, "global"), noalias=True, readonly=True)
    v = b.param(
        "v",
        PtrType(_ir_dtype(spec.operand_dtype), "global"),
        noalias=True,
        writeonly=True,
    )
    e_ptr = b.param("e", PtrType(I32, "global"), noalias=True)
    x_bytes = b.param("x_bytes", I32)
    v_bytes = b.param("v_bytes", I32)
    e_bytes = b.param("e_bytes", I32)
    g = {nm: b.param(nm, I32) for nm in _INPUT_GEOM}
    nvec = b.param("nvec", I32)
    tiles_h = b.param("tiles_h", I32)
    tiles_w = b.param("tiles_w", I32)
    cx = _Div(b, "cx")
    nbw = _Div(b, "nbw")
    nbh = _Div(b, "nbh")
    ddw = _Div(b, "ddw")
    ddh = _Div(b, "ddh")
    x_rsrc = b.buffer_rsrc(x, x_bytes)
    v_rsrc = b.buffer_rsrc(v, v_bytes)
    e_rsrc = b.buffer_rsrc(e_ptr, e_bytes) if ps else None
    C, Hi, Wi = g["C"], g["Hi"], g["Wi"]
    cb2 = b.mul(C, _c(b, 2))

    tid = b.thread_id_x()
    bid = b.block_id_x()
    r, cxi = cx.divmod(bid)
    r, bwi = nbw.divmod(r)
    sub, bhi = nbh.divmod(r)
    r, bsub = ddw.divmod(sub)
    img, a = ddh.divmod(r)
    ti0 = _mul_c(b, bhi, bh)
    tj0 = _mul_c(b, bwi, bw)
    h0 = b.add(b.sub(a, g["pH"]), b.mul(ti0, g["dHm"]))
    w0 = b.add(b.sub(bsub, g["pW"]), b.mul(tj0, g["dWm"]))
    img_row0 = b.mul(img, Hi)
    c0 = _mul_c(b, cxi, cc)

    lds = b.smem_alloc(dt, [loads * block * lv], name_hint="wino_xin")
    for k in range(loads):
        q = b.add(tid, _c(b, k * block)) if k else tid
        px = b.lshr(q, _c(b, ppx.bit_length() - 1)) if ppx > 1 else q
        pc = b.land(q, _c(b, ppx - 1)) if ppx > 1 else _c(b, 0)
        rr = b.div(px, _c(b, rw))
        cc_ = b.sub(px, b.mul(rr, _c(b, rw)))
        hi = b.add(h0, b.mul(rr, g["dH"]))
        wi = b.add(w0, b.mul(cc_, g["dW"]))
        ch = b.add(c0, _mul_c(b, pc, lv))
        ok = b.land(b.cmp_ult(hi, Hi), b.cmp_ult(wi, Wi))
        if (k + 1) * block > npx * ppx:
            ok = b.land(ok, b.cmp_lt(px, _c(b, npx)))
        ok = b.land(ok, b.cmp_lt(ch, C))
        pix = b.add(b.mul(b.add(img_row0, hi), Wi), wi)
        off = _oob(b, ok, b.add(b.mul(pix, cb2), _mul_c(b, ch, 2)))
        val = _load_raw(b, x_rsrc, off, spec.io_dtype, lv)
        b.smem_store_vN(lds, [_mul_c(b, q, lv)], val, lv)
    pm_lds = None
    if ps:
        T = g["T"]
        pm_lds = b.smem_alloc(
            I32, [ceil_div(npx, block) * block], name_hint="wino_pmax"
        )
        for k in range(ceil_div(npx, block)):
            px = b.add(tid, _c(b, k * block)) if k else tid
            rr = b.div(px, _c(b, rw))
            cc_ = b.sub(px, b.mul(rr, _c(b, rw)))
            hi = b.add(h0, b.mul(rr, g["dH"]))
            wi = b.add(w0, b.mul(cc_, g["dW"]))
            ok = b.land(b.cmp_ult(hi, Hi), b.cmp_ult(wi, Wi))
            if (k + 1) * block > npx:
                ok = b.land(ok, b.cmp_lt(px, _c(b, npx)))
            pix = b.add(b.mul(b.add(img_row0, hi), Wi), wi)
            off = _oob(b, ok, b.mul(b.add(pix, T), _c(b, 4)))
            b.smem_store_vN(pm_lds, [px], b.buffer_load(e_rsrc, off, _c(b, 0), I32), 1)
    b.sync()

    cl = b.land(tid, _c(b, tpc - 1)) if tpc > 1 else _c(b, 0)
    tb0 = b.lshr(tid, _c(b, tpc.bit_length() - 1)) if tpc > 1 else tid
    cv = b.add(_mul_c(b, cxi, tpc), cl)
    c_ok = b.cmp_lt(cv, nvec)
    ps_scale = b.bitcast(g["ps_bits"], F32) if ps else None
    for rep in range(tpt):
        tb = b.add(tb0, _c(b, rep * (block // tpc))) if rep else tb0
        bi = b.div(tb, _c(b, bw)) if bw > 1 else tb
        bj = b.sub(tb, _mul_c(b, bi, bw)) if bw > 1 else _c(b, 0)
        ti = b.add(ti0, bi)
        tj = b.add(tj0, bj)
        tile_ok = b.land(
            b.land(b.cmp_lt(ti, tiles_h), b.cmp_lt(tj, tiles_w)),
            c_ok,
        )
        t = b.add(b.mul(b.add(b.mul(sub, tiles_h), ti), tiles_w), tj)
        pbase = b.add(_mul_c(b, bi, m * rw), _mul_c(b, bj, mw))
        raw = []
        for uu in range(n):
            row = []
            for vv in range(nw):
                idx = b.add(
                    _mul_c(b, b.add(pbase, _c(b, uu * rw + vv)), cc), _mul_c(b, cl, vec)
                )
                ld = b.smem_load_vN(lds, idx, dtype=dt, n=vec)
                row.append(b.vec_extract(ld, 0) if vec == 1 else ld)
            raw.append(row)
        scales: List[Optional[Value]] = [None]
        if ps:
            acc = None
            for uu in range(n):
                for vv in range(nw):
                    pm = b.vec_extract(
                        b.smem_load_vN(
                            pm_lds, b.add(pbase, _c(b, uu * rw + vv)), dtype=I32, n=1
                        ),
                        0,
                    )
                    acc = pm if acc is None else b.smax(acc, pm)
            e_t = _prescale_exp_rt(b, _bits_to_f32(b, acc, spec.io_dtype), ps_scale)
            ok_e = b.land(tile_ok, b.cmp_eq(cv, _c(b, 0)))
            b.buffer_store_f32(
                e_rsrc, _oob(b, ok_e, b.mul(t, _c(b, 4))), _c(b, 0), b.bitcast(e_t, F32)
            )
            scales = [_pow2_f32(b, b.sub(_c(b, 0), e_t))]
        vbase0 = b.add(
            b.mul(t, _mul_c(b, cb2, spec.nh * spec.nw)), _mul_c(b, cv, vec * 2)
        )
        _input_tiles_transform(b, spec, raw, 1, [tile_ok], scales, vbase0, v_rsrc, cb2)
    return b.kernel


# ---------------------------------------------------------------------------
# K4: output transform
# ---------------------------------------------------------------------------

_OUTPUT_GEOM = ("K", "Ho", "Wo", "dH", "dW", "dHm", "dWm")


def output_signature(spec: RtOutputSpec) -> list:
    sb = (
        SignatureBuilder()
        .ptr("mm", spec.operand_dtype)
        .ptr("e", "i32")
        .ptr("y", spec.io_dtype)
        .scalar("m_bytes", "i32")
        .scalar("y_bytes", "i32")
    )
    for nm in _OUTPUT_GEOM:
        sb = sb.scalar(nm, "i32")
    return _run_sig(sb).build()


def build_output(
    spec: RtOutputSpec, *, arch: str = "gfx1250"
) -> KernelDef:  # noqa: ARG001
    """K4 of :func:`conv_winograd_fwd.build_winograd_output_transform`, extents at run time."""
    n, m = spec.mh + 2, spec.mh
    nw, mw = spec.mw + 2, spec.mw
    vec = spec.vec

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    mm = b.param(
        "mm",
        PtrType(_ir_dtype(spec.operand_dtype), "global"),
        noalias=True,
        readonly=True,
    )
    e_ptr = b.param("e", PtrType(I32, "global"), noalias=True, readonly=True)
    y = b.param(
        "y", PtrType(_ir_dtype(spec.io_dtype), "global"), noalias=True, writeonly=True
    )
    m_bytes = b.param("m_bytes", I32)
    y_bytes = b.param("y_bytes", I32)
    g = {nm: b.param(nm, I32) for nm in _OUTPUT_GEOM}
    m_rsrc = b.buffer_rsrc(mm, m_bytes)
    y_rsrc = b.buffer_rsrc(y, y_bytes)
    rc = _run_coords(b, 1)

    K, Ho, Wo = g["K"], g["Ho"], g["Wo"]
    kb2 = b.mul(K, _c(b, 2))
    t = rc.t0
    ok = b.land(rc.ok, rc.c_ok)
    k = _mul_c(b, rc.cv, vec)
    k2 = _mul_c(b, k, 2)
    mbase = _oob(b, ok, b.add(b.mul(t, _mul_c(b, kb2, n * nw)), k2))
    raw = []
    off = mbase
    for uu in range(n):
        row = []
        for vv in range(nw):
            if uu or vv:
                off = b.add(off, kb2)
            row.append(_load_raw(b, m_rsrc, off, spec.operand_dtype, vec))
        raw.append(row)
    scale = None
    if spec.prescale:
        t_safe = b.select(ok, t, _c(b, 0))
        scale = _pow2_f32(b, b.global_load_i32(e_ptr, t_safe))

    ho0 = b.add(rc.a, b.mul(rc.ti, g["dHm"]))
    wo0 = b.add(rc.bsub, b.mul(rc.tj0, g["dWm"]))
    img_row0 = b.mul(rc.n, Ho)
    yrow = []
    ho = ho0
    for i in range(m):
        if i:
            ho = b.add(ho, g["dH"])
        okr = b.land(ok, b.cmp_lt(ho, Ho))
        yrow.append(_oob(b, okr, b.add(b.mul(b.mul(b.add(img_row0, ho), Wo), kb2), k2)))
    ycol = []
    wo = wo0
    for j in range(mw):
        if j:
            wo = b.add(wo, g["dW"])
        ycol.append(_oob(b, b.cmp_lt(wo, Wo), b.mul(wo, kb2)))

    ops = _IROps(b)
    b.sched_barrier(0)
    colt = []
    for vv in range(nw):
        data = [_raw_to_f32(b, raw[uu][vv], vec) for uu in range(n)]
        colt.append(
            [at_1d(ops, m, [data[uu][ch] for uu in range(n)]) for ch in range(vec)]
        )
    pending = []
    for i in range(m):
        outs = [
            at_1d(ops, mw, [colt[vv][ch][i] for vv in range(nw)]) for ch in range(vec)
        ]
        for j in range(mw):
            vals = [outs[ch][j] for ch in range(vec)]
            if scale is not None:
                vals = [b.fmul(x_, scale) for x_ in vals]
            pending.append(
                (b.add(yrow[i], ycol[j]), _pack_vec_f32(b, vals, spec.io_dtype))
            )
    b.sched_barrier(0)
    cpol = _CPOL_NT if spec.nt_stores else 0
    for off_, packed in pending:
        _store_packed(b, y_rsrc, off_, packed, spec.io_dtype, cpol)
    return b.kernel


def output_launch(spec: RtOutputSpec, geo: RtGeometry):
    p = geo.problem
    vals = dict(
        K=p.K,
        Ho=p.Ho,
        Wo=p.Wo,
        dH=p.dH,
        dW=p.dW,
        dHm=p.dH * spec.mh,
        dWm=p.dW * spec.mw,
    )
    rv, grid = _run_values(geo, p.K // spec.vec, 1, spec.block_size)
    vals.update(rv)
    return grid, spec.block_size, vals


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------

_STAGES = {
    RtFilterSpec: (build_filter, filter_signature, filter_launch),
    RtPrescaleSpec: (build_prescale, prescale_signature, prescale_launch),
    RtInputSpec: (build_input, input_signature, input_launch),
    RtOutputSpec: (build_output, output_signature, output_launch),
}


def build_stage(spec, *, arch: str = "gfx1250") -> KernelDef:
    return _STAGES[type(spec)][0](spec, arch=arch)


def stage_signature(spec) -> list:
    return _STAGES[type(spec)][1](spec)


def stage_launch(
    spec, geo: RtGeometry
) -> Tuple[Tuple[int, int, int], int, Dict[str, int]]:
    """``(grid, block, scalar kernarg values)`` of one launch; the buffer
    pointers and ``<buf>_bytes`` sizes are the caller's."""
    why = spec.reason(geo)
    if why:
        raise ValueError(f"{spec.kernel_name()}: {why}")
    grid, block, vals = _STAGES[type(spec)][2](spec, geo)
    if min(grid) <= 0:
        raise ValueError("empty grid")
    if grid[0] >= 2**31:
        raise ValueError(f"grid {grid} too large")
    return grid, block, vals


# ---------------------------------------------------------------------------
# The AOT grid and the default pick
# ---------------------------------------------------------------------------

#: Tiles of the AOT grid: every tile the 3x3 selector considers
#: (``conv_winograd_fwd.AUTO_TILES``).
AOT_TILES: Tuple[Tuple[int, int], ...] = (
    (4, 4),
    (6, 4),
    (2, 2),
    (6, 2),
    (4, 2),
    (2, 4),
    (2, 6),
    (4, 6),
)
AOT_VECS: Tuple[int, ...] = (8, 4, 2, 1)
#: K2's LDS region for tiles with fewer than 36 bins (the JIT pipeline's
#: tuned LDS knobs) and the fill below which the register path is used.
_LDS_REGION = (4, 8)
_LDS_TPT = 4
_LDS_LANES = 16
_LDS_MIN_FILL = 0.75


def pipeline_dtypes(tile, io_dtype: str) -> Tuple[str, bool]:
    """``(operand dtype, prescale)`` of a pipeline, as the JIT pipeline's
    ``auto`` rules: bf16 operands only for F(2,3) with bf16 I/O; the prescale
    for bf16 I/O with a tile side of 4 or more."""
    mh, mw = _split_tile(tile)
    operand = "bf16" if ((mh, mw) == (2, 2) and io_dtype == "bf16") else "fp16"
    return operand, io_dtype == "bf16" and max(mh, mw) >= 4


def _small_tile(mh: int, mw: int) -> bool:
    return (mh + 2) * (mw + 2) < 36


def aot_stage_specs() -> list:
    """Every stage spec of the AOT grid: per I/O dtype, K2a at each vector
    width; per tile and operand dtype (bf16 F(2,3) also with fp16 operands),
    K1, K2 and K4 at each vector width -- K2 on the register path, with the
    fused prescale at blocks 256 and 1024 when the tile takes the prescale,
    and for tiles under 36 bins also with 2-tile runs and through the LDS
    region."""
    out = []
    for io in _IO:
        for v in AOT_VECS:
            out.append(RtPrescaleSpec(io, v))
        for mh, mw in AOT_TILES:
            _, ps = pipeline_dtypes((mh, mw), io)
            ops = (
                ("bf16", "fp16") if ((mh, mw) == (2, 2) and io == "bf16") else ("fp16",)
            )
            fr = 2 if mh >= 4 else mh + 2
            for op in ops:
                for v in AOT_VECS:
                    out.append(RtFilterSpec(mh, mw, io, op, v, fr))
                    out.append(RtOutputSpec(mh, mw, io, op, v, 256, prescale=ps))
                    out.append(RtInputSpec(mh, mw, io, op, v, 256, prescale=ps))
                    if ps:
                        for blk in (256, 1024):
                            out.append(
                                RtInputSpec(
                                    mh, mw, io, op, v, blk, prescale=True, fused=True
                                )
                            )
                    if _small_tile(mh, mw):
                        out.append(
                            RtInputSpec(
                                mh, mw, io, op, v, 256, prescale=ps, tiles_per_thread=2
                            )
                        )
                        out.append(
                            RtInputSpec(
                                mh, mw, io, op, v, prescale=ps, lds=_LDS_REGION,
                                tiles_per_thread=_LDS_TPT, lds_lanes=_LDS_LANES,
                            )
                        )  # fmt: skip
    return out


def _pow2_div(n: int, cap: int) -> int:
    v = cap
    while v > 1 and n % v:
        v //= 2
    return v


def default_stage_specs(problem: ConvProblem, tile, io_dtype: str) -> dict:
    """The AOT binaries a pipeline uses by default, ``{"K1", ("K2a",) "K2",
    "K4"}`` -> spec: the JIT pipeline's tuned knobs (8-wide channel vectors
    for F(2,3)-sized tiles, 4-wide for larger ones, narrowed to divide the
    channel extent; the fused prescale while a workgroup can hold all
    channels, else the split one; the LDS region for small tiles whose tile
    grid fills it). A pure function of the problem; every spec is in
    :func:`aot_stage_specs`."""
    mh, mw = _split_tile(tile)
    operand, ps = pipeline_dtypes((mh, mw), io_dtype)
    p = problem
    cap = 4 if max(mh, mw) >= 4 else 8
    vc = _pow2_div(p.C, cap)
    out = {
        "K1": RtFilterSpec(
            mh, mw, io_dtype, operand, _pow2_div(p.cpg, cap), 2 if mh >= 4 else mh + 2
        ),
        "K4": RtOutputSpec(
            mh, mw, io_dtype, operand, _pow2_div(p.K, cap), 256, prescale=ps
        ),
    }
    if ps:
        lanes = _pow2_ceil(p.C // vc)
        if lanes <= 1024:
            blk = 256 if lanes <= 256 else 1024
            out["K2"] = RtInputSpec(
                mh, mw, io_dtype, operand, vc, blk, prescale=True, fused=True
            )
        else:
            out["K2a"] = RtPrescaleSpec(io_dtype, _pow2_div(p.C, 8))
            out["K2"] = RtInputSpec(mh, mw, io_dtype, operand, vc, 256, prescale=True)
        return out
    geo = RtGeometry(p, mh, mw)
    bh, bw = _LDS_REGION
    th, tw = geo.tiles_h, geo.tiles_w
    fill = th * tw / (ceil_div(th, bh) * bh * ceil_div(tw, bw) * bw)
    if _small_tile(mh, mw) and fill >= _LDS_MIN_FILL:
        out["K2"] = RtInputSpec(
            mh, mw, io_dtype, operand, vc, lds=_LDS_REGION,
            tiles_per_thread=_LDS_TPT, lds_lanes=_LDS_LANES,
        )  # fmt: skip
    else:
        out["K2"] = RtInputSpec(mh, mw, io_dtype, operand, vc, 256)
    return out


__all__: Sequence[str] = (
    "RtFilterSpec",
    "RtGeometry",
    "RtInputSpec",
    "RtOutputSpec",
    "RtPrescaleSpec",
    "aot_stage_specs",
    "build_stage",
    "default_stage_specs",
    "pipeline_dtypes",
    "magic_div",
    "stage_launch",
    "stage_signature",
)
