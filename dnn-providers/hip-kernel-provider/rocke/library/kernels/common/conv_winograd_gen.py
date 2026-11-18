# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generalized Winograd ``F(my x mx, ry x rx)`` forward conv: transform kernels.

The 3x3 pipeline (:mod:`kernels.common.conv_winograd_fwd`) is specialized to
``F(m x m, 3 x 3)`` with ``m`` in {2, 4}. This module generalizes the three
transform kernels to any filter extent per axis, with selectable Toom-Cook
interpolation points, and adds *segmented* Winograd for long 1-D filters:

* Per axis, ``F(m, r)`` on ``n - 1`` finite points plus infinity
  (:func:`toom_cook_gen`, exact rationals). A 1-D conv (``Y = 1``) uses the
  trivial ``F(1, 1)`` on the H axis.
* Segmented along W (``seg = S > 1``): the ``X`` taps are split into ``S``
  segments of ``mx`` taps (the last one zero padded); each segment is an
  ``F(mx, mx)`` conv on the input shifted by ``s * mx``, which is the input
  tile ``t + s``. All segments therefore share the input transform::

      M_b[t] = sum_s V_b[t + s] U_{b,s}

  i.e. K3 becomes a batched GEMM with ``S`` taps along the tile axis
  (``WinogradGemmWmmaSpec.taps``). ``n = 2 mx - 1``.

Pipeline and layouts (``bins = ny * nx``, group index ``G' = bin * g + gi``)::

    w KYXC  --K1-->  U [G'][S][Kg][Cg]
    x NHWC  --K2-->  V [T + S - 1][bins][C]      (rows past T: never stored)
    V, U    --K3-->  M [T][bins][K]              (batched GEMM, S taps)
    M       --K4-->  y NHWK

Tiles. Dilation ``d`` is handled as ``dH * dW`` sub-grids (as in the 3x3
pipeline). Each sub-grid row of tiles holds ``tw_in = tw + S - 1`` tiles:
the last ``S - 1`` are *halo* tiles whose V feeds the shifted taps of the row
and whose M is garbage (K4 skips them). ``T = N * dH * dW * th * tw_in``.
The tile index is ``t = (((n * dH + a) * dW + bs) * th + ti) * tw_in + tj``.

Arithmetic: transforms run in fp32 with the rational coefficients rounded to
fp32. Every load/store goes through a buffer resource; padding, dilation
holes and out-of-range outputs redirect the offset to an out-of-range value.

Thread mapping (K2/K4): one thread per (tile, ``vec`` channels), channels
fastest so a wave reads/writes contiguous channel runs. K1: one thread per
(output channel, ``vec`` input channels) and segment ``s`` (grid y).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import List, Optional, Tuple

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_fwd import (
    _CPOL_NT,
    _MAX_BUFFER_BYTES,
    _ir_dtype,
    _load_raw,
    _oob_sel,
    _pack_vec_f32,
    _pow2_vec,
    _raw_to_f32,
    _solve_exact,
    _store_packed,
)

F = Fraction

# Interpolation points (finite ones; infinity is always added) per F(m, r).
# Chosen by a CPU numerics study (fp16 I/O and operands, M stored fp16):
# the set with the lowest worst-case
# rocke metric over gauss / leaky-relu / silu / offset inputs.
POINTS = {
    (1, 1): (),
    (2, 3): (0, 1, -1),
    (4, 3): (0, 1, -1, 2, -2),
    (6, 3): (0, 1, -1, 2, -2, F(1, 2), F(-1, 2)),
    (8, 3): (0, 2, -2, F(1, 2), F(-1, 2), F(3, 4), F(-3, 4), F(4, 3), F(-4, 3)),
    (2, 5): (0, 1, -1, 2, -2),
    (3, 5): (0, 1, -1, 2, -2, F(1, 2)),
    (4, 5): (0, 1, -1, F(1, 2), F(-1, 2), F(3, 2), F(-3, 2)),
    (6, 5): (0, 2, -2, F(1, 2), F(-1, 2), F(3, 4), F(-3, 4), F(4, 3), F(-4, 3)),
    (2, 7): (0, 1, -1, 2, -2, F(1, 2), F(-1, 2)),
    (4, 7): (0, 2, -2, F(1, 2), F(-1, 2), F(3, 4), F(-3, 4), F(4, 3), F(-4, 3)),
    (6, 7): (
        0,
        1,
        -1,
        3,
        -3,
        F(1, 3),
        F(-1, 3),
        F(2, 3),
        F(-2, 3),
        F(4, 3),
        F(-4, 3),
    ),  # fmt: skip
    (2, 11): (
        0,
        1,
        -1,
        3,
        -3,
        F(1, 3),
        F(-1, 3),
        F(3, 2),
        F(-3, 2),
        F(3, 4),
        F(-3, 4),
    ),  # fmt: skip
    # Segment transforms F(m, m) of the segmented long-filter variants.
    (3, 3): (F(1, 2), F(-1, 2), F(4, 3), F(-4, 3)),
    (4, 4): (1, -1, 2, -2, F(1, 4), F(-1, 4)),
    (5, 5): (2, -2, F(1, 4), F(-1, 4), F(2, 3), F(-2, 3), F(4, 3), F(-4, 3)),
    (6, 6): (1, -1, F(1, 2), F(-1, 2), 3, -3, F(1, 4), F(-1, 4), F(4, 3), F(-4, 3)),
}

_IO_DTYPES = ("fp16", "bf16")


@lru_cache(maxsize=None)
def toom_cook_gen(m: int, r: int, pts: Optional[tuple] = None):
    """Exact ``(A^T [m x n], G [n x r], B^T [n x n])`` of ``F(m, r)``.

    ``pts``: the ``n - 1`` finite interpolation points (default
    :data:`POINTS`); infinity is the last point. ``y_i = sum_j A^T[i, j]
    (G g)_j (B^T d)_j`` is the correlation ``sum_q g_q d_{i + q}``; ``B^T`` is
    the unique solution, computed exactly.
    """
    if pts is None:
        if (m, r) not in POINTS:
            raise ValueError(f"no default points for F({m},{r})")
        pts = POINTS[(m, r)]
    n = m + r - 1
    pts = [Fraction(p) for p in pts]
    if len(pts) != n - 1 or len(set(pts)) != n - 1:
        raise ValueError(f"F({m},{r}) needs {n - 1} distinct finite points")
    at = [[Fraction(0)] * n for _ in range(m)]
    g = [[Fraction(0)] * r for _ in range(n)]
    for j, a in enumerate(pts):
        f = Fraction(1)
        for k, bb in enumerate(pts):
            if k != j:
                f *= a - bb
        for i in range(m):
            at[i][j] = a**i
        for k in range(r):
            g[j][k] = a**k / f
    at[m - 1][n - 1] = Fraction(1)
    g[n - 1][r - 1] = Fraction(1)
    bt = [[Fraction(0)] * n for _ in range(n)]
    for p in range(n):
        rows, rhs = [], []
        for i in range(m):
            for q in range(r):
                rows.append([at[i][j] * g[j][q] for j in range(n)])
                rhs.append(Fraction(1 if p == i + q else 0))
        col = _solve_exact(rows, rhs)
        for j in range(n):
            bt[j][p] = col[j]
    return (
        tuple(tuple(r_) for r_ in at),
        tuple(tuple(r_) for r_ in g),
        tuple(tuple(r_) for r_ in bt),
    )


@dataclass(frozen=True)
class WinoAxis:
    """One axis of the transform: ``F(m, r)`` or segmented ``S x F(m, m)``."""

    m: int
    r: int  # filter taps on this axis
    seg: int = 1  # segments (S); > 1 means S x F(m, m)
    pts: Optional[tuple] = None

    @property
    def r1(self) -> int:
        """Taps of the per-segment transform."""
        return self.m if self.seg > 1 else self.r

    @property
    def n(self) -> int:
        return self.m + self.r1 - 1

    def matrices(self):
        return toom_cook_gen(self.m, self.r1, self.pts)

    def check(self) -> None:
        if self.m < 1 or self.r < 1 or self.seg < 1:
            raise ValueError(f"bad axis {self}")
        if self.seg > 1 and self.seg * self.m < self.r:
            raise ValueError(f"{self.seg} segments of {self.m} taps < r={self.r}")
        if self.seg > 1 and (self.seg - 1) * self.m >= self.r:
            raise ValueError("an all-zero segment")
        self.matrices()  # raises when no point set exists

    def label(self) -> str:
        if self.seg > 1:
            return f"{self.seg}xF{self.m}_{self.m}"
        return f"F{self.m}_{self.r}"


@dataclass(frozen=True)
class WinogradGenSpec:
    """One generalized Winograd forward pipeline (K1, K2, K4).

    Args:
        problem: the conv (stride 1, any pad / dilation / groups, 2-D).
        ax_y, ax_x: per-axis transforms; ``Y == 1`` must use ``F(1, 1)``.
            Segments (``seg > 1``) are supported on the W axis only.
        io_dtype: x, w, y dtype. operand_dtype: U, V, M dtype (fp16).
        vec: channel vector width (largest power of two <= vec dividing the
            channel extent is used).
    """

    problem: ConvProblem
    ax_y: WinoAxis
    ax_x: WinoAxis
    io_dtype: str = "fp16"
    operand_dtype: str = "fp16"
    vec: int = 4
    block_size: int = 256
    nt_stores: bool = True
    name: str = "conv_winograd_gen"

    def __post_init__(self) -> None:
        p = self.problem
        if p.is_3d:
            raise ValueError("2-D convolutions only")
        if (p.sH, p.sW) != (1, 1):
            raise ValueError("stride 1 only")
        for ax in (self.ax_y, self.ax_x):
            ax.check()
        if self.ax_y.r != p.Y or self.ax_x.r != p.X:
            raise ValueError("axis taps must match the filter")
        if self.ax_y.seg != 1:
            raise ValueError("segments are supported along W only")
        if self.io_dtype not in _IO_DTYPES or self.operand_dtype not in _IO_DTYPES:
            raise ValueError("dtype must be fp16 / bf16")
        if self.vec not in (1, 2, 4, 8):
            raise ValueError("vec must be 1/2/4/8")
        if self.block_size < 32 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two >= 32")
        if p.Ho <= 0 or p.Wo <= 0:
            raise ValueError("empty output")
        for nm, nbytes in self.buffer_bytes().items():
            if nbytes > _MAX_BUFFER_BYTES:
                raise ValueError(f"{nm} needs {nbytes} bytes (> 32-bit offsets)")

    # The 3x3 pipeline's host class reads these.
    prescale = False

    # -- geometry --
    @property
    def S(self) -> int:
        return self.ax_x.seg

    @property
    def bins(self) -> int:
        return self.ax_y.n * self.ax_x.n

    @property
    def tiles_h(self) -> int:
        return ceil_div(ceil_div(self.problem.Ho, self.problem.dH), self.ax_y.m)

    @property
    def tiles_w(self) -> int:
        return ceil_div(ceil_div(self.problem.Wo, self.problem.dW), self.ax_x.m)

    @property
    def tiles_w_in(self) -> int:
        return self.tiles_w + self.S - 1

    @property
    def num_tiles(self) -> int:
        p = self.problem
        return p.N * p.dH * p.dW * self.tiles_h * self.tiles_w_in

    @property
    def vec_c(self) -> int:
        return _pow2_vec(self.problem.C, self.vec)

    @property
    def vec_k(self) -> int:
        return _pow2_vec(self.problem.K, self.vec)

    @property
    def vec_cg(self) -> int:
        return _pow2_vec(self.problem.cpg, self.vec)

    # -- sizes --
    def buffer_bytes(self) -> dict:
        p = self.problem
        t = self.num_tiles
        return {
            "x": p.N * p.Hi * p.Wi * p.C * 2,
            "w": p.K * p.Y * p.X * p.cpg * 2,
            "y": p.N * p.Ho * p.Wo * p.K * 2,
            "U": self.bins * self.S * p.K * p.cpg * 2,
            # S - 1 extra rows: the shifted taps of the last tiles read them.
            "V": (t + self.S - 1) * self.bins * p.C * 2,
            "M": t * self.bins * p.K * 2,
            "e": 4,
        }

    def gemm_problem(self):
        from kernels.common.conv_winograd_gemm_wmma import WinogradGemmProblem

        p = self.problem
        return WinogradGemmProblem(
            T=self.num_tiles, G=self.bins * p.groups, Cg=p.cpg, Kg=p.kpg
        )

    def gemm_flops(self) -> int:
        return self.gemm_problem().flops * self.S

    def short(self) -> str:
        tags = [
            self.problem.short(),
            self.ax_y.label(),
            self.ax_x.label(),
            self.io_dtype,
            f"v{self.vec}",
            f"b{self.block_size}",
        ]
        if not self.nt_stores:
            tags.append("rt")
        return "_".join(tags)

    def kernel_name(self, stage: str) -> str:
        return kernel_name_join(self.name, stage, self.short())


def default_axes(problem: ConvProblem, mx: int, seg: int = 1, my: int = 0):
    """``(ax_y, ax_x)`` for a filter: ``F(1,1)`` on a unit axis; ``my = 0``
    means "same m as x" for 2-D filters."""
    if problem.Y == 1:
        ay = WinoAxis(1, 1)
    else:
        ay = WinoAxis(my or mx, problem.Y)
    if problem.X == 1:
        ax = WinoAxis(1, 1)
    else:
        ax = WinoAxis(mx, problem.X, seg)
    return ay, ax


def _f(v) -> float:
    return float(v)


def _lincomb_f(b: IRBuilder, terms):
    """``sum(c * v)`` with float coefficients (fp32), skipping zeros."""
    acc = None
    for c, v in terms:
        c = float(c)
        if c == 0.0:
            continue
        if acc is None:
            if c == 1.0:
                acc = v
            elif c == -1.0:
                acc = b.fneg(v)
            else:
                acc = b.fmul(b.const_f32(c), v)
        elif c == 1.0:
            acc = b.fadd(acc, v)
        elif c == -1.0:
            acc = b.fsub(acc, v)
        else:
            acc = b.fma(b.const_f32(c), v, acc)
    return acc if acc is not None else b.const_f32(0.0)


@dataclass(frozen=True)
class _TileCoords:
    ok: object
    cv: object
    n: object
    a: object
    bs: object
    ti: object
    tj: object
    t: object


def _flat_tile(b: IRBuilder, spec: WinogradGenSpec, num_vecs: int, tw_tiles: int):
    """Thread -> (tile over ``tw_tiles`` per row, channel vector); channels fastest."""
    p = spec.problem
    gid = b.add(b.mul(b.block_id_x(), b.const_i32(spec.block_size)), b.thread_id_x())
    total = spec.num_tiles // spec.tiles_w_in * tw_tiles * num_vecs
    ok = b.cmp_lt(gid, b.const_i32(total))
    cv = b.mod(gid, b.const_i32(num_vecs)) if num_vecs > 1 else b.const_i32(0)
    r = b.div(gid, b.const_i32(num_vecs)) if num_vecs > 1 else gid
    tj = b.mod(r, b.const_i32(tw_tiles))
    r = b.div(r, b.const_i32(tw_tiles))
    ti = b.mod(r, b.const_i32(spec.tiles_h)) if spec.tiles_h > 1 else b.const_i32(0)
    r = b.div(r, b.const_i32(spec.tiles_h)) if spec.tiles_h > 1 else r
    bs = b.mod(r, b.const_i32(p.dW)) if p.dW > 1 else b.const_i32(0)
    r = b.div(r, b.const_i32(p.dW)) if p.dW > 1 else r
    a = b.mod(r, b.const_i32(p.dH)) if p.dH > 1 else b.const_i32(0)
    nn = b.div(r, b.const_i32(p.dH)) if p.dH > 1 else r
    row = b.add(
        b.mul(
            b.add(b.mul(b.add(b.mul(nn, b.const_i32(p.dH)), a), b.const_i32(p.dW)), bs),
            b.const_i32(spec.tiles_h),
        ),
        ti,
    )
    t = b.add(b.mul(row, b.const_i32(spec.tiles_w_in)), tj)
    return _TileCoords(ok, cv, nn, a, bs, ti, tj, t)


def _grid(spec: WinogradGenSpec, items: int) -> Tuple[int, int, int]:
    gx = ceil_div(items, spec.block_size)
    if gx <= 0:
        raise ValueError("empty grid")
    return (gx, 1, 1)


# ---------------------------------------------------------------------------
# K1: filter transform
# ---------------------------------------------------------------------------


def filter_transform_grid(spec: WinogradGenSpec) -> Tuple[int, int, int]:
    p = spec.problem
    gx, _, _ = _grid(spec, p.K * (p.cpg // spec.vec_cg))
    return (gx, spec.S, 1)


def filter_transform_signature(spec: WinogradGenSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("w", spec.io_dtype)
        .ptr("u", spec.operand_dtype)
        .scalar("w_bytes", "i32")
        .scalar("u_bytes", "i32")
        .build()
    )


def build_filter_transform(
    spec: WinogradGenSpec, *, arch: str = "gfx1250"
) -> KernelDef:
    """K1: ``w[k][ky][kx][c] -> U[G'][s][kg][c] = (G_y g_s G_x^T)[bin]``."""
    del arch
    p = spec.problem
    vec = spec.vec_cg
    cg, kg, g = p.cpg, p.kpg, p.groups
    cgv = cg // vec
    items = p.K * cgv
    S = spec.S
    ay, ax = spec.ax_y, spec.ax_x
    _, gy, _ = ay.matrices()
    _, gx, _ = ax.matrices()
    ny, nx = ay.n, ax.n
    r1x = ax.r1

    b = IRBuilder(spec.kernel_name("filter"))
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    w = b.param(
        "w", PtrType(_ir_dtype(spec.io_dtype), "global"), noalias=True, readonly=True
    )
    u = b.param(
        "u",
        PtrType(_ir_dtype(spec.operand_dtype), "global"),
        noalias=True,
        writeonly=True,
    )
    w_rsrc = b.buffer_rsrc(w, b.param("w_bytes", I32))
    u_rsrc = b.buffer_rsrc(u, b.param("u_bytes", I32))
    gid = b.add(b.mul(b.block_id_x(), b.const_i32(spec.block_size)), b.thread_id_x())
    s_idx = b.block_id_y()  # segment (uniform)
    ok = b.cmp_lt(gid, b.const_i32(items))
    k = b.div(gid, b.const_i32(cgv))
    c = b.mul(b.mod(gid, b.const_i32(cgv)), b.const_i32(vec))
    wbase = _oob_sel(
        b, ok, b.mul(b.add(b.mul(k, b.const_i32(p.Y * p.X * cg)), c), b.const_i32(2))
    )
    kx0 = b.mul(s_idx, b.const_i32(r1x)) if S > 1 else b.const_i32(0)
    taps = []
    for ky in range(p.Y):
        row = []
        for q in range(r1x):
            kx = b.add(kx0, b.const_i32(q))
            okx = b.cmp_lt(kx, b.const_i32(p.X))
            off = b.add(
                wbase, b.mul(b.add(b.const_i32(ky * p.X), kx), b.const_i32(cg * 2))
            )
            off = _oob_sel(b, okx, off)
            row.append(
                _raw_to_f32(b, _load_raw(b, w_rsrc, off, spec.io_dtype, vec), vec)
            )
        taps.append(row)
    # tmp[ky][jx][ch] = sum_q Gx[jx][q] g[ky][q]
    tmp = [
        [
            [
                _lincomb_f(b, [(gx[jx][q], taps[ky][q][ch]) for q in range(r1x)])
                for ch in range(vec)
            ]
            for jx in range(nx)
        ]
        for ky in range(p.Y)
    ]
    gi = b.div(k, b.const_i32(kg)) if g > 1 else b.const_i32(0)
    kk = b.mod(k, b.const_i32(kg)) if g > 1 else k
    cpol = 0
    for jy in range(ny):
        for jx in range(nx):
            bin_ = jy * nx + jx
            vals = [
                _lincomb_f(b, [(gy[jy][ky], tmp[ky][jx][ch]) for ky in range(p.Y)])
                for ch in range(vec)
            ]
            # U[((bin * g + gi) * S + s) * kg + kk][c]
            gp = b.add(b.const_i32(bin_ * g), gi)
            row = b.add(
                b.mul(b.add(b.mul(gp, b.const_i32(S)), s_idx), b.const_i32(kg)), kk
            )
            off = _oob_sel(
                b, ok, b.mul(b.add(b.mul(row, b.const_i32(cg)), c), b.const_i32(2))
            )
            _store_packed(
                b,
                u_rsrc,
                off,
                _pack_vec_f32(b, vals, spec.operand_dtype),
                spec.operand_dtype,
                cpol,
            )
    return b.kernel


# ---------------------------------------------------------------------------
# K2: input transform
# ---------------------------------------------------------------------------


def input_transform_grid(spec: WinogradGenSpec) -> Tuple[int, int, int]:
    return _grid(spec, spec.num_tiles * (spec.problem.C // spec.vec_c))


def input_transform_signature(spec: WinogradGenSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("x", spec.io_dtype)
        .ptr("v", spec.operand_dtype)
        .scalar("x_bytes", "i32")
        .scalar("v_bytes", "i32")
        .build()
    )


def build_input_transform(spec: WinogradGenSpec, *, arch: str = "gfx1250") -> KernelDef:
    """K2: ``V[t][bin][c] = (B_y^T d B_x)[bin]`` over the tile's input patch."""
    del arch
    p = spec.problem
    vec = spec.vec_c
    ay, ax = spec.ax_y, spec.ax_x
    _, _, bty = ay.matrices()
    _, _, btx = ax.matrices()
    ny, nx = ay.n, ax.n
    my, mx = ay.m, ax.m

    b = IRBuilder(spec.kernel_name("input"))
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
    x_rsrc = b.buffer_rsrc(x, b.param("x_bytes", I32))
    v_rsrc = b.buffer_rsrc(v, b.param("v_bytes", I32))
    tc = _flat_tile(b, spec, p.C // vec, spec.tiles_w_in)
    c = b.mul(tc.cv, b.const_i32(vec))
    # row part (n, hi) and column part (wi), each OOB when outside the image
    img = b.mul(tc.n, b.const_i32(p.Hi))
    hi0 = b.sub(b.add(tc.a, b.mul(tc.ti, b.const_i32(p.dH * my))), b.const_i32(p.pH))
    rows = []
    for uu in range(ny):
        hi = b.add(hi0, b.const_i32(uu * p.dH))
        okr = b.land(
            tc.ok, b.land(b.cmp_ge(hi, b.const_i32(0)), b.cmp_lt(hi, b.const_i32(p.Hi)))
        )
        off = b.mul(
            b.add(b.mul(b.mul(b.add(img, hi), b.const_i32(p.Wi)), b.const_i32(p.C)), c),
            b.const_i32(2),
        )
        rows.append(_oob_sel(b, okr, off))
    wi0 = b.sub(b.add(tc.bs, b.mul(tc.tj, b.const_i32(p.dW * mx))), b.const_i32(p.pW))
    cols = []
    for vv in range(nx):
        wi = b.add(wi0, b.const_i32(vv * p.dW))
        okc = b.land(b.cmp_ge(wi, b.const_i32(0)), b.cmp_lt(wi, b.const_i32(p.Wi)))
        cols.append(_oob_sel(b, okc, b.mul(wi, b.const_i32(p.C * 2))))
    data = [
        [
            _raw_to_f32(
                b,
                _load_raw(b, x_rsrc, b.add(rows[uu], cols[vv]), spec.io_dtype, vec),
                vec,
            )
            for vv in range(nx)
        ]
        for uu in range(ny)
    ]
    b.sched_barrier(0)
    # x pass: tx[uu][jx][ch]
    tx = [
        [
            [
                _lincomb_f(b, [(btx[jx][vv], data[uu][vv][ch]) for vv in range(nx)])
                for ch in range(vec)
            ]
            for jx in range(nx)
        ]
        for uu in range(ny)
    ]
    vbase = _oob_sel(
        b,
        tc.ok,
        b.mul(b.add(b.mul(tc.t, b.const_i32(spec.bins * p.C)), c), b.const_i32(2)),
    )
    pending = []
    for jy in range(ny):
        for jx in range(nx):
            vals = [
                _lincomb_f(b, [(bty[jy][uu], tx[uu][jx][ch]) for uu in range(ny)])
                for ch in range(vec)
            ]
            off = b.add(vbase, b.const_i32((jy * nx + jx) * p.C * 2))
            pending.append((off, _pack_vec_f32(b, vals, spec.operand_dtype)))
    b.sched_barrier(0)
    cpol = _CPOL_NT if spec.nt_stores else 0
    for off, packed in pending:
        _store_packed(b, v_rsrc, off, packed, spec.operand_dtype, cpol)
    return b.kernel


# ---------------------------------------------------------------------------
# K4: output transform
# ---------------------------------------------------------------------------


def output_transform_grid(spec: WinogradGenSpec) -> Tuple[int, int, int]:
    rows = spec.num_tiles // spec.tiles_w_in
    return _grid(spec, rows * spec.tiles_w * (spec.problem.K // spec.vec_k))


def output_transform_signature(spec: WinogradGenSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("mm", spec.operand_dtype)
        .ptr("y", spec.io_dtype)
        .scalar("m_bytes", "i32")
        .scalar("y_bytes", "i32")
        .build()
    )


def build_output_transform(
    spec: WinogradGenSpec, *, arch: str = "gfx1250"
) -> KernelDef:
    """K4: ``y = A_y^T M A_x`` per tile (halo tiles skipped), cropped."""
    del arch
    p = spec.problem
    vec = spec.vec_k
    ay, ax = spec.ax_y, spec.ax_x
    aty, _, _ = ay.matrices()
    atx, _, _ = ax.matrices()
    ny, nx = ay.n, ax.n
    my, mx = ay.m, ax.m

    b = IRBuilder(spec.kernel_name("output"))
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    mm = b.param(
        "mm",
        PtrType(_ir_dtype(spec.operand_dtype), "global"),
        noalias=True,
        readonly=True,
    )
    y = b.param(
        "y", PtrType(_ir_dtype(spec.io_dtype), "global"), noalias=True, writeonly=True
    )
    m_rsrc = b.buffer_rsrc(mm, b.param("m_bytes", I32))
    y_rsrc = b.buffer_rsrc(y, b.param("y_bytes", I32))
    tc = _flat_tile(b, spec, p.K // vec, spec.tiles_w)
    k = b.mul(tc.cv, b.const_i32(vec))
    mbase = _oob_sel(
        b,
        tc.ok,
        b.mul(b.add(b.mul(tc.t, b.const_i32(spec.bins * p.K)), k), b.const_i32(2)),
    )
    raw = [
        [
            _load_raw(
                b,
                m_rsrc,
                b.add(mbase, b.const_i32((uu * nx + vv) * p.K * 2)),
                spec.operand_dtype,
                vec,
            )
            for vv in range(nx)
        ]
        for uu in range(ny)
    ]
    ho0 = b.add(tc.a, b.mul(tc.ti, b.const_i32(p.dH * my)))
    wo0 = b.add(tc.bs, b.mul(tc.tj, b.const_i32(p.dW * mx)))
    img = b.mul(tc.n, b.const_i32(p.Ho))
    yrow = []
    for i in range(my):
        ho = b.add(ho0, b.const_i32(i * p.dH))
        okr = b.land(tc.ok, b.cmp_lt(ho, b.const_i32(p.Ho)))
        off = b.mul(
            b.add(b.mul(b.mul(b.add(img, ho), b.const_i32(p.Wo)), b.const_i32(p.K)), k),
            b.const_i32(2),
        )
        yrow.append(_oob_sel(b, okr, off))
    ycol = []
    for j in range(mx):
        wo = b.add(wo0, b.const_i32(j * p.dW))
        ycol.append(
            _oob_sel(
                b, b.cmp_lt(wo, b.const_i32(p.Wo)), b.mul(wo, b.const_i32(p.K * 2))
            )
        )
    b.sched_barrier(0)
    data = [[_raw_to_f32(b, raw[uu][vv], vec) for vv in range(nx)] for uu in range(ny)]
    # x pass then y pass
    tx = [
        [
            [
                _lincomb_f(b, [(atx[j][vv], data[uu][vv][ch]) for vv in range(nx)])
                for ch in range(vec)
            ]
            for j in range(mx)
        ]
        for uu in range(ny)
    ]
    pending = []
    for i in range(my):
        for j in range(mx):
            vals = [
                _lincomb_f(b, [(aty[i][uu], tx[uu][j][ch]) for uu in range(ny)])
                for ch in range(vec)
            ]
            pending.append(
                (b.add(yrow[i], ycol[j]), _pack_vec_f32(b, vals, spec.io_dtype))
            )
    b.sched_barrier(0)
    cpol = _CPOL_NT if spec.nt_stores else 0
    for off, packed in pending:
        _store_packed(b, y_rsrc, off, packed, spec.io_dtype, cpol)
    return b.kernel


@dataclass(frozen=True)
class GenStage:
    name: str
    build: object
    grid: Tuple[int, int, int]
    block: int
    signature: list


def transform_stages(spec: WinogradGenSpec) -> List[GenStage]:
    bs = spec.block_size
    return [
        GenStage(
            "K1",
            build_filter_transform,
            filter_transform_grid(spec),
            bs,
            filter_transform_signature(spec),
        ),
        GenStage(
            "K2",
            build_input_transform,
            input_transform_grid(spec),
            bs,
            input_transform_signature(spec),
        ),
        GenStage(
            "K4",
            build_output_transform,
            output_transform_grid(spec),
            bs,
            output_transform_signature(spec),
        ),
    ]


# ---------------------------------------------------------------------------
# numpy model of the four stages (CPU tests)
# ---------------------------------------------------------------------------


def winograd_gen_reference_np(x, w, spec: WinogradGenSpec, *, round_fn=None):
    """Numpy model of K1..K4 with the kernels' tiling and layouts.

    ``x`` NHWC, ``w`` KYXC. Returns ``(y, U, V, M)``; ``round_fn(name, a)``
    rounds the workspaces (None keeps fp64).
    """
    import numpy as np

    p = spec.problem
    ay, ax = spec.ax_y, spec.ax_x
    f64 = lambda mat: np.array([[float(v) for v in r_] for r_ in mat])  # noqa: E731
    aty, gy, bty = (f64(a_) for a_ in ay.matrices())
    atx, gx, btx = (f64(a_) for a_ in ax.matrices())
    ny, nx, my, mx = ay.n, ax.n, ay.m, ax.m
    S, r1x = spec.S, ax.r1
    rnd = round_fn or (lambda _n, a_: a_)
    g, cg, kg = p.groups, p.cpg, p.kpg
    x = np.asarray(x, np.float64)
    w = np.asarray(w, np.float64)
    bins = ny * nx
    # K1
    wp = np.zeros((p.K, p.Y, S * r1x, cg))
    wp[:, :, : p.X] = w
    U = np.zeros((bins, g, S, kg, cg))
    for s in range(S):
        u4 = np.einsum("ia,kabc,jb->ijkc", gy, wp[:, :, s * r1x : (s + 1) * r1x], gx)
        u4 = u4.reshape(bins, g, kg, cg)
        U[:, :, s] = u4
    U = rnd("U", U.reshape(bins * g, S, kg, cg))
    # K2
    th, twin, tw = spec.tiles_h, spec.tiles_w_in, spec.tiles_w
    T = spec.num_tiles
    V = np.zeros((T + S - 1, bins, p.C))
    coords = []
    for nn in range(p.N):
        for a in range(p.dH):
            for bs in range(p.dW):
                for ti in range(th):
                    for tj in range(twin):
                        coords.append((nn, a, bs, ti, tj))
    xp_cache = {}
    for t, (nn, a, bs, ti, tj) in enumerate(coords):
        d = np.zeros((ny, nx, p.C))
        for uu in range(ny):
            hi = a - p.pH + p.dH * (ti * my + uu)
            if not 0 <= hi < p.Hi:
                continue
            for vv in range(nx):
                wi = bs - p.pW + p.dW * (tj * mx + vv)
                if 0 <= wi < p.Wi:
                    d[uu, vv] = x[nn, hi, wi]
        V[t] = np.einsum("iu,uvc,jv->ijc", bty, d, btx).reshape(bins, p.C)
    del xp_cache
    V = rnd("V", V)
    # K3
    M = np.zeros((T, bins, p.K))
    Ug = U.reshape(bins, g, S, kg, cg)
    for gi in range(g):
        for s in range(S):
            vg = V[s : s + T, :, gi * cg : (gi + 1) * cg]
            M[:, :, gi * kg : (gi + 1) * kg] += np.einsum(
                "tbc,bkc->tbk", vg, Ug[:, gi, s]
            )
    M = rnd("M", M)
    # K4
    y = np.zeros((p.N, p.Ho, p.Wo, p.K))
    for t, (nn, a, bs, ti, tj) in enumerate(coords):
        if tj >= tw:
            continue
        Y = np.einsum("iu,uvk,jv->ijk", aty, M[t].reshape(ny, nx, p.K), atx)
        for i in range(my):
            ho = a + p.dH * (ti * my + i)
            if ho >= p.Ho:
                continue
            for j in range(mx):
                wo = bs + p.dW * (tj * mx + j)
                if wo < p.Wo:
                    y[nn, ho, wo] = Y[i, j]
    return y, U, V, M


__all__ = [
    "POINTS",
    "WinoAxis",
    "WinogradGenSpec",
    "build_filter_transform",
    "build_input_transform",
    "build_output_transform",
    "default_axes",
    "toom_cook_gen",
    "transform_stages",
    "winograd_gen_reference_np",
]
