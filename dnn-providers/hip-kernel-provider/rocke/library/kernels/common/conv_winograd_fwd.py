# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Winograd ``F(m x m, 3 x 3)`` forward convolution: the three transform kernels.

The forward conv runs as an *unfused* four-kernel pipeline::

    w KYXC --(K1 filter transform)--> U [n2][K][Cg]          (operand dtype)
    x NHWC --(K2 input transform)---> V [T][n2][C]           (operand dtype)
                                      e [T]  int32           (prescale only)
    V, U ---(K3 grouped 1x1 GEMM, groups = n2*g)--> M [T][n2][K]  (operand dtype)
    M ------(K4 output transform, x 2**e)---------> y NHWK   (io dtype)

with ``n = m + 2`` and ``n2 = n * n``. K3 is the existing implicit-GEMM conv
launched on the grouped 1x1 problem :func:`winograd_gemm_problem` returns; this
module only holds K1, K2 and K4 plus the host-side sizing helpers.

Layouts. The GEMM group index is ``G = bin * g + gi`` (bin-major), so every
workspace keeps the conv's own channel numbering inside a bin:

* ``V[t][bin][c]`` with ``c`` the absolute input channel (``gi * Cg + c_in``),
* ``U[bin][k][c_in]`` with ``k`` the absolute output channel,
* ``M[t][bin][k]``.

Read as a grouped 1x1 NHWC conv with ``N = T``, ``C = n2 * C``, ``K = n2 * K``
and ``groups = n2 * g`` these are exactly its input / GKYXC weight / output, and
the transform kernels never need to know the group of a channel.

Tiles. A dilated 3x3 conv with dilation ``d`` is ``d_h * d_w`` independent
undilated convs on the sub-grids ``ho = a + d_h * i``, ``wo = b + d_w * j``.
Each sub-grid is tiled into ``m x m`` output tiles; ``T = N * d_h * d_w * th *
tw`` with ``th = ceil(ceil(Ho / d_h) / m)`` (same for w). The tile index is
``t = (((n * d_h + a) * d_w + b) * th + ti) * tw + tj``. The input patch of a
tile is rows ``hi(u) = a - pH + d_h * (ti * m + u)`` for ``u < n``; rows outside
the image read as zero (buffer OOB), output rows past ``Ho`` are dropped (buffer
OOB store).

Thread mapping. Lanes run along channels (``vec`` per lane), so a wave reads /
writes a contiguous channel run of one pixel / ``(t, bin)`` row.

* K4: one thread per (tile, ``vec_k`` channels), ``block / tpc`` tiles per
  workgroup with ``tpc = min(pow2ceil(K / vec), block)``.
* K2, register path: one thread per *run* of ``tiles_per_thread`` tiles along
  W x ``vec_c`` channels; the run's ``n x (m * tiles + 2)`` columns are loaded
  once and their column transforms are shared by adjacent tiles.
  ``wg_tile_rows`` stacks runs of vertically adjacent tile rows in one
  workgroup.
* K2, LDS path (``input_block_tiles = (bh, bw)``): a workgroup loads the
  ``(m*bh + 2) x (m*bw + 2)``-pixel region of ``bh x bw`` tiles (x
  ``lds_lanes * vec_c`` channels) into LDS once, then each thread transforms
  ``tiles_per_thread`` tiles from LDS. Neighbouring patches overlap by two
  rows and columns, so without LDS each pixel is read up to ``(n / m)**2``
  times through L2; that over-read, not DRAM, was K2's main cost.
* K1: one thread per (output channel, ``vec_cg`` channels, row ``i`` of
  ``G g``); ``i`` is the grid's y index.

Every load / store offset is the sum of a row part and a column part, each
forced to :data:`_OOB` when out of range, so padding, dilation holes and tail
threads need no branch. All stores of a thread are issued last (after a
scheduling barrier): a store's data VGPRs cannot be reused while it is in
flight (``s_wait_xcnt``), and interleaving stores with the arithmetic stalled
the waves. V and y are stored non-temporal (``nt_stores``).

Prescale (optional, K2 + K4). ``e[t] = max(0, ceil(log2(beta * max|x_patch| /
2**L)))`` over the tile's whole input patch and all C, with ``beta`` the
input-transform growth bound (the largest row 1-norm of ``B^T``, squared: 4 for
F2, 100 for F4) and ``L = 14 - ceil(log2(C/g) / 2)`` (V at most ``2**14``, minus
headroom for M's sum over ``C/g`` channels, see
:attr:`WinogradFwdSpec.prescale_limit_log2`). K2 scales V by ``2**-e`` and K4
scales y by ``2**e``; both are exact power-of-two multiplies, and M is linear
in V. The max is taken on the 16-bit patterns of ``|x|`` (integer order equals
magnitude order), in one of two ways:

* ``fused`` (default): the K2 workgroup holds all channels of its runs; each
  thread's patch max is reduced with xor shuffles and LDS, while the patch
  stays in registers (x is read once);
* ``split``: K2a writes the per-pixel max over C after ``e[T]``; K2 takes a
  tile's max over its patch pixels. Used with the LDS path and when
  ``C / vec > 1024``.

All transform arithmetic is fp32. ``B^T`` and ``A^T`` are small integers,
applied with shared subexpressions (:func:`bt_1d`, :func:`at_1d`: 14 instead of
22 operations per F4 input column); ``G`` is applied as integer numerators
followed by one fp32 scale by ``1 / den**2`` (``den`` = 2 for F2, 24 for F4).

The problem extents are baked into the kernels (compile-time constants), so
divisions by them fold to multiply-high sequences. The workspaces use 32-bit
buffer offsets: every buffer must be at most :data:`_MAX_BUFFER_BYTES`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from rocke.core.ir import (
    BF16,
    F16,
    F32,
    I16,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join
from kernels.common._conv_implicit_gemm_common import ConvProblem

# Interpolation points of the Toom-Cook construction (plus the point at
# infinity).
_POINTS = {
    2: (0, 1, -1),
    4: (0, 1, -1, 2, -2),
    6: (0, 1, -1, 2, -2, Fraction(1, 2), Fraction(-1, 2)),
}
_R = 3  # filter taps per axis

# Largest |V| / max|x_patch| for each m: (max row 1-norm of B^T) ** 2 (for a
# mixed F(mh x mw) tile the product of the two row norms).
PRESCALE_BETA = {2: 4.0, 4: 100.0, 6: 225.0}
# Scaled V is kept at or below 2**14 so fp16 V and M stay finite.
_PRESCALE_LOG2_LIMIT = 14
# Largest exponent K2 emits; anything larger means a non-finite input.
_PRESCALE_MAX_EXP = 100

# Byte offset that is out of range for every buffer: a buffer load from it
# returns 0 and a buffer store to it is dropped. Every buffer is at most this
# size, so ``_OOB`` plus any in-range offset (< 2**31) stays below 2**32 and is
# still out of range: K2 / K4 add a row part and a column part that may each
# be ``_OOB`` without a wrap.
_OOB = 0x7FFF0000
_MAX_BUFFER_BYTES = _OOB
# gfx12 cache-policy immediate of a non-temporal (streaming) store:
# ``th:TH_STORE_NT``. V is written once and read once by K3; marking its
# stores streaming keeps them from displacing x in the caches.
_CPOL_NT = 1

_IO_DTYPES = ("fp16", "bf16")


# ---------------------------------------------------------------------------
# Toom-Cook matrices (exact rationals)
# ---------------------------------------------------------------------------


def _solve_exact(rows: List[List[Fraction]], rhs: List[Fraction]) -> List[Fraction]:
    """Unique solution of an over-determined consistent system (Gauss-Jordan)."""
    nr, nc = len(rows), len(rows[0])
    a = [list(r) + [v] for r, v in zip(rows, rhs)]
    r = 0
    for c in range(nc):
        piv = next((i for i in range(r, nr) if a[i][c] != 0), None)
        if piv is None:
            raise ValueError("Toom-Cook system is rank deficient")
        a[r], a[piv] = a[piv], a[r]
        inv = 1 / a[r][c]
        a[r] = [x * inv for x in a[r]]
        for i in range(nr):
            if i != r and a[i][c] != 0:
                f = a[i][c]
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
        r += 1
    if any(a[i][-1] != 0 for i in range(r, nr)):
        raise ValueError("Toom-Cook system is inconsistent")
    return [a[i][-1] for i in range(nc)]


@lru_cache(maxsize=None)
def toom_cook(m: int) -> Tuple[tuple, tuple, tuple]:
    """``(A^T, G, B^T)`` of ``F(m, 3)`` as tuples of :class:`Fraction` rows.

    ``A^T`` is ``m x n``, ``G`` is ``n x 3``, ``B^T`` is ``n x n``. ``A^T`` and
    ``G`` follow the Lagrange construction on :data:`_POINTS` plus infinity;
    ``B^T`` is the unique matrix that makes ``A^T [(G g) * (B^T d)]`` the 1-D
    correlation of ``d`` with ``g``, solved exactly.
    """
    if m not in _POINTS:
        raise ValueError(f"F({m},3) is not supported; m must be one of {_POINTS}")
    pts = [Fraction(p) for p in _POINTS[m]]
    n = m + _R - 1
    at = [[Fraction(0)] * n for _ in range(m)]
    g = [[Fraction(0)] * _R for _ in range(n)]
    for j, a in enumerate(pts):
        f = Fraction(1)
        for k, bb in enumerate(pts):
            if k != j:
                f *= a - bb
        for i in range(m):
            at[i][j] = a**i
        for k in range(_R):
            g[j][k] = a**k / f
    at[m - 1][n - 1] = Fraction(1)
    g[n - 1][_R - 1] = Fraction(1)
    bt = [[Fraction(0)] * n for _ in range(n)]
    for p in range(n):
        rows, rhs = [], []
        for i in range(m):
            for q in range(_R):
                rows.append([at[i][j] * g[j][q] for j in range(n)])
                rhs.append(Fraction(1 if p == i + q else 0))
        col = _solve_exact(rows, rhs)
        for j in range(n):
            bt[j][p] = col[j]
    return (
        tuple(tuple(r) for r in at),
        tuple(tuple(r) for r in g),
        tuple(tuple(r) for r in bt),
    )


def _common_denominator(mat) -> int:
    den = 1
    for row in mat:
        for v in row:
            den = den * v.denominator // _gcd(den, v.denominator)
    return den


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def filter_matrix_int(m: int) -> Tuple[Tuple[Tuple[int, ...], ...], int]:
    """``(G * den, den)``: integer numerators of ``G`` and their denominator."""
    _, g, _ = toom_cook(m)
    den = _common_denominator(g)
    gi = tuple(tuple(int(v * den) for v in row) for row in g)
    return gi, den


def _int_matrix(mat) -> Tuple[Tuple[int, ...], ...]:
    out = []
    for row in mat:
        r = []
        for v in row:
            if v.denominator != 1:
                raise ValueError("expected an integer matrix")
            r.append(int(v))
        out.append(tuple(r))
    return tuple(out)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def _pow2_vec(n: int, cap: int) -> int:
    """Largest power of two <= ``cap`` that divides ``n``."""
    v = cap
    while v > 1 and n % v:
        v //= 2
    return v


def _pow2_ceil(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@dataclass(frozen=True)
class WinogradFwdSpec:
    """One Winograd ``F(m x m, 3 x 3)`` forward pipeline.

    Args:
        problem:       The conv (3x3, stride 1, any pad, any dilation, groups).
        m:             Output tile size, 2 or 4.
        io_dtype:      dtype of x, w and y (``fp16`` / ``bf16``).
        operand_dtype: dtype of U, V and M (the GEMM's operands and output).
                       ``fp16`` always works; ``bf16`` only for F2 with bf16 I/O
                       (F4 needs the fp16 mantissa).
        vec:           Requested channel vector width (elements); each kernel
                       uses the largest power of two <= ``vec`` that divides its
                       channel extent, so loads/stores stay aligned.
        block_size:    Threads per workgroup for K2/K4 (K2 with the fused
                       prescale widens it to cover all channels of a run).
        prescale:      Per-tile power-of-two prescale (see module doc).
        tiles_per_thread: K2 tiles per thread along W (a "run"); adjacent
                       tiles share two input columns, which are loaded and
                       column-transformed once.
        input_block_tiles: ``(rows, cols)`` of tiles per K2 workgroup, whose
                       input region (shared patch borders included once) is
                       staged through LDS; ``(0, 0)`` = no LDS, every thread
                       loads its own patch(es) (``tiles_per_thread``).
        prescale_mode: ``fused`` (K2 reduces the max over the workgroup that
                       holds all channels of its runs) or ``split`` (a
                       separate per-pixel max pass, K2a). ``fused`` needs
                       ``C / vec <= 1024`` and falls back to ``split``.
    """

    problem: ConvProblem
    m: int = 4
    io_dtype: str = "fp16"
    operand_dtype: str = "fp16"
    vec: int = 8
    block_size: int = 256
    prescale: bool = False
    tiles_per_thread: int = 1
    prescale_mode: str = "fused"
    input_block_tiles: Tuple[int, int] = (0, 0)
    wg_tile_rows: int = 1
    nt_stores: bool = True
    lds_lanes: int = 32
    # K1: rows of ``G g`` per thread (a divisor of n); n reads w once.
    filter_rows: int = 1
    # K1 channel vector width (0: ``vec``).
    filter_vec: int = 0
    # Output tile width (0: ``m``): ``F(m x m_w, 3 x 3)``, ``m`` rows by
    # ``m_w`` columns per tile, ``(m + 2) * (m_w + 2)`` bins.
    m_w: int = 0
    name: str = "conv_winograd_fwd"

    def __post_init__(self) -> None:
        p = self.problem
        if self.m not in _POINTS:
            raise ValueError(f"m must be one of {tuple(_POINTS)} (got {self.m})")
        if self.m_w and self.m_w not in _POINTS:
            raise ValueError(f"m_w must be 0 or one of {tuple(_POINTS)}")
        if p.is_3d:
            raise ValueError("Winograd fwd supports 2-D convolutions only")
        if (p.Y, p.X) != (3, 3):
            raise ValueError(f"Winograd F(m,3) needs a 3x3 filter (got {p.Y}x{p.X})")
        if (p.sH, p.sW) != (1, 1):
            raise ValueError("Winograd fwd needs stride 1")
        if self.io_dtype not in _IO_DTYPES:
            raise ValueError(f"io_dtype must be one of {_IO_DTYPES}")
        if self.operand_dtype not in _IO_DTYPES:
            raise ValueError(f"operand_dtype must be one of {_IO_DTYPES}")
        if self.operand_dtype == "bf16" and (
            (self.mh, self.mw) != (2, 2) or self.io_dtype != "bf16"
        ):
            raise ValueError(
                "bf16 operands are only accurate enough for F(2,3) with bf16 I/O"
            )
        if self.vec not in (1, 2, 4, 8):
            raise ValueError(f"vec must be 1/2/4/8 (got {self.vec})")
        if self.block_size < 32 or self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two >= 32")
        if self.block_size > 1024:
            raise ValueError("block_size must be <= 1024")
        if p.Ho <= 0 or p.Wo <= 0:
            raise ValueError("empty output")
        if self.filter_vec not in (0, 1, 2, 4, 8):
            raise ValueError(f"filter_vec must be 0/1/2/4/8 (got {self.filter_vec})")
        if self.filter_rows < 1 or self.nh % self.filter_rows:
            raise ValueError(f"filter_rows must divide n = {self.nh}")
        if self.tiles_per_thread < 1:
            raise ValueError("tiles_per_thread must be >= 1")
        if self.prescale_mode not in ("fused", "split"):
            raise ValueError("prescale_mode must be 'fused' or 'split'")
        bh, bw = self.input_block_tiles
        if (bh > 0) != (bw > 0) or bh < 0 or bw < 0:
            raise ValueError("input_block_tiles must be (0, 0) or two positive ints")
        for nm, nbytes in self.buffer_bytes().items():
            if nbytes > _MAX_BUFFER_BYTES:
                raise ValueError(
                    f"{nm} needs {nbytes} bytes; the transform kernels use 32-bit "
                    "buffer offsets (< 2 GiB)"
                )

    # -- geometry --

    @property
    def mh(self) -> int:
        """Output tile rows."""
        return self.m

    @property
    def mw(self) -> int:
        """Output tile columns."""
        return self.m_w or self.m

    @property
    def tile_mn(self) -> Tuple[int, int]:
        return (self.mh, self.mw)

    @property
    def nh(self) -> int:
        """Input patch rows (= bins along H)."""
        return self.mh + _R - 1

    @property
    def nw(self) -> int:
        return self.mw + _R - 1

    @property
    def n(self) -> int:
        """Patch size of a square tile (``nh``)."""
        return self.nh

    @property
    def n2(self) -> int:
        """Bins per tile."""
        return self.nh * self.nw

    @property
    def sub_h(self) -> int:
        """Output rows per dilation sub-grid (the largest one)."""
        return ceil_div(self.problem.Ho, self.problem.dH)

    @property
    def sub_w(self) -> int:
        return ceil_div(self.problem.Wo, self.problem.dW)

    @property
    def tiles_h(self) -> int:
        return ceil_div(self.sub_h, self.mh)

    @property
    def tiles_w(self) -> int:
        return ceil_div(self.sub_w, self.mw)

    @property
    def num_tiles(self) -> int:
        p = self.problem
        return p.N * p.dH * p.dW * self.tiles_h * self.tiles_w

    @property
    def vec_c(self) -> int:
        """Vector width over C (K2)."""
        return _pow2_vec(self.problem.C, self.vec)

    @property
    def vec_k(self) -> int:
        """Vector width over K (K4)."""
        return _pow2_vec(self.problem.K, self.vec)

    @property
    def vec_cg(self) -> int:
        """Vector width over C/groups (K1; ``filter_vec`` if set)."""
        return _pow2_vec(self.problem.cpg, self.filter_vec or self.vec)

    @property
    def input_lds(self) -> bool:
        """K2 stages each workgroup's input region through LDS."""
        return self.input_block_tiles[0] > 0

    @property
    def prescale_split(self) -> bool:
        """Prescale through the separate per-pixel max pass (K2a)."""
        if self.prescale_mode == "split" or self.input_lds:
            return True
        return _pow2_ceil(self.problem.C // self.vec_c) > 1024

    @property
    def prescale_beta(self) -> float:
        return math.sqrt(PRESCALE_BETA[self.mh] * PRESCALE_BETA[self.mw])

    @property
    def prescale_limit_log2(self) -> int:
        """log2 of the largest |V| the prescale allows.

        ``2**14`` keeps V itself finite in fp16, but M sums ``C/g`` products
        of V and U, and a random-sign sum grows like ``sqrt(C/g)``: with only
        the V bound, M overflowed fp16 at C = 512 in testing. So
        ``ceil(log2(C/g) / 2)`` bits are reserved for the reduction. The
        scale is a power of two, so this costs no precision (elements pushed
        into fp16 subnormals are ~2**-24 below the tile maximum).
        """
        headroom = math.ceil(math.log2(self.problem.cpg) / 2)
        return _PRESCALE_LOG2_LIMIT - headroom

    # -- sizes --

    def buffer_bytes(self) -> dict:
        p = self.problem
        io = 2
        op = 2
        t = self.num_tiles
        return {
            "x": p.N * p.Hi * p.Wi * p.C * io,
            "w": p.K * 9 * p.cpg * io,
            "y": p.N * p.Ho * p.Wo * p.K * io,
            "U": self.n2 * p.K * p.cpg * op,
            "V": t * self.n2 * p.C * op,
            "M": t * self.n2 * p.K * op,
            # e[T], then (split prescale) the per-pixel maxima of x.
            "e": t * 4
            + (p.N * p.Hi * p.Wi * 4 if self.prescale and self.prescale_split else 0),
        }

    def workspace_bytes(self) -> dict:
        """Bytes of each intermediate buffer (U, V, M, e)."""
        b = self.buffer_bytes()
        out = {k: b[k] for k in ("U", "V", "M")}
        out["e"] = b["e"] if self.prescale else 0
        return out

    def total_workspace_bytes(self) -> int:
        return sum(self.workspace_bytes().values())

    def short(self) -> str:
        mt = f"f{self.m}" if self.mw == self.mh else f"f{self.mh}x{self.mw}"
        tags = [self.problem.short(), mt, self.io_dtype]
        if self.operand_dtype != self.io_dtype:
            tags.append(f"op{self.operand_dtype}")
        tags.append(f"v{self.vec}")
        tags.append(f"b{self.block_size}")
        if self.tiles_per_thread > 1:
            tags.append(f"r{self.tiles_per_thread}")
        if self.input_lds:
            tags.append("lds{}x{}".format(*self.input_block_tiles))
        if self.wg_tile_rows > 1:
            tags.append(f"wr{self.wg_tile_rows}")
        if not self.nt_stores:
            tags.append("rt")
        if self.input_lds and self.lds_lanes != 32:
            tags.append(f"ll{self.lds_lanes}")
        if self.prescale:
            tags.append("pss" if self.prescale_split else "ps")
        if self.filter_rows != 1:
            tags.append(f"fr{self.filter_rows}")
        if self.filter_vec:
            tags.append(f"fv{self.filter_vec}")
        return "_".join(tags)

    def kernel_name(self, stage: str) -> str:
        return kernel_name_join(self.name, stage, self.short())


# Smallest fraction of an LDS region's tiles that must be real tiles for K2
# to use the LDS path (dilated sub-grids of a few tiles waste most of a 4x8
# region: a dilated layer with 3x3 tiles per sub-grid runs K2 faster on
# the register path).
_LDS_MIN_FILL = 0.75


def tuned_transform_options(
    m, prescale: bool, problem: Optional[ConvProblem] = None
) -> dict:
    """Transform-kernel knobs measured best on gfx1250 (``WinogradFwdSpec`` kwargs).

    ``m``: the tile, an int or ``(mh, mw)`` (the returned dict holds no tile
    size). Without prescale K2 stages 4 x 8-tile regions through LDS (each
    input pixel read ~1.2x instead of up to (n / m)**2 times), 4 tiles per
    thread, 16 lanes x vec channels per workgroup. With prescale the fused
    register path (one pass over x) beats the split max pass + LDS path. F4
    and larger tiles use 8-byte vectors (16-byte ones raise the VGPR count
    without a gain); F2 keeps the phase-1 width of 8. Given ``problem``, the
    LDS path is skipped when the tile grid of a (dilation) sub-grid fills
    less than ``_LDS_MIN_FILL`` of the regions.
    """
    mh, mw = _split_m(m)
    mw = mw or mh
    big = max(mh, mw) >= 4
    # K1: F2 computes all 4 rows of G g per thread (w read once), F4 two of
    # six.
    out = dict(vec=4 if big else 8, filter_rows=2 if mh >= 4 else mh + 2)
    if not prescale:
        if (mh + 2) * (mw + 2) >= 36:
            # F4 and larger tiles: the register path, one tile per thread.
            # The LDS path is as fast or faster *alone*, but inside the
            # pipeline the kernel after it ran intermittently slower.
            # Root cause not found.
            return out
        bh, bw = 4, 8
        if problem is not None:
            th = ceil_div(ceil_div(problem.Ho, problem.dH), mh)
            tw = ceil_div(ceil_div(problem.Wo, problem.dW), mw)
            fill = th * tw / (ceil_div(th, bh) * bh * ceil_div(tw, bw) * bw)
            if fill < _LDS_MIN_FILL:
                return out
        out.update(input_block_tiles=(bh, bw), tiles_per_thread=4, lds_lanes=16)
    return out


def _split_m(m) -> Tuple[int, int]:
    """``m`` (square tile) or ``(mh, mw)`` -> ``(m, m_w)`` spec arguments."""
    if isinstance(m, (tuple, list)):
        return int(m[0]), int(m[1])
    return int(m), 0


def winograd_gemm_problem(problem: ConvProblem, m) -> ConvProblem:
    """The grouped 1x1 conv K3 computes: ``M = V (x) U`` per (bin, group).

    ``N = T`` tiles, ``Hi = Wi = 1``, ``C = n2 * C``, ``K = n2 * K``,
    ``groups = n2 * g``; group index ``bin * g + gi`` (see the module doc).
    """
    mh, mw = _split_m(m)
    spec = WinogradFwdSpec(problem=problem, m=mh, m_w=mw)
    n2 = spec.n2
    return ConvProblem(
        N=spec.num_tiles,
        Hi=1,
        Wi=1,
        C=n2 * problem.C,
        K=n2 * problem.K,
        Y=1,
        X=1,
        groups=n2 * problem.groups,
    )


def winograd_gemm_flops(problem: ConvProblem, m) -> int:
    """MACs * 2 of the middle GEMM (for the reduction-vs-direct report)."""
    gp = winograd_gemm_problem(problem, m)
    return gp.flops


# ---------------------------------------------------------------------------
# 1-D transforms with shared subexpressions
# ---------------------------------------------------------------------------


class _IROps:
    """Scalar fp32 arithmetic on IR values (the ops the 1-D transforms use)."""

    def __init__(self, b: IRBuilder):
        self.b = b

    def add(self, x, y):
        return self.b.fadd(x, y)

    def sub(self, x, y):
        return self.b.fsub(x, y)

    def fma(self, c: float, x, y):
        """``c * x + y`` for a constant ``c``."""
        return self.b.fma(self.b.const_f32(float(c)), x, y)


def bt_1d(ops, m: int, d: Sequence):
    """``B^T d`` for one length-``n`` column, with shared subexpressions.

    The rows of ``B^T`` (see :func:`toom_cook`) share their pairwise sums and
    differences, so F4 needs 14 operations instead of the 22 of a dot product
    per row, F2 needs 4. Exactly the integer rows of ``B^T``.
    """
    if m == 2:
        return [
            ops.sub(d[2], d[0]),
            ops.add(d[1], d[2]),
            ops.sub(d[2], d[1]),
            ops.sub(d[3], d[1]),
        ]
    if m == 4:
        s12 = ops.add(d[1], d[2])
        s34 = ops.add(d[3], d[4])
        d12 = ops.sub(d[1], d[2])
        d43 = ops.sub(d[4], d[3])
        d31 = ops.sub(d[3], d[1])
        d42 = ops.sub(d[4], d[2])
        return [
            ops.fma(4.0, d[0], ops.fma(-5.0, d[2], d[4])),
            ops.fma(-4.0, s12, s34),
            ops.fma(4.0, d12, d43),
            ops.fma(2.0, d31, d42),
            ops.fma(-2.0, d31, d42),
            ops.fma(4.0, d[1], ops.fma(-5.0, d[3], d[5])),
        ]
    _, _, bt = toom_cook(m)
    return _matvec_1d(ops, bt, d)


def _matvec_1d(ops, mat, d: Sequence):
    """``mat @ d`` for constant rows (no shared subexpressions): every
    non-zero coefficient is one add / sub / fma (exact binary fractions)."""
    out = []
    for row in mat:
        terms = [(float(c), d[k]) for k, c in enumerate(row) if c != 0]
        unit = [t for t in terms if abs(t[0]) == 1.0]
        rest = [t for t in terms if abs(t[0]) != 1.0]
        acc = None
        for c, v in unit:
            if acc is None:
                acc = v if c > 0 else ops.sub(ops.sub(v, v), v)
            else:
                acc = ops.add(acc, v) if c > 0 else ops.sub(acc, v)
        for c, v in rest:
            acc = (
                ops.fma(c, v, acc) if acc is not None else ops.fma(c, v, ops.sub(v, v))
            )
        out.append(acc if acc is not None else ops.sub(d[0], d[0]))
    return out


def at_1d(ops, m: int, d: Sequence):
    """``A^T d`` for one length-``n`` column, with shared subexpressions."""
    if m == 2:
        return [ops.add(ops.add(d[0], d[1]), d[2]), ops.add(ops.sub(d[1], d[2]), d[3])]
    if m == 4:
        s12 = ops.add(d[1], d[2])
        d12 = ops.sub(d[1], d[2])
        s34 = ops.add(d[3], d[4])
        d34 = ops.sub(d[3], d[4])
        return [
            ops.add(ops.add(d[0], s12), s34),
            ops.fma(2.0, d34, d12),
            ops.fma(4.0, s34, s12),
            ops.add(ops.fma(8.0, d34, d12), d[5]),
        ]
    at, _, _ = toom_cook(m)
    return _matvec_1d(ops, at, d)


# ---------------------------------------------------------------------------
# IR helpers
# ---------------------------------------------------------------------------


def _ir_dtype(dtype: str):
    return F16 if dtype == "fp16" else BF16


def _lincomb(b: IRBuilder, terms: Sequence[Tuple[int, Value]]) -> Value:
    """``sum(c * v)`` over the non-zero integer coefficients, in fp32."""
    acc: Optional[Value] = None
    for c, v in terms:
        if c == 0:
            continue
        if acc is None:
            if c == 1:
                acc = v
            elif c == -1:
                acc = b.fneg(v)
            else:
                acc = b.fmul(b.const_f32(float(c)), v)
        elif c == 1:
            acc = b.fadd(acc, v)
        elif c == -1:
            acc = b.fsub(acc, v)
        else:
            acc = b.fma(b.const_f32(float(c)), v, acc)
    return acc if acc is not None else b.const_f32(0.0)


def _load_raw(b: IRBuilder, rsrc: Value, byte_off: Value, dtype: str, n: int) -> Value:
    """Buffer load of ``n`` 16-bit elements, kept packed (OOB -> 0)."""
    ty = _ir_dtype(dtype)
    zero = b.const_i32(0)
    if n == 1:
        return b.buffer_load(rsrc, byte_off, zero, ty)
    return b.buffer_load_vN(rsrc, byte_off, zero, ty, n)


def _raw_to_f32(b: IRBuilder, raw: Value, n: int) -> List[Value]:
    if n == 1:
        return [b.cast_to_f32(raw)]
    return [b.cast_to_f32(b.vec_extract(raw, i)) for i in range(n)]


def _pack_vec_f32(b: IRBuilder, vals: Sequence[Value], dtype: str):
    """Round fp32 values to ``dtype``: a scalar (one value) or a vector."""
    ty = _ir_dtype(dtype)
    casts = [b.cast_f32_to(v, ty) for v in vals]
    return casts[0] if len(casts) == 1 else b.vec_pack(casts, ty)


def _store_packed(
    b: IRBuilder, rsrc: Value, byte_off: Value, packed, dtype: str, cpol: int = 0
):
    """Buffer-store a :func:`_pack_vec_f32` value (OOB dropped).

    ``cpol``: the store's cache-policy immediate (``_CPOL_NT``: streaming).
    """
    zero = b.const_i32(0)
    if not isinstance(packed.type, VectorType):
        if dtype == "fp16":
            b.buffer_store_f16(rsrc, byte_off, zero, packed, cachepolicy=cpol)
        else:
            b.buffer_store_bf16(rsrc, byte_off, zero, packed, cachepolicy=cpol)
        return
    n = packed.type.count
    if dtype == "fp16":
        b.buffer_store_vN_f16(rsrc, byte_off, zero, packed, n // 2, cachepolicy=cpol)
    else:
        b.buffer_store_vN_bf16(rsrc, byte_off, zero, packed, n // 2, cachepolicy=cpol)


def _store_vec_f32(
    b: IRBuilder,
    rsrc: Value,
    byte_off: Value,
    vals: Sequence[Value],
    dtype: str,
) -> None:
    """Round fp32 values to ``dtype`` and buffer-store them (OOB dropped)."""
    _store_packed(b, rsrc, byte_off, _pack_vec_f32(b, vals, dtype), dtype)


def _pow2_f32(b: IRBuilder, e: Value) -> Value:
    """``2.0 ** e`` for an i32 ``e`` in [-126, 127], built from its bits (exact)."""
    bits = b.shl(b.add(e, b.const_i32(127)), b.const_i32(23))
    return b.bitcast(bits, F32)


def _abs_bits_vec(b: IRBuilder, raw: Value, n: int) -> Value:
    """``|raw|`` of 16-bit floats as integers (``<n x i16>`` or one i32).

    For non-negative IEEE values the integer order of the bit patterns is the
    magnitude order, so the max over them is an integer max (packed
    ``v_pk_max_i16``) and needs no conversion to fp32.
    """
    if n == 1:
        return b.land(b.zext(b.bitcast(raw, I16), I32), b.const_i32(0x7FFF))
    bits = b.vec_bitcast(raw, VectorType(I16, n))
    mask = b.vector_splat(b.trunc(b.const_i32(0x7FFF), I16), n)
    return b.vector_and(bits, mask)


def _max_bits(b: IRBuilder, x: Value, y: Value, n: int) -> Value:
    return b.smax(x, y) if n == 1 else b.vector_smax(x, y)


def _hmax_bits(b: IRBuilder, v: Value, n: int) -> Value:
    """Horizontal max of :func:`_abs_bits_vec` values, as an i32 (0..0x7FFF)."""
    if n == 1:
        return v
    acc = None
    for i in range(n):
        e = b.zext(b.vec_extract(v, i), I32)
        acc = e if acc is None else b.smax(acc, e)
    return acc


def _bits_to_f32(b: IRBuilder, bits: Value, dtype: str) -> Value:
    """A 16-bit float pattern (in the low half of an i32) as fp32 (exact)."""
    if dtype == "bf16":
        return b.bitcast(b.shl(bits, b.const_i32(16)), F32)
    return b.cast_to_f32(b.bitcast(b.trunc(bits, I16), F16))


def _prescale_exp(b: IRBuilder, spec: "WinogradFwdSpec", amax: Value) -> Value:
    """``e = clamp(ceil(log2(beta * amax / 2**L)), 0, 100)`` from the fp32 bits."""
    sv = b.fmul(
        amax, b.const_f32(spec.prescale_beta / float(2**spec.prescale_limit_log2))
    )
    bits = b.bitcast(sv, I32)
    expo = b.sub(
        b.land(b.lshr(bits, b.const_i32(23)), b.const_i32(0xFF)), b.const_i32(127)
    )
    frac = b.land(bits, b.const_i32(0x7FFFFF))
    e_val = b.add(
        expo, b.select(b.cmp_ne(frac, b.const_i32(0)), b.const_i32(1), b.const_i32(0))
    )
    return b.smin(b.smax(e_val, b.const_i32(0)), b.const_i32(_PRESCALE_MAX_EXP))


def _oob_sel(b: IRBuilder, ok: Value, off: Value) -> Value:
    return b.select(ok, off, b.const_i32(_OOB))


# ---------------------------------------------------------------------------
# Thread mapping shared by K2 and K4
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RunMap:
    """Per-thread (run of tiles along W, channel vector) assignment.

    A *run* is ``tiles`` consecutive tiles of one tile row (fixed image,
    dilation sub-grid and ``ti``); the last run of a row may be partial.
    ``tpc`` threads cover the run's channel vectors (lanes along channels so a
    wave reads / writes a contiguous channel run), ``runs_per_wg`` runs share
    a workgroup, ``cx`` workgroups split the channel axis.
    """

    tpc: int
    runs_per_wg: int
    cx: int
    num_vecs: int
    block: int
    tiles: int  # tiles per run
    runs_per_row: int
    num_runs: int
    rows: int  # tile rows (all images and dilation sub-grids)
    wg_rows: int  # tile rows per workgroup (a wg_rows x runs block)

    @property
    def wg_runs(self) -> int:
        """Runs per tile row of a workgroup."""
        return self.runs_per_wg // self.wg_rows

    @property
    def run_blocks(self) -> int:
        return ceil_div(self.runs_per_row, self.wg_runs)


def _run_map(
    spec: "WinogradFwdSpec", num_vecs: int, tiles: int, all_c: bool, wg_rows: int = 1
):
    """``all_c``: one workgroup covers all channels of a run (fused prescale).

    ``wg_rows > 1``: a workgroup's runs form a ``wg_rows x (runs_per_wg /
    wg_rows)`` block, so vertically adjacent tiles (which share two input rows)
    run on the same CU at the same time.
    """
    block = spec.block_size
    tpc = _pow2_ceil(num_vecs)
    if all_c:
        if tpc > 1024:
            raise ValueError("fused prescale needs C / vec <= 1024")
        block = max(block, tpc)
    else:
        tpc = min(tpc, block)
    rpr = ceil_div(spec.tiles_w, tiles)
    rows = spec.num_tiles // spec.tiles_w
    rpw = block // tpc
    wg_rows = max(1, min(wg_rows, rpw))
    while rpw % wg_rows:
        wg_rows -= 1
    return _RunMap(
        tpc=tpc,
        runs_per_wg=rpw,
        cx=ceil_div(num_vecs, tpc),
        num_vecs=num_vecs,
        block=block,
        tiles=tiles,
        runs_per_row=rpr,
        num_runs=rows * rpr,
        rows=rows,
        wg_rows=wg_rows,
    )


def _run_map_grid(rm: _RunMap) -> Tuple[int, int, int]:
    if rm.wg_rows > 1:
        gx = ceil_div(rm.rows, rm.wg_rows) * rm.run_blocks * rm.cx
    else:
        gx = ceil_div(rm.num_runs, rm.runs_per_wg) * rm.cx
    if gx <= 0:
        raise ValueError("empty grid")
    return (gx, 1, 1)


@dataclass(frozen=True)
class _RunCoords:
    ok: Value  # run index in range
    cv: Value  # channel vector
    c_ok: Value
    n: Value
    a: Value  # dilation sub-grid row
    bsub: Value  # dilation sub-grid col
    ti: Value
    tj0: Value  # first tile column of the run
    t0: Value  # first tile index of the run


def _run_coords(b: IRBuilder, spec: "WinogradFwdSpec", rm: _RunMap) -> _RunCoords:
    p = spec.problem
    tid = b.thread_id_x()
    bid = b.block_id_x()
    if rm.tpc > 1:
        lc = b.mod(tid, b.const_i32(rm.tpc))
        lr = b.div(tid, b.const_i32(rm.tpc))
    else:
        lc, lr = b.const_i32(0), tid
    if rm.cx > 1:
        cxi = b.mod(bid, b.const_i32(rm.cx))
        rb = b.div(bid, b.const_i32(rm.cx))
        cv = b.add(b.mul(cxi, b.const_i32(rm.tpc)), lc)
    else:
        rb, cv = bid, lc
    c_ok = b.cmp_lt(cv, b.const_i32(rm.num_vecs))
    if rm.wg_rows > 1:
        wr = rm.wg_runs
        rblk = b.mod(rb, b.const_i32(rm.run_blocks))
        hblk = b.div(rb, b.const_i32(rm.run_blocks))
        dr = b.div(lr, b.const_i32(wr)) if wr > 1 else lr
        dj = b.mod(lr, b.const_i32(wr)) if wr > 1 else b.const_i32(0)
        row = b.add(b.mul(hblk, b.const_i32(rm.wg_rows)), dr)
        jr = b.add(b.mul(rblk, b.const_i32(wr)), dj)
        ok = b.land(
            b.cmp_lt(row, b.const_i32(rm.rows)),
            b.cmp_lt(jr, b.const_i32(rm.runs_per_row)),
        )
        tj0 = b.mul(jr, b.const_i32(rm.tiles)) if rm.tiles > 1 else jr
    elif rm.runs_per_row > 1:
        rho = b.add(b.mul(rb, b.const_i32(rm.runs_per_wg)), lr)
        ok = b.cmp_lt(rho, b.const_i32(rm.num_runs))
        row = b.div(rho, b.const_i32(rm.runs_per_row))
        jr = b.mod(rho, b.const_i32(rm.runs_per_row))
        tj0 = b.mul(jr, b.const_i32(rm.tiles)) if rm.tiles > 1 else jr
    else:
        rho = b.add(b.mul(rb, b.const_i32(rm.runs_per_wg)), lr)
        ok = b.cmp_lt(rho, b.const_i32(rm.num_runs))
        row, tj0 = rho, b.const_i32(0)
    t0 = b.add(b.mul(row, b.const_i32(spec.tiles_w)), tj0)
    r = row
    if spec.tiles_h > 1:
        ti = b.mod(r, b.const_i32(spec.tiles_h))
        r = b.div(r, b.const_i32(spec.tiles_h))
    else:
        ti = b.const_i32(0)
    if p.dW > 1:
        bsub = b.mod(r, b.const_i32(p.dW))
        r = b.div(r, b.const_i32(p.dW))
    else:
        bsub = b.const_i32(0)
    if p.dH > 1:
        a = b.mod(r, b.const_i32(p.dH))
        r = b.div(r, b.const_i32(p.dH))
    else:
        a = b.const_i32(0)
    return _RunCoords(ok, cv, c_ok, r, a, bsub, ti, tj0, t0)


def _wave_max_bits(b: IRBuilder, v: Value, lanes: int) -> Value:
    s = 1
    while s < lanes:
        v = b.smax(v, b.warp_shuffle_xor(v, s))
        s *= 2
    return v


def _group_max_bits(
    b: IRBuilder, vals: List[Value], tpc: int, block: int, name: str
) -> List[Value]:
    """Max of each i32 in ``vals`` over the ``tpc`` threads sharing a run.

    xor shuffles within a 32-lane row, then (``tpc > 32``) LDS across rows.
    """
    vals = [_wave_max_bits(b, v, min(tpc, 32)) for v in vals]
    if tpc <= 32:
        return vals
    rows = block // 32
    per = tpc // 32
    nv = len(vals)
    red = b.smem_alloc_f32([rows * nv], name_hint=name)
    tid = b.thread_id_x()
    row_id = b.div(tid, b.const_i32(32))
    with b.scf_if(b.cmp_eq(b.mod(tid, b.const_i32(32)), b.const_i32(0))):
        for i, v in enumerate(vals):
            idx = b.add(b.mul(row_id, b.const_i32(nv)), b.const_i32(i))
            b.smem_store_vN_f32(red, [idx], b.vec_pack([b.bitcast(v, F32)], F32), 1)
    b.sync()
    first = b.mul(b.div(tid, b.const_i32(tpc)), b.const_i32(per * nv))
    out = []
    for i in range(nv):
        acc = None
        for r in range(per):
            idx = b.add(first, b.const_i32(r * nv + i))
            x = b.bitcast(b.vec_extract(b.smem_load_vN_f32(red, idx, n=1), 0), I32)
            acc = x if acc is None else b.smax(acc, x)
        out.append(acc)
    return out


# ---------------------------------------------------------------------------
# K1: filter transform
# ---------------------------------------------------------------------------

_FILTER_BLOCK = 128


def filter_transform_grid(spec: WinogradFwdSpec) -> Tuple[int, int, int]:
    p = spec.problem
    items = p.K * (p.cpg // spec.vec_cg)
    return (ceil_div(items, _FILTER_BLOCK), spec.nh // spec.filter_rows, 1)


def filter_transform_block(spec: WinogradFwdSpec) -> int:  # noqa: ARG001
    return _FILTER_BLOCK


def filter_transform_signature(spec: WinogradFwdSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("w", spec.io_dtype)
        .ptr("u", spec.operand_dtype)
        .scalar("w_bytes", "i32")
        .scalar("u_bytes", "i32")
        .build()
    )


def build_winograd_filter_transform(
    spec: WinogradFwdSpec, *, arch: str = "gfx1250"  # noqa: ARG001
) -> KernelDef:
    """K1: ``w[k][ky][kx][c] -> U[bin][k][c] = (G g G^T)[bin]``.

    One thread per (output channel, ``vec_cg`` input channels, row ``i`` of
    ``G g``); ``i`` is the grid's y index, so its ``G`` row is selected with
    scalar ops. Nine vector loads, ``n`` vector stores; lanes run along
    ``(k, c)``, so each store of a wave is one contiguous run of ``U[bin]``.
    """
    p = spec.problem
    vec = spec.vec_cg
    cg = p.cpg
    cgv = cg // vec
    items = p.K * cgv
    n, nw = spec.nh, spec.nw
    gint, den = filter_matrix_int(spec.mh)
    gint_w, den_w = filter_matrix_int(spec.mw)
    scale = 1.0 / float(den * den_w)  # power of two for F2; one rounding for F4

    b = _new_builder(spec.kernel_name("filter"))
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
    w_rsrc = b.buffer_rsrc(w, w_bytes)
    u_rsrc = b.buffer_rsrc(u, u_bytes)

    gid = b.add(b.mul(b.block_id_x(), b.const_i32(_FILTER_BLOCK)), b.thread_id_x())
    row_y = b.block_id_y()
    ok = b.cmp_lt(gid, b.const_i32(items))
    k = b.div(gid, b.const_i32(cgv))
    c = b.mul(b.mod(gid, b.const_i32(cgv)), b.const_i32(vec))
    # w[k][ky][kx][c]: element (k * 9 + tap) * cg + c
    wbase = _oob_sel(
        b, ok, b.mul(b.add(b.mul(k, b.const_i32(9 * cg)), c), b.const_i32(2))
    )
    taps = []
    for ky in range(3):
        row = []
        for kx in range(3):
            off = b.add(wbase, b.const_i32((ky * 3 + kx) * cg * 2))
            row.append(
                _raw_to_f32(b, _load_raw(b, w_rsrc, off, spec.io_dtype, vec), vec)
            )
        taps.append(row)
    sc = b.const_f32(scale)
    nr = spec.filter_rows
    for rr in range(nr):
        if nr == n:
            row_i = b.const_i32(rr)
            coef = [b.const_f32(float(gint[rr][ky])) for ky in range(3)]
        else:
            row_i = b.add(b.mul(row_y, b.const_i32(nr)), b.const_i32(rr))
            # G[i][ky] for the runtime row i (uniform): a select chain.
            coef = []
            for ky in range(3):
                cval = b.const_f32(float(gint[0][ky]))
                for i in range(1, n):
                    cval = b.select(
                        b.cmp_eq(row_i, b.const_i32(i)),
                        b.const_f32(float(gint[i][ky])),
                        cval,
                    )
                coef.append(cval)
        # tmp[kx][ch] = sum_ky G[i][ky] * g[ky][kx][ch]
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
        # U[i * n + j][k][c]
        ubase = _oob_sel(
            b,
            ok,
            b.mul(
                b.add(
                    b.mul(
                        b.add(b.mul(row_i, b.const_i32(nw * p.K)), k), b.const_i32(cg)
                    ),
                    c,
                ),
                b.const_i32(2),
            ),
        )
        for j in range(nw):
            vals = [
                b.fmul(
                    _lincomb(b, [(gint_w[j][kx], tmp[kx][ch]) for kx in range(3)]), sc
                )
                for ch in range(vec)
            ]
            off = b.add(ubase, b.const_i32(j * p.K * cg * 2))
            _store_vec_f32(b, u_rsrc, off, vals, spec.operand_dtype)
    return b.kernel


# ---------------------------------------------------------------------------
# K2: input transform (+ prescale)
# ---------------------------------------------------------------------------


def _input_run_map(spec: WinogradFwdSpec) -> _RunMap:
    p = spec.problem
    return _run_map(
        spec,
        p.C // spec.vec_c,
        spec.tiles_per_thread,
        all_c=spec.prescale and not spec.prescale_split,
        wg_rows=spec.wg_tile_rows,
    )


def input_transform_grid(spec: WinogradFwdSpec) -> Tuple[int, int, int]:
    if spec.input_lds:
        return _lds_map(spec).grid
    return _run_map_grid(_input_run_map(spec))


def input_transform_block(spec: WinogradFwdSpec) -> int:
    if spec.input_lds:
        return _lds_map(spec).block
    return _input_run_map(spec).block


def input_transform_signature(spec: WinogradFwdSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("x", spec.io_dtype)
        .ptr("v", spec.operand_dtype)
        .ptr("e", "i32")
        .scalar("x_bytes", "i32")
        .scalar("v_bytes", "i32")
        .scalar("e_bytes", "i32")
        .build()
    )


def build_winograd_input_transform(
    spec: WinogradFwdSpec, *, arch: str = "gfx1250"  # noqa: ARG001
) -> KernelDef:
    """K2: ``x NHWC -> V[t][bin][c] = (B^T d B)[bin]`` (and ``e[t]``).

    One thread per (run of ``tiles_per_thread`` tiles along W, ``vec_c``
    channels). The run's ``n x (m * tiles + 2)`` input columns are loaded
    once: adjacent tiles share ``n - m = 2`` columns, and so do their column
    transforms ``B^T d[:, v]``, which are computed once per column and reused.
    Each load / store offset is the sum of a row part and a column part that
    are forced to :data:`_OOB` for padding, dilation holes and tail threads,
    so out-of-range pixels read as zero without a branch.

    Prescale: ``e`` of each tile from the max of ``|x|`` over its patch and all
    channels; fused (the workgroup covers all C of its runs, the max is
    reduced through shuffles / LDS while the patch stays in registers) or
    split (per-pixel maxima from :func:`build_winograd_prescale`).
    """
    if spec.input_lds:
        return _build_input_lds(spec)
    p = spec.problem
    n, m = spec.nh, spec.mh  # rows
    nw, mw = spec.nw, spec.mw  # columns
    vec = spec.vec_c
    rm = _input_run_map(spec)
    R = rm.tiles
    ncols = mw * R + (nw - mw)
    T = spec.num_tiles
    ps = spec.prescale
    split = ps and spec.prescale_split

    b = _new_builder(spec.kernel_name("input"))
    b.kernel.attrs["max_workgroup_size"] = rm.block
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
    x_rsrc = b.buffer_rsrc(x, x_bytes)
    v_rsrc = b.buffer_rsrc(v, v_bytes)
    e_rsrc = b.buffer_rsrc(e_ptr, e_bytes) if ps else None

    rc = _run_coords(b, spec, rm)
    c = b.mul(rc.cv, b.const_i32(vec))
    h0 = b.add(b.sub(rc.a, b.const_i32(p.pH)), b.mul(rc.ti, b.const_i32(p.dH * m)))
    w0 = b.add(b.sub(rc.bsub, b.const_i32(p.pW)), b.mul(rc.tj0, b.const_i32(p.dW * mw)))
    img_row0 = b.mul(rc.n, b.const_i32(p.Hi))
    # Row part (with the channel offset) and column part of each patch offset.
    row_ok, row_pix = [], []
    for uu in range(n):
        hi = b.add(h0, b.const_i32(uu * p.dH))
        row_ok.append(b.land(rc.ok, b.cmp_ult(hi, b.const_i32(p.Hi))))
        row_pix.append(b.mul(b.add(img_row0, hi), b.const_i32(p.Wi)))
    col_ok, col_w = [], []
    for vv in range(ncols):
        wi = b.add(w0, b.const_i32(vv * p.dW))
        col_ok.append(b.cmp_ult(wi, b.const_i32(p.Wi)))
        col_w.append(wi)
    xrow = [
        _oob_sel(
            b,
            b.land(row_ok[uu], rc.c_ok),
            b.mul(b.add(b.mul(row_pix[uu], b.const_i32(p.C)), c), b.const_i32(2)),
        )
        for uu in range(n)
    ]
    xcol = [
        _oob_sel(b, col_ok[vv], b.mul(col_w[vv], b.const_i32(p.C * 2)))
        for vv in range(ncols)
    ]
    raw = [
        [
            _load_raw(b, x_rsrc, b.add(xrow[uu], xcol[vv]), spec.io_dtype, vec)
            for vv in range(ncols)
        ]
        for uu in range(n)
    ]
    tile_ok = [
        (
            b.land(
                b.land(rc.ok, rc.c_ok),
                b.cmp_lt(b.add(rc.tj0, b.const_i32(r)), b.const_i32(spec.tiles_w)),
            )
            if spec.tiles_w % R
            else b.land(rc.ok, rc.c_ok)
        )
        for r in range(R)
    ]

    scales: List[Optional[Value]] = [None] * R
    if ps:
        if split:
            # Per-pixel maxima (bit patterns) after e[T] in the e buffer.
            prow = [
                _oob_sel(
                    b,
                    row_ok[uu],
                    b.mul(b.add(row_pix[uu], b.const_i32(T)), b.const_i32(4)),
                )
                for uu in range(n)
            ]
            pcol = [
                _oob_sel(b, col_ok[vv], b.mul(col_w[vv], b.const_i32(4)))
                for vv in range(ncols)
            ]
            colmax = []
            for vv in range(ncols):
                acc = None
                for uu in range(n):
                    pm = b.buffer_load(
                        e_rsrc, b.add(prow[uu], pcol[vv]), b.const_i32(0), I32
                    )
                    acc = pm if acc is None else b.smax(acc, pm)
                colmax.append(acc)
            tmax = []
            for r in range(R):
                acc = colmax[mw * r]
                for vv in range(1, nw):
                    acc = b.smax(acc, colmax[mw * r + vv])
                tmax.append(acc)
        else:
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
            tmax = _group_max_bits(b, tmax, rm.tpc, rm.block, "wino_amax")
        lane0 = b.cmp_eq(rc.cv, b.const_i32(0))
        for r in range(R):
            e_r = _prescale_exp(b, spec, _bits_to_f32(b, tmax[r], spec.io_dtype))
            t_r = b.add(rc.t0, b.const_i32(r))
            ok_e = b.land(lane0, tile_ok[r])
            b.buffer_store_f32(
                e_rsrc,
                _oob_sel(b, ok_e, b.mul(t_r, b.const_i32(4))),
                b.const_i32(0),
                b.bitcast(e_r, F32),
            )
            scales[r] = _pow2_f32(b, b.sub(b.const_i32(0), e_r))

    vbase0 = b.add(
        b.mul(rc.t0, b.const_i32(spec.n2 * p.C * 2)), b.mul(c, b.const_i32(2))
    )
    _input_tiles_transform(b, spec, raw, R, tile_ok, scales, vbase0, v_rsrc, vec)
    return b.kernel


def _input_tiles_transform(b, spec, raw, R, tile_ok, scales, vbase0, v_rsrc, vec):
    """``V = B^T d B`` of ``R`` tiles adjacent along W and their stores.

    ``raw[u][v]`` holds the packed input columns of the run (tile ``r`` uses
    columns ``m*r .. m*r+n-1``). Column transforms are computed once per
    column and shared by the two tiles that use it; ``V`` of tile ``r`` goes to
    ``vbase0 + r * n2 * C * 2`` (bytes), forced out of range unless
    ``tile_ok[r]``.
    """
    p = spec.problem
    n, m = spec.nh, spec.mh
    nw, mw = spec.nw, spec.mw
    ops = _IROps(b)
    vstride = spec.n2 * p.C * 2
    colt = {}
    pending = []
    b.sched_barrier(0)

    def col_transform(vv):
        if vv not in colt:
            data = [_raw_to_f32(b, raw[uu][vv], vec) for uu in range(n)]
            colt[vv] = [
                bt_1d(ops, m, [data[uu][ch] for uu in range(n)]) for ch in range(vec)
            ]  # [ch][i]
        return colt[vv]

    for r in range(R):
        cols = [col_transform(mw * r + vv) for vv in range(nw)]
        vbase = _oob_sel(b, tile_ok[r], b.add(vbase0, b.const_i32(r * vstride)))
        for i in range(n):
            outs = [
                bt_1d(ops, mw, [cols[vv][ch][i] for vv in range(nw)])
                for ch in range(vec)
            ]
            for j in range(nw):
                vals = [outs[ch][j] for ch in range(vec)]
                if scales[r] is not None:
                    vals = [b.fmul(x_, scales[r]) for x_ in vals]
                off = b.add(vbase, b.const_i32((i * nw + j) * p.C * 2))
                pending.append((off, _pack_vec_f32(b, vals, spec.operand_dtype)))
    # All stores last, after a scheduling barrier: a store's data VGPRs must
    # not be reused while the store is in flight (``s_wait_xcnt``), which
    # stalls the wave when the stores are interleaved with the arithmetic.
    b.sched_barrier(0)
    cpol = _CPOL_NT if spec.nt_stores else 0
    for off, packed in pending:
        _store_packed(b, v_rsrc, off, packed, spec.operand_dtype, cpol)


@dataclass(frozen=True)
class _LdsMap:
    """K2 with LDS staging: a workgroup = ``bh x bw`` tiles x ``cc`` channels."""

    bh: int
    bw: int
    tpc: int  # lanes along channels (compute: vec_c channels each)
    cc: int  # channels per workgroup
    cx: int  # workgroups along C
    lv: int  # channels per global load
    rh: int  # region rows / cols (pixels)
    rw: int
    nbh: int
    nbw: int
    block: int
    loads: int  # global loads per thread
    grid: Tuple[int, int, int]
    tpt: int  # tiles per thread (sequential)


def _lds_map(spec: WinogradFwdSpec) -> _LdsMap:
    p = spec.problem
    bh, bw = spec.input_block_tiles
    vec = spec.vec_c
    nv = p.C // vec
    tpc = min(_pow2_ceil(nv), spec.lds_lanes)
    cc = tpc * vec
    cx = ceil_div(nv, tpc)
    lv = min(_pow2_vec(p.C, 8), cc)
    rh = spec.mh * bh + spec.nh - spec.mh
    rw = spec.mw * bw + spec.nw - spec.mw
    nbh = ceil_div(spec.tiles_h, bh)
    nbw = ceil_div(spec.tiles_w, bw)
    tpt = spec.tiles_per_thread
    if (bh * bw) % tpt:
        raise ValueError("tiles_per_thread must divide the LDS block's tile count")
    block = bh * bw * tpc // tpt
    if block > 1024:
        raise ValueError("input_block_tiles too large for one workgroup")
    pieces = rh * rw * (cc // lv)
    gx = p.N * p.dH * p.dW * nbh * nbw * cx
    if gx <= 0:
        raise ValueError("empty grid")
    return _LdsMap(
        bh,
        bw,
        tpc,
        cc,
        cx,
        lv,
        rh,
        rw,
        nbh,
        nbw,
        block,
        ceil_div(pieces, block),
        (gx, 1, 1),
        tpt,
    )


def _build_input_lds(spec: WinogradFwdSpec) -> KernelDef:
    """K2, LDS-staged: the workgroup loads its region once, then transforms.

    The region of ``bh x bw`` tiles is ``(m*bh + 2) x (m*bw + 2)`` pixels:
    every input pixel is read from memory once per workgroup (vs up to 4x
    without LDS, as neighbouring patches overlap by two rows and columns).
    Loads are ``lv``-channel vectors spread over all threads; then each
    thread transforms one tile x ``vec_c`` channels from LDS. Prescale (split
    only): the region's per-pixel maxima are staged too.
    """
    p = spec.problem
    n, m = spec.nh, spec.mh
    nw, mw = spec.nw, spec.mw
    vec = spec.vec_c
    lm = _lds_map(spec)
    T = spec.num_tiles
    ps = spec.prescale
    dt = _ir_dtype(spec.io_dtype)

    b = _new_builder(spec.kernel_name("input"))
    b.kernel.attrs["max_workgroup_size"] = lm.block
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
    x_rsrc = b.buffer_rsrc(x, x_bytes)
    v_rsrc = b.buffer_rsrc(v, v_bytes)
    e_rsrc = b.buffer_rsrc(e_ptr, e_bytes) if ps else None

    tid = b.thread_id_x()
    bid = b.block_id_x()
    cxi = b.mod(bid, b.const_i32(lm.cx)) if lm.cx > 1 else b.const_i32(0)
    r = b.div(bid, b.const_i32(lm.cx)) if lm.cx > 1 else bid
    bwi = b.mod(r, b.const_i32(lm.nbw))
    r = b.div(r, b.const_i32(lm.nbw))
    bhi = b.mod(r, b.const_i32(lm.nbh))
    sub = b.div(r, b.const_i32(lm.nbh))  # (n * dH + a) * dW + bsub
    bsub = b.mod(sub, b.const_i32(p.dW)) if p.dW > 1 else b.const_i32(0)
    r = b.div(sub, b.const_i32(p.dW)) if p.dW > 1 else sub
    a = b.mod(r, b.const_i32(p.dH)) if p.dH > 1 else b.const_i32(0)
    img = b.div(r, b.const_i32(p.dH)) if p.dH > 1 else r
    ti0 = b.mul(bhi, b.const_i32(lm.bh))
    tj0 = b.mul(bwi, b.const_i32(lm.bw))
    h0 = b.add(b.sub(a, b.const_i32(p.pH)), b.mul(ti0, b.const_i32(p.dH * m)))
    w0 = b.add(b.sub(bsub, b.const_i32(p.pW)), b.mul(tj0, b.const_i32(p.dW * mw)))
    img_row0 = b.mul(img, b.const_i32(p.Hi))
    c0 = b.mul(cxi, b.const_i32(lm.cc))

    # ---- load the region (lv-channel pieces) into LDS ----
    ppx = lm.cc // lm.lv
    npx = lm.rh * lm.rw
    lds = b.smem_alloc(dt, [lm.loads * lm.block * lm.lv], name_hint="wino_xin")
    for k in range(lm.loads):
        q = b.add(tid, b.const_i32(k * lm.block))
        px = b.div(q, b.const_i32(ppx)) if ppx > 1 else q
        pc = b.mod(q, b.const_i32(ppx)) if ppx > 1 else b.const_i32(0)
        rr = b.div(px, b.const_i32(lm.rw))
        cc = b.mod(px, b.const_i32(lm.rw))
        hi = b.add(h0, b.mul(rr, b.const_i32(p.dH)))
        wi = b.add(w0, b.mul(cc, b.const_i32(p.dW)))
        ch = b.add(c0, b.mul(pc, b.const_i32(lm.lv)))
        ok = b.land(b.cmp_ult(hi, b.const_i32(p.Hi)), b.cmp_ult(wi, b.const_i32(p.Wi)))
        if (k + 1) * lm.block > npx * ppx:
            ok = b.land(ok, b.cmp_lt(px, b.const_i32(npx)))
        if lm.cx * lm.cc > p.C:
            ok = b.land(ok, b.cmp_lt(ch, b.const_i32(p.C)))
        pix = b.add(b.mul(b.add(img_row0, hi), b.const_i32(p.Wi)), wi)
        off = _oob_sel(
            b, ok, b.mul(b.add(b.mul(pix, b.const_i32(p.C)), ch), b.const_i32(2))
        )
        val = _load_raw(b, x_rsrc, off, spec.io_dtype, lm.lv)
        b.smem_store_vN(lds, [b.mul(q, b.const_i32(lm.lv))], val, lm.lv)
    pm_lds = None
    if ps:
        # Per-pixel maxima of the region (written by K2a after e[T]).
        pm_lds = b.smem_alloc(
            I32, [ceil_div(npx, lm.block) * lm.block], name_hint="wino_pmax"
        )
        for k in range(ceil_div(npx, lm.block)):
            px = b.add(tid, b.const_i32(k * lm.block))
            rr = b.div(px, b.const_i32(lm.rw))
            cc = b.mod(px, b.const_i32(lm.rw))
            hi = b.add(h0, b.mul(rr, b.const_i32(p.dH)))
            wi = b.add(w0, b.mul(cc, b.const_i32(p.dW)))
            ok = b.land(
                b.cmp_ult(hi, b.const_i32(p.Hi)), b.cmp_ult(wi, b.const_i32(p.Wi))
            )
            if (k + 1) * lm.block > npx:
                ok = b.land(ok, b.cmp_lt(px, b.const_i32(npx)))
            pix = b.add(b.mul(b.add(img_row0, hi), b.const_i32(p.Wi)), wi)
            off = _oob_sel(b, ok, b.mul(b.add(pix, b.const_i32(T)), b.const_i32(4)))
            b.smem_store_vN(
                pm_lds, [px], b.buffer_load(e_rsrc, off, b.const_i32(0), I32), 1
            )
    b.sync()

    # ---- transform: ``tiles_per_thread`` tiles x vec channels per thread ----
    cl = b.mod(tid, b.const_i32(lm.tpc)) if lm.tpc > 1 else b.const_i32(0)
    tb0 = b.div(tid, b.const_i32(lm.tpc)) if lm.tpc > 1 else tid
    cv = b.add(b.mul(cxi, b.const_i32(lm.tpc)), cl)
    c_ok = b.cmp_lt(cv, b.const_i32(p.C // vec))
    for rep in range(lm.tpt):
        tb = b.add(tb0, b.const_i32(rep * (lm.block // lm.tpc))) if rep else tb0
        bi = b.div(tb, b.const_i32(lm.bw)) if lm.bw > 1 else tb
        bj = b.mod(tb, b.const_i32(lm.bw)) if lm.bw > 1 else b.const_i32(0)
        ti = b.add(ti0, bi)
        tj = b.add(tj0, bj)
        tile_ok = b.land(
            b.land(
                b.cmp_lt(ti, b.const_i32(spec.tiles_h)),
                b.cmp_lt(tj, b.const_i32(spec.tiles_w)),
            ),
            c_ok,
        )
        t = b.add(
            b.mul(
                b.add(b.mul(sub, b.const_i32(spec.tiles_h)), ti),
                b.const_i32(spec.tiles_w),
            ),
            tj,
        )
        pbase = b.add(b.mul(bi, b.const_i32(m * lm.rw)), b.mul(bj, b.const_i32(mw)))
        raw = []
        for uu in range(n):
            row = []
            for vv in range(nw):
                idx = b.add(
                    b.mul(
                        b.add(pbase, b.const_i32(uu * lm.rw + vv)), b.const_i32(lm.cc)
                    ),
                    b.mul(cl, b.const_i32(vec)),
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
                            pm_lds,
                            b.add(pbase, b.const_i32(uu * lm.rw + vv)),
                            dtype=I32,
                            n=1,
                        ),
                        0,
                    )
                    acc = pm if acc is None else b.smax(acc, pm)
            e_t = _prescale_exp(b, spec, _bits_to_f32(b, acc, spec.io_dtype))
            ok_e = b.land(tile_ok, b.cmp_eq(cv, b.const_i32(0)))
            b.buffer_store_f32(
                e_rsrc,
                _oob_sel(b, ok_e, b.mul(t, b.const_i32(4))),
                b.const_i32(0),
                b.bitcast(e_t, F32),
            )
            scales = [_pow2_f32(b, b.sub(b.const_i32(0), e_t))]
        vbase0 = b.add(
            b.mul(t, b.const_i32(spec.n2 * p.C * 2)),
            b.mul(b.mul(cv, b.const_i32(vec)), b.const_i32(2)),
        )
        _input_tiles_transform(b, spec, raw, 1, [tile_ok], scales, vbase0, v_rsrc, vec)
    return b.kernel


# ---------------------------------------------------------------------------
# K2a: per-pixel max |x| over C (split prescale)
# ---------------------------------------------------------------------------

_PRESCALE_BLOCK = 256


def _prescale_lanes(spec: WinogradFwdSpec) -> Tuple[int, int]:
    """``(tpc, loads)``: lanes per pixel (<= 32, one wave row) and vectors each."""
    nv = spec.problem.C // spec.vec_c
    tpc = min(_pow2_ceil(nv), 32)
    return tpc, ceil_div(nv, tpc)


def prescale_grid(spec: WinogradFwdSpec) -> Tuple[int, int, int]:
    p = spec.problem
    tpc, _ = _prescale_lanes(spec)
    gx = ceil_div(p.N * p.Hi * p.Wi, _PRESCALE_BLOCK // tpc)
    if gx <= 0:
        raise ValueError("empty grid")
    return (gx, 1, 1)


def prescale_block(spec: WinogradFwdSpec) -> int:  # noqa: ARG001
    return _PRESCALE_BLOCK


def prescale_signature(spec: WinogradFwdSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("x", spec.io_dtype)
        .ptr("e", "i32")
        .scalar("x_bytes", "i32")
        .scalar("e_bytes", "i32")
        .build()
    )


def build_winograd_prescale(
    spec: WinogradFwdSpec, *, arch: str = "gfx1250"  # noqa: ARG001
) -> KernelDef:
    """K2a: ``pmax[pixel] = max_c |x[pixel][c]|`` (as the 16-bit pattern).

    Written after ``e[T]`` in the ``e`` workspace; K2 takes each tile's
    ``e`` from the maxima of its patch pixels, which equals the max over the
    patch and all channels.
    """
    p = spec.problem
    vec = spec.vec_c
    nv = p.C // vec
    tpc, loads = _prescale_lanes(spec)
    npix = p.N * p.Hi * p.Wi
    T = spec.num_tiles

    b = _new_builder(spec.kernel_name("pmax"))
    b.kernel.attrs["max_workgroup_size"] = _PRESCALE_BLOCK
    x = b.param(
        "x", PtrType(_ir_dtype(spec.io_dtype), "global"), noalias=True, readonly=True
    )
    e_ptr = b.param("e", PtrType(I32, "global"), noalias=True, writeonly=True)
    x_bytes = b.param("x_bytes", I32)
    e_bytes = b.param("e_bytes", I32)
    x_rsrc = b.buffer_rsrc(x, x_bytes)
    e_rsrc = b.buffer_rsrc(e_ptr, e_bytes)
    tid = b.thread_id_x()
    lc = b.mod(tid, b.const_i32(tpc)) if tpc > 1 else b.const_i32(0)
    lp = b.div(tid, b.const_i32(tpc)) if tpc > 1 else tid
    pix = b.add(b.mul(b.block_id_x(), b.const_i32(_PRESCALE_BLOCK // tpc)), lp)
    pok = b.cmp_lt(pix, b.const_i32(npix))
    base = b.mul(
        b.add(b.mul(pix, b.const_i32(p.C)), b.mul(lc, b.const_i32(vec))), b.const_i32(2)
    )
    acc = None
    for q in range(loads):
        ok = pok
        if (q + 1) * tpc > nv:
            ok = b.land(ok, b.cmp_lt(b.add(lc, b.const_i32(q * tpc)), b.const_i32(nv)))
        off = _oob_sel(b, ok, b.add(base, b.const_i32(q * tpc * vec * 2)))
        a_ = _abs_bits_vec(b, _load_raw(b, x_rsrc, off, spec.io_dtype, vec), vec)
        acc = a_ if acc is None else _max_bits(b, acc, a_, vec)
    mx = _wave_max_bits(b, _hmax_bits(b, acc, vec), tpc)
    ok_s = b.land(pok, b.cmp_eq(lc, b.const_i32(0)))
    b.buffer_store_f32(
        e_rsrc,
        _oob_sel(b, ok_s, b.mul(b.add(pix, b.const_i32(T)), b.const_i32(4))),
        b.const_i32(0),
        b.bitcast(mx, F32),
    )
    return b.kernel


# ---------------------------------------------------------------------------
# K4: output transform
# ---------------------------------------------------------------------------


def _output_run_map(spec: WinogradFwdSpec) -> _RunMap:
    p = spec.problem
    return _run_map(spec, p.K // spec.vec_k, 1, False)


def output_transform_grid(spec: WinogradFwdSpec) -> Tuple[int, int, int]:
    return _run_map_grid(_output_run_map(spec))


def output_transform_block(spec: WinogradFwdSpec) -> int:
    return _output_run_map(spec).block


def output_transform_signature(spec: WinogradFwdSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("mm", spec.operand_dtype)
        .ptr("e", "i32")
        .ptr("y", spec.io_dtype)
        .scalar("m_bytes", "i32")
        .scalar("y_bytes", "i32")
        .build()
    )


def build_winograd_output_transform(
    spec: WinogradFwdSpec, *, arch: str = "gfx1250"  # noqa: ARG001
) -> KernelDef:
    """K4: ``M[t][bin][k] -> y = A^T M A`` (``x 2**e[t]``), cropped at the borders.

    One thread per (tile, ``vec_k`` channels): ``n2`` vector loads, the
    separable transform with shared subexpressions (column pass first, so
    only ``m x n`` intermediates per channel are live), ``m2`` vector stores.
    """
    p = spec.problem
    n, m = spec.nh, spec.mh
    nw, mw = spec.nw, spec.mw
    vec = spec.vec_k
    rm = _output_run_map(spec)

    b = _new_builder(spec.kernel_name("output"))
    b.kernel.attrs["max_workgroup_size"] = rm.block
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
    m_rsrc = b.buffer_rsrc(mm, m_bytes)
    y_rsrc = b.buffer_rsrc(y, y_bytes)

    rc = _run_coords(b, spec, rm)
    t = rc.t0
    ok = b.land(rc.ok, rc.c_ok)
    k = b.mul(rc.cv, b.const_i32(vec))
    mbase = _oob_sel(
        b, ok, b.mul(b.add(b.mul(t, b.const_i32(spec.n2 * p.K)), k), b.const_i32(2))
    )
    raw = [
        [
            _load_raw(
                b,
                m_rsrc,
                b.add(mbase, b.const_i32((uu * nw + vv) * p.K * 2)),
                spec.operand_dtype,
                vec,
            )
            for vv in range(nw)
        ]
        for uu in range(n)
    ]
    scale = None
    if spec.prescale:
        t_safe = b.select(ok, t, b.const_i32(0))
        scale = _pow2_f32(b, b.global_load_i32(e_ptr, t_safe))

    ho0 = b.add(rc.a, b.mul(rc.ti, b.const_i32(p.dH * m)))
    wo0 = b.add(rc.bsub, b.mul(rc.tj0, b.const_i32(p.dW * mw)))
    img_row0 = b.mul(rc.n, b.const_i32(p.Ho))
    yrow = []
    for i in range(m):
        ho = b.add(ho0, b.const_i32(i * p.dH))
        okr = b.land(ok, b.cmp_lt(ho, b.const_i32(p.Ho)))
        off = b.mul(
            b.add(
                b.mul(b.mul(b.add(img_row0, ho), b.const_i32(p.Wo)), b.const_i32(p.K)),
                k,
            ),
            b.const_i32(2),
        )
        yrow.append(_oob_sel(b, okr, off))
    ycol = []
    for j in range(mw):
        wo = b.add(wo0, b.const_i32(j * p.dW))
        ycol.append(
            _oob_sel(
                b, b.cmp_lt(wo, b.const_i32(p.Wo)), b.mul(wo, b.const_i32(p.K * 2))
            )
        )

    ops = _IROps(b)
    b.sched_barrier(0)
    # Column pass: colt[v][ch][i] = (A^T M[:, v])[i]
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
    # Stores last (see _input_tiles_transform).
    b.sched_barrier(0)
    cpol = _CPOL_NT if spec.nt_stores else 0
    for off, packed in pending:
        _store_packed(b, y_rsrc, off, packed, spec.io_dtype, cpol)
    return b.kernel


# ---------------------------------------------------------------------------
# Stage table (host side)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformStage:
    """One transform kernel of the pipeline and how to launch it.

    ``signature`` argument names map to the buffers ``w``/``u`` (U), ``x``,
    ``v`` (V), ``e``, ``mm`` (M), ``y``; a ``<name>_bytes`` scalar is that
    buffer's size (see :meth:`WinogradFwdSpec.buffer_bytes`).
    """

    name: str
    build: object
    grid: Tuple[int, int, int]
    block: int
    signature: list


def transform_stages(spec: WinogradFwdSpec) -> List[TransformStage]:
    """K1, (K2a,) K2, K4 in launch order. K3 runs between K2 and K4."""
    out = [
        TransformStage(
            "K1",
            build_winograd_filter_transform,
            filter_transform_grid(spec),
            filter_transform_block(spec),
            filter_transform_signature(spec),
        )
    ]
    if spec.prescale and spec.prescale_split:
        out.append(
            TransformStage(
                "K2a",
                build_winograd_prescale,
                prescale_grid(spec),
                prescale_block(spec),
                prescale_signature(spec),
            )
        )
    out.append(
        TransformStage(
            "K2",
            build_winograd_input_transform,
            input_transform_grid(spec),
            input_transform_block(spec),
            input_transform_signature(spec),
        )
    )
    out.append(
        TransformStage(
            "K4",
            build_winograd_output_transform,
            output_transform_grid(spec),
            output_transform_block(spec),
            output_transform_signature(spec),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Tile choice (analytic cost model)
# ---------------------------------------------------------------------------

# Tiles the automatic choice considers. F(6,3) on both axes is excluded: with
# fp16 operands its rocke error metric is 1.4-2.1e-2 (about 2x F(4,3), over
# the ~1.5e-2 budget); one F6 axis (6x4: 1.0-1.6e-2) is within it.
AUTO_TILES: Tuple[Tuple[int, int], ...] = (
    (4, 4),
    (6, 4),
    (2, 2),
    (6, 2),
    (4, 2),
    (2, 4),
    (2, 6),
    (4, 6),
)
# The tile ranking is an analytic roofline sum in byte equivalents: the K3
# GEMM flops at the occupancy of its first round of blocks, divided by a
# nominal machine balance (GEMM flops that cost as much as one byte of
# transform traffic), plus the bytes the transforms move. Every tile of a
# problem runs the same launches, so a launch term would drop out of the
# ranking. The balance is a nominal parameter, not a fit; pass the target's
# own ratio to rank for a different part.
COST_FLOPS_PER_BYTE = 256.0
# Edge of the K3 output block the occupancy is counted in.
_COST_GEMM_BLOCK = 256


def tile_cost(
    problem: ConvProblem,
    tile,
    io_dtype: str = "fp16",
    *,
    num_cus: int,
    flops_per_byte: float = COST_FLOPS_PER_BYTE,
) -> float:
    """Relative cost (byte equivalents) of the 4-kernel pipeline with
    ``tile`` (``m`` or ``(mh, mw)``), with the default operand / prescale
    choice of ``io_dtype``: GEMM flops over the block occupancy of the first
    round (``256 x 256`` blocks over ``num_cus``) and ``flops_per_byte``,
    plus the bytes the transforms move (x, V twice, M twice, y, w, U twice)
    and the extra x pass of the prescale. Only meant to rank tiles of one
    problem."""
    import dataclasses

    if num_cus < 1:
        raise ValueError(f"num_cus must be >= 1, got {num_cus}")
    mh, mw = _split_m(tile)
    one = dataclasses.replace(problem, N=1)  # linear in N; one image fits 2 GiB
    s = WinogradFwdSpec(problem=one, m=mh, m_w=mw)
    b = s.buffer_bytes()
    n = problem.N
    blocks = (
        ceil_div(s.num_tiles * n, _COST_GEMM_BLOCK)
        * ceil_div(problem.kpg, _COST_GEMM_BLOCK)
        * s.n2
        * problem.groups
    )
    occ = min(1.0, blocks / num_cus)
    gemm = n * winograd_gemm_flops(one, (mh, s.mw)) / occ / flops_per_byte
    moved = n * (b["x"] + 2 * b["V"] + 2 * b["M"] + b["y"]) + b["w"] + 2 * b["U"]
    ps_x = n * b["x"] if (io_dtype == "bf16" and max(mh, s.mw) >= 4) else 0
    return gemm + moved + ps_x


def auto_tiles(
    problem: ConvProblem,
    io_dtype: str = "fp16",
    count: int = 2,
    tiles: Sequence[Tuple[int, int]] = AUTO_TILES,
    *,
    num_cus: int,
    flops_per_byte: float = COST_FLOPS_PER_BYTE,
) -> List[Tuple[int, int]]:
    """The ``count`` tiles with the lowest :func:`tile_cost` (ties keep the
    order of ``tiles``; a transposed tile of the same cost is dropped). With
    bf16 I/O the transposed 4x6 tile is skipped: its fused-prescale input
    transform spills (6x4 is the same work)."""
    cands = []
    for i, t in enumerate(tiles):
        if io_dtype == "bf16" and tuple(t) == (4, 6):
            continue
        try:
            c = tile_cost(
                problem, t, io_dtype, num_cus=num_cus, flops_per_byte=flops_per_byte
            )
        except ValueError:
            continue
        # A transposed tile of equal cost (square sub-grids) adds nothing.
        if any(abs(c - c2) <= 1e-9 * c and t2 == tuple(t)[::-1] for c2, _, t2 in cands):
            continue
        cands.append((c, i, tuple(t)))
    cands.sort()
    return [t for _, _, t in cands[:count]]


# ---------------------------------------------------------------------------
# Merged launches: several transform kernels in one grid
# ---------------------------------------------------------------------------

# Set while :func:`build_winograd_merged` emits one stage's body: the stage
# builders then write into the merged kernel through a :class:`_SubBuilder`
# instead of creating their own kernel.
_MERGE_TARGET: Optional["_SubBuilder"] = None


def _new_builder(name: str) -> IRBuilder:
    if _MERGE_TARGET is not None:
        return _MERGE_TARGET
    return IRBuilder(name)


class _SubBuilder(IRBuilder):
    """An :class:`IRBuilder` view that emits into another builder's kernel.

    It shares every piece of state with ``base`` (value counter, region stack,
    params, kernel), so ops land in the region ``base`` is currently filling;
    only the work-item ids are replaced by the stage's own (decoded from the
    merged flat grid) and parameters of the same name are shared.
    """

    def __init__(self, base: IRBuilder, block_ids, tid: Value):  # noqa: D107
        self.__dict__ = base.__dict__
        object.__setattr__(self, "_sub_ids", (tuple(block_ids), tid))

    def block_id_x(self) -> Value:
        return self._sub_ids[0][0]

    def block_id_y(self) -> Value:
        return self._sub_ids[0][1]

    def block_id_z(self) -> Value:
        return self._sub_ids[0][2]

    def thread_id_x(self) -> Value:
        return self._sub_ids[1]

    def param(self, name: str, t, **attrs) -> Value:
        if name in self._param_values:
            return self._param_values[name]
        return super().param(name, t, **attrs)


@dataclass(frozen=True)
class MergedStage:
    """Launch data of a merged transform kernel (see
    :func:`build_winograd_merged`)."""

    name: str
    parts: Tuple[str, ...]
    grid: Tuple[int, int, int]
    block: int
    signature: list
    # (first flat block, flat blocks) of each part
    ranges: Tuple[Tuple[int, int], ...]
    # stage blocks per merged block (> 0) or merged blocks per stage block (< 0)
    packs: Tuple[int, ...]


# Stages with no barrier / LDS: their blocks may be packed into, or split
# across, the merged kernel's blocks (thread ids are remapped).
_BARRIER_FREE_STAGES = ("K1",)


def _merge_layout(stages: Sequence["TransformStage"]):
    """``(block, ranges, packs, flat blocks)``. The merged block is the block
    of the stages that use barriers / LDS (they must agree; they run with
    their own block unchanged), else the largest one. ``packs[i]`` > 0: that
    many stage blocks per merged block; < 0: ``-packs[i]`` merged blocks per
    stage block (only for barrier-free stages)."""
    fixed = {st.block for st in stages if st.name not in _BARRIER_FREE_STAGES}
    if len(fixed) > 1:
        raise ValueError(f"stages with barriers need one block size, got {fixed}")
    block = fixed.pop() if fixed else max(st.block for st in stages)
    ranges, packs = [], []
    start = 0
    for st in stages:
        gx, gy, gz = st.grid
        sub = gx * gy * gz
        if st.block == block:
            pack, nflat = 1, sub
        elif st.name not in _BARRIER_FREE_STAGES:
            raise ValueError(f"{st.name}: block {st.block} != merged block {block}")
        elif st.block < block:
            if block % st.block:
                raise ValueError(f"{st.name}: block {st.block} does not divide {block}")
            pack = block // st.block
            nflat = ceil_div(sub, pack)
        else:
            if st.block % block:
                raise ValueError(f"{st.name}: block {block} does not divide {st.block}")
            pack = -(st.block // block)
            nflat = sub * -pack
        ranges.append((start, nflat))
        packs.append(pack)
        start += nflat
    return block, tuple(ranges), tuple(packs), start


def merged_stage(
    spec: WinogradFwdSpec, names: Sequence[str] = ("K1", "K2")
) -> MergedStage:
    """Launch layout of ``names`` (stages of :func:`transform_stages`) merged
    into one kernel: disjoint ranges of a flat 1-D grid, in the given order."""
    by = {st.name: st for st in transform_stages(spec)}
    missing = [nm for nm in names if nm not in by]
    if missing:
        raise ValueError(f"no stage(s) {missing} in this pipeline")
    stages = [by[nm] for nm in names]
    block, ranges, packs, total = _merge_layout(stages)
    sig: list = []
    seen = set()
    for st in stages:
        for a in st.signature:
            if a["name"] not in seen:
                seen.add(a["name"])
                sig.append(a)
    return MergedStage(
        "+".join(names), tuple(names), (total, 1, 1), block, sig, ranges, packs
    )


def check_merged_layout(spec: WinogradFwdSpec, lay: MergedStage) -> None:
    """Host-side check of a merged launch: the ranges tile the flat grid,
    and every (stage block, stage thread) of every part is covered exactly
    once by the remapping :func:`build_winograd_merged` emits (enumerated
    here in Python). Raises ``ValueError``."""
    by = {st.name: st for st in transform_stages(spec)}
    nxt = 0
    for nm, (start, nflat), pack in zip(lay.parts, lay.ranges, lay.packs):
        st = by[nm]
        if start != nxt or nflat <= 0:
            raise ValueError(f"{nm}: range {(start, nflat)} not contiguous")
        nxt = start + nflat
        nsub = st.grid[0] * st.grid[1] * st.grid[2]
        seen = set()
        for loc in range(nflat):
            for tid in range(0, lay.block, 32):  # one probe per wave
                if pack > 1:
                    sb, stid = loc * pack + tid // st.block, tid % st.block
                    if sb >= nsub:
                        continue  # guarded off in the kernel
                elif pack < 0:
                    sb, stid = loc // -pack, (loc % -pack) * lay.block + tid
                else:
                    sb, stid = loc, tid
                if not (0 <= sb < nsub and 0 <= stid < st.block):
                    raise ValueError(f"{nm}: flat {start + loc} -> block {sb}/{stid}")
                if (sb, stid) in seen:
                    raise ValueError(f"{nm}: block {sb} thread {stid} twice")
                seen.add((sb, stid))
        if len(seen) != nsub * ceil_div(st.block, 32):
            raise ValueError(f"{nm}: {len(seen)} wave slots of {nsub} blocks covered")
    if nxt != lay.grid[0]:
        raise ValueError(f"ranges cover {nxt} of {lay.grid[0]} blocks")


# A merged kernel's LDS is reserved by every block, K1's included: past this
# the K1 part loses most of its occupancy (seen with a 114 KB K2 region,
# where the merged launch was slower than two separate ones).
MERGE_MAX_LDS_BYTES = 32 * 1024


def merge_worthwhile(spec: WinogradFwdSpec) -> bool:
    """Whether merging K1 with the first input-transform stage should pay:
    only when that stage reserves little LDS (the register path, or a small
    LDS region)."""
    if spec.prescale and spec.prescale_split:
        return True  # K2a uses no LDS
    if not spec.input_lds:
        return True
    lm = _lds_map(spec)
    return lm.loads * lm.block * lm.lv * 2 <= MERGE_MAX_LDS_BYTES


@dataclass(frozen=True)
class WinogradMergedSpec:
    """Independent transform stages of ``fwd`` (``names``, e.g. ``("K2",
    "K1")``) merged into one launch (:func:`build_winograd_merged`)."""

    fwd: WinogradFwdSpec
    names: Tuple[str, ...] = ("K1", "K2")


def build_winograd_merged(
    spec: WinogradMergedSpec, *, arch: str = "gfx1250"
) -> KernelDef:
    """Several independent transform stages as one kernel.

    Each stage gets a disjoint range of the flat grid (:func:`merged_stage`);
    a block decodes its stage-local block ids from its flat id and runs that
    stage's body unchanged. Stages with barriers / LDS (K2, K2a) keep their
    own block, which is the merged block; a barrier-free stage (K1) whose
    block differs is packed into (smaller) or split across (larger) merged
    blocks with remapped thread ids (see :func:`_merge_layout`). This saves one launch and the
    tail of the shorter stage overlaps the longer one; it is only valid for
    stages without a data dependency (K1 and K2, or K1 and K2a).
    """
    spec, names = spec.fwd, spec.names
    lay = merged_stage(spec, names)
    check_merged_layout(spec, lay)
    by = {st.name: st for st in transform_stages(spec)}
    stages = [by[nm] for nm in lay.parts]
    b = IRBuilder(spec.kernel_name("merged_" + "_".join(lay.parts).lower()))
    flat = b.block_id_x()
    tid = b.thread_id_x()

    def emit(i: int) -> None:
        st = stages[i]
        start, nflat = lay.ranges[i]
        pack = lay.packs[i]
        local = b.sub(flat, b.const_i32(start))
        if pack > 1:
            # Barrier-free stage with a smaller block: sub-block = tid / bs.
            sub = b.add(
                b.mul(local, b.const_i32(pack)),
                b.div(tid, b.const_i32(st.block)),
            )
            stid = b.mod(tid, b.const_i32(st.block))
        elif pack < 0:
            # Barrier-free stage with a larger block: -pack merged blocks each.
            sub = b.div(local, b.const_i32(-pack))
            stid = b.add(
                b.mul(b.mod(local, b.const_i32(-pack)), b.const_i32(lay.block)), tid
            )
        else:
            sub = local
            stid = tid
        gx, gy, gz = st.grid
        bx = b.mod(sub, b.const_i32(gx)) if gx > 1 else b.const_i32(0)
        r = b.div(sub, b.const_i32(gx)) if gx > 1 else sub
        by_ = b.mod(r, b.const_i32(gy)) if gy > 1 else b.const_i32(0)
        bz = b.div(r, b.const_i32(gy)) if gy > 1 else r

        def body():
            global _MERGE_TARGET
            _MERGE_TARGET = _SubBuilder(b, (bx, by_, bz), stid)
            try:
                st.build(spec, arch=arch)
            finally:
                _MERGE_TARGET = None

        if pack > 1 and nflat * pack != gx * gy * gz:
            with b.scf_if(b.cmp_lt(sub, b.const_i32(gx * gy * gz))):
                body()
        else:
            body()

    def chain(i: int) -> None:
        if i == len(stages) - 1:
            emit(i)
            return
        start, nflat = lay.ranges[i]
        # if / else (not two ifs): barriers inside a branch must survive.
        with b.scf_if_else(b.cmp_lt(flat, b.const_i32(start + nflat))) as (t, e):
            with t:
                emit(i)
            with e:
                chain(i + 1)

    chain(0)
    b.kernel.attrs["max_workgroup_size"] = lay.block
    # The launcher packs kernargs by ``lay.signature``: it must match the
    # parameter order the stage bodies declared.
    got = [prm.name for prm in b.kernel.params]
    want = [a["name"] for a in lay.signature]
    if got != want:
        raise ValueError(f"merged kernel params {got} != signature {want}")
    return b.kernel


# ---------------------------------------------------------------------------
# Host-side reference of the 4-stage algorithm (numpy; CPU tests)
# ---------------------------------------------------------------------------


def winograd_fwd_reference_np(x, w, problem: ConvProblem, m, *, round_fn=None):
    """Numpy model of K1..K4 with the same tiling, layouts and index maps.

    ``x`` is NHWC, ``w`` KYXC (``Cg`` innermost). ``round_fn(name, arr)`` lets
    a caller round each workspace (``"U"``, ``"V"``, ``"M"``) to the operand
    dtype; ``None`` keeps fp64 throughout. Returns ``(y, U, V, M)`` in the
    kernels' layouts.
    """
    import numpy as np

    mh, mw = _split_m(m)
    spec = WinogradFwdSpec(problem=problem, m=mh, m_w=mw)
    mw = spec.mw
    p = problem
    n, nw, n2 = spec.nh, spec.nw, spec.n2

    def mats(mm):
        return [np.array([[float(v) for v in r] for r in f]) for f in toom_cook(mm)]

    at, g, bt = mats(mh)
    at_w, g_w, bt_w = mats(mw)
    rnd = round_fn or (lambda _name, a: a)
    G, Cg, Kg = p.groups, p.cpg, p.kpg
    x = np.asarray(x, np.float64)
    w = np.asarray(w, np.float64)

    # K1: U[bin][k][c] (bin = i*n + j)
    wk = w.transpose(0, 3, 1, 2)  # K, Cg, 3, 3
    u4 = np.einsum("ia,kcab,jb->ijkc", g, wk, g_w)
    U = rnd("U", u4.reshape(n2, p.K, Cg))

    # K2: V[t][bin][c]
    th, tw, dH, dW = spec.tiles_h, spec.tiles_w, p.dH, p.dW
    T = spec.num_tiles
    V = np.zeros((T, n2, p.C))
    ts = []
    for nn in range(p.N):
        for a in range(dH):
            for bs in range(dW):
                for ti in range(th):
                    for tj in range(tw):
                        ts.append((nn, a, bs, ti, tj))
    for t, (nn, a, bs, ti, tj) in enumerate(ts):
        d = np.zeros((n, nw, p.C))
        for uu in range(n):
            hi = a - p.pH + dH * (ti * mh + uu)
            if not 0 <= hi < p.Hi:
                continue
            for vv in range(nw):
                wi = bs - p.pW + dW * (tj * mw + vv)
                if 0 <= wi < p.Wi:
                    d[uu, vv] = x[nn, hi, wi]
        V[t] = np.einsum("iu,uvc,jv->ijc", bt, d, bt_w).reshape(n2, p.C)
    V = rnd("V", V)

    # K3: M[t][bin][k] = sum_c V[t][bin][g*Cg + c] * U[bin][g*Kg + k][c]
    M = np.zeros((T, n2, p.K))
    for gi in range(G):
        vg = V[:, :, gi * Cg : (gi + 1) * Cg]  # T, n2, Cg
        ug = U[:, gi * Kg : (gi + 1) * Kg, :]  # n2, Kg, Cg
        M[:, :, gi * Kg : (gi + 1) * Kg] = np.einsum("tbc,bkc->tbk", vg, ug)
    M = rnd("M", M)

    # K4
    y = np.zeros((p.N, p.Ho, p.Wo, p.K))
    for t, (nn, a, bs, ti, tj) in enumerate(ts):
        Y = np.einsum("iu,uvk,jv->ijk", at, M[t].reshape(n, nw, p.K), at_w)
        for i in range(mh):
            ho = a + dH * (ti * mh + i)
            if ho >= p.Ho:
                continue
            for j in range(mw):
                wo = bs + dW * (tj * mw + j)
                if wo < p.Wo:
                    y[nn, ho, wo] = Y[i, j]
    return y, U, V, M


__all__ = [
    "PRESCALE_BETA",
    "WinogradFwdSpec",
    "build_winograd_filter_transform",
    "build_winograd_input_transform",
    "build_winograd_output_transform",
    "build_winograd_prescale",
    "TransformStage",
    "at_1d",
    "bt_1d",
    "filter_matrix_int",
    "filter_transform_block",
    "filter_transform_grid",
    "filter_transform_signature",
    "input_transform_block",
    "input_transform_grid",
    "input_transform_signature",
    "output_transform_block",
    "output_transform_grid",
    "output_transform_signature",
    "prescale_block",
    "prescale_grid",
    "prescale_signature",
    "toom_cook",
    "transform_stages",
    "tuned_transform_options",
    "winograd_fwd_reference_np",
    "winograd_gemm_flops",
    "winograd_gemm_problem",
]
