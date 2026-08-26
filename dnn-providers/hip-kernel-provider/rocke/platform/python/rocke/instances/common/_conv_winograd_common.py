# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared helpers for Winograd convolution kernels.

Internal module (underscore-prefixed). External callers should import from
:mod:`rocke.instances.common.conv_winograd`.

What lives here
---------------
* :class:`WinogradTile` — transform-matrix constants for each supported
  (output_tile, filter_size) pair.
* :class:`WinogradProblem` — 2-D convolution shape for Winograd (stride=1,
  dilation=1, 3x3 filter only).
* ``_emit_winograd_data_transform`` — IRBuilder helper that emits the
  B^T * d * B arithmetic for one (n, tile_h, tile_w) combination.
* ``_emit_winograd_filter_transform`` — G * g * G^T arithmetic for one
  (k, c) pair.
* ``_emit_winograd_output_transform`` — A^T * m * A arithmetic for one
  (n, tile_h, tile_w, k) output tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ...core.ir import F16, F32, IRBuilder, Value


# ---------------------------------------------------------------------------
# Winograd transform matrices
# ---------------------------------------------------------------------------

# Each entry: (out_tile, filter_size) -> {"xform": int, "B_t": ..., "G": ..., "A_t": ...}
# Matrices are stored row-major as flat Python float lists.
# Convention matches MIOpen / xform_bidirect_winograd_code.inc.

# F(2,3): 2-output, 3-filter -> 4x4 transform domain
# B^T (4x4), G (4x3), A^T (2x4)
_F2X3_B_T: List[List[float]] = [
    [1, 0, -1, 0],
    [0, 1, 1, 0],
    [0, -1, 1, 0],
    [0, 1, 0, -1],
]
_F2X3_G: List[List[float]] = [
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.0, 0.0, 1.0],
]
_F2X3_A_T: List[List[float]] = [
    [1, 1, 1, 0],
    [0, 1, -1, -1],
]

# F(4,3): 4-output, 3-filter -> 6x6 transform domain
# Matches CK / MIOpen ConvMPBidirectWinograd<4,3>
# B^T (6x6), G (6x3), A^T (4x6)
_F4X3_B_T: List[List[float]] = [
    [4, 0, -5, 0, 1, 0],
    [0, -4, -4, 1, 1, 0],
    [0, 4, -4, -1, 1, 0],
    [0, -2, -1, 2, 1, 0],
    [0, 2, -1, -2, 1, 0],
    [0, 4, 0, -5, 0, 1],
]
_F4X3_G: List[List[float]] = [
    [1 / 4, 0, 0],
    [-1 / 6, -1 / 6, -1 / 6],
    [-1 / 6, 1 / 6, -1 / 6],
    [1 / 24, 1 / 12, 1 / 6],
    [1 / 24, -1 / 12, 1 / 6],
    [0, 0, 1],
]
_F4X3_A_T: List[List[float]] = [
    [1, 1, 1, 1, 1, 0],
    [0, 1, -1, 2, -2, 0],
    [0, 1, 1, 4, 4, 0],
    [0, 1, -1, 8, -8, 1],
]


@dataclass(frozen=True)
class WinogradTile:
    """Winograd transform-matrix constants for one (out_tile, filter_size) variant.

    ``out_tile``: number of output elements per 1-D tile (2 or 4).
    ``filter_size``: convolution filter radius (always 3 for 3x3).
    ``xform_size``: transform domain size = out_tile + filter_size - 1.

    Matrices are stored as row-major nested Python float lists:
      ``B_t``  (xform_size x xform_size) — input transform
      ``G``    (xform_size x filter_size) — filter transform
      ``A_t``  (out_tile x xform_size)   — output transform
    """

    out_tile: int
    filter_size: int
    xform_size: int
    B_t: List[List[float]]
    G: List[List[float]]
    A_t: List[List[float]]

    @classmethod
    def f2x3(cls) -> "WinogradTile":
        return cls(
            out_tile=2,
            filter_size=3,
            xform_size=4,
            B_t=_F2X3_B_T,
            G=_F2X3_G,
            A_t=_F2X3_A_T,
        )

    @classmethod
    def f4x3(cls) -> "WinogradTile":
        return cls(
            out_tile=4,
            filter_size=3,
            xform_size=6,
            B_t=_F4X3_B_T,
            G=_F4X3_G,
            A_t=_F4X3_A_T,
        )


WINOGRAD_TILES = {
    (2, 3): WinogradTile.f2x3(),
    (4, 3): WinogradTile.f4x3(),
}


# ---------------------------------------------------------------------------
# Problem dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WinogradProblem:
    """2-D convolution shape for the Winograd path.

    Restrictions (checked by :meth:`validate`):
      - stride = 1 (both H and W)
      - dilation = 1 (both H and W)
      - filter = 3x3 only
      - NHWC input layout, KYXC filter layout

    Shape:
      Input:  [N, Hi, Wi, C]   (NHWC)
      Filter: [K, 3, 3, C]     (KYXC)
      Output: [N, Ho, Wo, K]   (NHWK),  Ho = Hi - 2 + 2*pH,  Wo = Wi - 2 + 2*pW
    """

    N: int
    Hi: int
    Wi: int
    C: int
    K: int
    pH: int = 1
    pW: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.N < 1 or self.Hi < 1 or self.Wi < 1:
            raise ValueError("N, Hi, Wi must be >= 1")
        if self.C < 1 or self.K < 1:
            raise ValueError("C, K must be >= 1")
        if self.pH not in (0, 1) or self.pW not in (0, 1):
            raise ValueError("Winograd supports pH/pW in {0, 1}")

    @property
    def Ho(self) -> int:
        return self.Hi + 2 * self.pH - 2

    @property
    def Wo(self) -> int:
        return self.Wi + 2 * self.pW - 2

    @property
    def flops(self) -> int:
        return 2 * self.N * self.Ho * self.Wo * self.K * self.C * 9


# ---------------------------------------------------------------------------
# IR emit helpers: 1-D Winograd transforms
# ---------------------------------------------------------------------------


def _matvec_f32(
    b: IRBuilder,
    mat: List[List[float]],
    vec: List[Value],
) -> List[Value]:
    """Emit a matrix-vector multiply in fp32 SSA.

    ``mat`` is an (rows x cols) Python float list; ``vec`` is a list of
    ``cols`` IRBuilder fp32 Values. Returns a list of ``rows`` fp32 Values.
    Zeros are elided (no op emitted); ±1 and ±2 use fadd/fsub/fmul(2);
    all other coefficients use fmul.
    """
    rows = len(mat)
    cols = len(mat[0])
    assert len(vec) == cols, f"matvec: got {len(vec)} vec elems, expected {cols}"

    out: List[Value] = []
    c_zero = b.const_f32(0.0)
    for r in range(rows):
        accum: Optional[Value] = None
        for c_idx in range(cols):
            coeff = mat[r][c_idx]
            if coeff == 0.0:
                continue
            v = vec[c_idx]
            if coeff == 1.0:
                term = v
            elif coeff == -1.0:
                term = b.fsub(c_zero, v)
            elif coeff == 2.0:
                term = b.fadd(v, v)
            elif coeff == -2.0:
                term = b.fsub(c_zero, b.fadd(v, v))
            elif coeff == 4.0:
                t = b.fadd(v, v)
                term = b.fadd(t, t)
            elif coeff == -4.0:
                t = b.fadd(v, v)
                term = b.fsub(c_zero, b.fadd(t, t))
            elif coeff == -5.0:
                t2 = b.fadd(v, v)
                t4 = b.fadd(t2, t2)
                term = b.fsub(c_zero, b.fadd(t4, v))
            elif coeff == 5.0:
                t2 = b.fadd(v, v)
                t4 = b.fadd(t2, t2)
                term = b.fadd(t4, v)
            elif coeff == 8.0:
                t = b.fadd(v, v)
                t = b.fadd(t, t)
                term = b.fadd(t, t)
            elif coeff == -8.0:
                t = b.fadd(v, v)
                t = b.fadd(t, t)
                term = b.fsub(c_zero, b.fadd(t, t))
            else:
                c_coeff = b.const_f32(coeff)
                term = b.fmul(c_coeff, v)
            accum = term if accum is None else b.fadd(accum, term)
        out.append(accum if accum is not None else c_zero)
    return out


def _apply_transform_2d(
    b: IRBuilder,
    mat: List[List[float]],
    tile: List[List[Value]],
) -> List[List[Value]]:
    """Apply ``mat`` to rows then columns of a 2-D tile of fp32 Values.

    ``tile`` is (in_rows x in_cols).  Returns (out_rows x in_cols) after
    the row transform, then another (out_rows x out_cols) after the column
    transform.  The 2-D Winograd transform Y = M * X * M^T is decomposed as:
      1. tmp[out_row][col] = sum_r M[out_row][r] * tile[r][col]   (row transform)
      2. out[out_row][out_col] = sum_c tmp[out_row][c] * M^T[c][out_col]  (col transform)

    For symmetric matrices (B^T applied to both dims, or A^T), M^T = M.T.
    We use the same matrix for both passes because all three transforms have
    the property: apply M to rows, then M to columns of the transposed result.
    """
    in_rows = len(tile)
    in_cols = len(tile[0])

    # Row transform: apply mat to each column (treat each column as a vector)
    mid_rows = len(mat)
    mid: List[List[Value]] = []
    for _ in range(mid_rows):
        mid.append([None] * in_cols)  # type: ignore[list-item]
    for col in range(in_cols):
        col_vec = [tile[row][col] for row in range(in_rows)]
        transformed_col = _matvec_f32(b, mat, col_vec)
        for out_row, val in enumerate(transformed_col):
            mid[out_row][col] = val

    # Column transform: apply mat to each row of mid
    out_cols = len(mat)
    out: List[List[Value]] = []
    for _ in range(mid_rows):
        out.append([None] * out_cols)  # type: ignore[list-item]
    for row in range(mid_rows):
        row_vec = mid[row]
        transformed_row = _matvec_f32(b, mat, row_vec)
        for out_col, val in enumerate(transformed_row):
            out[row][out_col] = val

    return out


# ---------------------------------------------------------------------------
# Higher-level emit helpers called by the kernel builders
# ---------------------------------------------------------------------------


def emit_data_transform(
    b: IRBuilder,
    tile: WinogradTile,
    input_patch: List[List[Value]],
) -> List[List[Value]]:
    """Emit B^T * patch * B for one xform_size x xform_size input patch.

    ``input_patch`` is (xform_size x xform_size) fp32 Values loaded from the
    padded NHWC input.  Returns (xform_size x xform_size) fp32 Values ready
    to be stored to the transformed data workspace.
    """
    return _apply_transform_2d(b, tile.B_t, input_patch)


def emit_filter_transform(
    b: IRBuilder,
    tile: WinogradTile,
    filter_patch: List[List[Value]],
) -> List[List[Value]]:
    """Emit G * filter * G^T for one (filter_size x filter_size) filter patch.

    ``filter_patch`` is (filter_size x filter_size) fp32 Values.
    Returns (xform_size x xform_size) fp32 Values.

    The filter transform is G * g * G^T.  We apply G to rows (gives
    xform_size x filter_size intermediate), then G to columns of the
    transposed intermediate (equiv. G^T applied to columns).
    """
    xs = tile.xform_size
    fs = tile.filter_size
    # Step 1: mid = G * filter  (xform_size x filter_size)
    mid: List[List[Value]] = []
    for _ in range(xs):
        mid.append([None] * fs)  # type: ignore[list-item]
    for col in range(fs):
        col_vec = [filter_patch[row][col] for row in range(fs)]
        transformed_col = _matvec_f32(b, tile.G, col_vec)
        for out_row, val in enumerate(transformed_col):
            mid[out_row][col] = val

    # Step 2: out = mid * G^T  (xform_size x xform_size)
    # mid has shape (xs, fs). G^T has shape (xs, fs) when written as a
    # left-multiplier: out[row][col] = sum_k mid[row][k] * G[col][k].
    # Equivalently: for each row of mid, we apply G as a (xs x fs) matrix
    # to produce xs output values.
    out: List[List[Value]] = []
    for _ in range(xs):
        out.append([None] * xs)  # type: ignore[list-item]
    for row in range(xs):
        row_vec = mid[row]  # length fs
        # G is (xs x fs): _matvec_f32(G, row_vec) gives xs outputs
        transformed_row = _matvec_f32(b, tile.G, row_vec)
        for out_col, val in enumerate(transformed_row):
            out[row][out_col] = val
    return out


def emit_output_transform(
    b: IRBuilder,
    tile: WinogradTile,
    xformed_output: List[List[Value]],
) -> List[List[Value]]:
    """Emit A^T * xformed_output * A for one (xform_size x xform_size) tile.

    Returns (out_tile x out_tile) fp32 Values — the recovered spatial output.
    """
    ot = tile.out_tile
    xs = tile.xform_size

    # Step 1: mid = A^T * xformed_output  (out_tile x xform_size)
    mid: List[List[Value]] = []
    for _ in range(ot):
        mid.append([None] * xs)  # type: ignore[list-item]
    for col in range(xs):
        col_vec = [xformed_output[row][col] for row in range(xs)]
        transformed_col = _matvec_f32(b, tile.A_t, col_vec)
        for out_row, val in enumerate(transformed_col):
            mid[out_row][col] = val

    # Step 2: out = mid * A  (out_tile x out_tile)
    # mid has shape (ot, xs). We want out[row][col] = sum_k mid[row][k] * A_t[col][k].
    # Using A_t as a (ot x xs) left-multiplier on mid's rows (length xs) gives
    # ot outputs per row — same as applying A_t.
    out: List[List[Value]] = []
    for _ in range(ot):
        out.append([None] * ot)  # type: ignore[list-item]
    for row in range(ot):
        row_vec = mid[row]  # length xs
        # A_t is (ot x xs): _matvec_f32(A_t, row_vec) gives ot outputs
        transformed_row = _matvec_f32(b, tile.A_t, row_vec)
        for out_col, val in enumerate(transformed_row):
            out[row][out_col] = val
    return out
