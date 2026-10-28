# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.
#
# Provenance carried over from the sweep snapshot:
# Adapted from: https://github.com/ROCm/aiter
#   commit:   8253efc4059516f492df597b80ce5311e9fda8bd
#   path:     aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py
# via the gfx11 forward `attention_rdna/flash_attn_func_rdna.py` (ROCm/rocm-libraries PR #14111 @ d850254d45e),
# which supplies the WMMA ABI split (modification 1), the xor-16 exchange and the epilogue exchange re-expressed here as functions.

"""WMMA layout layer of the gfx11 (RDNA3/3.5) and gfx12 (RDNA4) SDPA backward kernels: index algebra, lane exchanges, LDS indexing.

Every function uses plain integer arithmetic (`+ - * // %`, comparisons) on its index arguments, so the same call runs on Python ints
(`tools/test_rdna_layout.py`, numpy arrays of lane ids) and on FlyDSL values (kernels). No FlyDSL import. A kernel that needs a
data select passes a precomputed `klane_is_zero` predicate and an `exchange` callable; the kernel's `exchange` is
`lambda v: Float32(v).shuffle_xor(Int32(16), Int32(32))`, the model's is a lane permutation (`lane ^ 16`).

ABI switch. `v8` is `WMMA_V8` of the vendored forward: `False` = gfx11 (`wmma_v8_for_arch("gfx11-generic")`), `True` = gfx12.
`v_wmma_f32_16x16x16_{f16,bf16}` on a wave32, `l` = lane, `klane = l // 16`, `l16 = l % 16`:

                            gfx11 (v16)                               gfx12 (v8)
    A operand (row i)       lane l16 = i; all 16 K values;            lane l16 = i; 8 K values, K = 8 klane + t;
    B operand (col j)       lanes 16-31 duplicate lanes 0-15          lanes 16-31 carry the upper K half
    C/D, element r (0..7)   row 2 r + klane, column l16               row r + 8 klane, column l16

Orientation of an S-type tile (S, P, dP, dS), the two the backward uses:

    Q_ROWS   rows = q, cols = kv.   dK/dV kernel. S = Q K^T: A = Q (lane row q), B = K^T (lane col kv). dV^T = dO^T P and
             dK^T = Q^T dS take P / dS as the B operand (K = q, col = kv) and dO^T / Q^T as the A operand (row = head dim,
             K = q; `lds_transposed_operand`). LSE and Delta are per element: row `acc_row(r, klane)` of the tile.
    KV_ROWS  rows = kv, cols = q.   dQ kernel (and dB). S^T = K Q^T: A = K, B = Q^T. dQ^T = K^T dS^T takes dS^T as the B operand
             (K = kv, col = q). LSE and Delta are per lane (column l16): one value per lane for all 8 elements.
    An accumulator that holds a transposed gradient (dV^T, dK^T, dQ^T) has rows = head dim, cols = kv (dK/dV) or q (dQ).

The `*_op` and `acc_*` names below are the layer's API; the contract per function is its docstring.
"""

WARP_SIZE = 32
WMMA_DIM = 16
ACC_ELEMS = 8

Q_ROWS = "q_rows"
KV_ROWS = "kv_rows"


def wmma_v8_for_arch(arch):
    """`WMMA_V8` for a ROCm arch string: gfx12* passes 8 K values per lane, gfx11* 16 (the forward's `get_rocm_arch().startswith("gfx12")`)."""
    return arch.lower().startswith("gfx12")


def op_width(v8):
    """K values one lane passes per A or B operand: 16 (gfx11), 8 (gfx12)."""
    return 8 if v8 else 16


# -- lane decomposition -----------------------------------------------------------------------------------------------------------


def lane_col(lane):
    """`l16`: the operand row (A) / column (B) of a lane and the column of its accumulator elements."""
    return lane % 16


def lane_half(lane):
    """`klane`: 0 for lanes 0-15, 1 for lanes 16-31."""
    return lane // 16


# -- A/B operands -----------------------------------------------------------------------------------------------------------------


def op_k0(klane, v8):
    """Offset of a lane's first operand element from the K-step origin: 0 (gfx11, all 16 K values) or `8 klane` (gfx12)."""
    return 8 * klane if v8 else 0


def op_origin(k_origin, klane, v8):
    """Index (head dim, q or kv) of the lane's first operand element for the K-step starting at `k_origin`; the lane's operand is
    `op_width(v8)` consecutive indices from here."""
    return k_origin + op_k0(klane, v8)


def lds_rowmajor_operand(base, stride, row, k_origin, klane, v8):
    """Element offset of the lane's operand read from a row-major LDS tile (K contiguous): `base + row * stride + op_origin`; the read is
    `op_width(v8)` contiguous elements. `row` is `lane_col(lane)` plus the tile row origin (A: M row, B: N column).
    """
    return base + row * stride + op_origin(k_origin, klane, v8)


def lds_transposed_operand(base, stride, col, k_origin, klane, v8):
    """Element offsets (a list of `op_width(v8)`, in K order) of the lane's operand read from a row-major LDS tile whose K axis is the ROW
    axis (the transposed A operand: Q^T, dO^T, K^T). Element t is `base + (op_origin + t) * stride + col`; `col` is `lane_col(lane)` plus the
    column (head dim) origin. The list is a Python-level constant length: iterate it with `range_constexpr`.
    """
    k0 = op_origin(k_origin, klane, v8)
    return [base + (k0 + t) * stride + col for t in range(op_width(v8))]


# -- accumulator coordinates ------------------------------------------------------------------------------------------------------


def acc_row(r, klane, v8):
    """Tile row of accumulator element `r` (0..7) of a lane: `2 r + klane` (gfx11), `r + 8 klane` (gfx12)."""
    return 8 * klane + r if v8 else 2 * r + klane


def acc_row_base(klane, v8):
    """Row of element 0; element `r` is at `acc_row_base + acc_row_step * r`."""
    return 8 * klane if v8 else klane


def acc_row_step(v8):
    """Row distance between consecutive elements of a lane: 1 (gfx12), 2 (gfx11)."""
    return 1 if v8 else 2


def acc_col(lane):
    """Tile column of every accumulator element of a lane."""
    return lane % 16


def acc_coord(r, lane, v8):
    """(row, col) of accumulator element `r` of `lane`."""
    return acc_row(r, lane // 16, v8), lane % 16


def acc_q_kv(orient, r, lane, q0, kv0, v8):
    """(q, kv) of element `r` of `lane` in an S-type tile whose origin is (`q0`, `kv0`): `Q_ROWS` rows are q, `KV_ROWS` rows are kv."""
    row, col = acc_coord(r, lane, v8)
    return (q0 + row, kv0 + col) if orient == Q_ROWS else (q0 + col, kv0 + row)


def acc_q(orient, r, lane, q0, v8):
    """q index of element `r` of `lane`: where its LSE and Delta are read. `Q_ROWS`: per element (`q0 + acc_row`); `KV_ROWS`: `q0 + l16`, the same for all `r`."""
    return q0 + (acc_row(r, lane // 16, v8) if orient == Q_ROWS else lane % 16)


def acc_kv(orient, r, lane, kv0, v8):
    """kv index of element `r` of `lane` (`Q_ROWS`: `kv0 + l16`; `KV_ROWS`: `kv0 + acc_row`)."""
    return kv0 + (lane % 16 if orient == Q_ROWS else acc_row(r, lane // 16, v8))


# -- exchanges --------------------------------------------------------------------------------------------------------------------


def klane_select(klane_is_zero, if_zero, if_one):
    """`if_zero` where the lane is in lanes 0-15, else `if_one`. Dispatches on the predicate's type: a FlyDSL value (`.select`), a numpy
    array (`where`), or a Python bool."""
    sel = getattr(klane_is_zero, "select", None)
    if sel is not None:
        return sel(if_zero, if_one)
    if hasattr(klane_is_zero, "dtype"):
        import numpy as np

        return np.where(klane_is_zero, if_zero, if_one)
    return if_zero if klane_is_zero else if_one


def acc_to_operand(own, exchange, klane_is_zero, v8):
    """The B-operand values of a lane from its 8 accumulator values `own[r]` (tile row `acc_row(r)`, K = tile row): `op_width(v8)` values
    whose element k is tile row `k` (gfx11) or `8 klane + k` (gfx12): the operand of the next GEMM contracting over the tile rows (P / dS as
    the B operand). gfx12: `own` as it is. gfx11: the lane holds only the even (klane 0) or odd (klane 1) rows, its xor-16 peer the others;
    trade all 8 (`exchange(v)` returns the xor-16 peer's `v`) and interleave, so both lanes of a column hold all 16 rows (the duplication
    the hardware needs)."""
    if v8:
        return list(own)
    peer = [exchange(v) for v in own]
    full = []
    for j in range(8):
        full.append(klane_select(klane_is_zero, own[j], peer[j]))
        full.append(klane_select(klane_is_zero, peer[j], own[j]))
    return full


def acc_to_rows8(own, exchange, klane_is_zero, v8):
    """The epilogue exchange: 8 values `rows[j]` = tile row `8 klane + j` of the lane's column, so each lane stores 8 contiguous elements of
    a transposed gradient (dV^T, dK^T, dQ^T rows = head dim) or of a dB row (`KV_ROWS` tile: rows = kv). gfx12 already holds rows
    `8 klane + r` (`own`). gfx11: trade with the xor-16 peer once (`exchange`), the forward's O store.
    """
    if v8:
        return list(own)
    peer = [exchange(v) for v in own]
    rows = []
    for j in range(8):
        lo_src, hi_src = (own, peer) if j % 2 == 0 else (peer, own)
        rows.append(klane_select(klane_is_zero, lo_src[j // 2], hi_src[4 + j // 2]))
    return rows


def rows8_origin(klane):
    """First tile row held by `acc_to_rows8` for a lane: `8 klane`."""
    return 8 * klane


# -- masks, bias, dropout, elementwise ---------------------------------------------------------------------------------------------


def kv_window(q, Sq, Sk, kind, wl, wr, vmax=max, vmin=min):
    """(first, span) of the kv columns row `q` sees, so that the element `kv` is visible iff `0 <= kv - first < span` (one unsigned compare of
    `kv - first` against `span`). `kind` in tl | br | win | winbr as `h/refmath.visible`; `wl`, `wr` the window extents (win kinds).
    `vmax` / `vmin` are the element-wise max / min (`max`, `min` on ints; `np.maximum`, `np.minimum` on arrays). A row that sees no key
    has span 0."""
    off = Sk - Sq if kind in ("br", "winbr") else 0
    if kind in ("tl", "br"):
        lo, hi = 0, q + off + 1
    else:
        lo, hi = q + off - wl, q + off + wr + 1
    first = vmax(lo, 0)
    end = vmin(hi, Sk)
    span = vmax(end - first, 0)
    return first, span


def kv_visible(kv, first, span):
    """`0 <= kv - first < span` as the two-compare form (kernels use the single unsigned compare of `kv - first`)."""
    rel = kv - first
    return (rel >= 0) & (rel < span)


def bias_offset(q, kv, stride_q, stride_kv):
    """Element offset of the bias entry of (q, kv) in a plane; a broadcast dim has stride 0."""
    return q * stride_q + kv * stride_kv


DB_WRITER_SHARD = 0  # dQ kernel: of the `contraction_shards` waves that each hold a Q tile's whole reduced dS, only this shard stores dB


def dropout_row_stride(Sk):
    """Philox offsets per score row: `ceil(Sk / 4)` (four randoms per counter)."""
    return (Sk + 3) // 4


def dropout_plane_base(offset_base, plane, Sq, Sk):
    """Counter of element (0, 0) of plane `plane = sequence * Hq + q_head`: `offset_base + plane * Sq * dropout_row_stride(Sk)`."""
    return offset_base + plane * Sq * dropout_row_stride(Sk)


def dropout_counter(plane_base, row_stride, q, kv):
    """(counter, slot) of element (q, kv): the Philox call at `counter` yields 4 randoms and the element keeps slot `kv % 4`. In a `Q_ROWS`
    tile all 8 elements of a lane share the slot (`kv0 + l16`) and differ in counter (one call each). In a `KV_ROWS` tile (kv0 a multiple of
    16) gfx11 elements `r` and `r + 1` (even `r`) share a counter (`r // 2`; slots `2 (r % 2) + klane`), gfx12 elements `4 i .. 4 i + 3` share
    one (slots `r % 4`), so one call can serve a group."""
    return plane_base + q * row_stride + kv // 4, kv % 4


def dropout_kv_rows_call(r, klane, v8):
    """`KV_ROWS` tile (kv0 a multiple of 16), accumulator element `r` of a lane of half `klane` (0 or 1): `(group, slot)` with `group` the Philox
    call's counter offset from `kv0 // 4` (the call at `dropout_counter(.., q, kv0 + 4 group)`) and `slot` the random the element keeps. gfx11
    (`acc_row = 2 r + klane`): elements `2 p` and `2 p + 1` share group `p`, slots `klane` and `2 + klane`. gfx12 (`acc_row = r + 8 klane`):
    elements `4 i .. 4 i + 3` share group `2 klane + i`, slots `r % 4`. A kernel evaluates it at klane 0 and 1 at trace time; a group that
    differs between the two is `g0 + (g1 - g0) klane`, a slot that differs is a `klane` select.
    """
    row = acc_row(r, klane, v8)
    return row // 4, row % 4


def dropout_mult(keep, scale):
    """The factor a kept element is multiplied by (`scale = 1 / (1 - p)`) and a dropped one by 0: `keep * scale`."""
    return keep * scale


def bwd_elements(p, dp, delta, mult):
    """Elementwise backward of one score: with `p` the softmax probability (before dropout), `dp = dO V^T` (raw), `delta = rowsum(dO * O)` of
    the row and `mult` the dropout factor (1 without dropout): returns `(p_dropped, ds)` = `(p * mult, p * (dp * mult - delta))`. `p_dropped`
    feeds dV, `ds` feeds dK, dQ and dB (dB = ds; dK and dQ carry the softmax scale at the epilogue).
    """
    return p * mult, p * (dp * mult - delta)
