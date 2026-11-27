# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); its sources are listed below.
#
# Provenance carried over from the sweep snapshot:
# Adapted from: https://github.com/ROCm/aiter
#   commit:   8253efc4059516f492df597b80ce5311e9fda8bd
#   path:     aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py
# via the gfx11 forward `attention_rdna/flash_attn_func_rdna.py` (ROCm/rocm-libraries PR #14111 @ d850254d45e),
# which supplies the WMMA ABI split (modification 1), the xor-16 exchange and the epilogue exchange re-expressed here as functions.

"""Layout layer of the gfx11 (RDNA3/3.5) and gfx12 (RDNA4) WMMA and the gfx90a / gfx942 (CDNA2/3) MFMA16 SDPA backward kernels: index algebra, lane exchanges, LDS indexing.

Every function uses plain integer arithmetic (`+ - * // %`, comparisons) on its index arguments, so the same call runs on Python ints
(`tools/test_rdna_layout.py`, numpy arrays of lane ids) and on FlyDSL values (kernels). No FlyDSL import. A kernel that needs a
data select passes a precomputed `klane_is_zero` predicate and an `exchange` callable; the kernel's `exchange` is
`lambda v: Float32(v).shuffle_xor(Int32(16), Int32(32))`, the model's is a lane permutation (`lane ^ 16`).

ABI token. Every `abi` argument is one of `ABI_V16` ("v16", gfx11 WMMA), `ABI_V8` ("v8", gfx12 WMMA) or `ABI_MFMA16` ("mfma16", gfx90a / gfx942
`v_mfma_f32_16x16x16{f16,bf16_1k}`); `abi_for(arch)` maps a knob-record arch string to it. `l` = lane, `klane = l // 16`, `l16 = l % 16`; a wave
is `warp_size(abi)` lanes (32 WMMA, 64 MFMA16: klane 0..3):

                            gfx11 (v16, wave32)                       gfx12 (v8, wave32)                         gfx90a/gfx942 (mfma16, wave64)
    A operand (row i)       lane l16 = i; all 16 K values;            lane l16 = i; 8 K values, K = 8 klane + t;  lane l16 = i; 4 K values, K = 4 klane + t
    B operand (col j)       lanes 16-31 duplicate lanes 0-15          lanes 16-31 carry the upper K half         lane l16 = j; 4 K values, K = 4 klane + t
    C/D, element r          row 2 r + klane (r 0..7), column l16      row r + 8 klane (r 0..7), column l16        row 4 klane + r (r 0..3), column l16

On MFMA16 the accumulator of a lane (n, g) holds D[4 g + r][n] for r = 0..3, which is exactly the B operand of lane (n, g) of the next GEMM
contracting over those rows: P / dS feed the next MFMA with no exchange.

Orientation of an S-type tile (S, P, dP, dS), the two the backward uses:

    Q_ROWS   rows = q, cols = kv.   dK/dV kernel. S = Q K^T: A = Q (lane row q), B = K^T (lane col kv). dV^T = dO^T P and
             dK^T = Q^T dS take P / dS as the B operand (K = q, col = kv) and dO^T / Q^T as the A operand (row = head dim,
             K = q; `lds_transposed_operand`). LSE and Delta are per element: row `acc_row(r, klane)` of the tile.
    KV_ROWS  rows = kv, cols = q.   dQ kernel (and dB). S^T = K Q^T: A = K, B = Q^T. dQ^T = K^T dS^T takes dS^T as the B operand
             (K = kv, col = q). LSE and Delta are per lane (column l16): one value per lane for all `acc_elems` elements.
    An accumulator that holds a transposed gradient (dV^T, dK^T, dQ^T) has rows = head dim, cols = kv (dK/dV) or q (dQ).

The `*_op` and `acc_*` names below are the layer's API; the contract per function is its docstring.
"""

WMMA_DIM = 16

ABI_V16 = "v16"
ABI_V8 = "v8"
ABI_MFMA16 = "mfma16"

Q_ROWS = "q_rows"
KV_ROWS = "kv_rows"


def abi_for(arch):
    """The ABI token of a knob-record arch string: `ABI_MFMA16` for gfx90a / gfx942, `ABI_V8` for gfx12*, `ABI_V16` for gfx11*."""
    arch = arch.lower()
    if arch.startswith(("gfx90a", "gfx942")):
        return ABI_MFMA16
    if arch.startswith("gfx12"):
        return ABI_V8
    if arch.startswith("gfx11"):
        return ABI_V16
    raise ValueError(f"no backward layout ABI for arch {arch!r}")


def warp_size(abi):
    """Lanes of a wave: 64 (MFMA16), 32 (WMMA)."""
    return 64 if abi == ABI_MFMA16 else 32


def acc_elems(abi):
    """f32 accumulator elements per lane of one 16x16 tile: 4 (MFMA16), 8 (WMMA)."""
    return 4 if abi == ABI_MFMA16 else 8


def op_width(abi):
    """K values one lane passes per A or B operand: 4 (MFMA16), 8 (gfx12), 16 (gfx11)."""
    return {ABI_MFMA16: 4, ABI_V8: 8, ABI_V16: 16}[abi]


def read_width(abi):
    """Elements of one global operand read (16 B, or 8 B on MFMA16): `min(op_width, 8)`. An operand is `op_width // read_width` reads (`max(., 1)` of them)."""
    return min(op_width(abi), 8)


def read_count(abi):
    """Operand reads (of `read_width` elements) a lane issues per operand: 2 (gfx11), 1 (gfx12, MFMA16)."""
    return max(op_width(abi) // 8, 1)


# -- lane decomposition -----------------------------------------------------------------------------------------------------------


def lane_col(lane):
    """`l16`: the operand row (A) / column (B) of a lane and the column of its accumulator elements."""
    return lane % 16


def lane_half(lane):
    """`klane`: lane // 16 (0 or 1 on a wave32, 0..3 on a wave64)."""
    return lane // 16


# -- A/B operands -----------------------------------------------------------------------------------------------------------------


def op_k0(klane, abi):
    """Offset of a lane's first operand element from the K-step origin: 0 (gfx11, all 16 K values), `8 klane` (gfx12), `4 klane` (MFMA16)."""
    if abi == ABI_MFMA16:
        return 4 * klane
    return 8 * klane if abi == ABI_V8 else 0


def op_origin(k_origin, klane, abi):
    """Index (head dim, q or kv) of the lane's first operand element for the K-step starting at `k_origin`; the lane's operand is
    `op_width(abi)` consecutive indices from here."""
    return k_origin + op_k0(klane, abi)


def lds_rowmajor_operand(base, stride, row, k_origin, klane, abi):
    """Element offset of the lane's operand read from a row-major LDS tile (K contiguous): `base + row * stride + op_origin`; the read is
    `op_width(abi)` contiguous elements. `row` is `lane_col(lane)` plus the tile row origin (A: M row, B: N column).
    """
    return base + row * stride + op_origin(k_origin, klane, abi)


def lds_transposed_operand(base, stride, col, k_origin, klane, abi):
    """Element offsets (a list of `op_width(abi)`, in K order) of the lane's operand read from a row-major LDS tile whose K axis is the ROW
    axis (the transposed A operand: Q^T, dO^T, K^T). Element t is `base + (op_origin + t) * stride + col`; `col` is `lane_col(lane)` plus the
    column (head dim) origin. The list is a Python-level constant length: iterate it with `range_constexpr`.
    """
    k0 = op_origin(k_origin, klane, abi)
    return [base + (k0 + t) * stride + col for t in range(op_width(abi))]


# -- accumulator coordinates ------------------------------------------------------------------------------------------------------


def acc_row(r, klane, abi):
    """Tile row of accumulator element `r` (0..`acc_elems`-1) of a lane: `2 r + klane` (gfx11), `r + 8 klane` (gfx12), `r + 4 klane` (MFMA16)."""
    if abi == ABI_MFMA16:
        return 4 * klane + r
    return 8 * klane + r if abi == ABI_V8 else 2 * r + klane


def acc_row_base(klane, abi):
    """Row of element 0; element `r` is at `acc_row_base + acc_row_step * r`."""
    if abi == ABI_MFMA16:
        return 4 * klane
    return 8 * klane if abi == ABI_V8 else klane


def acc_row_step(abi):
    """Row distance between consecutive elements of a lane: 1 (gfx12, MFMA16), 2 (gfx11)."""
    return 2 if abi == ABI_V16 else 1


def acc_klane_step(abi):
    """Rows a lane's `klane` adds to `acc_row(r, 0)`: 4 (MFMA16), 8 (gfx12), 1 (gfx11)."""
    return {ABI_MFMA16: 4, ABI_V8: 8, ABI_V16: 1}[abi]


def acc_col(lane):
    """Tile column of every accumulator element of a lane."""
    return lane % 16


def acc_coord(r, lane, abi):
    """(row, col) of accumulator element `r` of `lane`."""
    return acc_row(r, lane // 16, abi), lane % 16


def acc_q_kv(orient, r, lane, q0, kv0, abi):
    """(q, kv) of element `r` of `lane` in an S-type tile whose origin is (`q0`, `kv0`): `Q_ROWS` rows are q, `KV_ROWS` rows are kv."""
    row, col = acc_coord(r, lane, abi)
    return (q0 + row, kv0 + col) if orient == Q_ROWS else (q0 + col, kv0 + row)


def acc_q(orient, r, lane, q0, abi):
    """q index of element `r` of `lane`: where its LSE and Delta are read. `Q_ROWS`: per element (`q0 + acc_row`); `KV_ROWS`: `q0 + l16`, the same for all `r`."""
    return q0 + (acc_row(r, lane // 16, abi) if orient == Q_ROWS else lane % 16)


def acc_kv(orient, r, lane, kv0, abi):
    """kv index of element `r` of `lane` (`Q_ROWS`: `kv0 + l16`; `KV_ROWS`: `kv0 + acc_row`)."""
    return kv0 + (lane % 16 if orient == Q_ROWS else acc_row(r, lane // 16, abi))


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


def acc_to_operand(own, exchange, klane_is_zero, abi):
    """The B-operand values of a lane from its `acc_elems(abi)` accumulator values `own[r]` (tile row `acc_row(r)`, K = tile row): `op_width(abi)` values
    whose element k is tile row `k` (gfx11), `8 klane + k` (gfx12) or `4 klane + k` (MFMA16): the operand of the next GEMM contracting over the tile
    rows (P / dS as the B operand). gfx12 and MFMA16: `own` as it is. gfx11: the lane holds only the even (klane 0) or odd (klane 1) rows, its xor-16 peer the others;
    trade all 8 (`exchange(v)` returns the xor-16 peer's `v`) and interleave, so both lanes of a column hold all 16 rows (the duplication
    the hardware needs)."""
    if abi != ABI_V16:
        return list(own)
    peer = [exchange(v) for v in own]
    full = []
    for j in range(8):
        full.append(klane_select(klane_is_zero, own[j], peer[j]))
        full.append(klane_select(klane_is_zero, peer[j], own[j]))
    return full


def acc_to_rows8(own, exchange, klane_is_zero, abi):
    """The epilogue exchange: `acc_elems(abi)` values `rows[j]` = tile row `rows8_origin + j` of the lane's column, so each lane stores `acc_elems` contiguous elements of
    a transposed gradient (dV^T, dK^T, dQ^T rows = head dim) or of a dB row (`KV_ROWS` tile: rows = kv). gfx12 already holds rows
    `8 klane + r` (`own`); MFMA16 holds the 4 rows `4 klane + r` (`own`, 4 values: its store is 8 bytes). gfx11: trade with the xor-16 peer once
    (`exchange`), the forward's O store.
    """
    if abi != ABI_V16:
        return list(own)
    peer = [exchange(v) for v in own]
    rows = []
    for j in range(8):
        lo_src, hi_src = (own, peer) if j % 2 == 0 else (peer, own)
        rows.append(klane_select(klane_is_zero, lo_src[j // 2], hi_src[4 + j // 2]))
    return rows


def rows8_origin(klane, abi):
    """First tile row held by `acc_to_rows8` for a lane: `8 klane` (`4 klane` on MFMA16)."""
    return 4 * klane if abi == ABI_MFMA16 else 8 * klane


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
    tile all elements of a lane share the slot (`kv0 + l16`) and differ in counter (one call each). In a `KV_ROWS` tile (kv0 a multiple of
    16) gfx11 elements `r` and `r + 1` (even `r`) share a counter (`r // 2`; slots `2 (r % 2) + klane`), gfx12 elements `4 i .. 4 i + 3` share
    one (slots `r % 4`), MFMA16 elements `0 .. 3` of a lane share one (slots `r`), so one call can serve a group.
    """
    return plane_base + q * row_stride + kv // 4, kv % 4


def dropout_kv_rows_call(r, klane, abi):
    """`KV_ROWS` tile (kv0 a multiple of 16), accumulator element `r` of a lane of half `klane` (0 or 1): `(group, slot)` with `group` the Philox
    call's counter offset from `kv0 // 4` (the call at `dropout_counter(.., q, kv0 + 4 group)`) and `slot` the random the element keeps. gfx11
    (`acc_row = 2 r + klane`): elements `2 p` and `2 p + 1` share group `p`, slots `klane` and `2 + klane`. gfx12 (`acc_row = r + 8 klane`):
    elements `4 i .. 4 i + 3` share group `2 klane + i`, slots `r % 4`. MFMA16 (`acc_row = r + 4 klane`): all 4 elements share group `klane`, slots
    `r`. A kernel evaluates it at klane 0 and 1 at trace time (a lane's klane runs 0..3 on a wave64: the group is `g0 + (g1 - g0) klane` on every ABI, linear in klane); a slot that differs is a `klane` select.
    """
    row = acc_row(r, klane, abi)
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
