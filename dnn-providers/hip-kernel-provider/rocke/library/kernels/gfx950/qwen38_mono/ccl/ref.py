# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU references of the CCL collectives (numpy, bit-exact).

Every device reduction adds the ranks' parts in f32 in rank order 0..N-1;
numpy float32 elementwise adds round the same way (IEEE round to nearest
even), so the references below reproduce the device bits exactly. bf16
rounding is round-to-nearest-even on the f32 bits (LLVM ``fptrunc`` to
``bfloat``); inputs here are finite.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "allgather_chain",
    "allreduce_chain",
    "argmax_merge",
    "bf16_to_f32",
    "LM_GROWTH",
    "chain_scales",
    "delay_units",
    "f32_to_bf16",
    "lse_merge",
    "topk_merge",
    "wire_to_f32",
]


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bits (uint16), round to nearest even (NaN -> quiet NaN)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    if nan.any():
        r = np.where(nan, ((u >> 16) | 0x40).astype(np.uint16), r)
    return r


def bf16_to_f32(h: np.ndarray) -> np.ndarray:
    return (np.asarray(h, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def wire_to_f32(x: np.ndarray, dtype: str) -> np.ndarray:
    return bf16_to_f32(x) if dtype == "bf16" else np.asarray(x, dtype=np.float32)


def f32_to_wire(x: np.ndarray, dtype: str) -> np.ndarray:
    return f32_to_bf16(x) if dtype == "bf16" else np.asarray(x, dtype=np.float32)


def chain_scales(ranks: int, growth: float = 1.0) -> List[float]:
    """Per-rank factors of the all-reduce chain: ``growth (r + 1) / T`` with
    ``T = N (N + 1) / 2`` (as f32). The factors sum to ``growth``, so collective
    k + 1's sum is ``growth`` times collective k's.

    ``growth = 1`` (the default, W3's chain) keeps the magnitude bounded, but the
    chain then reaches a BITWISE fixed point after 2-3 collectives: a build that
    reads the previous collective's partial (wrong slot, read-before-publish)
    reproduces the reference. A non-contracting ``growth`` (CCL-LM checks use
    :data:`LM_GROWTH`) changes every collective's values, so such a read is a
    mismatch; keep ``growth ** iters`` far from the f32 / bf16 range limit
    (1.25 ** 96 is about 2e9)."""
    t = ranks * (ranks + 1) / 2
    return [float(np.float32(growth * (r + 1) / t)) for r in range(ranks)]


# non-contracting chain growth of the CCL-LM numerics gates (bench_ccl_lm, lm_* suite tests)
LM_GROWTH = 1.25


def rank_sum(parts: Sequence[np.ndarray], reverse: bool = False) -> np.ndarray:
    order = list(range(len(parts)))
    if reverse:
        order.reverse()
    acc = np.array(parts[order[0]], dtype=np.float32, copy=True)
    for r in order[1:]:
        acc = (acc + np.asarray(parts[r], dtype=np.float32)).astype(np.float32)
    return acc


def allreduce_chain(
    xs: Sequence[np.ndarray], dtype: str, iters: int, scales: Sequence[float], twoshot: bool = False, *, wire: str = "", canon: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """``xs[r]``: rank r's data-dtype input of the first collective (bf16 as
    uint16 bits). Returns (the last collective's f32 output, every rank's next
    input in the data dtype).

    ``wire`` (CCL-LM ``ar_dtype``): the dtype of the partials on the wire when
    it differs from ``dtype`` (``"bf16"`` for f32 data: each partial is
    rounded to bf16 before the f32 rank-order sum, the two-shot all-gather
    carries bf16 too). ``canon`` (lamport form): wire words with the empty
    pattern (bf16 ``0xFFFF``, f32 ``0xFFFFFFFF``) are replaced by the quiet
    NaN the producer sends (``0x7FC0`` / ``0x7FC00000``)."""
    wire = wire or dtype
    x = [np.asarray(v) for v in xs]
    out = None
    for _ in range(iters):
        ws = [f32_to_wire(wire_to_f32(v, dtype), wire) if wire != dtype else v for v in x]
        if canon:
            ws = [_lamport_canon(w, wire) for w in ws]
        s = rank_sum([wire_to_f32(w, wire) for w in ws])
        out = wire_to_f32(f32_to_wire(s, wire), wire) if twoshot else s
        x = [f32_to_wire((out * np.float32(sc)).astype(np.float32), dtype) for sc in scales]
    return out, x


def _lamport_canon(w: np.ndarray, wire: str) -> np.ndarray:
    w = np.array(w, copy=True)
    if wire == "bf16":
        u = w.view(np.uint16)
        u[u == 0xFFFF] = 0x7FC0
    else:
        u = w.view(np.uint32)
        u[u == 0xFFFFFFFF] = 0x7FC00000
    return w


def allgather_chain(xs: Sequence[np.ndarray], iters: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """``xs[r]``: rank r's (rows, cols) slice. Output of the last collective:
    (rows, N cols) concatenation; the next input of rank r is slice r + 1."""
    x = [np.asarray(v) for v in xs]
    n = len(x)
    out = None
    for _ in range(iters):
        out = np.concatenate(x, axis=1)
        x = [x[(r + 1) % n] for r in range(n)]
    return out, x


def argmax_merge(vals: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``vals[r, row]`` / ``ids[r, row]``: rank r's max and its global id.
    Rank-order merge: a larger value wins, an equal value takes the lower id."""
    vals = np.asarray(vals, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int32)
    bv, bi = vals[0].copy(), ids[0].copy()
    for r in range(1, vals.shape[0]):
        better = (vals[r] > bv) | ((vals[r] == bv) & (ids[r] < bi))
        bv = np.where(better, vals[r], bv)
        bi = np.where(better, ids[r], bi)
    return bv, bi


def lse_merge(m: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-rank (max, sum exp(x - max)) -> global (M, S) in rank order:
    ``M = max_r m_r``, ``S = sum_r s_r exp(m_r - M)`` (f32)."""
    m = np.asarray(m, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    M = m.max(axis=0)
    S = np.zeros_like(M)
    for r in range(m.shape[0]):
        S = (S + (s[r] * np.exp(m[r] - M).astype(np.float32)).astype(np.float32)).astype(np.float32)
    return M, S


def topk_merge(vals: np.ndarray, ids: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """``vals[r, row, :K']`` / ``ids``: per-rank local top-K' -> global top-k
    by (value desc, global id asc)."""
    vals = np.asarray(vals, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    R, rows, kk = vals.shape
    v = vals.transpose(1, 0, 2).reshape(rows, R * kk)
    i = ids.transpose(1, 0, 2).reshape(rows, R * kk)
    ov = np.empty((rows, k), np.float32)
    oi = np.empty((rows, k), np.int64)
    for row in range(rows):
        order = np.lexsort((i[row], -v[row]))[:k]
        ov[row], oi[row] = v[row, order], i[row, order]
    return ov, oi


def delay_units(salt: int, gw: int, use: int, point: int, mx: int) -> int:
    """Host twin of the hooks' delay hash (``s_sleep 1`` units)."""
    m = 0xFFFFFFFF
    h = (salt ^ ((gw << 12) & m) ^ ((use << 2) & m)) & m
    h ^= point
    for sh, left in ((13, True), (17, False), (5, True), (13, True), (17, False), (5, True)):
        h = (h ^ ((h << sh) & m)) if left else (h ^ (h >> sh))
        h &= m
    return (h & 0x7FFFFFFF) % (mx + 1)
