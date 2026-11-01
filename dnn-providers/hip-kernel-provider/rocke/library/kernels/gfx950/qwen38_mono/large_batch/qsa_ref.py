# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host references for probe P-L5 (``qsa_lb``): float64 math on the device's own inputs, numpy only.

Every check takes the device buffers after one launch and recomputes each stage from that stage's device inputs, so a
failure names the stage: heads (q / k norm + RoPE, the KV append), the indexer key (ring append or block completion:
mean, norm, RoPE at the block start, E4M3 codes + power-of-two scale), the indexer query, the scores, the selection
(exact: recomputed from the DEVICE scores with FlyTape's tie rule), and the gated attention output (from the device's
q, KV cache and selection). Tolerances are relative to the per-(row, head) scale of the quantity, set by the storage
precision (bf16 outputs, bf16 P in the PV product, E4M3 keys) with margin, and every must-fail mutant of ``qsa_lb``
breaks at least one of them by a wide factor (``pl5_bench.py`` numerics asserts both directions).
"""

from __future__ import annotations

import numpy as np

HD, ROT, IH, IHN, RATIO, NBLK = 256, 64, 128, 4, 4, 512
EPS = 1e-6
LOG2E = 1.4426950408889634


# ------------------------------------------------------------------ number formats


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """RNE f32 -> bf16 bits (uint16); NaN not expected."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def _e4m3_table() -> np.ndarray:
    t = np.zeros(256, np.float64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e = (c >> 3) & 0xF
        m = c & 7
        if e == 15 and m == 7:
            t[c] = np.nan
        elif e == 0:
            t[c] = s * (m / 8.0) * 2.0 ** -6
        else:
            t[c] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3 = _e4m3_table()


def e4m3_decode(codes_u8: np.ndarray) -> np.ndarray:
    return E4M3[codes_u8.astype(np.int64)]


def key_scale(amax: float) -> tuple[float, float]:
    """FlyTape ``qsa_state._key_scale`` (power of two from the exponent bits of the f32 amax)."""
    a = np.float32(max(float(amax), 1e-4))
    bits = int(np.array([a], np.float32).view(np.uint32)[0])
    floor = int(np.array([1e-4], np.float32).view(np.uint32)[0])
    bits = max(bits, floor)
    e = bits >> 23
    up = 1 if (bits & 0x7FFFFF) > 0x600000 else 0
    k = e - 127 - 8 + up
    return 2.0 ** k, 2.0 ** -k


def rope_table(ctx: int, theta: float = 10_000_000.0) -> np.ndarray:
    """``[ctx, 64]`` f32: cos then sin of the fp32 angles (FlyTape ``model.rope_table``)."""
    inv = (1.0 / (np.float32(theta) ** (np.arange(0, ROT, 2, dtype=np.float32) / np.float32(ROT)))).astype(np.float32)
    pos = np.arange(ctx, dtype=np.float32)[:, None]
    fr = (pos * inv[None, :]).astype(np.float32).astype(np.float64)
    return np.concatenate([np.cos(fr), np.sin(fr)], axis=1).astype(np.float32)


# ------------------------------------------------------------------ stages


def norm_rope(x: np.ndarray, w: np.ndarray, rope_row: np.ndarray, rotate: bool = True) -> np.ndarray:
    """Zero-centred RMSNorm (``x rrms (1 + w)``) then rotate-half RoPE on the first 64 dims (f64)."""
    x = x.astype(np.float64)
    n = x.shape[-1]
    rr = 1.0 / np.sqrt((x * x).sum(-1, keepdims=True) / n + EPS)
    y = x * rr * (1.0 + w.astype(np.float64))
    if not rotate:
        return y
    cos = rope_row[: ROT // 2].astype(np.float64)
    sin = rope_row[ROT // 2:].astype(np.float64)
    lo, hi = y[..., : ROT // 2].copy(), y[..., ROT // 2: ROT].copy()
    out = y.copy()
    out[..., : ROT // 2] = lo * cos - hi * sin
    out[..., ROT // 2: ROT] = hi * cos + lo * sin
    return out


def indexer_block_key(raw3: np.ndarray, new: np.ndarray, wik: np.ndarray, rope_start: np.ndarray):
    """The completed block's compressed key: f64 mean of the 4 raw keys (bf16 values), norm, RoPE; and its
    reference E4M3 quantization (scale, ref value per dim)."""
    tot = new.astype(np.float64) + raw3.astype(np.float64).sum(0)
    mean = tot / RATIO
    k = norm_rope(mean, wik, rope_start)
    scale, _inv = key_scale(np.abs(k).max())
    return k, scale


def scores_ref(iq: np.ndarray, kc_codes: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """``[blocks]`` scores: sum over the 4 heads of ReLU(q_h . key) x scale / sqrt(128); keys as stored (E4M3 codes,
    ``kc_codes`` uint8 ``[blocks, 128]``)."""
    keys = e4m3_decode(kc_codes)  # [blocks, 128]
    dots = keys @ iq.astype(np.float64).reshape(IHN, IH).T  # [blocks, 4]
    return np.maximum(dots, 0.0).sum(1) * ks.astype(np.float64) * IH ** -0.5


def select_ref(scores_f32: np.ndarray, blocks: int) -> np.ndarray:
    """FlyTape ``_select_blocks`` from the device's f32 scores: 16-bit keys (bits 30..15), the 512 largest, ties to
    the lower block, ascending."""
    s = np.ascontiguousarray(scores_f32[:blocks], np.float32).view(np.uint32).astype(np.int64)
    key = s >> 15
    order = np.lexsort((np.arange(blocks), -key))  # key descending, block ascending
    return np.sort(order[:NBLK])


def attention_ref(q: np.ndarray, kv_k: np.ndarray, kv_v: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Gated attention of one kv head's q heads ``q [g, 256]`` over tokens ``kv_k / kv_v [n, 256]`` (f64):
    ``softmax(q k^T / 16) v * sigmoid(gate)``."""
    s = q.astype(np.float64) @ kv_k.astype(np.float64).T / 16.0
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    o = (p @ kv_v.astype(np.float64)) / p.sum(1, keepdims=True)
    return o / (1.0 + np.exp(-gate.astype(np.float64)))


def positions(pos: int, sel: np.ndarray | None) -> np.ndarray:
    """The row's attended token positions: every token (within the budget) or the selected blocks' 4 tokens plus the
    incomplete tail (FlyTape ``_window`` / ``_pos_of``)."""
    blocks = (pos + 1) // RATIO
    tail = (pos + 1) % RATIO
    if blocks <= NBLK:
        return np.arange(pos + 1)
    toks = (sel[:, None] * RATIO + np.arange(RATIO)[None, :]).reshape(-1)
    return np.concatenate([toks, blocks * RATIO + np.arange(tail)])


def rel_err(dev: np.ndarray, ref: np.ndarray, axis=-1) -> float:
    """max |dev - ref| / max |ref| per slice along ``axis``, the worst slice."""
    d = np.abs(dev.astype(np.float64) - ref.astype(np.float64)).max(axis)
    m = np.abs(ref.astype(np.float64)).max(axis)
    return float((d / np.maximum(m, 1e-30)).max())
