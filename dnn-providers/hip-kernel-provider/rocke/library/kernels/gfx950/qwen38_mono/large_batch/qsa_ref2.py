# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host references for WP-L2e (``qsa_lb2`` with ``sem="atom"``): ATOM's QSA semantics in numpy.

Two kinds of reference:

* **Exact emulations** of ATOM's arithmetic where the LB kernel claims ATOM's bits (``qsa.py`` Q2, ``qsa_fast.py``
  Q3, Q4's rule): the q / k heads (direct-scale RMSNorm with the in-lane pair tree, the natural butterfly and the
  f64 ``inv``; RoPE from bf16 tables with ATOM's roundings), the indexer query and block keys (zero-centred norm,
  ``_rope16``, the pooled mean), the scores (f64 sums of the exact f32 products, ATOM's butterfly order; relu; the
  head pair tree; the IEEE divide by fl(sqrt(128))) and the selection (top ``min(blocks, 512)`` by the 31-bit score,
  ties to the lower block). Every f32 operation is a numpy float32 operation (IEEE, round to nearest even), every f64
  one a float64 operation, in ATOM's order.
* **The f64 golden** of the gated attention (``softmax(q k^T / 16) v * sigmoid(gate)`` over the attended list) and an
  **ATOM-precision model** of its output (f32 scores, f32 ``e``, ``p = bf16(e / den)``, f32 PV, ``bf16(bf16(o) sig)``):
  the LB attention is MFMA (tolerance class), so it is checked against the golden with a bound and its drift is
  reported next to the model's (DESIGN 12.4's T0 style).
"""

from __future__ import annotations

import numpy as np

from .qsa_ref import bf16_to_f32, f32_to_bf16, rel_err  # noqa: F401  (re-exported for the bench)

HD, ROT, IH, IHN, RATIO, NBLK, PAGE = 256, 64, 128, 4, 4, 512, 16
EPS32 = np.float32(np.array([0x358637BD], np.uint32).view(np.float32)[0])  # fl(1e-6)
SQRT128 = np.float32(np.array([0x413504F3], np.uint32).view(np.float32)[0])
F32 = np.float32


def bf16r(x) -> np.ndarray:
    """f32 -> bf16 (RNE) -> f32."""
    return bf16_to_f32(f32_to_bf16(np.asarray(x, np.float32)))


def butterfly(v: np.ndarray, offs=(1, 2, 4, 8, 16, 32)) -> np.ndarray:
    """XOR butterfly sum over the last axis (64 lanes) in ``offs`` order; every lane ends with the same bits."""
    v = v.copy()
    lanes = np.arange(v.shape[-1])
    for o in offs:
        v = (v + v[..., lanes ^ o]).astype(v.dtype)
    return v


def rope_tables(npos: int, theta: float = 10_000_000.0) -> tuple[np.ndarray, np.ndarray]:
    """ATOM-style bf16 ``cos`` / ``sin`` tables ``[npos, 64]`` (rotate-half: dim d uses frequency d % 32), as uint16."""
    inv = (1.0 / (np.float32(theta) ** (np.arange(0, ROT, 2, dtype=np.float32) / np.float32(ROT)))).astype(np.float32)
    pos = np.arange(npos, dtype=np.float32)[:, None]
    fr = (pos * inv[None, :]).astype(np.float32).astype(np.float64)
    fr = np.concatenate([fr, fr], axis=1)
    return f32_to_bf16(np.cos(fr).astype(np.float32)), f32_to_bf16(np.sin(fr).astype(np.float32))


def _inv(ss: np.ndarray, r: float) -> np.ndarray:
    stab = (ss * F32(r)).astype(F32) + EPS32
    return (1.0 / np.sqrt(stab.astype(np.float64))).astype(F32)


def norm_head(x: np.ndarray, w: np.ndarray, zero_centered: bool = False) -> np.ndarray:
    """ATOM ``qsa.norm`` of a 256-dim head held 4 dims per lane: ``bf16((x inv) w)`` (or ``1 + w``)."""
    x = x.astype(F32).reshape(64, 4)
    sq = ((x[:, 0] * x[:, 0]).astype(F32) + (x[:, 1] * x[:, 1]).astype(F32)).astype(F32) + \
        ((x[:, 2] * x[:, 2]).astype(F32) + (x[:, 3] * x[:, 3]).astype(F32)).astype(F32)
    ss = butterfly(sq.astype(F32))[0]
    inv = _inv(np.asarray(ss, F32), 1.0 / 256.0)
    ws = w.astype(F32).reshape(64, 4)
    if zero_centered:
        ws = (F32(1.0) + ws).astype(F32)
    return bf16r(((x * inv).astype(F32) * ws).astype(F32)).reshape(-1)


def norm_index(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ATOM's zero-centred norm of a 128-dim vector held 2 dims per lane (``R128``)."""
    x = x.astype(F32).reshape(64, 2)
    sq = ((x[:, 0] * x[:, 0]).astype(F32) + (x[:, 1] * x[:, 1]).astype(F32)).astype(F32)
    ss = butterfly(sq)[0]
    inv = _inv(np.asarray(ss, F32), 1.0 / 128.0)
    ws = (F32(1.0) + w.astype(F32).reshape(64, 2)).astype(F32)
    return bf16r(((x * inv).astype(F32) * ws).astype(F32)).reshape(-1)


def rope(y: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray, mutant: bool = False) -> np.ndarray:
    """ATOM's RoPE on bf16 values: dims < 64 rotate with the partner 32 dims away (``-partner`` below 32):
    ``bf16(bf16(y c) +- bf16(partner s))``; ``cos_row`` / ``sin_row`` are the table rows (bf16 values as f32)."""
    y = y.astype(F32)
    out = y.copy()
    d = np.arange(ROT)
    partner = y[d] if mutant else y[d ^ 32]
    t1 = bf16r((y[:ROT] * cos_row.astype(F32)).astype(F32))
    t2 = bf16r((partner * sin_row.astype(F32)).astype(F32))
    out[:ROT] = bf16r(np.where(d < 32, (t1 - t2).astype(F32), (t1 + t2).astype(F32)))
    return out


def head(x, w, cos_row, sin_row, zero_centered=False, rot=True):
    y = norm_head(x, w, zero_centered)
    return rope(y, cos_row, sin_row) if rot else y


def index_query(x, w, cos_row, sin_row, mutant=False):
    return rope(norm_index(x, w), cos_row, sin_row, mutant)


def block_key(raw4: np.ndarray, w, cos_row, sin_row) -> np.ndarray:
    """``qsa_fast._block_key``: the 4 raw rows pooled ``bf16(((r0 + r1) + (r2 + r3)) / 4)``, normed, rotated."""
    r = raw4.astype(F32)
    pooled = bf16r((((r[0] + r[1]).astype(F32) + (r[2] + r[3]).astype(F32)).astype(F32) * F32(0.25)).astype(F32))
    return rope(norm_index(pooled, w), cos_row, sin_row)


def scores_exact(iq: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """ATOM's ``_score`` of every block: per head the f64 butterfly (32, 16, .., 1) of the lanes' f64 sums of their
    two f32 products, rounded to f32, relu; ``((r0 + r1) + (r2 + r3)) / fl(sqrt(128))`` in f32. ``iq`` [4, 128],
    ``keys`` [n, 128] (bf16 values as f32)."""
    k = keys.astype(F32).reshape(-1, 64, 2)
    rel = []
    for h in range(IHN):
        qh = iq[h].astype(F32).reshape(64, 2)
        part = (k[:, :, 0] * qh[None, :, 0]).astype(F32).astype(np.float64) + \
            (k[:, :, 1] * qh[None, :, 1]).astype(F32).astype(np.float64)
        s = butterfly(part, (32, 16, 8, 4, 2, 1))[:, 0].astype(F32)
        rel.append(np.maximum(s, F32(0.0)).astype(F32))
    tot = ((rel[0] + rel[1]).astype(F32) + (rel[2] + rel[3]).astype(F32)).astype(F32)
    return (tot / SQRT128).astype(F32)


def scores_f64(iq: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """The same scores in f64 throughout (the golden the tolerance arms are measured against)."""
    d = keys.astype(np.float64) @ iq.astype(np.float64).T
    return np.maximum(d, 0.0).sum(1) / float(SQRT128)


def select(scores_f32: np.ndarray, blocks: int, key16: bool = False) -> np.ndarray:
    """Top ``min(blocks, 512)`` blocks by the 31-bit f32 score (``key16``: FlyTape's 16-bit keys), ties to the lower
    block, ascending."""
    s = np.ascontiguousarray(scores_f32[:blocks], np.float32).view(np.uint32).astype(np.int64)
    key = (s >> 15) if key16 else (s & 0x7FFFFFFF)
    order = np.lexsort((np.arange(blocks), -key))
    return np.sort(order[:NBLK])


def positions(pos: int, sel: np.ndarray | None) -> np.ndarray:
    blocks = (pos + 1) // RATIO
    tail = (pos + 1) % RATIO
    if blocks <= NBLK:
        return np.arange(pos + 1)
    toks = (sel[:, None] * RATIO + np.arange(RATIO)[None, :]).reshape(-1)
    return np.concatenate([toks, blocks * RATIO + np.arange(tail)])


def attention_golden(q: np.ndarray, k: np.ndarray, v: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """f64 ``softmax(q k^T / 16) v * sigmoid(gate)``; ``q`` [g, 256], ``k`` / ``v`` [n, 256], ``gate`` [g, 256]."""
    s = q.astype(np.float64) @ k.astype(np.float64).T / 16.0
    s -= s.max(1, keepdims=True)
    p = np.exp(s)
    o = (p @ v.astype(np.float64)) / p.sum(1, keepdims=True)
    return o / (1.0 + np.exp(-gate.astype(np.float64)))


def attention_atom_model(q: np.ndarray, k: np.ndarray, v: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """A model of ATOM's QSA precision (Q5-Q7): f32 scores x 1/16, f32 ``e = exp(s - m)``, an f32 denominator,
    ``p = bf16(e / den)``, an f32 PV sum, ``bf16(bf16(o) bf16(sigmoid(g)))``. Not bitwise (ATOM's summation orders are
    not reproduced); the T0 drift ratio compares the LB output's error to this model's."""
    s = (q.astype(np.float64) @ k.astype(np.float64).T).astype(F32)
    s = (s * F32(1.0 / 16.0)).astype(F32)
    m = s.max(1, keepdims=True)
    e = np.exp((s - m).astype(np.float64)).astype(F32)
    den = e.astype(np.float64).sum(1, keepdims=True).astype(F32)
    p = bf16r((e * (F32(1.0) / den).astype(F32)).astype(F32))
    a = (p.astype(np.float64) @ v.astype(np.float64)).astype(F32)
    sig = bf16r((F32(1.0) / (F32(1.0) + np.exp(-gate.astype(np.float64)).astype(F32))).astype(F32))
    return bf16r((bf16r(a) * sig).astype(F32))


def drift(out: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    """(max abs error / max |gold|, relative L2) of one (row, kv head) block."""
    d = out.astype(np.float64) - gold
    return float(np.abs(d).max() / max(np.abs(gold).max(), 1e-30)), float(np.linalg.norm(d) /
                                                                         max(np.linalg.norm(gold), 1e-30))
