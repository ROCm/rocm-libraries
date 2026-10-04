# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free float64 scaled-dot-product-attention oracle.

Semantics follow the hipDNN CPU reference for the SDPA forward node:

* ``scores = (Q @ K^T) * scale`` then ``+ bias`` (bias is added AFTER scaling).
* The diagonal band keeps ``k`` iff
  ``(right < 0 or k <= q + offset + right) and (left < 0 or k >= q + offset - left)``
  with ``offset = 0`` for a top-left diagonal and ``s_kv - s_q`` for a
  bottom-right one. ``-1`` means unbounded; values below ``-1`` are invalid.
  ``left = W`` therefore keeps ``W + 1`` keys (the diagonal plus ``W`` to its left).
* Softmax is over the kept keys; ``LSE`` is the natural log of the softmax
  denominator (max-shifted: ``LSE = max + log(sum exp(s - max))``).
* A row with no kept key (fully masked by band, padding or an all ``-inf``
  bias row) yields ``O = 0`` and ``LSE = -inf``. This deliberately differs from
  ``dense_attention_reference``, which uses a finite sentinel.
* Query rows at or beyond the per-batch ``seq_len_q`` (padded layout) also
  yield ``O = 0`` and ``LSE = -inf``; keys at or beyond ``seq_len_kv`` are
  never attended. In the band, ``s_q``/``s_kv`` are the per-batch valid counts.

Layouts
-------
Dense (padded) layout: ``Q[B, Hq, Sq, D]``, ``K[B, Hk, Skv, D]``,
``V[B, Hv, Skv, Dv]`` -> ``O[B, Hq, Sq, Dv]``, ``LSE[B, Hq, Sq]``.
Ragged (THD) layout: ``Q[T_q, Hq, D]``, ``K[T_kv, Hk, D]``, ``V[T_kv, Hv, Dv]``
with ``q_offsets[B+1]`` / ``kv_offsets[B+1]`` token offsets -> ``O[T_q, Hq, Dv]``,
``LSE[T_q, Hq]``. Rows outside every batch segment are zero / ``-inf``.

GQA/MQA: ``Hq % Hk == 0`` and ``Hq % Hv == 0``; query head ``h`` uses KV head
``h // (Hq // Hk)`` (contiguous grouping).

Bias: rank 1-4, right-aligned against ``[B, Hq, Sq, Skv]``; any size-1 axis
broadcasts. In the ragged layout the bias is indexed by in-batch positions
and its Sq/Skv extents are the longest per-batch lengths.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["keep", "resolve_diagonal_band", "sdpa_reference"]


def resolve_diagonal_band(
    *,
    left_bound=None,
    right_bound=None,
    top_left=True,
    causal=False,
    causal_bottom_right=False,
):
    """Resolve user mask settings to ``(left_bound, right_bound, top_left)``.

    ``None`` bounds mean unbounded (-1). The deprecated causal booleans take
    precedence over explicit bounds and alignment: ``causal`` forces
    ``(-1, 0, top_left=True)`` and ``causal_bottom_right`` forces
    ``(-1, 0, top_left=False)``. Setting both booleans is an error.
    """
    left = -1 if left_bound is None else int(left_bound)
    right = -1 if right_bound is None else int(right_bound)
    if left < -1 or right < -1:
        raise ValueError(f"band bounds must be >= -1 (got left={left}, right={right})")
    if causal and causal_bottom_right:
        raise ValueError("causal and causal_bottom_right are mutually exclusive")
    if causal:
        return -1, 0, True
    if causal_bottom_right:
        return -1, 0, False
    return left, right, bool(top_left)


def keep(q, k, s_q, s_kv, diagonal, left_bound, right_bound):
    """True iff query ``q`` may attend key ``k``.

    ``diagonal`` is ``"top_left"`` / ``"bottom_right"`` (or a bool where True
    means top-left). Scalar arguments only.
    """
    if left_bound < -1 or right_bound < -1:
        raise ValueError("band bounds must be >= -1")
    if isinstance(diagonal, str):
        if diagonal not in ("top_left", "bottom_right"):
            raise ValueError(f"unknown diagonal alignment {diagonal!r}")
        top_left = diagonal == "top_left"
    else:
        top_left = bool(diagonal)
    offset = 0 if top_left else s_kv - s_q
    if right_bound >= 0 and k > q + offset + right_bound:
        return False
    if left_bound >= 0 and k < q + offset - left_bound:
        return False
    return True


def _band_mask(s_q, s_kv, top_left, left, right):
    """Vectorised ``keep`` over a ``[s_q, s_kv]`` grid."""
    qi = np.arange(s_q)[:, None]
    ki = np.arange(s_kv)[None, :]
    offset = 0 if top_left else s_kv - s_q
    m = np.ones((s_q, s_kv), dtype=bool)
    if right >= 0:
        m &= ki <= qi + offset + right
    if left >= 0:
        m &= ki >= qi + offset - left
    return m


def _expand_bias(bias, B, H, Sq, Skv):
    bias = np.asarray(bias, dtype=np.float64)
    if not 1 <= bias.ndim <= 4:
        raise ValueError("bias rank must be 1..4")
    bias = bias.reshape((1,) * (4 - bias.ndim) + bias.shape)
    want = (B, H, Sq, Skv)
    for got, w in zip(bias.shape, want):
        if got not in (1, w):
            raise ValueError(f"bias shape {bias.shape} not broadcastable to {want}")
    return np.broadcast_to(bias, want)


def _attend(q, k, v, scale, bias2d, band):
    """One (batch, head) problem. q[sq,D], k[skv,D], v[skv,Dv] -> (o, lse)."""
    sq, skv = q.shape[0], k.shape[0]
    if skv == 0:
        return np.zeros((sq, v.shape[1])), np.full(sq, -np.inf)
    s = (q @ k.T) * scale
    if bias2d is not None:
        s = s + bias2d
    s = np.where(band, s, -np.inf)
    mx = s.max(axis=1)
    live = np.isfinite(mx)
    safe = np.where(live, mx, 0.0)
    e = np.where(live[:, None], np.exp(s - safe[:, None]), 0.0)
    den = e.sum(axis=1)
    den1 = np.where(live, den, 1.0)
    o = (e / den1[:, None]) @ v
    lse = np.where(live, mx + np.log(den1), -np.inf)
    return o, lse


def sdpa_reference(
    q,
    k,
    v,
    *,
    scale=None,
    bias=None,
    left_bound=None,
    right_bound=None,
    top_left=True,
    causal=False,
    causal_bottom_right=False,
    seq_len_q=None,
    seq_len_kv=None,
    q_offsets=None,
    kv_offsets=None,
):
    """Float64 SDPA forward. Returns ``(O, LSE)``, both float64.

    ``LSE`` is natural-log (``-inf`` for empty rows) and can be cast to fp32.
    Provide ``q_offsets`` and ``kv_offsets`` for the ragged THD layout,
    otherwise inputs are dense 4-D with optional per-batch ``seq_len_q`` /
    ``seq_len_kv`` counts. ``scale`` defaults to ``1/sqrt(D)``.
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    left, right, tl = resolve_diagonal_band(
        left_bound=left_bound,
        right_bound=right_bound,
        top_left=top_left,
        causal=causal,
        causal_bottom_right=causal_bottom_right,
    )
    ragged = q_offsets is not None or kv_offsets is not None
    if ragged:
        if q_offsets is None or kv_offsets is None:
            raise ValueError("ragged layout needs both q_offsets and kv_offsets")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("ragged layout expects [T, H, D] tensors")
        if seq_len_q is not None or seq_len_kv is not None:
            raise ValueError("seq_len_* and offsets are mutually exclusive")
        qo = [int(x) for x in q_offsets]
        ko = [int(x) for x in kv_offsets]
        if len(qo) != len(ko) or len(qo) < 2:
            raise ValueError("offsets must have B+1 entries")
        B = len(qo) - 1
        Hq, D = q.shape[1], q.shape[2]
        Hk, Hv, Dv = k.shape[1], v.shape[1], v.shape[2]
        lens_q = [qo[b + 1] - qo[b] for b in range(B)]
        lens_kv = [ko[b + 1] - ko[b] for b in range(B)]
        if min(lens_q + lens_kv) < 0 or qo[-1] > q.shape[0] or ko[-1] > k.shape[0]:
            raise ValueError("offsets out of range")
        Smax_q, Smax_kv = max(lens_q), max(lens_kv)
        o = np.zeros((q.shape[0], Hq, Dv))
        lse = np.full((q.shape[0], Hq), -np.inf)
    else:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("dense layout expects [B, H, S, D] tensors")
        B, Hq, Smax_q, D = q.shape
        Hk, Hv, Dv = k.shape[1], v.shape[1], v.shape[3]
        Smax_kv = k.shape[2]
        lens_q = (
            [Smax_q] * B
            if seq_len_q is None
            else [int(x) for x in np.reshape(seq_len_q, -1)]
        )
        lens_kv = (
            [Smax_kv] * B
            if seq_len_kv is None
            else [int(x) for x in np.reshape(seq_len_kv, -1)]
        )
        if len(lens_q) != B or len(lens_kv) != B:
            raise ValueError("seq_len arrays must have one entry per batch")
        if any(not 0 <= n <= Smax_q for n in lens_q) or any(
            not 0 <= n <= Smax_kv for n in lens_kv
        ):
            raise ValueError("seq_len out of range")
        o = np.zeros((B, Hq, Smax_q, Dv))
        lse = np.full((B, Hq, Smax_q), -np.inf)
    if k.shape[-1] != D:
        raise ValueError("Q and K head dims differ")
    if Hq % Hk or Hq % Hv:
        raise ValueError("Hq must be a multiple of Hk and Hv")
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    bias_full = None if bias is None else _expand_bias(bias, B, Hq, Smax_q, Smax_kv)

    for b in range(B):
        sq, skv = lens_q[b], lens_kv[b]
        if sq == 0:
            continue
        band = _band_mask(sq, skv, tl, left, right)
        for h in range(Hq):
            hk, hv = h // (Hq // Hk), h // (Hq // Hv)
            bb = None if bias_full is None else bias_full[b, h, :sq, :skv]
            if ragged:
                qs = slice(qo[b], qo[b] + sq)
                ks = slice(ko[b], ko[b] + skv)
                o[qs, h], lse[qs, h] = _attend(
                    q[qs, h], k[ks, hk], v[ks, hv], scale, bb, band
                )
            else:
                o[b, h, :sq], lse[b, h, :sq] = _attend(
                    q[b, h, :sq], k[b, hk, :skv], v[b, hv, :skv], scale, bb, band
                )
    return o, lse
