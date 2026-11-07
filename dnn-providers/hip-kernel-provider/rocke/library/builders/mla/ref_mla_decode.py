# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numerical reference for weight-absorbed MLA decode.

Test oracle for the ``mla_decode`` kernel family, beside the prefill oracle in
:mod:`builders.mla.ref_mla_attn` whose conventions it shares (``[in, out]``
weights, paged cache, host-supplied ``scale``, natural-log LSE). numpy only.

Decode is one query per sequence (``S_q in {0, 1}``), weight-absorbed: the
query arrives as ``Qa = [Q_nope @ W_UK_nopeᵀ | Q_rope]`` (``[total_q, H_q,
r_KV + d_rope]``), every head attends to the same 576-wide key row
``[c_kv | k_rope]``, and the output is the normalized *latent*
``[total_q, H_q, r_KV]``. ``W_UV`` is applied afterwards, outside the kernel.
At ``S_q = 1`` the bottom-right causal mask admits every key, so decode has no
mask beyond the sequence's own length. Everything here computes in fp64: an
oracle for a bf16 kernel should not add rounding of its own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from builders.mla.ref_mla_attn import gather_paged_kv, mla_scale_log2

# Ordered by pipeline stage, as in ref_mla_attn: absorb -> latent -> partials
# -> merge -> whole op.
__all__ = [  # noqa: RUF022
    "absorb_query",
    "ref_mla_decode_latent",
    "ref_mla_decode_partials",
    "ref_mla_decode_merge",
    "ref_mla_decode_absorb",
]


def _require_scale(scale: float) -> None:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"scale must be a finite positive float, got {scale!r}. It is host-supplied "
            "and must be the same value the kernel receives; do not derive it here. "
            "A non-positive value is the signature of passing log2(scale) where the "
            "linear scale was wanted -- see mla_scale_log2 for the ABI's base-2 form."
        )


def _validate_decode_packing(
    total_q: int, cu_seqlens_q: Sequence[int], cu_seqlens_k: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Packed-varlen offsets of a decode batch: at most one query per sequence.

    Decode is not ``total_q == num_seqs``: a packed batch may hold sequences
    with no query at all, which is why the offsets are taken rather than a
    rectangular ``[B, H, ...]`` query.
    """
    cu_q = np.asarray(cu_seqlens_q, dtype=np.int64)
    cu_k = np.asarray(cu_seqlens_k, dtype=np.int64)
    if cu_q.ndim != 1 or cu_q.shape != cu_k.shape:
        raise ValueError(
            f"cu_seqlens_q {cu_q.shape} and cu_seqlens_k {cu_k.shape} must both be "
            "1-D [B+1] arrays describing the same batch"
        )
    if cu_q[0] != 0 or cu_k[0] != 0:
        raise ValueError("cu_seqlens arrays must start at 0")
    q_lens = np.diff(cu_q)
    if np.any(q_lens < 0) or np.any(np.diff(cu_k) < 0):
        raise ValueError("cu_seqlens arrays must be non-decreasing")
    if np.any(q_lens > 1):
        b = int(np.argmax(q_lens > 1))
        raise ValueError(
            f"decode takes at most one query per sequence; sequence {b} has "
            f"S_q={int(q_lens[b])}. Use ref_mla_prefill for S_q > 1."
        )
    if int(cu_q[-1]) != total_q:
        raise ValueError(
            f"cu_seqlens_q[-1]={int(cu_q[-1])} does not match total_q={total_q}"
        )
    return cu_q, cu_k


def absorb_query(
    q_nope: np.ndarray, q_rope: np.ndarray, w_uk: np.ndarray
) -> np.ndarray:
    """``Qa = [Q_nope @ W_UK[:, :, :d_nope]ᵀ | Q_rope]``, the decode kernel's query.

    ``q_nope`` is ``[T, H_q, d_nope]`` and ``q_rope`` ``[T, H_q, d_rope]``, the
    projected query with RoPE already applied (``W_UQ`` and the rotation stay
    upstream of decode, as they do for prefill). Returns
    ``[T, H_q, r_KV + d_rope]`` in fp64. Two-stage: the weight product is never
    materialized as a ``[H_q, d_q, r_KV]`` matrix.
    """
    if w_uk.ndim != 3:
        raise ValueError(
            f"W_UK must be per head [H_q, r_KV, d_nope + d_V] (3-D); got shape "
            f"{w_uk.shape}. Dropping the head axis is not a simplification, it is a "
            "different operator: the shared latent expands to a different K/V for "
            "every head."
        )
    if q_nope.ndim != 3 or q_rope.ndim != 3 or q_nope.shape[:2] != q_rope.shape[:2]:
        raise ValueError(
            f"q_nope {q_nope.shape} and q_rope {q_rope.shape} must be [T, H_q, *] "
            "with matching leading axes"
        )
    d_nope = q_nope.shape[-1]
    if w_uk.shape[0] != q_nope.shape[1]:
        raise ValueError(
            f"W_UK head count {w_uk.shape[0]} does not match the query's {q_nope.shape[1]}"
        )
    if w_uk.shape[2] <= d_nope:
        raise ValueError(
            f"W_UK output width {w_uk.shape[2]} must exceed d_nope={d_nope}; it is "
            "d_nope + d_V."
        )
    qa_nope = np.einsum(
        "thn,hrn->thr",
        q_nope.astype(np.float64),
        w_uk[:, :, :d_nope].astype(np.float64),
        optimize=True,
    )
    return np.concatenate([qa_nope, q_rope.astype(np.float64)], axis=-1)


def _decode_keys(
    qa: np.ndarray,
    c_kv: np.ndarray,
    k_rope: np.ndarray,
    block_table: np.ndarray,
    cu_seqlens_q: Sequence[int],
    cu_seqlens_k: Sequence[int],
) -> list[tuple[int, np.ndarray]]:
    """``(token, keys [S_k, r_KV + d_rope])`` per sequence that holds a query.

    The key row is the absorbed form's: the latent ``c_kv`` and the head-shared
    ``k_rope`` side by side, one row for every head.
    """
    if qa.ndim != 3:
        raise ValueError(
            f"qa must be [total_q, H_q, r_KV + d_rope] (3-D); got shape {qa.shape}"
        )
    if c_kv.ndim != 3:
        raise ValueError(
            f"c_kv must be paged [num_blocks, block_size, r_KV] (3-D); got {c_kv.shape}"
        )
    if k_rope.ndim != 3:
        raise ValueError(
            f"k_rope must be paged [num_blocks, block_size, d_rope] (3-D); got shape "
            f"{k_rope.shape}. k_rope is head-shared -- it is the only part of K that "
            "is, and giving it a head axis is a different operator."
        )
    if k_rope.shape[:2] != c_kv.shape[:2]:
        raise ValueError(
            f"k_rope paging {k_rope.shape[:2]} must match c_kv paging {c_kv.shape[:2]}: "
            "both index the same cache blocks."
        )
    width = c_kv.shape[2] + k_rope.shape[2]
    if qa.shape[2] != width:
        raise ValueError(
            f"qa width {qa.shape[2]} must be r_KV + d_rope = {width}. A 192-wide query "
            "is the un-absorbed one; run absorb_query first."
        )
    cu_q, cu_k = _validate_decode_packing(qa.shape[0], cu_seqlens_q, cu_seqlens_k)
    per_seq = gather_paged_kv(c_kv, k_rope, block_table, cu_k)
    out = []
    for b, (latent, rope) in enumerate(per_seq):
        if cu_q[b + 1] == cu_q[b]:
            continue
        keys = np.concatenate([latent, rope], axis=-1).astype(np.float64)
        out.append((int(cu_q[b]), keys))
    return out


def ref_mla_decode_latent(
    qa: np.ndarray,
    c_kv: np.ndarray,
    k_rope: np.ndarray,
    block_table: np.ndarray,
    cu_seqlens_q: Sequence[int],
    cu_seqlens_k: Sequence[int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The decode kernel's contract: ``(latent [T, H_q, r_KV], lse [T, H_q])``.

    ``latent[t, h] = softmax_k(scale * qa[t, h] . key[k]) @ c_kv[k]`` -- the
    normalized latent, before ``W_UV``. ``lse`` is natural-log over the scaled
    scores. A sequence with ``S_k = 0`` yields a zero latent and ``lse = -inf``.
    Single pass, natural-log domain: deliberately not built on
    :func:`ref_mla_decode_merge`, so that the split-KV merge is checked against an
    independent computation.
    """
    _require_scale(scale)
    r_kv = c_kv.shape[2]
    latent = np.zeros(qa.shape[:2] + (r_kv,), dtype=np.float64)
    lse = np.full(qa.shape[:2], -np.inf, dtype=np.float64)
    for token, keys in _decode_keys(
        qa, c_kv, k_rope, block_table, cu_seqlens_q, cu_seqlens_k
    ):
        if keys.shape[0] == 0:
            continue
        s = scale * (qa[token].astype(np.float64) @ keys.T)  # [H, S_k]
        m = s.max(axis=-1, keepdims=True)
        p = np.exp(s - m)
        denom = p.sum(axis=-1)
        latent[token] = (p @ keys[:, :r_kv]) / denom[:, None]
        lse[token] = m[:, 0] + np.log(denom)
    return latent, lse


def ref_mla_decode_partials(
    qa: np.ndarray,
    c_kv: np.ndarray,
    k_rope: np.ndarray,
    block_table: np.ndarray,
    cu_seqlens_q: Sequence[int],
    cu_seqlens_k: Sequence[int],
    scale: float,
    *,
    num_seg: int,
    block_k: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split-KV partials in the kernel's segmentation and workspace layout.

    A sequence's ``ceil(S_k / block_k)`` key tiles are cut into ``num_seg``
    contiguous runs of ``ceil(n_tiles / num_seg)`` tiles; trailing runs may be
    empty. Returns ``part_acc [T, num_seg, H_q, r_KV]`` (unnormalized),
    ``part_m [T, num_seg, H_q]`` and ``part_l [T, num_seg, H_q]``, with ``m`` in
    the kernel's log2 units (scores times :func:`mla_scale_log2`). An empty
    segment is ``(acc, m, l) = (0, -inf, 0)``.
    """
    _require_scale(scale)
    if num_seg < 1 or block_k < 1:
        raise ValueError(
            f"num_seg and block_k must be positive, got {num_seg}, {block_k}"
        )
    scale_log2 = mla_scale_log2(scale)
    r_kv = c_kv.shape[2]
    t, h = qa.shape[:2]
    part_acc = np.zeros((t, num_seg, h, r_kv), dtype=np.float64)
    part_m = np.full((t, num_seg, h), -np.inf, dtype=np.float64)
    part_l = np.zeros((t, num_seg, h), dtype=np.float64)
    for token, keys in _decode_keys(
        qa, c_kv, k_rope, block_table, cu_seqlens_q, cu_seqlens_k
    ):
        n_tiles = -(-keys.shape[0] // block_k)
        per_seg = -(-n_tiles // num_seg) if n_tiles else 0
        s2 = scale_log2 * (qa[token].astype(np.float64) @ keys.T)  # [H, S_k]
        for seg in range(num_seg):
            lo = seg * per_seg * block_k
            hi = min((seg + 1) * per_seg * block_k, keys.shape[0])
            if lo >= hi:
                continue
            m = s2[:, lo:hi].max(axis=-1)
            p = np.exp2(s2[:, lo:hi] - m[:, None])
            part_m[token, seg] = m
            part_l[token, seg] = p.sum(axis=-1)
            part_acc[token, seg] = p @ keys[lo:hi, :r_kv]
    return part_acc, part_m, part_l


def ref_mla_decode_merge(
    part_acc: np.ndarray, part_m: np.ndarray, part_l: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Combine split-KV partials: ``(latent [T, H, r], lse [T, H])``.

    Shapes as :func:`ref_mla_decode_partials` returns them; ``m`` is in log2
    units. With ``M = max_s m_s`` and ``w_s = 2**(m_s - M)``::

        latent = sum_s w_s acc_s / sum_s w_s l_s
        lse    = (M + log2(sum_s w_s l_s)) * ln(2)

    Empty segments (``m = -inf``) get weight zero; a row whose segments are all
    empty is the ``S_k = 0`` case and yields a zero latent and ``lse = -inf``.
    The partials only need to be *mutually* consistent -- each segment's
    ``(m, l, acc)`` relative to the same ``m`` -- so a kernel that keeps a stale
    running max (lazy rescale) merges to the same result.
    """
    if part_acc.ndim != 4 or part_m.shape != part_acc.shape[:3]:
        raise ValueError(
            f"expected part_acc [T, S, H, r] and part_m/part_l [T, S, H]; got "
            f"{part_acc.shape}, {part_m.shape}"
        )
    if part_l.shape != part_m.shape:
        raise ValueError(f"part_l {part_l.shape} must match part_m {part_m.shape}")
    big_m = part_m.max(axis=1)  # [T, H]
    alive = np.isfinite(big_m)
    m_safe = np.where(alive, big_m, 0.0)
    w = np.where(np.isfinite(part_m), np.exp2(part_m - m_safe[:, None, :]), 0.0)
    big_l = (w * part_l).sum(axis=1)
    acc = (w[..., None] * part_acc).sum(axis=1)
    live = alive & (big_l > 0)
    latent = np.where(live[..., None], acc / np.where(live, big_l, 1.0)[..., None], 0.0)
    lse = np.where(
        live, (m_safe + np.log2(np.where(live, big_l, 1.0))) * math.log(2.0), -np.inf
    )
    return latent, lse


def ref_mla_decode_absorb(
    q_nope: np.ndarray,
    q_rope: np.ndarray,
    c_kv: np.ndarray,
    k_rope: np.ndarray,
    w_uk: np.ndarray,
    block_table: np.ndarray,
    cu_seqlens_q: Sequence[int],
    cu_seqlens_k: Sequence[int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The whole decode op: absorb [A], latent attention [B, C], ``W_UV`` [D].

    Returns ``(out [T, H_q, d_V], lse [T, H_q])`` in fp64. ``scale`` is the
    model's ``1/sqrt(d_nope + d_rope)`` (or its YaRN-adjusted value), never
    ``1/sqrt(r_KV + d_rope)``: absorption rewrites the dot product, not the
    softmax temperature.
    """
    qa = absorb_query(q_nope, q_rope, w_uk)
    latent, lse = ref_mla_decode_latent(
        qa, c_kv, k_rope, block_table, cu_seqlens_q, cu_seqlens_k, scale
    )
    d_nope = q_nope.shape[-1]
    out = np.einsum(
        "thr,hrv->thv", latent, w_uk[:, :, d_nope:].astype(np.float64), optimize=True
    )
    return out, lse
