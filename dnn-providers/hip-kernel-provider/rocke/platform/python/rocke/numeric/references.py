# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free numpy numeric references for rocKE verify harnesses.

One canonical numpy definition per numeric contract, so attention verify
harnesses can share the same reference math instead of each carrying its own
copy. All math is fp32; the output dtype is the caller's choice
(``out_dtype=None`` keeps fp32, the honest full-precision reference; pass
``np.float16`` when the harness wants a dtype-truncated reference).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["dense_attention_reference"]


def dense_attention_reference(
    Q, K, V, *, causal: bool, seqlen_q=None, seqlen_kv=None, out_dtype=None
):
    """Dense softmax-attention reference (fp32 math).

    ``Q``/``K``/``V`` are a single batch of shape
    ``(seqlen, heads, head_size)`` with KV heads already expanded to the query
    head count -- GQA/MQA expansion (``np.repeat`` over the head axis) is the
    caller's responsibility, as is looping the batch axis.

    ``seqlen_q`` / ``seqlen_kv`` are this sequence's valid (unpadded) query and
    key counts; positions at or beyond them are padding. ``None`` (the default
    for both) keeps the full-length, unpadded behavior so existing callers are
    byte-unchanged. When set, keys ``k >= seqlen_kv`` are masked out of every
    softmax, a query row left with no valid key is fully masked and yields
    ``O = 0`` (hipDNN's ``LSE = -inf`` convention), and padded query rows
    ``q >= seqlen_q`` are zeroed. The padded-query-row zeroing is the one piece
    that should be confirmed against the hipDNN padding reference.

    Returns fp32 unless ``out_dtype`` is given, in which case the result is
    cast to it (e.g. ``np.float16`` to model a dtype-truncated reference).
    """
    d = Q.shape[-1]
    scores = np.einsum("ihd,jhd->ihj", Q.astype(np.float32), K.astype(np.float32))
    scores /= math.sqrt(d)
    if causal:
        q_pos = np.arange(Q.shape[0])[:, None, None]
        k_pos = np.arange(K.shape[0])[None, None, :]
        scores = np.where(k_pos <= q_pos, scores, -1e30)
    if seqlen_kv is not None:
        k_pos = np.arange(K.shape[0])[None, None, :]
        scores = np.where(k_pos < seqlen_kv, scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", probs, V.astype(np.float32))
    if seqlen_kv is not None:
        # A row with no valid key is fully masked -> O = 0; the -1e30 softmax
        # above would otherwise average all keys uniformly instead of zero.
        k_idx = np.arange(K.shape[0])[None, None, :]
        keep = k_idx < seqlen_kv
        if causal:
            keep = keep & (k_idx <= np.arange(Q.shape[0])[:, None, None])
        out = np.where(keep.any(axis=-1, keepdims=True), out, 0.0)
    if seqlen_q is not None:
        out = np.where(np.arange(Q.shape[0])[:, None, None] < seqlen_q, out, 0.0)
    return out if out_dtype is None else out.astype(out_dtype)
