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


def dense_attention_reference(Q, K, V, *, causal: bool, scale=None, out_dtype=None):
    """Dense softmax-attention reference (fp32 math).

    ``Q``/``K``/``V`` are a single batch of shape
    ``(seqlen, heads, head_size)`` with KV heads already expanded to the query
    head count -- GQA/MQA expansion (``np.repeat`` over the head axis) is the
    caller's responsibility, as is looping the batch axis.

    ``scale`` multiplies the QK scores before softmax. ``None`` keeps the
    historical ``scores /= sqrt(head_size)`` arithmetic exactly, so existing
    callers are byte-unchanged; pass a float to model an arbitrary attention
    scale (hipDNN supplies its own rather than assuming ``1 / sqrt(d)``).

    Returns fp32 unless ``out_dtype`` is given, in which case the result is
    cast to it (e.g. ``np.float16`` to model a dtype-truncated reference).
    """
    d = Q.shape[-1]
    scores = np.einsum("ihd,jhd->ihj", Q.astype(np.float32), K.astype(np.float32))
    if scale is None:
        scores /= math.sqrt(d)
    else:
        scores *= scale
    if causal:
        q_pos = np.arange(Q.shape[0])[:, None, None]
        k_pos = np.arange(K.shape[0])[None, None, :]
        scores = np.where(k_pos <= q_pos, scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", probs, V.astype(np.float32))
    return out if out_dtype is None else out.astype(out_dtype)
