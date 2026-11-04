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
    Q, K, V, *, causal: bool, scale=None, seqlen_q=None, seqlen_kv=None, out_dtype=None
):
    """Dense softmax-attention reference (fp32 math).

    ``Q``/``K``/``V`` are a single batch of shape
    ``(seqlen, heads, head_size)`` with KV heads already expanded to the query
    head count -- GQA/MQA expansion (``np.repeat`` over the head axis) is the
    caller's responsibility, as is looping the batch axis.

    ``scale`` multiplies the QK scores before softmax. ``None`` keeps the
    historical ``scores /= sqrt(head_size)`` arithmetic exactly, so existing
    callers are byte-unchanged; pass a float to model an arbitrary attention
    scale (hipDNN supplies its own rather than assuming ``1 / sqrt(d)``).

    ``seqlen_q`` / ``seqlen_kv`` are this sequence's valid (unpadded) query and
    key counts; positions at or beyond them are padding. ``None`` (the default
    for both) keeps the full-length, unpadded behavior so existing callers are
    byte-unchanged. When set, keys ``k >= seqlen_kv`` are masked out of every
    softmax, a query row left with no valid key is fully masked and yields
    ``O = 0`` (hipDNN's ``LSE = -inf`` convention), and padded query rows
    ``q >= seqlen_q`` are written as ``O = 0``. That padded-query-row convention
    is this reference's own definition; the hipDNN CPU reference does not model
    SEQ_LEN_Q / SEQ_LEN_KV yet, so there is nothing to cross-check it against
    today.

    Returns fp32 unless ``out_dtype`` is given, in which case the result is
    cast to it (e.g. ``np.float16`` to model a dtype-truncated reference).
    """
    d = Q.shape[-1]
    scores = np.einsum("ihd,jhd->ihj", Q.astype(np.float32), K.astype(np.float32))
    if scale is None:
        scores /= math.sqrt(d)
    else:
        scores *= scale
    # One keep mask, built up front and reused both to mask the scores and to
    # find rows left with no visible key, so the masking and the fully-masked
    # check cannot drift apart.
    keep = None
    if causal or seqlen_kv is not None:
        q_pos = np.arange(Q.shape[0])[:, None, None]
        k_pos = np.arange(K.shape[0])[None, None, :]
        keep = np.ones((Q.shape[0], 1, K.shape[0]), dtype=bool)
        if causal:
            keep = keep & (k_pos <= q_pos)
        if seqlen_kv is not None:
            keep = keep & (k_pos < seqlen_kv)
        scores = np.where(keep, scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", probs, V.astype(np.float32))
    if seqlen_kv is not None:
        # A row with no visible key is fully masked -> O = 0; the -1e30 softmax
        # above would otherwise average all keys uniformly instead of zero.
        out = np.where(keep.any(axis=-1, keepdims=True), out, 0.0)
    if seqlen_q is not None:
        out = np.where(np.arange(Q.shape[0])[:, None, None] < seqlen_q, out, 0.0)
    return out if out_dtype is None else out.astype(out_dtype)


from rocke.numeric.sdpa_reference import (  # noqa: E402,F401
    keep,
    resolve_diagonal_band,
    sdpa_reference,
)
from rocke.numeric.sdpa_reference import (  # noqa: E402,F401
    SdpaBwdResult,
    sdpa_reference_bwd,
)

__all__ += ["keep", "resolve_diagonal_band", "sdpa_reference"]
__all__ += ["SdpaBwdResult", "sdpa_reference_bwd"]
