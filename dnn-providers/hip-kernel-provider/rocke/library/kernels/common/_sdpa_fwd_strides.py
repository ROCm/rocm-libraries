# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side element strides for the free-stride SDPA forward ABI.

The generalized forward kernel takes per-tensor ``(batch, head, token)``
element strides and assumes a unit head-dim stride (the ``StridedTensorAddr``
contract). This module is the single source of truth for the stride values the
launcher feeds in for the two contiguous 4D orders; it emits no kernel IR.

General (non-contiguous) and packed-QKV strides are supplied by the caller
(hipDNN hands them straight from the framework tensors), so only the contiguous
``bhsd`` / ``bshd`` orders are derived here. The head dim is always innermost
with unit stride, so it is not part of the returned triple.
"""

from __future__ import annotations

_LAYOUTS = ("bhsd", "bshd")


def contiguous_strides(layout: str, *, heads: int, seqlen: int, head_size: int):
    """``(stride_batch, stride_head, stride_token)`` element strides of one
    contiguous 4D tensor.

    ``layout`` is ``"bhsd"`` ([B, H, S, D], heads-major) or ``"bshd"``
    ([B, S, H, D], tokens-major). ``heads`` / ``seqlen`` are this tensor's own
    counts (query vs KV tensors differ), ``head_size`` is D. The head dim is
    innermost with unit stride and is not returned.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")
    for name, val in (("heads", heads), ("seqlen", seqlen), ("head_size", head_size)):
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")
    h, s, d = heads, seqlen, head_size
    if layout == "bhsd":
        # [B, H, S, D]: batch = H*S*D, head = S*D, token(seq) = D
        return (h * s * d, s * d, d)
    # bshd -- [B, S, H, D]: batch = S*H*D, head = D, token(seq) = H*D
    return (s * h * d, d, h * d)
