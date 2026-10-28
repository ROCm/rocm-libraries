# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SDPA admission for the unified attention candidates.

Translates a :class:`~dispatch.sdpa.common.SdpaRequest` into the
``dispatch/attention/`` request the unified candidates already consume and
asks :data:`dispatch.attention.ATTENTION_ROUTE_REGISTRY` whether any of them
serves it. The existing ``dispatch/attention/`` candidates are not edited.

This module adds no shape, dtype or head-size rule of its own: "can it run"
is the candidates' own ``admits()``, so new kernel coverage reaches SDPA
routing without touching this file. Its only declines are ``SdpaRequest``
features the unified request cannot carry.

Binding request tensors to a kernel launch is not here yet; it needs on-GPU
validation against ``sdpa_reference`` and lands separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rocke.core.arch import known_arches

from dispatch.attention import (
    ATTENTION_ROUTE_REGISTRY,
    UNIFIED_HEAD_SIZES,
    AttentionMaskType,
    AttentionRequest,
)

if TYPE_CHECKING:
    from .common import SdpaRequest

__all__ = [
    "sdpa_unified_admits",
    "sdpa_unified_candidates",
]


def _to_attention_request(req: "SdpaRequest") -> AttentionRequest:
    """The ``AttentionRequest`` the unified candidates admit for ``req``.

    ``SdpaRequest.left_bound`` / ``right_bound`` / ``diagonal`` are in the
    same cuDNN/hipDNN band vocabulary as ``AttentionRequest.right_bound`` /
    ``diagonal_alignment``. The one adjustment is ``sliding_window``: it sees
    W keys, while hipDNN ``left_bound = W`` keeps W + 1, so lower with
    ``left_bound + 1``.
    """
    features = req.features()
    if "causal_tl" in features:
        mask_type = AttentionMaskType.TOP_LEFT_CAUSAL
    elif "causal_br" in features:
        mask_type = AttentionMaskType.BOTTOM_RIGHT_CAUSAL
    elif "window" in features:
        mask_type = AttentionMaskType.SLIDING_WINDOW
    else:
        mask_type = AttentionMaskType.NO_MASK

    is_window = mask_type == AttentionMaskType.SLIDING_WINDOW
    return AttentionRequest(
        batch=req.batch,
        nhead_q=req.h_q,
        nhead_k=req.h_k,
        seqlen_q=req.s_q,
        seqlen_k=req.s_kv,
        hdim_q=req.d_qk,
        hdim_v=req.d_v,
        arch=req.arch,
        dtype=req.dtype,
        mask_type=mask_type,
        sliding_window=(req.left_bound + 1) if is_window and req.left_bound >= 0 else 0,
        right_bound=req.right_bound if is_window else -1,
        diagonal_alignment=(
            (1 if req.diagonal == "bottom_right" else 0) if is_window else None
        ),
        use_sinks=req.sink_token is not None,
        use_alibi=req.alibi,
        emit_lse=req.stats is not None,
    )


def _sdpa_unified_rejects(req: "SdpaRequest") -> "tuple[bool, str] | None":
    """Return ``(False, reason)`` if this request cannot be served, else ``None``.

    The single gate shared by :func:`sdpa_unified_admits` and
    :func:`sdpa_unified_candidates`, so the two cannot drift. The request is
    validated (head_dim, arch) before any feature gap, so a malformed request
    reports its own cause.
    """
    if req.d_qk not in UNIFIED_HEAD_SIZES or req.d_v not in UNIFIED_HEAD_SIZES:
        return False, (
            f"head_dim (d_qk={req.d_qk}, d_v={req.d_v}) is not a unified head "
            f"size; supported: {sorted(UNIFIED_HEAD_SIZES)}"
        )
    if req.arch not in known_arches():
        return False, f"unknown arch {req.arch!r}"

    if req.h_k != req.h_v:
        return False, "kv_heads_differ: the unified request has one nhead_k for K and V"
    if req.paged is not None:
        return False, "paged: the page-table format is not mapped yet (see SdpaPaged)"
    if req.dropout is not None:
        return False, "dropout: the unified kernels have no dropout"
    if req.fp8 is not None:
        return False, "fp8: the unified kernels take no descale/scale tensors"
    if req.block_mask is not None:
        return False, "block_mask: the unified kernels take no block mask"
    if req.scale_source == "tensor":
        return False, "scale_tensor: the unified kernels take only a scalar scale"
    if req.bias is not None:
        return False, "bias: no unified candidate declares additive-bias support yet"
    if req.alibi:
        return False, "alibi: no unified candidate declares alibi support yet"

    return None


def sdpa_unified_admits(req: "SdpaRequest") -> "tuple[bool, str]":
    """Top-level SDPA admission check for the unified family.

    Returns ``(True, "ok")`` only when nothing in
    :func:`_sdpa_unified_rejects` applies and at least one unified candidate
    admits the translated request.
    """
    rejected = _sdpa_unified_rejects(req)
    if rejected is not None:
        return rejected

    if ATTENTION_ROUTE_REGISTRY.supported(_to_attention_request(req)):
        return True, "ok"
    return False, (
        f"no unified candidate covers arch={req.arch!r} dtype={req.dtype!r} "
        f"d_qk={req.d_qk} d_v={req.d_v}"
    )


def sdpa_unified_candidates(req: "SdpaRequest") -> "tuple[str, ...]":
    """Names of the unified candidates that serve ``req``, for tests and triage.

    In registry priority order; empty whenever :func:`sdpa_unified_admits`
    would return ``False``.
    """
    if _sdpa_unified_rejects(req) is not None:
        return ()
    attn_req = _to_attention_request(req)
    return tuple(c.name for c in ATTENTION_ROUTE_REGISTRY.supported(attn_req))
