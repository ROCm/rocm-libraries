# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SDPA admission for the unified attention candidates.

Translates a :class:`~dispatch.sdpa.common.SdpaRequest` into the
``dispatch/attention/`` request the unified candidates already consume and
asks :data:`dispatch.attention.ATTENTION_ROUTE_REGISTRY` whether a unified
route candidate serves it. The existing ``dispatch/attention/`` candidates
are not edited.

This module adds no shape, dtype or head-size rule of its own: "can it run"
is the candidates' own ``admits()``, so new kernel coverage reaches SDPA
routing without touching this file. What it does add is the guarantee that
nothing is admitted and then dropped: every feature in ``SDPA_FEATURES`` is
either carried (:data:`SDPA_UNIFIED_FEATURES`) or declined with a reason, and
a request is declined when the band the unified kernels would compute is not
the band it asked for.

Binding request tensors to a kernel launch is not here yet; it needs on-GPU
validation against ``sdpa_reference`` and lands separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rocke.core.arch import known_arches

from dispatch.attention import (
    ATTENTION_ROUTE_REGISTRY,
    UNIFIED_HEAD_SIZES,
    AttentionMaskType,
    AttentionRequest,
)
from dispatch.attention.common import Alignment, AttentionBand, normalize_band

if TYPE_CHECKING:
    from rocke.dispatch.core import KernelCandidate

    from .common import SdpaRequest

__all__ = [
    "SDPA_UNIFIED_FEATURES",
    "sdpa_unified_admits",
    "sdpa_unified_candidates",
]

# What the unified launch path routes among (``run_unified_attention_torch``
# with no tuning spec). Tuning geometries, dense, RDNA WMMA and strided-KV
# decode candidates are other launch paths.
_UNIFIED_ALGORITHMS = frozenset(
    {"unified_2d", "unified_3d", "d256_decode", "d256_gfx950", "dense_pipe"}
)
# Only the split-KV (3D) path writes LSE.
_LSE_ALGORITHMS = frozenset({"unified_3d", "d256_decode"})

# The binding's ``supports_features``: what the unified launch path carries.
# Layouts other than packed BSHD are carried by copying Q/O to and from a
# packed buffer; K/V are always copied into the paged cache.
SDPA_UNIFIED_FEATURES = frozenset(
    {
        "gqa",
        "d_v_differs",
        "layout_bshd",
        "layout_bhsd",
        "layout_strided",
        "kv_layout_differs",
        "o_layout_differs",
        "batch_stride",
        "d_stride",
        "unaligned",
        "causal_tl",
        "causal_br",
        "window",
        "window_br",
        "window_right",
        "padded",
        "ragged",
        "lse",
        "mma_core_mode",
        "implementation_hint",
    }
)

# Every other feature, with the reason it is declined. Checked in this order,
# so the reported reason is deterministic.
_DECLINED = {
    "kv_heads_differ": "kv_heads_differ: the unified request has one nhead_k for K and V",
    "dtype_mixed": "dtype_mixed: the unified kernels read and write one dtype",
    "compute_dtype": "compute_dtype: the unified kernels accumulate in fp32 only",
    "scale_tensor": "scale_tensor: the unified kernels take only a scalar scale",
    "bias": "bias: no unified candidate declares additive-bias support yet",
    "bias_broadcast": "bias_broadcast: no unified candidate declares additive-bias support yet",
    "alibi": "alibi: no unified candidate declares alibi support yet",
    "sinks": "sinks: no reference oracle for sink tokens yet",
    "dropout": "dropout: the unified kernels have no dropout",
    "paged": "paged: the page-table format is not mapped yet (see SdpaPaged)",
    "block_mask": "block_mask: the unified kernels take no block mask",
    "fp8": "fp8: the unified kernels take no descale/scale tensors",
    "fp8_s_o": "fp8_s_o: the unified kernels take no softmax/output quantization",
    "max_sum_exp": "max_sum_exp: the unified kernels write only LSE, not max and sum-exp",
}


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


def _keep_grid(s_q: int, s_kv: int, top_left: bool, left: int, right: int):
    """hipDNN band (``-1`` unbounded, ``left = W`` keeps W + 1 keys) over a grid."""
    q = np.arange(s_q)[:, None]
    k = np.arange(s_kv)[None, :]
    d = q + (0 if top_left else s_kv - s_q)
    keep = np.ones((s_q, s_kv), dtype=bool)
    if right >= 0:
        keep &= k <= d + right
    if left >= 0:
        keep &= k >= d - left
    return keep


def _band_reject(req: "SdpaRequest") -> "tuple[bool, str] | None":
    """Decline when the unified kernels would compute a different band.

    ``normalize_band`` is the one place the attention layer resolves the band
    its kernels actually compute, including arches whose kernels serve only
    some bands. Bands that differ in form but keep the same keys are equal:
    full vs bottom-right causal at ``s_q == 1`` always, and any other pair
    (e.g. top-left vs bottom-right at ``s_q == s_kv``) only for fixed lengths,
    since with per-sequence lengths the diagonal moves per batch.
    """
    try:
        band = normalize_band(_to_attention_request(req))
    except ValueError as exc:
        return False, f"band: {exc}"

    asked = AttentionBand(
        left_bound=None if req.left_bound < 0 else req.left_bound + 1,
        right_bound=None if req.right_bound < 0 else req.right_bound,
        alignment=(
            Alignment.TOP_LEFT if req.diagonal == "top_left" else Alignment.BOTTOM_RIGHT
        ),
    )
    if (band.left_bound, band.right_bound) == (
        asked.left_bound,
        asked.right_bound,
    ) and (band.alignment == asked.alignment or asked.is_full):
        return None
    # One query row per sequence: bottom-right causal keeps every key of each
    # sequence, whatever its length, so it is full attention.
    if (
        asked.is_full
        and req.s_q == 1
        and band.left_bound is None
        and band.right_bound == 0
        and band.alignment == Alignment.BOTTOM_RIGHT
    ):
        return None

    features = req.features()
    if "padded" not in features and "ragged" not in features:

        def grid(b: AttentionBand):
            return _keep_grid(
                req.s_q,
                req.s_kv,
                b.alignment == Alignment.TOP_LEFT,
                -1 if b.left_bound is None else b.left_bound - 1,
                -1 if b.right_bound is None else b.right_bound,
            )

        if np.array_equal(grid(band), grid(asked)):
            return None

    return False, (
        f"band: {req.arch} unified kernels compute {_band_text(band)} for "
        f"requested {_band_text(asked)}"
    )


def _band_text(b: AttentionBand) -> str:
    if b.is_full:
        return "full attention"
    return f"left={b.left_bound} right={b.right_bound} " f"{b.alignment.name.lower()}"


def _sdpa_unified_rejects(req: "SdpaRequest") -> "tuple[bool, str] | None":
    """Return ``(False, reason)`` if this request cannot be served, else ``None``.

    The single gate shared by :func:`sdpa_unified_admits` and
    :func:`sdpa_unified_candidates`, so the two cannot drift. The request is
    validated (head_dim, arch) first, so a malformed request reports its own
    cause rather than a feature it also happens to use.
    """
    if req.d_qk not in UNIFIED_HEAD_SIZES or req.d_v not in UNIFIED_HEAD_SIZES:
        return False, (
            f"head_dim (d_qk={req.d_qk}, d_v={req.d_v}) is not a unified head "
            f"size; supported: {sorted(UNIFIED_HEAD_SIZES)}"
        )
    if req.arch not in known_arches():
        return False, f"unknown arch {req.arch!r}"

    features = req.features()
    for feature, reason in _DECLINED.items():
        if feature in features:
            return False, reason

    if req.mma_core_mode is not None and req.mma_core_mode != req.dtype:
        return False, (
            f"mma_core_mode: the unified kernels run MMA in {req.dtype}, "
            f"not {req.mma_core_mode}"
        )
    if req.implementation == "composite":
        return False, "implementation_hint: composite was requested"
    if req.padding_mask and req.seq_len_q is None and req.seq_len_kv is None:
        return False, "padded: padding_mask is set but no seq_len tensor is given"

    return _band_reject(req)


def _unified_supported(req: "SdpaRequest") -> "tuple[KernelCandidate, ...]":
    """Unified route candidates that admit the translated request."""
    supported = tuple(
        c
        for c in ATTENTION_ROUTE_REGISTRY.supported(_to_attention_request(req))
        if c.algorithm in _UNIFIED_ALGORITHMS
    )
    if req.stats is not None:
        supported = tuple(c for c in supported if c.algorithm in _LSE_ALGORITHMS)
    return supported


def sdpa_unified_admits(req: "SdpaRequest") -> "tuple[bool, str]":
    """Top-level SDPA admission check for the unified family.

    Returns ``(True, "ok")`` only when nothing in
    :func:`_sdpa_unified_rejects` applies and at least one unified route
    candidate admits the translated request.
    """
    rejected = _sdpa_unified_rejects(req)
    if rejected is not None:
        return rejected
    if _unified_supported(req):
        return True, "ok"
    path = "split-KV (LSE) " if req.stats is not None else ""
    return False, (
        f"no unified {path}candidate covers arch={req.arch!r} "
        f"dtype={req.dtype!r} d_qk={req.d_qk} d_v={req.d_v} "
        f"s_q={req.s_q} s_kv={req.s_kv}"
    )


def sdpa_unified_candidates(req: "SdpaRequest") -> "tuple[str, ...]":
    """Names of the unified candidates that serve ``req``, for tests and triage.

    In registry priority order; empty whenever :func:`sdpa_unified_admits`
    would return ``False``.
    """
    if _sdpa_unified_rejects(req) is not None:
        return ()
    return tuple(c.name for c in _unified_supported(req))
