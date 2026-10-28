# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SDPA binding for the unified attention candidates (AICK-2613).

Translates a real :class:`~dispatch.sdpa.common.SdpaRequest` into the
``dispatch/attention/`` request the unified candidates already consume, and
binds the request's tensor-shaped fields (Q/K/V/O strides, scale, band
bounds, LSE, sequence lengths, ragged offsets) to a launchable kernel call.
The existing ``dispatch/attention/`` candidates are not edited; this module
only translates and adds the handful of declines a ``SdpaRequest`` needs that
an ``AttentionRequest`` cannot express.

Two entry points, both built from one shared translation
(:func:`_to_attention_request`), so they cannot disagree on what a request
means:

- :func:`sdpa_unified_admits` / :func:`sdpa_unified_candidates` -- routing,
  reusing :data:`dispatch.attention.ATTENTION_ROUTE_REGISTRY` directly on the
  translated request (no probe shapes: the real request carries real shapes).
- :func:`sdpa_unified_bind` -- kernel-argument binding, reusing the existing
  unified "tuning" Torch path (:func:`dispatch.attention.dispatch_attention_all`
  -> ``KernelCandidate.bind_torch``) rather than inventing a new launch
  mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from rocke.core.arch import known_arches

from dispatch.attention import (
    ATTENTION_ROUTE_REGISTRY,
    UNIFIED_HEAD_SIZES,
    AttentionMaskType,
    AttentionRequest,
    dispatch_attention_all,
)

if TYPE_CHECKING:
    from rocke.dispatch.core import TorchBinding

    from .common import SdpaRequest

__all__ = [
    "sdpa_unified_admits",
    "sdpa_unified_bind",
    "sdpa_unified_candidates",
]

# fp16 d=256 declined pending accuracy validation on the unified candidates.
# No tracking ID here: the reason is what a caller can act on, not where the
# policy came from (see PR #13115 review).
_FP16_D256_REASON = (
    "fp16 d=256 is declined pending accuracy validation on the unified "
    "candidates; use bf16 at d=256, or fp16 at d in {64, 128}"
)


def _to_attention_request(req: "SdpaRequest") -> AttentionRequest:
    """The ``AttentionRequest`` the unified candidates admit/bind for ``req``.

    ``SdpaRequest.left_bound`` / ``right_bound`` / ``diagonal`` are already in
    the same cuDNN/hipDNN band vocabulary ``AttentionRequest.right_bound`` /
    ``diagonal_alignment`` document (see both modules' docstrings), so this is
    close to an identity mapping -- the one adjustment is
    ``AttentionRequest.sliding_window``, whose "W keys" convention is one less
    than hipDNN's ``left_bound = W`` ("W + 1 keys"); lower with
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

    The single gate shared by every entry point in this module. Order
    matters: the request is *validated* (head_dim, arch) before any SDPA
    policy or capability-gap rule runs, so a malformed request reports its
    own cause instead of borrowing a policy rule's reason.

    Rules here are in addition to, not instead of, the per-candidate
    Capability and support-closure checks in ``dispatch/attention/``.
    """
    # 1. Request validity, cheapest first.
    if req.d_qk not in UNIFIED_HEAD_SIZES or req.d_v not in UNIFIED_HEAD_SIZES:
        return False, (
            f"head_dim (d_qk={req.d_qk}, d_v={req.d_v}) is not a unified head "
            f"size; supported: {sorted(UNIFIED_HEAD_SIZES)}"
        )
    if req.arch not in known_arches():
        return False, f"unknown arch {req.arch!r}"

    # 2. SdpaRequest features the unified Torch binding cannot carry today.
    #    Each of these is a real, verified gap in dispatch/attention/bindings.py
    #    (bind_tuning_attention_torch), not a restatement of candidate Capability.
    if req.h_k != req.h_v:
        return False, "kv_heads_differ: the unified request has one nhead_k for K and V"
    if req.paged is not None:
        return False, "paged: the page-table format is not mapped yet (see SdpaPaged)"
    if req.dropout is not None:
        return False, "dropout: the unified tuning binding has no dropout plumbing"
    if req.fp8 is not None:
        return (
            False,
            "fp8: the unified tuning binding has no descale/scale tensor inputs",
        )
    if req.block_mask is not None:
        return False, "block_mask: no equivalent tensor input on the unified binding"
    if req.scale_source == "tensor":
        return (
            False,
            "scale_tensor: the unified tuning binding takes only a scalar scale",
        )
    if req.bias is not None:
        return False, (
            "bias: explicit unified tuning specs do not support additive bias "
            "(the spec makers reject it; see bindings.py)"
        )
    if req.alibi:
        return False, "alibi: no unified candidate declares alibi support yet"

    # 3. SDPA policy declines, applied only to requests that are otherwise sane.
    if req.dtype == "fp16" and req.d_qk == 256:
        return False, _FP16_D256_REASON

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

    attn_req = _to_attention_request(req)
    if ATTENTION_ROUTE_REGISTRY.supported(attn_req):
        return True, "ok"
    return False, (
        f"no unified candidate covers arch={req.arch!r} dtype={req.dtype!r} "
        f"d_qk={req.d_qk} d_v={req.d_v}"
    )


def sdpa_unified_candidates(req: "SdpaRequest") -> "tuple[str, ...]":
    """Names of the unified candidates that serve ``req``, for tests and triage.

    In registry priority order (``ATTENTION_ROUTE_REGISTRY.supported`` already
    preserves it); empty whenever :func:`sdpa_unified_admits` would return
    ``False``.
    """
    if _sdpa_unified_rejects(req) is not None:
        return ()
    attn_req = _to_attention_request(req)
    return tuple(c.name for c in ATTENTION_ROUTE_REGISTRY.supported(attn_req))


def _dense_paging_tensors(attn_req: AttentionRequest, q, k, v, out) -> dict:
    """Identity ``cu_seqlens_q`` / ``seqused_k`` / ``block_table`` for a dense
    (non-ragged, non-paged) request onto the unified paged-KV ABI.

    Mirrors ``benchmarks/common/attention_combo_sweep.py``'s ``_unified_tensors``
    / ``_pack_paged`` adapter (not imported: ``benchmarks`` sits above
    ``dispatch`` in the ``kernels -> dispatch -> builders -> benchmarks``
    layering rocke/CLAUDE.md enforces, so dispatch may not import it).
    """
    import torch

    b, s_kv, h_kv, d = k.shape
    page = int(attn_req.kv_block_size)
    pages = (s_kv + page - 1) // page
    k_padded = torch.zeros(b, pages * page, h_kv, d, dtype=k.dtype, device=k.device)
    k_padded[:, :s_kv] = k
    v_padded = torch.zeros(
        b, pages * page, h_kv, v.shape[-1], dtype=v.dtype, device=v.device
    )
    v_padded[:, :s_kv] = v
    k_cache = k_padded.reshape(b * pages, page, h_kv, d).contiguous()
    v_cache = v_padded.reshape(b * pages, page, h_kv, v.shape[-1]).contiguous()
    block_table = (
        torch.arange(b * pages, device=k.device, dtype=torch.int32)
        .view(b, pages)
        .contiguous()
    )
    cu_seqlens_q = torch.arange(
        0,
        (attn_req.batch + 1) * attn_req.seqlen_q,
        attn_req.seqlen_q,
        dtype=torch.int32,
        device=q.device,
    )
    seqused_k = torch.full(
        (attn_req.batch,), attn_req.seqlen_k, dtype=torch.int32, device=q.device
    )
    return {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "out": out,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "block_table": block_table,
    }


def sdpa_unified_bind(
    req: "SdpaRequest", tensors: Mapping[str, Any], **kwargs: Any
) -> "TorchBinding":
    """Bind ``req`` plus caller-owned ``tensors`` to a launchable kernel call.

    ``tensors`` carries the live operands the frozen ``SdpaRequest`` cannot
    (its ``SdpaTensor`` fields are shape/stride/dtype metadata only, never
    runtime values): ``q``, ``k``, ``v``, ``out`` (all real tensors), and,
    when ``req`` is ragged, the real ragged-offset tensors keyed
    ``ragged_offset_q`` / ``ragged_offset_k``.

    Raises ``ValueError`` with the same reason :func:`sdpa_unified_admits`
    would on a declined request -- binding a declined request is a caller
    bug, not a soft failure, and also when no unified candidate's Torch
    binding admits the translated request (should not happen for a request
    :func:`sdpa_unified_admits` accepted).

    ``dispatch_attention_all`` is the sweep/autotune primitive and
    deliberately does not rank its results (``CandidateRegistry.iter_dispatch_all``
    docstring: "Does not rank or collapse"). When it returns more than one
    eligible geometry for the exact translated request, this picks
    ``(candidate.priority, candidate.name)`` -- the same tie-break
    :meth:`CandidateRegistry.candidates` uses -- for a deterministic choice.
    Whether that choice always matches the unpinned production route
    (:func:`dispatch.attention.dispatch_attention`) for every admitted shape
    is not yet verified on hardware; treat this as an open item for the
    on-GPU acceptance check.
    """
    rejected = _sdpa_unified_rejects(req)
    if rejected is not None:
        raise ValueError(rejected[1])

    attn_req = _to_attention_request(req)
    results = dispatch_attention_all(attn_req)
    if not results:
        raise ValueError(
            f"no executable unified candidate for arch={req.arch!r} "
            f"dtype={req.dtype!r} d_qk={req.d_qk} d_v={req.d_v}"
        )
    result = min(results, key=lambda r: (r.candidate.priority, r.candidate.name))

    q, k, v, out = tensors["q"], tensors["k"], tensors["v"], tensors["out"]
    is_ragged = req.q.ragged or req.k.ragged
    if is_ragged:
        payload = {
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "cu_seqlens_q": tensors["ragged_offset_q"],
            "seqused_k": tensors["seqused_k"],
            "block_table": tensors["block_table"],
        }
    else:
        payload = _dense_paging_tensors(attn_req, q, k, v, out)

    if req.sink_token is not None:
        payload["sinks"] = tensors["sinks"]
    if req.stats is not None:
        payload["lse"] = tensors["stats"]

    bind_kwargs = dict(kwargs)
    # hipDNN's "default" scale source bakes 1.0 (common.py), which is not the
    # unified kernel's own default of 1/sqrt(d) -- always pass it explicitly.
    bind_kwargs.setdefault("softmax_scale", req.scale)
    return result.bind_torch(payload, **bind_kwargs)
