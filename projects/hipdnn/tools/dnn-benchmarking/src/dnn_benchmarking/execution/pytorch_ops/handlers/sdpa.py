# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Scaled-dot-product-attention forward/backward reference handlers."""

from math import sqrt
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .._common import *  # noqa: F401,F403
from .._registry import register_handler


def _sdpa_bool(node: Dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(_node_param(node, key, default))


def _sdpa_unsupported_if_present(node: Dict[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        if _optional_uid(node, key) is not None:
            raise ValueError(
                f"Unsupported SDPA optional tensor '{key}' in PyTorch reference"
            )


def _sdpa_scale(
    node: Dict[str, Any], tensors: Dict[int, torch.Tensor]
) -> Optional[float]:
    scale_uid = _optional_uid(node, "scale_tensor_uid")
    if scale_uid is not None:
        return _scalar_value(tensors, scale_uid, node)
    value = _node_param(node, "attn_scale_value", None)
    return None if value is None else float(value)


def _sdpa_head_repeat(q_heads: int, kv_heads: int, label: str) -> int:
    """Validate and return the per-query-head repeat factor for K or V.

    hipDNN allows independent K and V head counts; each must divide the query
    head count (frontend SdpaBwdNode validation), and the CPU reference maps K
    and V with separate ratios.
    """
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError(
            f"Unsupported SDPA {label} head count: q_heads={q_heads}, "
            f"{label.lower()}_heads={kv_heads}"
        )
    return q_heads // kv_heads


def _sdpa_common(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], float, bool, Optional[float], int, int]:
    unsupported = [
        "seq_len_q_tensor_uid",
        "seq_len_kv_tensor_uid",
        "seed_tensor_uid",
        "offset_tensor_uid",
        "dropout_mask_tensor_uid",
        "dropout_scale_tensor_uid",
        "page_table_k_tensor_uid",
        "page_table_v_tensor_uid",
        "block_mask_tensor_uid",
        "sink_token_tensor_uid",
        "descale_q_tensor_uid",
        "descale_k_tensor_uid",
        "descale_v_tensor_uid",
        "descale_s_tensor_uid",
        "scale_s_tensor_uid",
        "scale_o_tensor_uid",
    ]
    _sdpa_unsupported_if_present(node, unsupported)

    if _sdpa_bool(node, "alibi_mask") or _sdpa_bool(node, "padding_mask"):
        raise ValueError(
            "SDPA alibi/padding masks are not supported by the PyTorch reference"
        )
    if _sdpa_bool(node, "causal_mask_bottom_right"):
        raise ValueError(
            "SDPA bottom-right causal mask is not supported by the PyTorch reference"
        )
    diagonal_alignment = _node_param(node, "diagonal_alignment", "TOP_LEFT")
    if diagonal_alignment not in ("TOP_LEFT", 0, None):
        raise ValueError("Only TOP_LEFT SDPA diagonal alignment is supported")
    if (
        _node_param(node, "left_bound", None) is not None
        or _node_param(node, "right_bound", None) is not None
    ):
        raise ValueError(
            "SDPA sliding-window bounds are not supported by the PyTorch reference"
        )

    dropout_probability = _node_param(node, "dropout_probability", 0.0)
    dropout_p = 0.0 if dropout_probability is None else float(dropout_probability)
    if dropout_p != 0.0:
        raise ValueError(
            "Nonzero SDPA dropout cannot be exactly validated against PyTorch"
        )

    mask_uid = _optional_uid(node, "attn_mask_tensor_uid")
    attn_mask = _tensor(tensors, mask_uid, node) if mask_uid is not None else None
    is_causal = _sdpa_bool(node, "causal_mask")
    if attn_mask is not None and is_causal:
        raise ValueError(
            "PyTorch SDPA reference does not support both attn_mask and causal_mask"
        )

    scale = _sdpa_scale(node, tensors)
    if q.ndim < 3 or k.ndim < 3 or v.ndim < 3:
        raise ValueError("SDPA expects q/k/v tensors with head and matrix dimensions")
    q_heads = int(q.shape[-3])
    rep_k = _sdpa_head_repeat(q_heads, int(k.shape[-3]), "K")
    rep_v = _sdpa_head_repeat(q_heads, int(v.shape[-3]), "V")
    return attn_mask, dropout_p, is_causal, scale, rep_k, rep_v


def _call_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    rep_k: int,
    rep_v: int,
) -> torch.Tensor:
    kwargs: Dict[str, Any] = {}
    if scale is not None:
        kwargs["scale"] = scale
    # Expand K and V independently to the query head count. PyTorch's
    # enable_gqa only models equal K/V head counts, so explicit repeat is the
    # only correct path when Hk != Hv.
    if rep_k > 1:
        k = k.repeat_interleave(rep_k, dim=-3)
    if rep_v > 1:
        v = v.repeat_interleave(rep_v, dim=-3)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        **kwargs,
    )


def _sdpa_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: Optional[float],
    rep_k: int,
) -> torch.Tensor:
    q_float = q.to(dtype=torch.float32)
    k_float = k.to(dtype=torch.float32)
    if rep_k > 1:
        k_float = k_float.repeat_interleave(rep_k, dim=-3)
    scale_value = (1.0 / sqrt(float(q.shape[-1]))) if scale is None else scale
    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale_value
    if attn_mask is not None:
        scores = scores + attn_mask.to(dtype=torch.float32)
    if is_causal:
        length_q = scores.shape[-2]
        length_k = scores.shape[-1]
        causal = torch.ones(
            length_q,
            length_k,
            dtype=torch.bool,
            device=scores.device,
        ).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    return torch.logsumexp(scores, dim=-1, keepdim=True)


# -----------------------------------------------------------------------------
# Operation Handlers
# -----------------------------------------------------------------------------


@register_handler("SdpaAttributes")
def handle_sdpa(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle scaled dot-product attention forward."""
    _sdpa_unsupported_if_present(
        node,
        [
            "max_tensor_uid",
            "sum_exp_tensor_uid",
            "rng_dump_tensor_uid",
            "amax_s_tensor_uid",
            "amax_o_tensor_uid",
        ],
    )
    q_uid = _required_input_uid(node, "q_tensor_uid")
    k_uid = _required_input_uid(node, "k_tensor_uid")
    v_uid = _required_input_uid(node, "v_tensor_uid")
    o_uid = _required_output_uid(node, "o_tensor_uid")

    q = _tensor(tensors, q_uid, node)
    k = _tensor(tensors, k_uid, node)
    v = _tensor(tensors, v_uid, node)
    attn_mask, dropout_p, is_causal, scale, rep_k, rep_v = _sdpa_common(
        node, tensors, q, k, v
    )
    o = _call_sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale, rep_k, rep_v)
    _store_tensor(tensors, o_uid, o)

    stats_uid = _optional_uid(node, "stats_tensor_uid")
    if stats_uid is not None:
        _store_tensor(
            tensors,
            stats_uid,
            _sdpa_stats(q, k, attn_mask, is_causal, scale, rep_k),
        )


@register_handler("SdpaBackwardAttributes")
def handle_sdpa_backward(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle scaled dot-product attention backward.

    Mirrors hipDNN's CPU reference (CpuFpReferenceSdpa::backward): the saved
    softmax statistics ``stats`` (forward log-sum-exp) are consumed directly to
    recompute probabilities as ``P = exp(scores - stats)`` without
    renormalization.  PyTorch's built-in SDPA autograd cannot consume an
    external ``stats`` tensor and always renormalizes its own softmax, so it
    would diverge from hipDNN whenever ``stats`` is not the exact, consistent
    forward LSE.  This handler therefore implements the gradient manually.
    """
    _sdpa_unsupported_if_present(node, ["dropout_scale_inv_tensor_uid"])
    if _optional_uid(node, "dbias_tensor_uid") is not None:
        raise ValueError(
            "SDPA backward dBias gradient is not supported by the PyTorch reference"
        )

    q_uid = _required_input_uid(node, "q_tensor_uid")
    k_uid = _required_input_uid(node, "k_tensor_uid")
    v_uid = _required_input_uid(node, "v_tensor_uid")
    o_uid = _required_input_uid(node, "o_tensor_uid")
    do_uid = _required_input_uid(node, "do_tensor_uid")
    stats_uid = _required_input_uid(node, "stats_tensor_uid")
    dq_uid = _required_output_uid(node, "dq_tensor_uid")
    dk_uid = _required_output_uid(node, "dk_tensor_uid")
    dv_uid = _required_output_uid(node, "dv_tensor_uid")

    q = _tensor(tensors, q_uid, node)
    k = _tensor(tensors, k_uid, node)
    v = _tensor(tensors, v_uid, node)
    o = _tensor(tensors, o_uid, node)
    do = _tensor(tensors, do_uid, node)
    stats = _tensor(tensors, stats_uid, node)
    attn_mask, _dropout_p, is_causal, scale, rep_k, rep_v = _sdpa_common(
        node, tensors, q, k, v
    )

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("SDPA backward expects rank-4 q/k/v tensors [B, H, S, D]")

    q_f = q.to(dtype=torch.float32)
    k_f = k.to(dtype=torch.float32)
    v_f = v.to(dtype=torch.float32)
    o_f = o.to(dtype=torch.float32)
    do_f = do.to(dtype=torch.float32)
    stats_f = stats.to(dtype=torch.float32)

    head_dim = int(q.shape[-1])
    scale_value = (1.0 / sqrt(float(head_dim))) if scale is None else float(scale)
    k_heads = int(k.shape[1])
    v_heads = int(v.shape[1])
    if rep_k > 1:
        k_f = k_f.repeat_interleave(rep_k, dim=1)
    if rep_v > 1:
        v_f = v_f.repeat_interleave(rep_v, dim=1)

    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale_value
    if attn_mask is not None:
        scores = scores + attn_mask.to(dtype=torch.float32)
    if is_causal:
        causal = torch.ones(
            scores.shape[-2],
            scores.shape[-1],
            dtype=torch.bool,
            device=scores.device,
        ).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

    probs = torch.exp(scores - stats_f)
    row_dot = (do_f * o_f).sum(dim=-1, keepdim=True)
    d_probs = torch.matmul(do_f, v_f.transpose(-2, -1))
    d_scores = probs * (d_probs - row_dot)
    d_scores_scaled = d_scores * scale_value

    dq = torch.matmul(d_scores_scaled, k_f)
    dk_full = torch.matmul(d_scores_scaled.transpose(-2, -1), q_f)
    dv_full = torch.matmul(probs.transpose(-2, -1), do_f)

    batch, seq_kv = dk_full.shape[0], dk_full.shape[2]
    if rep_k > 1:
        dk_f = dk_full.view(batch, k_heads, rep_k, seq_kv, head_dim).sum(dim=2)
    else:
        dk_f = dk_full
    if rep_v > 1:
        dv_f = dv_full.view(batch, v_heads, rep_v, seq_kv, int(v.shape[-1])).sum(dim=2)
    else:
        dv_f = dv_full

    _store_tensor(tensors, dq_uid, dq.to(dtype=q.dtype))
    _store_tensor(tensors, dk_uid, dk_f.to(dtype=k.dtype))
    _store_tensor(tensors, dv_uid, dv_f.to(dtype=v.dtype))
