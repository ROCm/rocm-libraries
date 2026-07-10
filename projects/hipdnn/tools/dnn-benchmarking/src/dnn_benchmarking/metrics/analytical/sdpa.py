# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handler for scaled-dot-product attention forward (SdpaAttributes).

Attention forward is two batched matmuls (``QKᵀ`` then ``P·V``); FMA = 2
FLOPs. Only query heads count (GQA repeats KV heads, which adds no
arithmetic); softmax/exp/scaling are ignored (matmul-only convention).

    flops = 2 * batch * num_q_heads * num_nonmasked * (head_dim_qk + head_dim_vo)

where ``num_nonmasked`` is the score-matrix area actually computed per
query head: ``Sq * Skv`` unmasked, or the causal lower-triangle count.
"""

from typing import Any, Dict, Optional


def sdpa_fwd_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for SdpaAttributes (forward attention).

    Returns None (marking the graph partial) when q/k/v tensor data is
    incomplete or a sliding-window mask is present (the bound->window
    mapping is not modelled).
    """
    inputs = node.get("inputs", {}) or {}
    q_uid = inputs.get("q_tensor_uid")
    k_uid = inputs.get("k_tensor_uid")
    v_uid = inputs.get("v_tensor_uid")
    if q_uid is None or k_uid is None or v_uid is None:
        return None
    q = tensors_by_uid.get(int(q_uid))
    k = tensors_by_uid.get(int(k_uid))
    v = tensors_by_uid.get(int(v_uid))
    if not q or not k or not v:
        return None

    q_dims = q.get("dims") or []
    k_dims = k.get("dims") or []
    v_dims = v.get("dims") or []
    if len(q_dims) < 3 or len(k_dims) < 3 or len(v_dims) < 3:
        return None

    q_heads = int(q_dims[-3])
    q_seqlen = int(q_dims[-2])
    head_dim_qk = int(q_dims[-1])
    kv_seqlen = int(k_dims[-2])
    head_dim_vo = int(v_dims[-1])

    batch = 1
    for d in q_dims[:-3]:
        batch *= int(d)

    attributes = node.get("attributes", {}) or {}

    # Sliding window not modelled: report unknown rather than a wrong count.
    if (
        attributes.get("left_bound") is not None
        or attributes.get("right_bound") is not None
    ):
        return None

    if attributes.get("causal_mask") is True:
        diagonal_alignment = attributes.get("diagonal_alignment", "TOP_LEFT")
        if diagonal_alignment in ("TOP_LEFT", 0, None):
            offset = 0
        else:
            offset = kv_seqlen - q_seqlen
        num_nonmasked = sum(
            min(max(i + 1 + offset, 0), kv_seqlen) for i in range(q_seqlen)
        )
    else:
        num_nonmasked = q_seqlen * kv_seqlen

    return 2 * batch * q_heads * num_nonmasked * (head_dim_qk + head_dim_vo)
