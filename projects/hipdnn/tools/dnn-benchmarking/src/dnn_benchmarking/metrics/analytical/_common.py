# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Shared helpers for analytical FLOP / IO handlers.

Each per-op module imports from here. Kept private so callers go
through :mod:`dnn_benchmarking.metrics.analytical` for the public API.
"""

from typing import Any, Dict, Optional


def tensor_dim_product(tensor: Dict[str, Any]) -> int:
    """Return the product of a tensor's dims, or 0 when dims are missing."""
    dims = tensor.get("dims") or []
    if not dims:
        return 0
    n = 1
    for d in dims:
        n *= int(d)
    return n


def tensor_lookup(graph_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Index the graph's tensor list by integer UID."""
    return {int(t["uid"]): t for t in graph_json.get("tensors", []) if "uid" in t}


def output_elements(
    node: Dict[str, Any],
    tensors_by_uid: Dict[int, Dict[str, Any]],
    output_key: str = "y_tensor_uid",
) -> Optional[int]:
    """Resolve the output tensor and return its element count, or None.

    Most element-wise ops scale with the output element count; this
    helper centralises the UID lookup + dim-product reduction.
    """
    outputs = node.get("outputs", {}) or {}
    out_uid = outputs.get(output_key)
    if out_uid is None:
        return None
    tensor = tensors_by_uid.get(int(out_uid))
    if not tensor:
        return None
    return tensor_dim_product(tensor)
