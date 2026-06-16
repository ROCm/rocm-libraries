# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handlers for element-wise ops driven by output element count.

Pointwise (relu, add, mul, …) and Rng both do O(num_output_elements)
work and are dominated by memory traffic. We don't distinguish unary
vs binary pointwise because the FLOP component is small relative to
fused-graph totals.
"""

from typing import Any, Dict, Optional

from ._common import tensor_dim_product


def pointwise_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """PointwiseAttributes: 1 FLOP per output element.

    Element-wise operations (relu, add, mul, sub, div, abs, neg, exp,
    log, tanh, sigmoid, sqrt) all do O(num_elements) work.
    """
    outputs = node.get("outputs", {}) or {}
    out_uid = outputs.get("out_0_tensor_uid")
    if out_uid is None:
        return None
    out = tensors_by_uid.get(int(out_uid))
    if not out:
        return None
    return tensor_dim_product(out)


def rng_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """Rng: 1 op per generated value (conservative — most PRNGs do more)."""
    outputs = node.get("outputs", {}) or {}
    out_uid = outputs.get("out_0_tensor_uid")
    if out_uid is None:
        out_uid = outputs.get("y_tensor_uid")
    if out_uid is None:
        return None
    tensor = tensors_by_uid.get(int(out_uid))
    if not tensor:
        return None
    return tensor_dim_product(tensor)
