# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handler for ReductionAttributes.

Reduction work scales with the input element count (the output is
typically a scalar or row), so this is the one normalization-adjacent
op that's input-driven instead of output-driven.
"""

from typing import Any, Dict, Optional

from ._common import tensor_dim_product


def reduction_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """Reduction: 1 op/elem of the *input* tensor."""
    inputs = node.get("inputs", {}) or {}
    in_uid = inputs.get("x_tensor_uid") or inputs.get("in_0_tensor_uid")
    if in_uid is None:
        return None
    tensor = tensors_by_uid.get(int(in_uid))
    if not tensor:
        return None
    return tensor_dim_product(tensor)
