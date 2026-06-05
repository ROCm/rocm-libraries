# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""FLOP handler for matrix multiplication (MatmulAttributes)."""

from typing import Any, Dict, Optional


def matmul_flops(
    node: Dict[str, Any], tensors_by_uid: Dict[int, Dict[str, Any]]
) -> Optional[int]:
    """FLOPs for MatmulAttributes: ``2 * batch * M * N * K``.

    Supports batched matmul by multiplying all leading dims of the
    output tensor.
    """
    inputs = node.get("inputs", {}) or {}
    outputs = node.get("outputs", {}) or {}
    a_uid = inputs.get("a_tensor_uid")
    b_uid = inputs.get("b_tensor_uid")
    c_uid = outputs.get("c_tensor_uid")
    if a_uid is None or b_uid is None or c_uid is None:
        return None
    a = tensors_by_uid.get(int(a_uid))
    b = tensors_by_uid.get(int(b_uid))
    c = tensors_by_uid.get(int(c_uid))
    if not a or not b or not c:
        return None

    a_dims = a.get("dims") or []
    b_dims = b.get("dims") or []
    c_dims = c.get("dims") or []
    if len(a_dims) < 2 or len(b_dims) < 2 or len(c_dims) < 2:
        return None

    m = int(c_dims[-2])
    n = int(c_dims[-1])
    k = int(a_dims[-1])

    batch = 1
    for d in c_dims[:-2]:
        batch *= int(d)

    return 2 * batch * m * n * k
