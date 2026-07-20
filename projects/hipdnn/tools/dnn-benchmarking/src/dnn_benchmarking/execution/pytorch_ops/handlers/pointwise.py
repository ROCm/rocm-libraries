# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pointwise reference handler."""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .._common import *  # noqa: F401,F403
from .._registry import CompiledOp, register_handler


@register_handler("PointwiseAttributes")
def compile_pointwise(node: Dict[str, Any], graph_json: Dict[str, Any]) -> CompiledOp:
    """Plan PointwiseAttributes (element-wise operations).

    Supports: relu_fwd, add, mul, sub, div, sqrt, abs, neg, exp, log, tanh_fwd, sigmoid_fwd
    """
    inputs = node.get("inputs", {})
    outputs = node.get("outputs", {})

    operation = inputs.get("operation", "")
    in0_uid = inputs.get("in_0_tensor_uid")
    in1_uid = inputs.get("in_1_tensor_uid")
    out_uid = outputs.get("out_0_tensor_uid")

    if in0_uid is None or out_uid is None:
        raise ValueError(f"Pointwise node missing required tensor UIDs: {node}")

    # Clipping bounds for ReLU6-style clamping.
    lower_clip = inputs.get("relu_lower_clip", 0.0)
    upper_clip = inputs.get("relu_upper_clip", float("inf"))

    def run(tensors: Dict[int, torch.Tensor]) -> None:
        in0 = tensors[in0_uid]
        in1 = tensors.get(in1_uid) if in1_uid is not None else None

        # Map operation to PyTorch equivalent
        if operation == "relu_fwd":
            if upper_clip == float("inf") or upper_clip >= 1e30:
                # Standard ReLU
                out = F.relu(in0)
            else:
                # Clipped ReLU (e.g., ReLU6)
                out = torch.clamp(in0, min=lower_clip, max=upper_clip)

        elif operation == "add":
            if in1 is None:
                raise ValueError("Add operation requires two inputs")
            out = in0 + in1

        elif operation == "mul":
            if in1 is None:
                raise ValueError("Mul operation requires two inputs")
            out = in0 * in1

        elif operation == "sub":
            if in1 is None:
                raise ValueError("Sub operation requires two inputs")
            out = in0 - in1

        elif operation == "div":
            if in1 is None:
                raise ValueError("Div operation requires two inputs")
            out = in0 / in1

        elif operation == "sqrt":
            out = torch.sqrt(in0)

        elif operation == "abs":
            out = torch.abs(in0)

        elif operation == "neg":
            out = -in0

        elif operation == "exp":
            out = torch.exp(in0)

        elif operation == "log":
            out = torch.log(in0)

        elif operation == "tanh_fwd":
            out = torch.tanh(in0)

        elif operation == "sigmoid_fwd":
            out = torch.sigmoid(in0)

        else:
            raise ValueError(f"Unsupported pointwise operation: {operation}")

        _store_tensor(tensors, out_uid, out)

    return run
