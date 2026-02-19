# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch operation implementations for graph execution.

These handlers execute on the device of the input tensors (CPU or CUDA).
Used by both PyTorchReferenceProvider (CPU) and PyTorchCudaExecutor (CUDA).
"""

from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.nn.functional as F

# Type alias for operation handlers
OpHandler = Callable[[Dict[str, Any], Dict[int, torch.Tensor], Dict[str, Any]], None]

# Registry of operation handlers
_OP_HANDLERS: Dict[str, OpHandler] = {}


def register_handler(op_type: str) -> Callable[[OpHandler], OpHandler]:
    """Decorator to register an operation handler.

    Args:
        op_type: The node type string to handle (e.g., "ConvolutionFwdAttributes").

    Returns:
        Decorator function.
    """

    def decorator(func: OpHandler) -> OpHandler:
        _OP_HANDLERS[op_type] = func
        return func

    return decorator


def get_handler(op_type: str) -> Optional[OpHandler]:
    """Get handler for operation type.

    Args:
        op_type: The node type string.

    Returns:
        Handler function or None if not found.
    """
    return _OP_HANDLERS.get(op_type)


def get_supported_operations() -> Set[str]:
    """Get set of supported operation types.

    Returns:
        Set of operation type strings that have handlers.
    """
    return set(_OP_HANDLERS.keys())


def supports_graph(graph_json: Dict[str, Any]) -> bool:
    """Check if all graph operations are supported.

    Args:
        graph_json: The graph as a parsed JSON dictionary.

    Returns:
        True if all node types have handlers.
    """
    for node in graph_json.get("nodes", []):
        if node.get("type") not in _OP_HANDLERS:
            return False
    return True


def get_unsupported_operations(graph_json: Dict[str, Any]) -> List[str]:
    """Get list of unsupported operation types in graph.

    Args:
        graph_json: The graph as a parsed JSON dictionary.

    Returns:
        List of unsupported operation type strings.
    """
    unsupported = []
    for node in graph_json.get("nodes", []):
        op_type = node.get("type")
        if op_type not in _OP_HANDLERS:
            unsupported.append(op_type)
    return unsupported


def execute_graph(
    graph_json: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
) -> None:
    """Execute all graph operations in order.

    Args:
        graph_json: The graph as a parsed JSON dictionary.
        tensors: Mapping of tensor UID to torch.Tensor.

    Raises:
        ValueError: If graph contains unsupported operations.
    """
    for node in graph_json.get("nodes", []):
        op_type = node.get("type")
        handler = _OP_HANDLERS.get(op_type)
        if handler:
            handler(node, tensors, graph_json)
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")


# -----------------------------------------------------------------------------
# Operation Handlers
# -----------------------------------------------------------------------------


@register_handler("ConvolutionFwdAttributes")
def handle_conv_fwd(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle ConvolutionFwdAttributes (2D convolution forward pass).

    Maps to torch.nn.functional.conv2d.
    """
    inputs = node.get("inputs", {})
    outputs = node.get("outputs", {})
    params = node.get("parameters", {})

    x_uid = inputs.get("x_tensor_uid")
    w_uid = inputs.get("w_tensor_uid")
    y_uid = outputs.get("y_tensor_uid")

    if x_uid is None or w_uid is None or y_uid is None:
        raise ValueError(f"Conv node missing required tensor UIDs: {node}")

    x = tensors[x_uid]
    w = tensors[w_uid]

    # Extract convolution parameters
    padding = tuple(params.get("pre_padding", [0, 0]))
    stride = tuple(params.get("stride", [1, 1]))
    dilation = tuple(params.get("dilation", [1, 1]))

    # Perform convolution
    # hipDNN uses NCHW format, same as PyTorch default
    y = F.conv2d(x, w, padding=padding, stride=stride, dilation=dilation)

    tensors[y_uid] = y


@register_handler("MatmulAttributes")
def handle_matmul(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle MatmulAttributes (matrix multiplication).

    Maps to torch.matmul.
    """
    inputs = node.get("inputs", {})
    outputs = node.get("outputs", {})

    a_uid = inputs.get("a_tensor_uid")
    b_uid = inputs.get("b_tensor_uid")
    c_uid = outputs.get("c_tensor_uid")

    if a_uid is None or b_uid is None or c_uid is None:
        raise ValueError(f"Matmul node missing required tensor UIDs: {node}")

    a = tensors[a_uid]
    b = tensors[b_uid]

    c = torch.matmul(a, b)

    tensors[c_uid] = c


@register_handler("PointwiseAttributes")
def handle_pointwise(
    node: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
    graph_json: Dict[str, Any],
) -> None:
    """Handle PointwiseAttributes (element-wise operations).

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

    in0 = tensors[in0_uid]
    in1 = tensors.get(in1_uid) if in1_uid is not None else None

    # Map operation to PyTorch equivalent
    if operation == "relu_fwd":
        # Check for clipping bounds (ReLU6-style)
        lower_clip = inputs.get("relu_lower_clip", 0.0)
        upper_clip = inputs.get("relu_upper_clip", float("inf"))

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

    tensors[out_uid] = out
