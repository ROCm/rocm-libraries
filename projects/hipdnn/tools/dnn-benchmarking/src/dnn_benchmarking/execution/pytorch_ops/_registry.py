# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Operation-handler registry and graph execution."""

from typing import Any, Callable, Dict, List, Optional, Set

import torch


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
