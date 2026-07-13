# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Operation-handler registry and graph execution."""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from ...common.exceptions import UnsupportedGraphError


# Type aliases for operation planners.
# CompiledOp replays a planned op against a live tensor map.
CompiledOp = Callable[[Dict[int, torch.Tensor]], None]
# OpHandler plans an op once from its node + graph JSON, returning a CompiledOp.
OpHandler = Callable[[Dict[str, Any], Dict[str, Any]], CompiledOp]

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


class CompiledGraph:
    """A graph compiled once into replayable per-op closures."""

    def __init__(self, ops: List[Tuple[str, CompiledOp]]) -> None:
        self._ops = ops

    def execute(self, tensors: Dict[int, torch.Tensor]) -> None:
        """Replay every compiled op against the provided tensor map.

        Args:
            tensors: Mapping of tensor UID to torch.Tensor.

        Raises:
            UnsupportedGraphError: If an op fails at runtime.
        """
        for op_type, run in self._ops:
            try:
                run(tensors)
            except UnsupportedGraphError:
                raise
            except ValueError as e:
                # A handler's ValueError signals an unsupported graph feature;
                # normalize to UnsupportedGraphError so callers skip it. Any
                # other exception is a real failure and propagates as an error.
                raise UnsupportedGraphError(
                    f"PyTorch reference could not execute {op_type!r} with the provided "
                    f"dtypes/parameters: {e}"
                ) from e


def compile_graph(graph_json: Dict[str, Any]) -> CompiledGraph:
    """Compile all graph operations into replayable closures once.

    Args:
        graph_json: The graph as a parsed JSON dictionary.

    Returns:
        A CompiledGraph that replays the planned ops on each execute().

    Raises:
        UnsupportedGraphError: If the graph contains an operation, attribute, or
            parameter the PyTorch reference does not support.
    """
    ops: List[Tuple[str, CompiledOp]] = []
    for node in graph_json.get("nodes", []):
        op_type = node.get("type")
        planner = _OP_HANDLERS.get(op_type)
        if planner is None:
            raise UnsupportedGraphError(f"Unsupported operation type: {op_type}")
        try:
            op = planner(node, graph_json)
        except UnsupportedGraphError:
            raise
        except ValueError as e:
            # A planner raises ValueError to signal a graph feature it cannot
            # represent; normalize to UnsupportedGraphError so callers skip it.
            # Any other exception is an unexpected failure and propagates.
            raise UnsupportedGraphError(
                f"PyTorch reference could not execute {op_type!r} with the provided "
                f"dtypes/parameters: {e}"
            ) from e
        ops.append((op_type, op))
    return CompiledGraph(ops)


def execute_graph(
    graph_json: Dict[str, Any],
    tensors: Dict[int, torch.Tensor],
) -> None:
    """Compile and execute all graph operations in order.

    Args:
        graph_json: The graph as a parsed JSON dictionary.
        tensors: Mapping of tensor UID to torch.Tensor.

    Raises:
        UnsupportedGraphError: If the graph contains an operation, attribute, or
            parameter the PyTorch reference does not support.
    """
    compile_graph(graph_json).execute(tensors)
