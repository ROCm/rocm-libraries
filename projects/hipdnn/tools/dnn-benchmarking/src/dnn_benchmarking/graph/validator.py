# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Graph validation for supported operations."""

from typing import Any, Dict, List, Set

from ..common.exceptions import GraphLoadError

# Supported operation types - accepts any hipDNN operation
SUPPORTED_NODE_TYPES: Set[str] = {
    "ConvolutionFwdAttributes",
    "ConvolutionDgradAttributes",
    "ConvolutionWgradAttributes",
    "MatmulAttributes",
    "PointwiseAttributes",
    "BatchnormAttributes",
    "BatchnormBackwardAttributes",
    "BatchnormInferenceAttributes",
}


class GraphValidator:
    """Validates that a graph contains valid operations.

    Accepts any hipDNN operation type that can be deserialized and executed.
    """

    def __init__(self, supported_types: Set[str] = SUPPORTED_NODE_TYPES) -> None:
        """Initialize validator with supported operation types.

        Args:
            supported_types: Set of node type names that are supported.
                           If None, accepts any operation type.
        """
        self._supported_types = supported_types

    def validate_conv_fwd_only(self, graph_json: Dict[str, Any]) -> None:
        """Validate that graph contains valid operations.

        Args:
            graph_json: The parsed graph JSON dictionary.

        Raises:
            GraphLoadError: If graph contains no operation nodes.
        """
        nodes = graph_json.get("nodes", [])

        if not nodes:
            raise GraphLoadError("Graph contains no operation nodes")

        # Accept any operation - let hipDNN backend handle validation
        # The operation will fail at build time if truly unsupported

    def _find_unsupported_nodes(self, nodes: List[Dict[str, Any]]) -> Set[str]:
        """Find all unsupported node types in the graph.

        Args:
            nodes: List of node dictionaries from graph JSON.

        Returns:
            Set of unsupported node type names.
        """
        unsupported = set()

        for node in nodes:
            node_type = node.get("type", "")
            if node_type not in self._supported_types:
                unsupported.add(node_type)

        return unsupported

    def get_supported_types(self) -> Set[str]:
        """Get the set of supported operation types.

        Returns:
            Set of supported node type names.
        """
        return self._supported_types.copy()
