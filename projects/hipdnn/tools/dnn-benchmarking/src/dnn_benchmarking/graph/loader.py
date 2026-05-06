# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Graph loading from JSON files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..common.exceptions import GraphLoadError
from .tensor_info import TensorInfo
from .validator import GraphValidator


class GraphLoader:
    """Loads and parses hipDNN graph JSON files.

    Handles JSON loading, validation, and tensor info extraction.
    """

    def __init__(self, validator: Optional[GraphValidator] = None) -> None:
        """Initialize loader with optional validator.

        Args:
            validator: GraphValidator instance for operation validation.
                      If None, a default validator is created.
        """
        self._validator = validator or GraphValidator()

    def load_json(self, path: Path) -> Dict[str, Any]:
        """Load and parse a graph JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            GraphLoadError: If file cannot be read or parsed.
        """
        if not path.exists():
            raise GraphLoadError(f"Graph file not found: {path}")

        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise GraphLoadError(f"Invalid JSON in graph file: {e}") from e
        except OSError as e:
            raise GraphLoadError(f"Cannot read graph file: {e}") from e

    def validate(self, graph_json: Dict[str, Any]) -> None:
        """Validate that graph contains only supported operations.

        Args:
            graph_json: Parsed graph JSON dictionary.

        Raises:
            GraphLoadError: If graph contains unsupported operations.
        """
        self._validator.validate(graph_json)

    def extract_tensor_info(self, graph_json: Dict[str, Any]) -> List[TensorInfo]:
        """Extract tensor information from graph JSON.

        Args:
            graph_json: Parsed graph JSON dictionary.

        Returns:
            List of TensorInfo objects for all non-virtual tensors.
        """
        tensors = graph_json.get("tensors", [])
        output_uids = self._get_output_tensor_uids(graph_json)

        result = []
        for tensor_json in tensors:
            is_output = tensor_json.get("uid") in output_uids
            tensor_info = TensorInfo.from_json(tensor_json, is_output=is_output)

            # Skip virtual tensors - they don't need buffers
            if not tensor_info.is_virtual:
                result.append(tensor_info)

        return result

    def _get_output_tensor_uids(self, graph_json: Dict[str, Any]) -> Set[int]:
        """Get UIDs of output tensors from graph nodes.

        Args:
            graph_json: Parsed graph JSON dictionary.

        Returns:
            Set of output tensor UIDs.
        """
        output_uids = set()

        for node in graph_json.get("nodes", []):
            outputs = node.get("outputs", {})
            for key, value in outputs.items():
                if isinstance(value, int):
                    output_uids.add(value)

        return output_uids

    def get_graph_name(self, graph_json: Dict[str, Any]) -> str:
        """Get the name of the graph.

        Args:
            graph_json: Parsed graph JSON dictionary.

        Returns:
            Graph name, or "unnamed_graph" if not specified.
        """
        return graph_json.get("name", "unnamed_graph")

    def get_engine_id(self, graph_json: Dict[str, Any]) -> Optional[int]:
        """Get the preferred engine ID from the graph.

        Args:
            graph_json: Parsed graph JSON dictionary.

        Returns:
            Preferred engine ID, or None if not specified.
        """
        return graph_json.get("preferred_engine_id")
