# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Graph validation for supported operations."""

from typing import Any, Dict

from ..common.exceptions import GraphLoadError


class GraphValidator:
    """Validates graph structure before passing to hipDNN.

    Only checks basic structural requirements (e.g. non-empty nodes).
    Operation-level validation is deferred to hipDNN at build time.
    """

    def validate(self, graph_json: Dict[str, Any]) -> None:
        """Validate basic graph structure.

        Args:
            graph_json: The parsed graph JSON dictionary.

        Raises:
            GraphLoadError: If graph contains no operation nodes.
        """
        nodes = graph_json.get("nodes", [])

        if not nodes:
            raise GraphLoadError("Graph contains no operation nodes")
