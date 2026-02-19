# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch reference provider for hipDNN graph validation.

Computes reference outputs by parsing graph JSON and executing
equivalent PyTorch operations on CPU.
"""

from typing import Any, Dict, List, Set

import numpy as np

from ...execution import pytorch_ops
from ..reference_provider import (
    ReferenceOutput,
    ReferenceProvider,
    ReferenceProviderRegistry,
)


@ReferenceProviderRegistry.register("pytorch")
class PyTorchReferenceProvider(ReferenceProvider):
    """Reference provider using PyTorch for computation.

    Parses hipDNN graph JSON and executes equivalent PyTorch operations
    to produce reference outputs for validation.

    Uses the shared operation handlers from pytorch_ops module, executing
    on CPU tensors for reference computation.

    Supported operations:
    - ConvolutionFwdAttributes: 2D convolution forward pass
    - MatmulAttributes: Matrix multiplication
    - PointwiseAttributes: Element-wise operations (relu, add, mul, etc.)
    """

    @property
    def name(self) -> str:
        """Provider name."""
        return "pytorch"

    def is_available(self) -> bool:
        """Check if PyTorch is available.

        Returns:
            True if torch can be imported.
        """
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_operations(self) -> Set[str]:
        """Get set of supported operation types.

        Returns:
            Set of operation type strings that have handlers.
        """
        return pytorch_ops.get_supported_operations()

    def supports_graph(self, graph_json: Dict[str, Any]) -> bool:
        """Check if all graph operations are supported.

        Args:
            graph_json: The graph as a parsed JSON dictionary.

        Returns:
            True if all node types have handlers.
        """
        return pytorch_ops.supports_graph(graph_json)

    def get_unsupported_operations(self, graph_json: Dict[str, Any]) -> List[str]:
        """Get list of unsupported operation types in graph.

        Args:
            graph_json: The graph as a parsed JSON dictionary.

        Returns:
            List of unsupported operation type strings.
        """
        return pytorch_ops.get_unsupported_operations(graph_json)

    def compute_reference(
        self,
        graph_json: Dict[str, Any],
        input_data: Dict[int, np.ndarray],
    ) -> Dict[int, ReferenceOutput]:
        """Compute reference outputs using PyTorch on CPU.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            input_data: Mapping of tensor UID to input numpy arrays.

        Returns:
            Mapping of output tensor UID to ReferenceOutput.

        Raises:
            ImportError: If PyTorch is not available.
            ValueError: If graph contains unsupported operations.
        """
        if not self.is_available():
            raise ImportError(
                "PyTorch is not available. Install with: pip install torch"
            )

        import torch

        # Check for unsupported operations
        unsupported = self.get_unsupported_operations(graph_json)
        if unsupported:
            raise ValueError(
                f"Graph contains unsupported operations: {unsupported}. "
                f"Supported: {list(self.supported_operations())}"
            )

        # Convert input data to torch tensors on CPU
        tensors: Dict[int, torch.Tensor] = {}
        for uid, data in input_data.items():
            tensors[uid] = torch.from_numpy(data.copy()).cpu()

        # Execute graph using shared handlers (works on CPU tensors)
        pytorch_ops.execute_graph(graph_json, tensors)

        # Extract output tensors
        # Build set of output UIDs from all nodes
        output_uids: Set[int] = set()
        for node in graph_json.get("nodes", []):
            outputs = node.get("outputs", {})
            for uid in outputs.values():
                if uid is not None:
                    output_uids.add(uid)

        # Return outputs that exist in our tensor dict
        results: Dict[int, ReferenceOutput] = {}
        for uid in output_uids:
            if uid in tensors:
                results[uid] = ReferenceOutput(
                    data=tensors[uid].cpu().numpy(),
                    tensor_uid=uid,
                )

        return results
