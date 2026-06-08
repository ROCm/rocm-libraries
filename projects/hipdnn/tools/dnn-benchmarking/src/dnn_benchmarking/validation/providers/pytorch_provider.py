# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch reference provider for hipDNN graph validation.

Computes reference outputs by parsing graph JSON and executing
equivalent PyTorch operations on CPU.
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np

from ...execution import pytorch_ops
from ..reference_provider import (
    ReferenceOutput,
    ReferenceProvider,
    ReferenceProviderRegistry,
)


_TORCH_DTYPE_BY_GRAPH_TYPE = {
    "float": "float32",
    "half": "float16",
    "bfloat16": "bfloat16",
    "double": "float64",
    "int8": "int8",
    "int32": "int32",
    "uint8": "uint8",
}


def _tensor_metadata(graph_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(tensor["uid"]): tensor
        for tensor in graph_json.get("tensors", [])
        if "uid" in tensor
    }


def _torch_dtype_for_tensor(
    torch: Any, tensor_json: Optional[Dict[str, Any]]
) -> Optional[Any]:
    if tensor_json is None:
        return None
    dtype_name = _TORCH_DTYPE_BY_GRAPH_TYPE.get(
        str(tensor_json.get("data_type", "float")).lower()
    )
    if dtype_name is None:
        raise ValueError(
            f"PyTorch reference does not support tensor data_type "
            f"'{tensor_json.get('data_type')}' for tensor UID {tensor_json.get('uid')}"
        )
    return getattr(torch, dtype_name)


def _numpy_output_for_tensor(tensor: Any, graph_dtype: Optional[str]) -> np.ndarray:
    # NumPy has no native bfloat16 dtype. Convert BF16 tensors to float32
    # numeric values rather than returning their uint16 storage encoding.
    if graph_dtype == "bfloat16":
        return tensor.detach().cpu().to(dtype=tensor.float().dtype).numpy()
    return tensor.detach().cpu().numpy()


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

        # Convert input data to torch tensors on CPU. When graph metadata names
        # a dtype, force PyTorch to execute with that dtype; validating a BF16
        # hipDNN graph against FP32 PyTorch math is a different computation.
        tensor_json_by_uid = _tensor_metadata(graph_json)
        tensors: Dict[int, torch.Tensor] = {}
        for uid, data in input_data.items():
            tensor = torch.from_numpy(data.copy()).cpu()
            graph_dtype = _torch_dtype_for_tensor(torch, tensor_json_by_uid.get(uid))
            if graph_dtype is not None:
                tensor = tensor.to(dtype=graph_dtype)
            tensors[uid] = tensor

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

        # Return outputs that exist in our tensor dict. NumPy has no native
        # bfloat16 dtype, so BF16 graph outputs are decoded to float32 for the
        # same comparison representation used by BufferManager.
        results: Dict[int, ReferenceOutput] = {}
        for uid in output_uids:
            if uid in tensors:
                tensor_json = tensor_json_by_uid.get(uid)
                graph_dtype = (
                    str(tensor_json.get("data_type", "")).lower()
                    if tensor_json is not None
                    else None
                )
                results[uid] = ReferenceOutput(
                    data=_numpy_output_for_tensor(tensors[uid], graph_dtype),
                    tensor_uid=uid,
                )

        return results
