# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PyTorch CUDA tensor management for graph execution."""

from typing import Dict, List, Optional

import numpy as np
import torch

from ..graph.tensor_info import TensorInfo
from .buffer_manager import generate_input_data

# Map data type strings to torch dtypes
TORCH_DTYPE_MAP = {
    "float": torch.float32,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "double": torch.float64,
    "int8": torch.int8,
    "int32": torch.int32,
    "uint8": torch.uint8,
}


class PyTorchCudaBufferManager:
    """Manages PyTorch CUDA tensor allocation for graph execution.

    This class handles:
    - Allocating CUDA tensors for all non-virtual tensors
    - Filling input tensors with random data
    - Zeroing output tensors
    - Providing tensors for graph execution
    """

    def __init__(
        self,
        tensor_infos: List[TensorInfo],
        device: str = "cuda:0",
    ) -> None:
        """Initialize buffer manager with tensor metadata.

        Args:
            tensor_infos: List of TensorInfo objects describing tensors.
            device: CUDA device to use (e.g., "cuda:0").
        """
        self._tensor_infos = tensor_infos
        self._device = torch.device(device)
        self._tensors: Dict[int, torch.Tensor] = {}
        self._host_data: Dict[int, np.ndarray] = {}

    def allocate_all(self) -> None:
        """Allocate CUDA tensors for all non-virtual tensors."""
        for tensor_info in self._tensor_infos:
            if tensor_info.is_virtual:
                continue

            dtype = TORCH_DTYPE_MAP.get(tensor_info.data_type.lower(), torch.float32)
            if tensor_info.is_pass_by_value:
                value = torch.tensor(
                    [tensor_info.value], dtype=dtype, device=self._device
                )
                self._tensors[tensor_info.uid] = value
                self._host_data[tensor_info.uid] = np.asarray([tensor_info.value])
                continue

            if tensor_info.strides:
                tensor = torch.empty_strided(
                    tensor_info.dims,
                    tensor_info.strides,
                    dtype=dtype,
                    device=self._device,
                )
            else:
                tensor = torch.empty(
                    tensor_info.dims,
                    dtype=dtype,
                    device=self._device,
                )
            self._tensors[tensor_info.uid] = tensor

    @staticmethod
    def _host_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Copy a tensor to numpy using the validation representation."""
        host = tensor.detach().cpu()
        if host.dtype == torch.bfloat16:
            host = host.to(dtype=torch.float32)
        return host.contiguous().numpy()

    def load_input_data(self, input_data: Dict[int, np.ndarray]) -> None:
        """Copy pre-generated graph input data into CUDA tensors.

        The host-to-device copy happens before timing starts so PyTorch
        reference rows preserve the same timing semantics as hipDNN rows.
        """
        for tensor_info in self._tensor_infos:
            if tensor_info.is_output or tensor_info.is_virtual:
                continue

            data = input_data.get(tensor_info.uid)
            if data is None:
                if tensor_info.is_pass_by_value:
                    data = np.asarray([tensor_info.value])
                else:
                    raise ValueError(
                        f"Missing input data for tensor UID {tensor_info.uid}"
                    )

            tensor = self._tensors.get(tensor_info.uid)
            if tensor is None:
                continue

            host_tensor = torch.from_numpy(np.asarray(data))
            tensor.copy_(host_tensor.to(dtype=tensor.dtype, device=self._device))
            self._host_data[tensor_info.uid] = self._host_numpy(tensor)

    def fill_inputs_random(self, seed: Optional[int] = None) -> None:
        """Fill input tensor buffers with random data.

        Args:
            seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)

        self.load_input_data(generate_input_data(self._tensor_infos, seed))

    def zero_outputs(self) -> None:
        """Zero output tensor buffers."""
        for tensor_info in self._tensor_infos:
            if not tensor_info.is_output:
                continue

            tensor = self._tensors.get(tensor_info.uid)
            if tensor is not None:
                tensor.zero_()

    def get_tensors(self) -> Dict[int, torch.Tensor]:
        """Get mapping of tensor UIDs to CUDA tensors.

        Returns:
            Dictionary mapping tensor UID to torch.Tensor on CUDA.
        """
        return self._tensors

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        """Copy output tensor data from CUDA to numpy array.

        Args:
            uid: Tensor UID.

        Returns:
            Numpy array with output data, or None if tensor not found.
        """
        tensor = self._tensors.get(uid)
        if tensor is None:
            return None

        return self._host_numpy(tensor)

    def get_input_data(self, uid: int) -> Optional[np.ndarray]:
        """Get the host-side input data for a tensor.

        Args:
            uid: Tensor UID.

        Returns:
            Numpy array with input data, or None if not found.
        """
        return self._host_data.get(uid)

    def get_output_tensors(self) -> List[TensorInfo]:
        """Get list of output tensor infos.

        Returns:
            List of TensorInfo objects for output tensors.
        """
        return [ti for ti in self._tensor_infos if ti.is_output]

    def cleanup(self) -> None:
        """Free all tensors."""
        self._tensors.clear()
        self._host_data.clear()
        # Let PyTorch handle CUDA memory cleanup via garbage collection
        torch.cuda.empty_cache()

    def __enter__(self) -> "PyTorchCudaBufferManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup tensors."""
        self.cleanup()
