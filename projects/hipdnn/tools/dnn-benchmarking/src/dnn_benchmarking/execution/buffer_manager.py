# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Device buffer management for graph execution."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..common.exceptions import ExecutionError
from ..graph.tensor_info import TensorInfo

# Map data type strings to numpy dtypes (bfloat16 handled separately via torch)
DTYPE_MAP = {
    "float": np.float32,
    "half": np.float16,
    "double": np.float64,
    "int8": np.int8,
    "int32": np.int32,
    "uint8": np.uint8,
}


def _generate_bfloat16_bytes(
    dims: List[int], rng: np.random.RandomState = None
) -> bytes:
    """Generate random data in bfloat16 format using torch.

    Args:
        dims: Tensor dimensions.
        rng: Random state for reproducibility. If None, creates a new one.

    Returns:
        Raw bytes in bfloat16 format.
    """
    if rng is None:
        rng = np.random.RandomState()
    data_f32 = rng.uniform(0.0, 1.0, dims).astype(np.float32)
    t = torch.from_numpy(data_f32).bfloat16()
    # Get raw bfloat16 bytes via untyped_storage
    storage = t.untyped_storage()
    return bytes(storage)


def _bfloat16_bytes_to_ndarray(data_bytes: bytes, dims: List[int]) -> np.ndarray:
    """Convert raw bfloat16 bytes to a float32 numpy array via torch.

    Args:
        data_bytes: Raw bytes in bfloat16 format.
        dims: Tensor dimensions for reshaping.

    Returns:
        Float32 numpy array with the bfloat16 values upcast.
    """
    t = torch.frombuffer(bytearray(data_bytes), dtype=torch.bfloat16)
    return t.float().numpy().reshape(dims)


class BufferManager:
    """Manages device buffer allocation and data transfer for graph execution.

    This class handles:
    - Allocating device buffers for all tensors
    - Creating variant packs (UID -> pointer mapping)
    - Filling input buffers with random data
    - Cleanup of device memory
    """

    def __init__(self, tensor_infos: List[TensorInfo]) -> None:
        """Initialize buffer manager with tensor metadata.

        Args:
            tensor_infos: List of TensorInfo objects describing tensors.
        """
        self._tensor_infos = tensor_infos
        self._buffers: Dict[int, "DeviceBuffer"] = {}  # UID -> DeviceBuffer
        self._host_data: Dict[int, np.ndarray] = {}  # UID -> numpy array

    def allocate_all(self) -> None:
        """Allocate device buffers for all tensors.

        Raises:
            ExecutionError: If hipdnn_frontend is not available.
        """
        try:
            import hipdnn_frontend as hipdnn
        except ImportError as e:
            raise ExecutionError(
                "hipdnn_frontend not available. Install hipDNN Python bindings."
            ) from e

        for tensor_info in self._tensor_infos:
            if tensor_info.is_virtual:
                continue

            buffer = hipdnn.DeviceBuffer(tensor_info.size_bytes)
            self._buffers[tensor_info.uid] = buffer

    def create_variant_pack(self) -> Dict[int, int]:
        """Create variant pack mapping tensor UIDs to device pointers.

        Returns:
            Dictionary mapping tensor UID to device pointer (as int).

        Raises:
            ExecutionError: If buffers not allocated.
        """
        if not self._buffers:
            raise ExecutionError("Buffers not allocated. Call allocate_all() first.")

        return {uid: buffer.ptr() for uid, buffer in self._buffers.items()}

    def fill_inputs_random(self, seed: Optional[int] = None) -> None:
        """Fill input tensor buffers with random data.

        Args:
            seed: Optional random seed for reproducibility.

        Raises:
            ExecutionError: If buffers not allocated.
        """
        if not self._buffers:
            raise ExecutionError("Buffers not allocated. Call allocate_all() first.")

        if seed is not None:
            np.random.seed(seed)

        for tensor_info in self._tensor_infos:
            if tensor_info.is_output or tensor_info.is_virtual:
                continue

            dtype_key = tensor_info.data_type.lower()
            buffer = self._buffers.get(tensor_info.uid)

            if dtype_key == "bfloat16":
                # bfloat16 has a different binary format than float16;
                # use torch for correct conversion
                raw_bytes = _generate_bfloat16_bytes(tensor_info.dims)
                # Store as float32 for host-side validation
                self._host_data[tensor_info.uid] = _bfloat16_bytes_to_ndarray(
                    raw_bytes, tensor_info.dims
                )
                if buffer:
                    buffer.copy_from_host(raw_bytes)
            else:
                dtype = DTYPE_MAP.get(dtype_key, np.float32)
                data = np.random.uniform(0.0, 1.0, tensor_info.dims).astype(dtype)
                self._host_data[tensor_info.uid] = data
                if buffer:
                    buffer.copy_from_host(data.tobytes())

    def zero_outputs(self) -> None:
        """Zero output tensor buffers.

        Raises:
            ExecutionError: If buffers not allocated.
        """
        if not self._buffers:
            raise ExecutionError("Buffers not allocated. Call allocate_all() first.")

        for tensor_info in self._tensor_infos:
            if not tensor_info.is_output:
                continue

            buffer = self._buffers.get(tensor_info.uid)
            if buffer:
                buffer.zeros()

    def get_output_data(self, uid: int) -> Optional[np.ndarray]:
        """Copy output tensor data from device to host.

        Args:
            uid: Tensor UID.

        Returns:
            Numpy array with output data, or None if tensor not found.
        """
        buffer = self._buffers.get(uid)
        if buffer is None:
            return None

        # Find tensor info for shape and dtype
        tensor_info = None
        for ti in self._tensor_infos:
            if ti.uid == uid:
                tensor_info = ti
                break

        if tensor_info is None:
            return None

        dtype_key = tensor_info.data_type.lower()
        data_bytes = buffer.copy_to_host()

        if dtype_key == "bfloat16":
            return _bfloat16_bytes_to_ndarray(data_bytes, tensor_info.dims)

        dtype = DTYPE_MAP.get(dtype_key, np.float32)
        return np.frombuffer(data_bytes, dtype=dtype).reshape(tensor_info.dims)

    def get_output_tensors(self) -> List[TensorInfo]:
        """Get list of output tensor infos.

        Returns:
            List of TensorInfo objects for output tensors.
        """
        return [ti for ti in self._tensor_infos if ti.is_output]

    def get_input_data(self, uid: int) -> Optional[np.ndarray]:
        """Get the host-side input data for a tensor.

        Args:
            uid: Tensor UID.

        Returns:
            Numpy array with input data, or None if not found.
        """
        return self._host_data.get(uid)

    def cleanup(self) -> None:
        """Free all device buffers."""
        self._buffers.clear()
        self._host_data.clear()

    def __enter__(self) -> "BufferManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup buffers."""
        self.cleanup()
