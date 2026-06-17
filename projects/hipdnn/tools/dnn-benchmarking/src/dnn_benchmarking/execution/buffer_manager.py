# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Device buffer management for graph execution."""

from typing import Dict, List, Optional

import numpy as np

from ..common.exceptions import ExecutionError
from ..graph.tensor_info import TensorInfo

# Map data type strings to numpy dtypes (bfloat16 handled separately via uint16 bit manipulation)
DTYPE_MAP = {
    "float": np.float32,
    "half": np.float16,
    "double": np.float64,
    "int8": np.int8,
    "int32": np.int32,
    "uint8": np.uint8,
}


def _f32_to_bf16_bytes(data_f32: np.ndarray) -> bytes:
    """Convert a float32 numpy array to raw bfloat16 bytes (round-to-nearest-even).

    bfloat16 is the upper 16 bits of an IEEE-754 float32. This helper
    rounds each f32 word to its high half-word using round-to-nearest,
    ties-to-even (RNE), matching ``torch.Tensor.bfloat16()``.

    NaN inputs are preserved as NaN: the quiet bit (high mantissa bit
    of the bf16) is forced on so truncation cannot collapse the value
    into an infinity, and the rounding bias is skipped so the exponent
    cannot overflow.

    Args:
        data_f32: Float32 numpy array.

    Returns:
        Raw bytes in bfloat16 format.
    """
    f32_bits = data_f32.astype(np.float32).view(np.uint32).copy()
    exp_mask = np.uint32(0x7F800000)
    mant_mask = np.uint32(0x007FFFFF)
    is_nan = ((f32_bits & exp_mask) == exp_mask) & ((f32_bits & mant_mask) != 0)
    # RNE bias: 0x7FFF rounds half away from zero; adding the LSB of
    # the eventual bf16 word ties to even.
    lsb = (f32_bits >> np.uint32(16)) & np.uint32(1)
    rounding_bias = lsb + np.uint32(0x7FFF)
    rounded = f32_bits + rounding_bias
    # Preserve NaN: keep original exponent, set the bf16 quiet bit.
    rounded = np.where(is_nan, f32_bits | np.uint32(0x00400000), rounded)
    bf16 = (rounded >> np.uint32(16)).astype(np.uint16)
    return bf16.tobytes()


def _generate_bfloat16_bytes(
    dims: List[int], rng: np.random.RandomState = None
) -> bytes:
    """Generate random data in bfloat16 format using numpy bit manipulation.

    Samples uniformly in [0, 1) as float32, then converts to bfloat16
    via :func:`_f32_to_bf16_bytes` (round-to-nearest-even).

    Args:
        dims: Tensor dimensions.
        rng: Random state for reproducibility. If None, creates a new one.

    Returns:
        Raw bytes in bfloat16 format.
    """
    if rng is None:
        rng = np.random.RandomState()
    data_f32 = rng.uniform(0.0, 1.0, dims).astype(np.float32)
    return _f32_to_bf16_bytes(data_f32)


def _bfloat16_bytes_to_ndarray(data_bytes: bytes, dims: List[int]) -> np.ndarray:
    """Convert raw bfloat16 bytes to a float32 numpy array via uint16 bit manipulation.

    Reverses :func:`_generate_bfloat16_bytes` by zero-padding each bfloat16
    value back to the upper half of a float32 word.

    Args:
        data_bytes: Raw bytes in bfloat16 format.
        dims: Tensor dimensions for reshaping.

    Returns:
        Float32 numpy array with the bfloat16 values upcast.
    """
    bf16 = np.frombuffer(data_bytes, dtype=np.uint16)
    f32_bits = bf16.astype(np.uint32) << np.uint32(16)
    return f32_bits.view(np.float32).reshape(dims)


def _storage_view(storage: np.ndarray, tensor_info: TensorInfo) -> np.ndarray:
    """Create a logical ndarray view over raw tensor storage."""
    if tensor_info.strides:
        byte_strides = tuple(
            stride * storage.dtype.itemsize for stride in tensor_info.strides
        )
        return np.lib.stride_tricks.as_strided(
            storage,
            shape=tuple(tensor_info.dims),
            strides=byte_strides,
        )
    return storage.reshape(tensor_info.dims)


def _dense_from_storage_bytes(
    data_bytes: bytes, tensor_info: TensorInfo, dtype: np.dtype
) -> np.ndarray:
    """Decode raw storage bytes into a dense logical ndarray."""
    storage = np.frombuffer(data_bytes, dtype=dtype, count=tensor_info.storage_elements)
    return np.ascontiguousarray(_storage_view(storage, tensor_info))


def _bfloat16_storage_bytes_to_ndarray(
    data_bytes: bytes, tensor_info: TensorInfo
) -> np.ndarray:
    """Decode raw bfloat16 storage bytes into a dense float32 logical ndarray.

    NumPy has no native bfloat16 dtype. The uint16 view is only the on-wire
    storage encoding; validation compares numeric values, so expose decoded
    bfloat16 values as float32.
    """
    storage = np.frombuffer(
        data_bytes, dtype=np.uint16, count=tensor_info.storage_elements
    )
    logical_bf16 = np.ascontiguousarray(_storage_view(storage, tensor_info))
    f32_bits = logical_bf16.astype(np.uint32) << np.uint32(16)
    return np.ascontiguousarray(f32_bits.view(np.float32).reshape(tensor_info.dims))


def _encode_dense_to_storage_bytes(data: np.ndarray, tensor_info: TensorInfo) -> bytes:
    """Encode a dense logical ndarray into raw storage bytes using graph strides."""
    storage = np.zeros(tensor_info.storage_elements, dtype=data.dtype)
    _storage_view(storage, tensor_info)[...] = data
    return storage.tobytes()


def _encode_bfloat16_dense_to_storage_bytes(
    data_f32: np.ndarray, tensor_info: TensorInfo
) -> bytes:
    """Encode dense float32 logical data into raw bfloat16 storage bytes."""
    logical_bf16 = np.frombuffer(_f32_to_bf16_bytes(data_f32), dtype=np.uint16).reshape(
        tensor_info.dims
    )
    storage = np.zeros(tensor_info.storage_elements, dtype=np.uint16)
    _storage_view(storage, tensor_info)[...] = logical_bf16
    return storage.tobytes()


def generate_input_data(
    tensor_infos: List[TensorInfo], seed: Optional[int] = None
) -> Dict[int, np.ndarray]:
    """Generate one graph-scoped logical input map.

    Returned arrays use the validation representation consumed by both hipDNN
    and reference providers: dense logical ndarrays keyed by tensor UID. BF16
    values are generated as FP32, rounded through BF16 storage, then decoded
    back to numeric FP32 because NumPy has no native bfloat16 dtype.
    """
    rng = np.random.RandomState(seed)
    input_data: Dict[int, np.ndarray] = {}

    for tensor_info in tensor_infos:
        if tensor_info.is_output or tensor_info.is_virtual:
            continue

        dtype_key = tensor_info.data_type.lower()
        if tensor_info.is_pass_by_value:
            dtype = DTYPE_MAP.get(dtype_key, np.float32)
            input_data[tensor_info.uid] = np.asarray([tensor_info.value], dtype=dtype)
            continue

        if dtype_key == "bfloat16":
            data_f32 = rng.uniform(0.0, 1.0, tensor_info.dims).astype(np.float32)
            raw_bytes = _encode_bfloat16_dense_to_storage_bytes(data_f32, tensor_info)
            input_data[tensor_info.uid] = _bfloat16_storage_bytes_to_ndarray(
                raw_bytes, tensor_info
            )
            continue

        dtype = DTYPE_MAP.get(dtype_key, np.float32)
        input_data[tensor_info.uid] = rng.uniform(0.0, 1.0, tensor_info.dims).astype(
            dtype
        )

    return input_data


class BufferManager:
    """Manages device buffer allocation and data transfer for graph execution.

    This class handles:
    - Allocating device buffers for all tensors
    - Creating variant packs (UID -> pointer mapping)
    - Filling input buffers with random data
    - Cleanup of device memory
    """

    def __init__(
        self,
        tensor_infos: List[TensorInfo],
    ) -> None:
        """Initialize buffer manager with tensor metadata.

        Args:
            tensor_infos: List of TensorInfo objects describing tensors.
        """
        self._tensor_infos = tensor_infos
        self._tensor_info_by_uid = {tensor.uid: tensor for tensor in tensor_infos}
        self._buffers: Dict[int, "DeviceBuffer"] = {}  # UID -> DeviceBuffer
        self._host_data: Dict[int, np.ndarray] = {}  # UID -> logical numpy array

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
            if tensor_info.is_virtual or tensor_info.is_pass_by_value:
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

    def _copy_logical_data_to_buffer(self, uid: int, data: np.ndarray) -> None:
        """Store logical host data and copy its graph-layout bytes to device."""
        tensor_info = self._tensor_info_by_uid.get(uid)
        if tensor_info is None:
            raise ExecutionError(f"Unknown tensor UID {uid}")

        dtype_key = tensor_info.data_type.lower()
        buffer = self._buffers.get(uid)

        if dtype_key == "bfloat16":
            raw_bytes = _encode_bfloat16_dense_to_storage_bytes(
                data.astype(np.float32), tensor_info
            )
            self._host_data[uid] = _bfloat16_storage_bytes_to_ndarray(
                raw_bytes, tensor_info
            )
        else:
            dtype = DTYPE_MAP.get(dtype_key, np.float32)
            logical_data = data.astype(dtype, copy=False)
            raw_bytes = _encode_dense_to_storage_bytes(logical_data, tensor_info)
            self._host_data[uid] = np.array(logical_data, copy=True)

        if buffer:
            buffer.copy_from_host(raw_bytes)

    def load_input_data(self, input_data: Dict[int, np.ndarray]) -> None:
        """Copy pre-generated graph input data into device buffers.

        Input generation and host-to-device copies are intentionally separate
        from benchmark timing. Call this after ``allocate_all`` and before
        creating the variant pack.
        """
        if not self._buffers:
            raise ExecutionError("Buffers not allocated. Call allocate_all() first.")

        for tensor_info in self._tensor_infos:
            if tensor_info.is_output or tensor_info.is_virtual:
                continue

            data = input_data.get(tensor_info.uid)
            if data is None:
                if tensor_info.is_pass_by_value:
                    dtype = DTYPE_MAP.get(tensor_info.data_type.lower(), np.float32)
                    self._host_data[tensor_info.uid] = np.asarray(
                        [tensor_info.value], dtype=dtype
                    )
                    continue
                raise ExecutionError(
                    f"Missing input data for tensor UID {tensor_info.uid}"
                )

            if tensor_info.is_pass_by_value:
                dtype = DTYPE_MAP.get(tensor_info.data_type.lower(), np.float32)
                self._host_data[tensor_info.uid] = np.asarray(data, dtype=dtype)
                continue

            self._copy_logical_data_to_buffer(tensor_info.uid, data)

    def fill_inputs_random(self, seed: Optional[int] = None) -> None:
        """Fill input tensor buffers with random data.

        Args:
            seed: Optional random seed for reproducibility.

        Raises:
            ExecutionError: If buffers not allocated.
        """
        if not self._buffers:
            raise ExecutionError("Buffers not allocated. Call allocate_all() first.")

        self.load_input_data(generate_input_data(self._tensor_infos, seed))

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
            return _bfloat16_storage_bytes_to_ndarray(data_bytes, tensor_info)

        dtype = DTYPE_MAP.get(dtype_key, np.float32)
        return _dense_from_storage_bytes(data_bytes, tensor_info, dtype)

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
