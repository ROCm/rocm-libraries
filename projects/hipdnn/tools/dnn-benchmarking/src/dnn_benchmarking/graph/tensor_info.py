# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tensor information dataclass."""

import warnings
from dataclasses import dataclass, field
from typing import List

# Map data type strings to byte sizes
DTYPE_SIZES = {
    "float": 4,
    "half": 2,
    "bfloat16": 2,
    "double": 8,
    "int8": 1,
    "int32": 4,
    "uint8": 1,
}


@dataclass
class TensorInfo:
    """Information about a tensor extracted from graph JSON.

    Attributes:
        uid: Unique identifier for the tensor.
        name: Human-readable name of the tensor.
        dims: Dimensions of the tensor (e.g., [N, C, H, W]).
        strides: Memory strides for each dimension.
        data_type: Data type as string (e.g., "float", "half").
        is_virtual: Whether this is a virtual (intermediate) tensor.
        is_output: Whether this tensor is marked as a graph output.
    """

    uid: int
    name: str
    dims: List[int]
    strides: List[int]
    data_type: str
    is_virtual: bool
    is_output: bool = False

    @property
    def element_size(self) -> int:
        """Get size of one element in bytes."""
        dtype_lower = self.data_type.lower()
        size = DTYPE_SIZES.get(dtype_lower)
        if size is None:
            warnings.warn(
                f"Unknown data type '{self.data_type}', defaulting to 4 bytes",
                stacklevel=2,
            )
            return 4
        return size

    @property
    def num_elements(self) -> int:
        """Get total number of elements."""
        result = 1
        for dim in self.dims:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        """Get total size in bytes.

        Uses strides to compute actual memory footprint when available,
        since non-contiguous tensors may require more memory than
        product(dims) * element_size.
        """
        if self.strides:
            # Memory footprint = max(dim_i * stride_i) * element_size
            max_extent = max(d * s for d, s in zip(self.dims, self.strides))
            return max_extent * self.element_size
        return self.num_elements * self.element_size

    @classmethod
    def from_json(cls, tensor_json: dict, is_output: bool = False) -> "TensorInfo":
        """Create TensorInfo from a JSON tensor object.

        Args:
            tensor_json: Dictionary containing tensor attributes from graph JSON.
            is_output: Whether this tensor is a graph output.

        Returns:
            TensorInfo instance.
        """
        return cls(
            uid=tensor_json["uid"],
            name=tensor_json.get("name", f"tensor_{tensor_json['uid']}"),
            dims=tensor_json["dims"],
            strides=tensor_json.get("strides", []),
            data_type=tensor_json.get("data_type", "float"),
            is_virtual=tensor_json.get("virtual", False),
            is_output=is_output,
        )
