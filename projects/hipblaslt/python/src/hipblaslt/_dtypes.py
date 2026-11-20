# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""NumPy and ml_dtypes types for hipBLASLt's data types, and the host side of DeviceArray.

nanobind only accepts NumPy's own scalar types, so an ml_dtypes array (bfloat16, the fp8 formats)
crosses into C++ as a same-width unsigned view, with its dtype name alongside. The C++ side
(csrc/device_array.cpp) does every check, so this module only does the viewing.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

from . import _core

__all__ = ["DTYPE_TO_NUMPY", "from_numpy", "numpy_dtype", "to_numpy"]

# Every hipDataType DeviceArray holds, by the NumPy dtype name its C++ side expects.
DTYPE_TO_NUMPY: dict[_core.DataType, np.dtype] = {
    _core.DataType(value): np.dtype(getattr(ml_dtypes, name, None) or name)
    for value, (name, _) in _core._host_types().items()
}

_NUMPY_TO_DTYPE = {v: k for k, v in DTYPE_TO_NUMPY.items()}


def numpy_dtype(dtype: _core.DataType) -> np.dtype:
    try:
        return DTYPE_TO_NUMPY[dtype]
    except KeyError:
        raise ValueError(f"DeviceArray doesn't support {dtype!r} yet") from None


def _wire(arr: np.ndarray) -> tuple[np.ndarray, str | None]:
    """The array nanobind accepts, plus the dtype name for an ml_dtypes array."""
    if arr.dtype.type.__module__ == "numpy":
        return arr, None
    width = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[arr.dtype.itemsize]
    return arr.view(width), arr.dtype.name


def from_numpy(
    arr: np.ndarray, dtype: _core.DataType | None = None
) -> _core.DeviceArray:
    """Copy a C-contiguous array to the device. `dtype` defaults to the array's own type."""
    if dtype is None:
        try:
            dtype = _NUMPY_TO_DTYPE[arr.dtype]
        except KeyError:
            raise ValueError(f"no hipBLASLt data type holds {arr.dtype}") from None
    wire, name = _wire(arr)
    return _core.DeviceArray.from_numpy(wire, dtype, name)


def to_numpy(self: _core.DeviceArray) -> np.ndarray:
    """Copy a DeviceArray back into a new NumPy array of its shape and type."""
    out = np.empty(self.shape, dtype=numpy_dtype(self.dtype))
    wire, name = _wire(out)
    self.copy_to_host(wire, name)
    return out
