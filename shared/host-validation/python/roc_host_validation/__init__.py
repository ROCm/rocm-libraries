# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from ._roc_host_validation import *  # noqa: F403


_NUMPY_DTYPES = {
    ScalarType.Boolean: np.bool_,  # noqa: F405
    ScalarType.UInt8: np.uint8,  # noqa: F405
    ScalarType.Int8: np.int8,  # noqa: F405
    ScalarType.UInt16: np.uint16,  # noqa: F405
    ScalarType.Int16: np.int16,  # noqa: F405
    ScalarType.UInt32: np.uint32,  # noqa: F405
    ScalarType.Int32: np.int32,  # noqa: F405
    ScalarType.UInt64: np.uint64,  # noqa: F405
    ScalarType.Int64: np.int64,  # noqa: F405
    ScalarType.Float16: np.float16,  # noqa: F405
    ScalarType.BFloat16: np.float32,  # noqa: F405
    ScalarType.Float32: np.float32,  # noqa: F405
    ScalarType.Float64: np.float64,  # noqa: F405
    ScalarType.ComplexFloat32: np.complex64,  # noqa: F405
    ScalarType.ComplexFloat64: np.complex128,  # noqa: F405
    ScalarType.Float8E4M3: np.float32,  # noqa: F405
    ScalarType.Float8E5M2: np.float32,  # noqa: F405
    ScalarType.Float8E4M3Fnuz: np.float32,  # noqa: F405
    ScalarType.Float8E5M2Fnuz: np.float32,  # noqa: F405
    ScalarType.Float6E2M3: np.float32,  # noqa: F405
    ScalarType.Float6E3M2: np.float32,  # noqa: F405
    ScalarType.Float4E2M1: np.float32,  # noqa: F405
    ScalarType.Int4: np.int8,  # noqa: F405
    ScalarType.Int12: np.int16,  # noqa: F405
    ScalarType.E8M0: np.float32,  # noqa: F405
    ScalarType.E5M3: np.float32,  # noqa: F405
}

_SCALAR_TYPES_FROM_NUMPY = {
    np.dtype(np.bool_): ScalarType.Boolean,  # noqa: F405
    np.dtype(np.uint8): ScalarType.UInt8,  # noqa: F405
    np.dtype(np.int8): ScalarType.Int8,  # noqa: F405
    np.dtype(np.uint16): ScalarType.UInt16,  # noqa: F405
    np.dtype(np.int16): ScalarType.Int16,  # noqa: F405
    np.dtype(np.uint32): ScalarType.UInt32,  # noqa: F405
    np.dtype(np.int32): ScalarType.Int32,  # noqa: F405
    np.dtype(np.uint64): ScalarType.UInt64,  # noqa: F405
    np.dtype(np.int64): ScalarType.Int64,  # noqa: F405
    np.dtype(np.float16): ScalarType.Float16,  # noqa: F405
    np.dtype(np.float32): ScalarType.Float32,  # noqa: F405
    np.dtype(np.float64): ScalarType.Float64,  # noqa: F405
    np.dtype(np.complex64): ScalarType.ComplexFloat32,  # noqa: F405
    np.dtype(np.complex128): ScalarType.ComplexFloat64,  # noqa: F405
}


def from_numpy(array: np.ndarray, scalar_type=None):
    """Create an owning host-validation tensor by quantizing NumPy values."""

    values = np.asarray(array)
    if scalar_type is None:
        try:
            scalar_type = _SCALAR_TYPES_FROM_NUMPY[values.dtype]
        except KeyError as error:
            raise TypeError(
                f"NumPy dtype {values.dtype} has no automatic ScalarType mapping"
            ) from error

    flat = values.reshape(-1)
    shape = list(values.shape)
    if np.issubdtype(values.dtype, np.complexfloating):
        return Tensor.from_complex_values(  # noqa: F405
            scalar_type, shape, [complex(value) for value in flat]
        )
    if np.issubdtype(values.dtype, np.signedinteger):
        return Tensor.from_signed_values(  # noqa: F405
            scalar_type, shape, [int(value) for value in flat]
        )
    if np.issubdtype(values.dtype, np.unsignedinteger) or values.dtype == np.bool_:
        return Tensor.from_unsigned_values(  # noqa: F405
            scalar_type, shape, [int(value) for value in flat]
        )
    return Tensor.from_values(  # noqa: F405
        scalar_type, shape, [float(value) for value in flat]
    )


def to_numpy(tensor, dtype=None) -> np.ndarray:
    """Decode a host-validation tensor into an owning NumPy array."""

    if dtype is None:
        dtype = _NUMPY_DTYPES[tensor.type]
    return np.asarray(tensor.values, dtype=dtype).reshape(tensor.shape)


def reference_gemm(*args, **kwargs):
    """Run reference GEMM and return only its owning output tensor."""

    return reference_gemm_result(*args, **kwargs).output  # noqa: F405
