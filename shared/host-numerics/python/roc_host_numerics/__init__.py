# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from ._roc_host_numerics import *  # noqa: F403


# Default NumPy containers for decoded numerical values. These are not storage
# dtypes: packed and custom encodings remain in tensor.storage and decode into
# the wider type listed here.
_DEFAULT_DECODED_DTYPES = {
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
    ScalarType.E8M0: np.float32,  # noqa: F405
    ScalarType.E8M0Zero: np.float32,  # noqa: F405
    ScalarType.E5M3: np.float32,  # noqa: F405
    ScalarType.E4M3: np.float32,  # noqa: F405
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


def default_decoded_dtype(scalar_type) -> np.dtype:
    """Return the default NumPy dtype for decoded values of scalar_type.

    This describes the owning array returned by to_numpy, not tensor.storage's
    encoded or packed representation.
    """

    try:
        return np.dtype(_DEFAULT_DECODED_DTYPES[scalar_type])
    except KeyError as error:
        raise ValueError(
            f"No default decoded NumPy dtype for scalar type {scalar_type!r}"
        ) from error


def from_numpy(array: np.ndarray, scalar_type=None):
    """Create an owning host-numerics tensor by quantizing NumPy values."""

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
    """Decode tensor values into an owning NumPy array.

    dtype selects the decoded output container. It does not reinterpret the
    encoded bytes exposed by tensor.storage.
    """

    if dtype is None:
        dtype = default_decoded_dtype(tensor.type)
    return np.asarray(tensor.values, dtype=dtype).reshape(tensor.shape)


def reference_gemm(
    a,
    b,
    c,
    output_type,
    accumulator_type,
    alpha=1.0,
    beta=0.0,
    scale_c=1.0,
    compute_type_a=None,
    compute_type_b=None,
    math_mode=MathMode.Default,  # noqa: F405
    activation=Activation.None_,  # noqa: F405
    activation_parameter0=0.0,
    activation_parameter1=0.0,
    output_selection=None,
    backend=GemmBackend.Pointwise,  # noqa: F405
    block_scale_a=None,
    block_scale_b=None,
    block_size_a=0,
    block_size_b=0,
    pre_quantization_scales_a=None,
    pre_quantization_scales_b=None,
    bias=None,
    scale_alpha=None,
    scale_a=None,
    scale_b=None,
    output_scale=1.0,
    output_conversion=OutputConversion.Default,  # noqa: F405
    accumulation_rounding=AccumulationRounding.TypeDefault,  # noqa: F405
    conjugate_a=False,
    conjugate_b=False,
    output_layout=None,
):
    """Compute a reference GEMM from tensor arguments and return its output tensor."""

    operand_a = GemmOperand(a)  # noqa: F405
    operand_b = GemmOperand(b)  # noqa: F405
    if compute_type_a is not None:
        operand_a.compute_type = compute_type_a
    if compute_type_b is not None:
        operand_b.compute_type = compute_type_b
    operand_a.conjugate = conjugate_a
    operand_b.conjugate = conjugate_b

    def as_default_vector_broadcast(scale, axis):
        return scale.expand_dims(axis) if len(scale.shape) == 1 else scale

    def add_pre_quantization_scales(operand, scales, default_axis):
        scales = [] if scales is None else list(scales)
        operand.pre_quantization_scales = [
            as_default_vector_broadcast(scale, default_axis) for scale in scales
        ]

    add_pre_quantization_scales(operand_a, pre_quantization_scales_a, 1)
    add_pre_quantization_scales(operand_b, pre_quantization_scales_b, 0)

    if block_scale_a is not None:
        if block_size_a == 0:
            raise ValueError(
                "Python reference_gemm A block scale requires a nonzero size."
            )
        operand_a.block_scale = BlockScaleBinding(  # noqa: F405
            block_scale_a, block_size_a
        )
    elif block_size_a != 0:
        raise ValueError("Python reference_gemm A block size requires a scale tensor.")

    if block_scale_b is not None:
        if block_size_b == 0:
            raise ValueError(
                "Python reference_gemm B block scale requires a nonzero size."
            )
        operand_b.block_scale = BlockScaleBinding(  # noqa: F405
            block_scale_b, block_size_b
        )
    elif block_size_b != 0:
        raise ValueError("Python reference_gemm B block size requires a scale tensor.")

    options = GemmOptions(accumulator_type)  # noqa: F405
    options.accumulation_rounding = accumulation_rounding
    options.math_mode = math_mode
    options.epilogue.alpha = alpha
    options.epilogue.beta = beta
    options.epilogue.scale_c = scale_c
    options.epilogue.bias = bias
    options.epilogue.scale_alpha = scale_alpha
    options.epilogue.scale_a = (
        None if scale_a is None else as_default_vector_broadcast(scale_a, 1)
    )
    options.epilogue.scale_b = (
        None if scale_b is None else as_default_vector_broadcast(scale_b, 0)
    )
    options.epilogue.output_scale = output_scale
    options.epilogue.output_conversion = output_conversion
    options.epilogue.activation = activation
    options.epilogue.activation_parameter0 = activation_parameter0
    options.epilogue.activation_parameter1 = activation_parameter1
    options.output_selection = (
        OutputSelection.all()  # noqa: F405
        if output_selection is None
        else output_selection
    )
    return reference_gemm_operands(  # noqa: F405
        operand_a,
        operand_b,
        c,
        output_type,
        options,
        output_layout,
        backend,
    )
