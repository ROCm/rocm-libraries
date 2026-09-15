# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

import roc_host_numerics as hv
from gemm_test_adapter import reference_gemm, reference_gemm_into
from _numeric_oracle_support import quantize_bfloat16, wrap_int32


def exact_int32_gemm(left, right, initial, alpha, beta, output_scale):
    """Use unbounded Python integers and wrap after every defined Int32 operation."""

    result = np.empty((left.shape[0], right.shape[1]), dtype=np.int32)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            accumulation = 0
            for reduction in range(left.shape[1]):
                product = wrap_int32(
                    int(left[row, reduction]) * int(right[reduction, column])
                )
                accumulation = wrap_int32(accumulation + product)
            combined = wrap_int32(
                wrap_int32(alpha * accumulation)
                + wrap_int32(beta * int(initial[row, column]))
            )
            result[row, column] = wrap_int32(combined * output_scale)
    return result


def per_k_block_scaled_gemm(left, right, scale_a, scale_b, block_a, block_b):
    """Apply each operand's scale directly at k, independent of block traversal."""

    result = np.empty((left.shape[0], right.shape[1]), dtype=np.float32)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            accumulation = np.float32(0.0)
            for reduction in range(left.shape[1]):
                term = np.float32(left[row, reduction] * right[reduction, column])
                term = np.float32(term * scale_a[row, reduction // block_a])
                term = np.float32(term * scale_b[column, reduction // block_b])
                accumulation = np.float32(accumulation + term)
            result[row, column] = accumulation
    return result


def stepwise_gemm(left, right, accumulator_type):
    """Round every product and sum in an independently implemented accumulator."""

    if accumulator_type == hv.ScalarType.Float16:
        quantize = lambda value: np.float32(np.float16(value))
    elif accumulator_type == hv.ScalarType.BFloat16:
        quantize = quantize_bfloat16
    else:
        raise ValueError("stepwise_gemm requires a reduced-precision accumulator")

    result = np.empty((left.shape[0], right.shape[1]), dtype=np.float32)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            accumulation = np.float32(0.0)
            for reduction in range(left.shape[1]):
                product = quantize(
                    np.float32(left[row, reduction] * right[reduction, column])
                )
                accumulation = quantize(np.float32(accumulation + product))
            result[row, column] = accumulation
    return result


def affine_tensor(values, scalar_type, strides, offset):
    values = np.asarray(values)
    maximum_offset = offset
    for extent, stride in zip(values.shape, strides):
        maximum_offset += (extent - 1) * stride
    storage = np.zeros(maximum_offset + 1, dtype=values.dtype)
    for coordinates in np.ndindex(values.shape):
        storage_index = offset + sum(
            coordinate * stride for coordinate, stride in zip(coordinates, strides)
        )
        storage[storage_index] = values[coordinates]
    return hv.Tensor.from_storage(
        scalar_type,
        list(values.shape),
        storage.tobytes(),
        strides=list(strides),
        offset=offset,
    )


def saturating_int8(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


def selected_values(values, selected):
    result = np.zeros_like(values)
    result.reshape(-1)[selected] = values.reshape(-1)[selected]
    return result


def modular_values(rows, columns, row_factor, column_factor, modulus, center, divisor):
    return (
        (
            np.arange(rows)[:, None] * row_factor
            + np.arange(columns)[None, :] * column_factor
        )
        % modulus
        - center
    ).astype(np.float32) / np.float32(divisor)


__all__ = [name for name in globals() if not name.startswith("__")]
