# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Storage and operand types are core contracts, independent of matrix transport."""

import pytest

from rocke.core.ir import dtype_to_ir_type
from rocke.core.ir_storage import (
    storage_element_type,
    storage_ir_type,
    storage_vector_type,
)
from rocke.core.mma_operand import mma_operand_ir_type


@pytest.mark.parametrize(
    "dtype,count,storage,operand",
    [
        ("fp16", 4, "vec<f16x4>", "vec<f16x4>"),
        ("tf32", 2, "vec<i32x2>", "vec<tf32x2>"),
        ("i8", 16, "vec<i8x16>", "vec<i8x16>"),
        ("fp8e4m3", 32, "vec<fp8e4m3x32>", "vec<fp8e4m3x32>"),
        ("fp4e2m1", 32, "vec<i32x4>", "vec<i32x4>"),
        ("fp6e2m3", 32, "vec<i32x6>", "vec<i32x6>"),
        ("fp6e2m3", 16, "vec<i32x3>", "vec<i32x3>"),
        ("i4", 8, "vec<i32x1>", "vec<i32x1>"),
        ("tf32", 8, "vec<i32x8>", "vec<tf32x8>"),
        ("i4", 16, "vec<i32x2>", "vec<i32x2>"),
        ("e8m0", 4, "vec<i8x4>", "vec<e8m0x4>"),
    ],
)
def test_live_storage_and_logical_operand_types(dtype, count, storage, operand):
    assert storage_vector_type(dtype_to_ir_type(dtype), count).name == storage
    assert mma_operand_ir_type(dtype, count).name == operand


@pytest.mark.parametrize(
    "dtype,pointer,element",
    [
        ("fp4e2m1", "i8", "i32"),
        ("fp6e2m3", "i8", "i32"),
        ("i4", "i8", "i32"),
        ("tf32", "i32", "i32"),
        ("e8m0", "i8", "i8"),
        ("e5m3", "i8", "i8"),
        ("bf16", "bf16", "bf16"),
        ("fp16", "f16", "f16"),
    ],
)
def test_memory_pointer_and_register_units_are_distinct(dtype, pointer, element):
    assert storage_ir_type(dtype).name == pointer
    assert storage_element_type(dtype_to_ir_type(dtype)).name == element


@pytest.mark.parametrize("dtype,count", [("fp4e2m1", 4), ("fp6e2m3", 8)])
def test_partial_storage_word_is_refused(dtype, count):
    with pytest.raises(ValueError, match="not a whole number of 32-bit registers"):
        storage_vector_type(dtype_to_ir_type(dtype), count)
    with pytest.raises(ValueError, match="not a whole number of 32-bit registers"):
        mma_operand_ir_type(dtype, count)
