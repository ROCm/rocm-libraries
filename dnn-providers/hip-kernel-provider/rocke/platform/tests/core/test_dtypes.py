# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Core dtype names are independent of target support and retain public aliases."""

import pytest

from rocke.core.dtypes import normalize_dtype
from rocke.core.dtypes import dtype_info
from rocke.core.ir import I8, IRBuilder, dtype_to_ir_type
from rocke.core.ir_serialize import parse, serialize
from rocke.helpers.quant import quant_ir_type, ir_to_qdtype, dequantize_scalar_to_f32


@pytest.mark.parametrize(
    "dtype,bits",
    [
        ("fp4", 4),
        ("fp6", 6),
        ("bf6", 6),
        ("fp8", 8),
        ("bf8", 8),
        ("f16", 16),
        ("bf16", 16),
        ("f32", 32),
        ("e8m0", 8),
        ("e4m3", 8),
        ("e5m3", 8),
    ],
)
def test_logical_dtype_and_serialization(dtype, bits):
    info = dtype_info(dtype)
    logical = dtype_to_ir_type(dtype)
    assert info.encoded_bits == bits
    assert logical != I8
    b = IRBuilder("dtype_transport")
    b.param("code", logical)
    assert serialize(parse(serialize(b.kernel))) == serialize(b.kernel)


@pytest.mark.parametrize("dtype", ["fp4", "fp6", "bf6"])
def test_quant_type_recognition_does_not_enable_integer_conversion(dtype):
    logical = quant_ir_type(dtype)
    assert logical == dtype_to_ir_type(dtype)
    assert ir_to_qdtype(logical) == dtype_info(dtype).name
    b = IRBuilder("no_lowbit_conversion")
    value = b.param("value", logical)
    scale = b.const_f32(1)
    with pytest.raises(ValueError, match="unsupported input type"):
        dequantize_scalar_to_f32(b, value, scale=scale)


def test_unknown_dtype_and_unrepresented_integer_family():
    with pytest.raises(ValueError, match="unknown dtype"):
        dtype_info("custom_format")
    with pytest.raises(ValueError, match="no logical IR type"):
        dtype_to_ir_type("iu4")


@pytest.mark.parametrize(
    "spelling,canonical",
    [
        (" HALF ", "fp16"),
        ("bfloat16", "bf16"),
        (" Float\t", "fp32"),
        ("FP8", "fp8e4m3"),
        ("BF8", "bf8e5m2"),
        ("FP6", "fp6e2m3"),
        ("BF6", "fp6e3m2"),
        ("FP4", "fp4e2m1"),
        ("int32", "i32"),
        (" Custom_Format ", "custom_format"),
        ("", ""),
    ],
)
def test_normalize_dtype(spelling, canonical):
    assert normalize_dtype(spelling) == canonical
    assert normalize_dtype(canonical) == canonical


def test_architecture_entry_points_reexport_core_normalization():
    from rocke.core import normalize_dtype as core_normalize
    from rocke.core.arch import normalize_dtype as arch_normalize
    from rocke.core.arch.target import normalize_dtype as target_normalize

    assert normalize_dtype.__module__ == "rocke.core.dtypes"
    assert core_normalize is arch_normalize is target_normalize is normalize_dtype
