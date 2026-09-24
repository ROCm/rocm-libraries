# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Core dtype names are independent of target support and retain public aliases."""

import ast

import pytest

from rocke.core.dtypes import normalize_dtype
from rocke.core.dtypes import dtype_info
from rocke.core.arch.target import MmaScaleDType
from rocke.core.ir import FP8E4M3, I8, IRBuilder, dtype_to_ir_type
from rocke.core.storage import TensorStorage
from rocke.helpers.mma_io import storage_ir_type
from rocke.core.ir_serialize import parse, serialize
from rocke.helpers.quant import (
    quant_ir_type,
    ir_to_qdtype,
    dequantize_scalar_to_f32,
    quant_max_abs,
)


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
        (" E4M3 ", "fp8e4m3"),
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


@pytest.mark.parametrize("spelling", ["e4m3", "fp8e4m3", "fp8", MmaScaleDType.E4M3])
def test_e4m3_shares_encoding_type_and_storage(spelling):
    assert dtype_info(spelling) is dtype_info("fp8e4m3")
    assert dtype_to_ir_type(spelling) is FP8E4M3
    assert storage_ir_type(spelling) is FP8E4M3
    assert TensorStorage(spelling, (2, 128)) == TensorStorage("fp8e4m3", (2, 128))
    assert quant_ir_type(spelling) is FP8E4M3
    b = IRBuilder("e4m3_conversion")
    value = b.param("value", dtype_to_ir_type(spelling))
    b.cvt_fp8_to_f32(value)
    assert serialize(b.kernel) == serialize(parse(serialize(b.kernel)))


def test_scale_role_remains_independent_of_encoding_aliases():
    assert MmaScaleDType("fp8e4m3") is MmaScaleDType.E4M3
    assert str(MmaScaleDType.E4M3) == "e4m3"
    assert dtype_info("e5m3") is not dtype_info("bf8e5m2")
    assert dtype_to_ir_type("e5m3") != dtype_to_ir_type("bf8e5m2")
    with pytest.raises(ValueError):
        MmaScaleDType("bf8e5m2")


def test_quant_type_diagnostic_lists_every_accepted_spelling():
    expected = {
        "i8",
        "int8",
        "fp8",
        "fp8e4m3",
        "fp8_e4m3",
        "e4m3",
        "bf8",
        "bf8e5m2",
        "fp8_e5m2",
        "fp4",
        "fp4e2m1",
        "fp6",
        "fp6e2m3",
        "bf6",
        "fp6e3m2",
    }
    with pytest.raises(ValueError) as error:
        quant_ir_type("invalid")
    accepted = ast.literal_eval(str(error.value).split("expected one of ")[1])
    assert accepted == sorted(expected)
    assert all(quant_ir_type(name) is not None for name in accepted)


@pytest.mark.parametrize(
    "dtype", ["fp4", "fp4e2m1", "fp6", "fp6e2m3", "bf6", "fp6e3m2"]
)
def test_type_alias_does_not_enable_scalar_quantization(dtype):
    assert quant_ir_type(dtype) is not None
    with pytest.raises(ValueError, match="unsupported quant dtype"):
        quant_max_abs(dtype)
