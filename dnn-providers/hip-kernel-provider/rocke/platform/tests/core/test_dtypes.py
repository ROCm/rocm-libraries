# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Core dtype names are independent of target support and retain public aliases."""

import ast
import os
import subprocess
from typing import get_args, get_type_hints

import pytest

from rocke.core.dtypes import normalize_dtype
from rocke.core.dtypes import dtype_info
from rocke.core.arch.target import MmaScaleDType
from rocke.core import ir
from rocke.core.ir import FP8E4M3, I8, IRBuilder, Type, dtype_to_ir_type, type_bits
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
        ("f64", 64),
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


# The width of every IR scalar type, written out by hand so that the format table
# in core/dtypes.py and type_bits are checked against an independent source.
# Adding an IR scalar type means adding its width here; see section 5.4,
# "Adding a dtype", in dsl_docs/development/extending.md.
_IR_SCALAR_BITS = {
    "i1": 1,
    "i4": 4,
    "u4": 4,
    "i8": 8,
    "u8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "bf16": 16,
    "f32": 32,
    "tf32": 32,
    "f64": 64,
    "fp8e4m3": 8,
    "bf8e5m2": 8,
    "fp6e2m3": 6,
    "fp6e3m2": 6,
    "fp4e2m1": 4,
    "e8m0": 8,
    "e5m3": 8,
}


def test_every_ir_scalar_type_has_a_reference_width():
    defined = {t.name for t in vars(ir).values() if type(t) is Type}
    assert not defined - set(_IR_SCALAR_BITS), (
        f"add the width of {sorted(defined - set(_IR_SCALAR_BITS))} to "
        "_IR_SCALAR_BITS"
    )
    assert not set(_IR_SCALAR_BITS) - defined


@pytest.mark.parametrize(("name", "bits"), sorted(_IR_SCALAR_BITS.items()))
def test_an_ir_scalar_type_has_its_reference_width(name, bits):
    assert dtype_info(name).encoded_bits == bits
    assert type_bits(Type(name)) == bits


def test_native_scalar_widths_match_the_reference():
    # The C++ engine lists every scalar kind it defines (a static_assert keeps
    # the list complete) with the width rocke_i_type_bits gives it.
    executable = os.environ.get("ROCKE_IR_TYPES_TEST")
    if not executable:
        pytest.skip("set ROCKE_IR_TYPES_TEST to the built rocke_ir_types test")
    lines = subprocess.run(
        [executable, "--scalar-bits"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    native = {name: int(bits) for name, bits in (line.split() for line in lines)}
    assert native == _IR_SCALAR_BITS


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


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError, match="unknown dtype"):
        dtype_info("custom_format")


def test_the_integer_dtypes_are_i8_and_i4_only():
    # "iu8" / "iu4" appear only in RDNA integer WMMA instruction names: signedness is
    # an instruction operand, not a property of the data, so they are not dtypes.
    from rocke.core.ir import I4

    assert dtype_to_ir_type("i8") is I8
    assert dtype_to_ir_type("i4") is I4
    assert dtype_info("i4").encoded_bits == 4
    assert storage_ir_type("i4") is I8
    for name in ("iu8", "iu4"):
        with pytest.raises(ValueError, match="unknown dtype"):
            dtype_info(name)


@pytest.mark.parametrize(
    "spelling,canonical",
    [
        (" HALF ", "fp16"),
        ("bfloat16", "bf16"),
        (" Float\t", "fp32"),
        (" Double ", "fp64"),
        ("F64", "fp64"),
        ("FP8", "fp8e4m3"),
        (" E4M3 ", "fp8e4m3"),
        ("BF8", "bf8e5m2"),
        ("FP6", "fp6e2m3"),
        ("BF6", "fp6e3m2"),
        ("FP4", "fp4e2m1"),
        ("int32", "i32"),
        ("XF32", "tf32"),
        (" TF32 ", "tf32"),
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


def test_quant_dtype_annotations_match_scalar_and_logical_contracts():
    from rocke import helpers
    from rocke.helpers import quant

    scalar = {"i8", "fp8e4m3", "bf8e5m2"}
    assert set(get_args(quant.QDType)) == scalar
    assert all(quant_max_abs(dtype) > 0 for dtype in get_args(quant.QDType))
    logical = scalar | {"fp4e2m1", "fp6e2m3", "fp6e3m2"}
    assert set(get_args(quant.LogicalQDType)) == logical
    assert get_type_hints(ir_to_qdtype)["return"] == quant.LogicalQDType
    assert all(ir_to_qdtype(quant_ir_type(dtype)) == dtype for dtype in logical)
    assert helpers.QDType is quant.QDType
    assert helpers.LogicalQDType is quant.LogicalQDType


def test_u8_and_u4_are_logical_unsigned_types():
    # Types only: they name unsigned data for matrix instructions with a signedness flag.
    import rocke
    from rocke.core.ir import U4, U8, IRBuilder, PtrType
    from rocke.tests.engines import (
        lower_hip_with_both_engines,
        lower_with_both_engines,
    )
    from rocke.tests.ir_validity import assert_target_accepts

    for spelling, ir_type, bits in (
        ("u8", U8, 8),
        ("uint8", U8, 8),
        ("u4", U4, 4),
        ("uint4", U4, 4),
    ):
        assert dtype_to_ir_type(spelling) is ir_type
        assert dtype_info(spelling).encoded_bits == bits
        assert dtype_info(spelling).category.value == "integer"
    assert U8 != I8
    assert rocke.U8 is U8 and rocke.U4 is U4
    assert storage_ir_type("u4") is I8
    # U8 values are LLVM i8 (LLVM integers carry no sign) and HIP uint8_t.
    b = IRBuilder("u8_pointer")
    src = b.param("SRC", PtrType(U8, "global"))
    dst = b.param("DST", PtrType(U8, "global"))
    tid = b.thread_id_x()
    loaded = b.global_load(src, tid, U8)
    b.global_store(dst, tid, loaded)
    ir_text = lower_with_both_engines(b.kernel, "gfx90a")
    assert "load i8, ptr addrspace(1)" in ir_text
    assert_target_accepts(ir_text, "gfx90a")
    hip = lower_hip_with_both_engines(b.kernel, "gfx90a")
    assert "uint8_t* __restrict__ SRC" in hip
