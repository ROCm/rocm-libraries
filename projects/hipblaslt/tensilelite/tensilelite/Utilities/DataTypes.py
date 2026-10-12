# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Datatype metadata shared by the kernel generator and standalone client tools.

The order matches rocisa's DataTypeEnum; Common.DataType adds those native enum
objects when the generator imports it. Register sizes and instruction spellings
live here so clients do not maintain a second table or import the extension.
"""

from types import MappingProxyType
from typing import NamedTuple


class _TypeProperties(NamedTuple):
    char: str
    nameAbbrev: str
    miOutTypeNameAbbrev: str
    reg: float
    hip: str
    isComplex: bool = False


# reg counts 32-bit registers per element. FP6/BF6 and FP4 use fractional
# registers; client-specific packing determines their allocation granularity.
# "NONE" marks types unsupported as matrix-instruction inputs. FP8 abbreviations
# match instruction spellings for both FNUZ (NANOO) and OCP formats.
TYPE_PROPERTIES = MappingProxyType(
    {
        "Float": _TypeProperties("S", "f32", "f32", 1, "float"),
        "Double": _TypeProperties("D", "f64", "f64", 2, "double"),
        "ComplexFloat": _TypeProperties("C", "f32c", "f32", 2, "TensileComplexFloat", True),
        "ComplexDouble": _TypeProperties("Z", "f64c", "f64", 4, "TensileComplexDouble", True),
        "Half": _TypeProperties("H", "f16", "f32", 0.5, "tensile_half"),
        "Int8x4": _TypeProperties("4xi8", "i8", "i32", 1, "uint32_t"),
        "Int32": _TypeProperties("I", "i32", "NONE", 1, "int32_t"),
        "BFloat16": _TypeProperties("B", "bf16", "f32", 0.5, "tensile_bfloat16"),
        "Int8": _TypeProperties("I8", "i8", "i32", 0.25, "int8_t"),
        "Int64": _TypeProperties("I64", "i64", "NONE", 1, "int64_t"),
        "XFloat32": _TypeProperties("X", "xf32", "f32", 1, "ERROR"),
        # FNUZ (NANOO) E4M3/E5M2 formats and mixed input pairs.
        "Float8_fnuz": _TypeProperties("F8N", "fp8", "f32", 0.25, "tensile_float8_fnuz"),
        "BFloat8_fnuz": _TypeProperties("B8N", "bf8", "f32", 0.25, "tensile_bfloat8_fnuz"),
        "Float8BFloat8_fnuz": _TypeProperties("F8B8N", "fp8_bf8", "f32", 0.25, "ERROR"),
        "BFloat8Float8_fnuz": _TypeProperties("B8F8N", "bf8_fp8", "f32", 0.25, "ERROR"),
        # OCP E4M3/E5M2 formats and mixed input pairs.
        "Float8": _TypeProperties("F8", "fp8", "f32", 0.25, "tensile_float8"),
        "BFloat8": _TypeProperties("B8", "bf8", "f32", 0.25, "tensile_bfloat8"),
        "Float8BFloat8": _TypeProperties("F8B8", "fp8_bf8", "f32", 0.25, "ERROR"),
        "BFloat8Float8": _TypeProperties("B8F8", "bf8_fp8", "f32", 0.25, "ERROR"),
        "Float6": _TypeProperties("F6", "fp6", "f32", 0.1875, "tensile_Float6x16"),
        "BFloat6": _TypeProperties("B6", "bf6", "f32", 0.1875, "tensile_bFloat6x16"),
        "Float4": _TypeProperties("F4", "fp4", "f32", 0.125, "tensile_float4x2"),
        "E8": _TypeProperties("E8", "e8", "f32", 0.25, "ERROR"),
        "E5M3": _TypeProperties("E5M3", "e5m3", "f32", 0.25, "ERROR"),
    }
)
