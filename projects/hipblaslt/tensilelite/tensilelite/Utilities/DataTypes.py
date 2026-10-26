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
    """Type and instruction abbreviations, generator register counts, and HIP spelling.

    Register counts use 32-bit units, except for Int64's legacy value below.
    """

    char: str
    nameAbbrev: str
    miOutTypeNameAbbrev: str
    reg: float
    hip: str
    isComplex: bool = False


# reg normally counts 32-bit registers per element; see Int64's legacy exception.
# FP6/BF6 and FP4 use fractional registers; client-specific packing determines
# their allocation granularity.
# "NONE" marks types unsupported as matrix-instruction inputs. FP8 abbreviations
# match instruction spellings for both FNUZ (NANOO) and OCP formats.
TYPE_PROPERTIES = MappingProxyType(
    {
        "Float": _TypeProperties(
            char="S", nameAbbrev="f32", miOutTypeNameAbbrev="f32", reg=1, hip="float"
        ),
        "Double": _TypeProperties(
            char="D", nameAbbrev="f64", miOutTypeNameAbbrev="f64", reg=2, hip="double"
        ),
        "ComplexFloat": _TypeProperties(
            char="C",
            nameAbbrev="f32c",
            miOutTypeNameAbbrev="f32",
            reg=2,
            hip="TensileComplexFloat",
            isComplex=True,
        ),
        "ComplexDouble": _TypeProperties(
            char="Z",
            nameAbbrev="f64c",
            miOutTypeNameAbbrev="f64",
            reg=4,
            hip="TensileComplexDouble",
            isComplex=True,
        ),
        "Half": _TypeProperties(
            char="H", nameAbbrev="f16", miOutTypeNameAbbrev="f32", reg=0.5, hip="tensile_half"
        ),
        "Int8x4": _TypeProperties(
            char="4xi8", nameAbbrev="i8", miOutTypeNameAbbrev="i32", reg=1, hip="uint32_t"
        ),
        "Int32": _TypeProperties(
            char="I", nameAbbrev="i32", miOutTypeNameAbbrev="NONE", reg=1, hip="int32_t"
        ),
        "BFloat16": _TypeProperties(
            char="B", nameAbbrev="bf16", miOutTypeNameAbbrev="f32", reg=0.5, hip="tensile_bfloat16"
        ),
        "Int8": _TypeProperties(
            char="I8", nameAbbrev="i8", miOutTypeNameAbbrev="i32", reg=0.25, hip="int8_t"
        ),
        # The generator's legacy reg=1 makes numBytes() report 4, although
        # int64_t occupies 8 bytes. Do not use this entry's register count
        # to calculate Int64 storage size.
        "Int64": _TypeProperties(
            char="I64", nameAbbrev="i64", miOutTypeNameAbbrev="NONE", reg=1, hip="int64_t"
        ),
        "XFloat32": _TypeProperties(
            char="X", nameAbbrev="xf32", miOutTypeNameAbbrev="f32", reg=1, hip="ERROR"
        ),
        # FNUZ (NANOO) E4M3/E5M2 formats and mixed input pairs.
        "Float8_fnuz": _TypeProperties(
            char="F8N",
            nameAbbrev="fp8",
            miOutTypeNameAbbrev="f32",
            reg=0.25,
            hip="tensile_float8_fnuz",
        ),
        "BFloat8_fnuz": _TypeProperties(
            char="B8N",
            nameAbbrev="bf8",
            miOutTypeNameAbbrev="f32",
            reg=0.25,
            hip="tensile_bfloat8_fnuz",
        ),
        "Float8BFloat8_fnuz": _TypeProperties(
            char="F8B8N", nameAbbrev="fp8_bf8", miOutTypeNameAbbrev="f32", reg=0.25, hip="ERROR"
        ),
        "BFloat8Float8_fnuz": _TypeProperties(
            char="B8F8N", nameAbbrev="bf8_fp8", miOutTypeNameAbbrev="f32", reg=0.25, hip="ERROR"
        ),
        # OCP E4M3/E5M2 formats and mixed input pairs.
        "Float8": _TypeProperties(
            char="F8", nameAbbrev="fp8", miOutTypeNameAbbrev="f32", reg=0.25, hip="tensile_float8"
        ),
        "BFloat8": _TypeProperties(
            char="B8", nameAbbrev="bf8", miOutTypeNameAbbrev="f32", reg=0.25, hip="tensile_bfloat8"
        ),
        "Float8BFloat8": _TypeProperties(
            char="F8B8", nameAbbrev="fp8_bf8", miOutTypeNameAbbrev="f32", reg=0.25, hip="ERROR"
        ),
        "BFloat8Float8": _TypeProperties(
            char="B8F8", nameAbbrev="bf8_fp8", miOutTypeNameAbbrev="f32", reg=0.25, hip="ERROR"
        ),
        "Float6": _TypeProperties(
            char="F6",
            nameAbbrev="fp6",
            miOutTypeNameAbbrev="f32",
            reg=0.1875,
            hip="tensile_Float6x16",
        ),
        "BFloat6": _TypeProperties(
            char="B6",
            nameAbbrev="bf6",
            miOutTypeNameAbbrev="f32",
            reg=0.1875,
            hip="tensile_bFloat6x16",
        ),
        "Float4": _TypeProperties(
            char="F4",
            nameAbbrev="fp4",
            miOutTypeNameAbbrev="f32",
            reg=0.125,
            hip="tensile_float4x2",
        ),
        "E8": _TypeProperties(
            char="E8", nameAbbrev="e8", miOutTypeNameAbbrev="f32", reg=0.25, hip="ERROR"
        ),
        "E5M3": _TypeProperties(
            char="E5M3", nameAbbrev="e5m3", miOutTypeNameAbbrev="f32", reg=0.25, hip="ERROR"
        ),
        "Int4": _TypeProperties(
            char="I4", nameAbbrev="i4", miOutTypeNameAbbrev="f32", reg=0.125, hip="tensile_int4x2"
        ),
    }
)
