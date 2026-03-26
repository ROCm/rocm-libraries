// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: types — DataType enum, constexpr queries. No runtime, no CK deps.
//
// Pure type definitions for data type metadata. Host-side conversion
// utilities (float_to_typed, typed_to_float, tolerance_for) are in
// datatype_convert.hpp.

#pragma once

#include <cstdint>

namespace rocm_ck {

/// Data type tag for compile-time kernel configuration.
/// Independent of CK Tile headers. No UNDEFINED — every config must specify
/// a valid type. Covers all types used by CK operations.
///
/// Naming: FP8 = e4m3, BF8 = e5m2 (CK convention). FNUZ = MI300 native
/// (gfx942 hardware), OCP = MI350 native (gfx950 hardware, software on MI300).
enum class DataType
{
    // Floating point — standard widths
    FP64,
    FP32,
    FP16,
    BF16,

    // FP8 variants
    FP8_FNUZ, // e4m3, gfx942 hardware
    BF8_FNUZ, // e5m2, gfx942 hardware
    FP8_OCP,  // e4m3, gfx950 hardware
    BF8_OCP,  // e5m2, gfx950 hardware

    // Integer types — signed and unsigned at each width
    I4,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64
};

/// Returns the bit-width of a DataType. Uses bits (not bytes) so future
/// sub-byte types (fp4, fp6, int4) are clean integers.
/// No default case — lets -Wswitch catch unhandled enum values.
constexpr int data_type_bits(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return 64;
    case DataType::FP32: return 32;
    case DataType::FP16: return 16;
    case DataType::BF16: return 16;
    case DataType::FP8_FNUZ: return 8;
    case DataType::BF8_FNUZ: return 8;
    case DataType::FP8_OCP: return 8;
    case DataType::BF8_OCP: return 8;
    case DataType::I4: return 4;
    case DataType::I8: return 8;
    case DataType::I16: return 16;
    case DataType::I32: return 32;
    case DataType::I64: return 64;
    case DataType::U8: return 8;
    case DataType::U16: return 16;
    case DataType::U32: return 32;
    case DataType::U64: return 64;
    }
    return 0;
}

/// Returns a short string name for the data type (e.g. "FP32").
constexpr const char* data_type_name(DataType dt)
{
    switch(dt)
    {
    case DataType::FP64: return "FP64";
    case DataType::FP32: return "FP32";
    case DataType::FP16: return "FP16";
    case DataType::BF16: return "BF16";
    case DataType::FP8_FNUZ: return "FP8_FNUZ";
    case DataType::BF8_FNUZ: return "BF8_FNUZ";
    case DataType::FP8_OCP: return "FP8_OCP";
    case DataType::BF8_OCP: return "BF8_OCP";
    case DataType::I4: return "I4";
    case DataType::I8: return "I8";
    case DataType::I16: return "I16";
    case DataType::I32: return "I32";
    case DataType::I64: return "I64";
    case DataType::U8: return "U8";
    case DataType::U16: return "U16";
    case DataType::U32: return "U32";
    case DataType::U64: return "U64";
    }
    return "???";
}

} // namespace rocm_ck
