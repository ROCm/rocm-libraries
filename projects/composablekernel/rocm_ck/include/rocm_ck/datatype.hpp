// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types — DataType enum, constexpr queries. No runtime, no CK deps.

#pragma once

#include <cstdint>

namespace rocm_ck {

// FP8 = e4m3, BF8 = e5m2 (CK convention).
// FNUZ = gfx942 hardware native, OCP = gfx950 hardware native (software on gfx942).
enum class DataType : uint8_t
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

// Bits (not bytes) so sub-byte types (I4) are clean integers.
constexpr int dataTypeBits(DataType dt)
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

constexpr const char* dataTypeName(DataType dt)
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
