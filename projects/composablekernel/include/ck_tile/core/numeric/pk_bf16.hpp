// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

namespace ck_tile {

// v_cvt_pk_bf16_f32 converts 2 floats to 2 bf16 in one instruction
// Available on gfx94x (gfx942, gfx950) and later
CK_TILE_DEVICE bf16x2_t cvt_pk_bf16_f32(float a, float b)
{
#if defined(__gfx94__)
    bf16x2_t result;
    asm volatile("v_cvt_pk_bf16_f32 %[result], %[a], %[b]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
#else
    return fp32x2_to_bf16x2(fp32x2_t{a, b});
#endif
}

// Packed bf16x2 to fp32x2 conversion using bit operations
// bf16 to fp32 is just left-shift by 16 bits (padding zeros in mantissa)
// bf16x2 layout: [bf16[1] | bf16[0]] in 32-bit register
CK_TILE_HOST_DEVICE constexpr fp32x2_t bf16x2_to_fp32x2(bf16x2_t x)
{
    uint32_t packed = bit_cast<uint32_t>(x);
    // Extract low bf16: shift left 16 to move to high bits of fp32
    float f0 = bit_cast<float>(packed << 16);
    // Extract high bf16: already in high 16 bits, just mask
    float f1 = bit_cast<float>(packed & 0xFFFF0000u);
    return fp32x2_t{f0, f1};
}

} // namespace ck_tile
