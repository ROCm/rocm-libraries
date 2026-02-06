// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp" // for fp32x2_t

namespace ck_tile {

// v_pk_add_f32 with neg modifiers for packed subtraction
// Computes: result = a - b (packed 2x fp32)
// Available on gfx94x (gfx942, gfx950) and later
#if defined(__gfx94__)
CK_TILE_DEVICE fp32x2_t pk_sub_f32(fp32x2_t a, fp32x2_t b)
{
    fp32x2_t result;
    asm volatile("v_pk_add_f32 %[result], %[a], %[b] neg_lo:[0,1] neg_hi:[0,1]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
}
#endif

} // namespace ck_tile
