// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "ConvArchFamily.h"

typedef _Float16 HkpHalf2 __attribute__((ext_vector_type(2)));

// Packed fp16 products accumulated in fp32. gfx1250 lacks the legacy dot10
// instruction; two native fp32 FMAs keep the same stored operand values.
__device__ inline float k1Dot2(HkpHalf2 a, HkpHalf2 b, float c)
{
#if HKP_ARCH_GFX1250
    c = __builtin_fmaf(static_cast<float>(a[0]), static_cast<float>(b[0]), c);
    return __builtin_fmaf(static_cast<float>(a[1]), static_cast<float>(b[1]), c);
#else
    return __builtin_amdgcn_fdot2(a, b, c, false);
#endif
}
