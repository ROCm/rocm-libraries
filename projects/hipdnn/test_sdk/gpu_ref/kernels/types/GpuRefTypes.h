// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Self-contained device header for GPU reference kernels.
// No host includes allowed - this is compiled by HipRTC.
// DATA_TYPE must be defined at compile time via -DDATA_TYPE=<type>

#pragma once

// Accumulation is always done in fp32
using AccumType = float;

// --- float overloads ---

__device__ inline float toFloat(float x)
{
    return x;
}

__device__ inline float fromFloat(float x, float* /*tag*/)
{
    return x;
}

// --- _Float16 (fp16) overloads ---

__device__ inline float toFloat(_Float16 x)
{
    return static_cast<float>(x);
}

__device__ inline _Float16 fromFloat(float x, _Float16* /*tag*/)
{
    return static_cast<_Float16>(x);
}

// --- unsigned short (bfloat16) overloads ---
// Uses manual bit conversion matching the Bfloat16Dev.hpp pattern.

typedef union
{
    unsigned int u32;
    float f32;
    unsigned short u16[2];
} CvtBf16Fp32;

__device__ inline float toFloat(unsigned short x)
{
    CvtBf16Fp32 cvt;
    cvt.u16[0] = 0;
    cvt.u16[1] = x;
    return cvt.f32;
}

__device__ inline unsigned short fromFloat(float x, unsigned short* /*tag*/)
{
    CvtBf16Fp32 cvt;
    cvt.f32 = x;
    if((~cvt.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        if((cvt.u32 & 0xffff) != 0)
        {
            cvt.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    return cvt.u16[1];
}
