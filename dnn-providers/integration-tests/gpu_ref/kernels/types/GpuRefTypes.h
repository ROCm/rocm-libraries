// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Self-contained device header for GPU reference kernels.
// No host includes allowed - this is compiled by HipRTC.
// X_TYPE, W_TYPE, Y_TYPE, COMPUTE_TYPE must be defined at compile time via
// -DX_TYPE=<type> -DW_TYPE=<type> -DY_TYPE=<type> -DCOMPUTE_TYPE=<type>

#pragma once

#include "GpuRefConvArgs.h"

// --- float overloads ---

__device__ inline COMPUTE_TYPE toAccum(float x)
{
    return static_cast<COMPUTE_TYPE>(x);
}

__device__ inline float fromAccum(COMPUTE_TYPE x, float* /*tag*/)
{
    return static_cast<float>(x);
}

// --- _Float16 (fp16) overloads ---

__device__ inline COMPUTE_TYPE toAccum(_Float16 x)
{
    return static_cast<COMPUTE_TYPE>(static_cast<float>(x));
}

__device__ inline _Float16 fromAccum(COMPUTE_TYPE x, _Float16* /*tag*/)
{
    return static_cast<_Float16>(static_cast<float>(x));
}

// --- unsigned short (bfloat16) overloads ---
// Uses manual bit conversion matching the Bfloat16Dev.hpp pattern.

typedef union
{
    unsigned int u32;
    float f32;
    unsigned short u16[2];
} CvtBf16Fp32;

__device__ inline COMPUTE_TYPE toAccum(unsigned short x)
{
    CvtBf16Fp32 cvt;
    cvt.u16[0] = 0;
    cvt.u16[1] = x;
    return static_cast<COMPUTE_TYPE>(cvt.f32);
}

__device__ inline unsigned short fromAccum(COMPUTE_TYPE x, unsigned short* /*tag*/)
{
    CvtBf16Fp32 cvt;
    cvt.f32 = static_cast<float>(x);
    if((~cvt.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        if((cvt.u32 & 0xffff) != 0)
        {
            cvt.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    return cvt.u16[1];
}

// --- signed char (int8) overloads ---

__device__ inline COMPUTE_TYPE toAccum(signed char x)
{
    return static_cast<COMPUTE_TYPE>(x);
}

__device__ inline signed char fromAccum(COMPUTE_TYPE x, signed char* /*tag*/)
{
    return static_cast<signed char>(x);
}

// --- int overloads ---

__device__ inline COMPUTE_TYPE toAccum(int x)
{
    return static_cast<COMPUTE_TYPE>(x);
}

__device__ inline int fromAccum(COMPUTE_TYPE x, int* /*tag*/)
{
    return static_cast<int>(x);
}

// --- double overloads ---

__device__ inline COMPUTE_TYPE toAccum(double x)
{
    return static_cast<COMPUTE_TYPE>(x);
}

__device__ inline double fromAccum(COMPUTE_TYPE x, double* /*tag*/)
{
    return static_cast<double>(x);
}

// --- TF32 truncation ---

#ifdef USE_TF32
__device__ inline float truncateToTf32(float x)
{
    typedef union
    {
        float f32;
        unsigned int u32;
    } CvtTf32;
    CvtTf32 cvt;
    cvt.f32 = x;
    cvt.u32 &= 0xFFFFE000u; // Zero bottom 13 mantissa bits
    return cvt.f32;
}
#endif
