// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Self-contained device header for GPU reference kernels.
// No host includes allowed - this is compiled by HipRTC.
// SRC_TYPE, DST_TYPE, ACC_TYPE must be defined at compile time via
// -DSRC_TYPE=<type> -DDST_TYPE=<type> -DACC_TYPE=<type>

#pragma once

// --- Stride structs for stride-based indexing ---

struct Strides3
{
    long long s[3];
};

struct Strides4
{
    long long s[4];
};

struct Strides5
{
    long long s[5];
};

// --- float overloads ---

__device__ inline ACC_TYPE toAccum(float x)
{
    return static_cast<ACC_TYPE>(x);
}

__device__ inline float fromAccum(ACC_TYPE x, float* /*tag*/)
{
    return static_cast<float>(x);
}

// --- _Float16 (fp16) overloads ---

__device__ inline ACC_TYPE toAccum(_Float16 x)
{
    return static_cast<ACC_TYPE>(static_cast<float>(x));
}

__device__ inline _Float16 fromAccum(ACC_TYPE x, _Float16* /*tag*/)
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

__device__ inline ACC_TYPE toAccum(unsigned short x)
{
    CvtBf16Fp32 cvt;
    cvt.u16[0] = 0;
    cvt.u16[1] = x;
    return static_cast<ACC_TYPE>(cvt.f32);
}

__device__ inline unsigned short fromAccum(ACC_TYPE x, unsigned short* /*tag*/)
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

__device__ inline ACC_TYPE toAccum(signed char x)
{
    return static_cast<ACC_TYPE>(x);
}

__device__ inline signed char fromAccum(ACC_TYPE x, signed char* /*tag*/)
{
    return static_cast<signed char>(x);
}

// --- int overloads ---

__device__ inline ACC_TYPE toAccum(int x)
{
    return static_cast<ACC_TYPE>(x);
}

__device__ inline int fromAccum(ACC_TYPE x, int* /*tag*/)
{
    return static_cast<int>(x);
}

// --- double overloads ---

__device__ inline ACC_TYPE toAccum(double x)
{
    return static_cast<ACC_TYPE>(x);
}

__device__ inline double fromAccum(ACC_TYPE x, double* /*tag*/)
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
