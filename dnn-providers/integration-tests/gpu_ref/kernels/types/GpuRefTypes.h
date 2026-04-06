// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Self-contained device header for GPU reference kernels.
// No host includes allowed - this is compiled by HipRTC.
// X_TYPE, W_TYPE, Y_TYPE, COMPUTE_TYPE must be defined at compile time via
// -DX_TYPE=<type> -DW_TYPE=<type> -DY_TYPE=<type> -DCOMPUTE_TYPE=<type>

#pragma once

#include "GpuRefConvArgs.h"

namespace gpu_ref
{

// --- safeConvert ---

template <typename TargetType, typename SourceType>
__device__ inline TargetType safeConvert(SourceType value)
{
    if constexpr(__is_same(TargetType, __bf16) || __is_same(TargetType, _Float16))
    {
        return static_cast<TargetType>(static_cast<float>(value));
    }
    else
    {
        return static_cast<TargetType>(value);
    }
}

// --- toAccum: convert input data to accumulation precision ---

template <typename T>
__device__ inline COMPUTE_TYPE toAccum(T x)
{
    return safeConvert<COMPUTE_TYPE>(x);
}

// --- fromAccum: convert accumulation result back to output type ---

template <typename T>
__device__ inline T fromAccum(COMPUTE_TYPE x, T* /*tag*/)
{
    return safeConvert<T>(x);
}

// --- fabs overloads ---

__device__ inline float fabs(float x)
{
    return __builtin_fabsf(x);
}

__device__ inline double fabs(double x)
{
    return __builtin_fabs(x);
}

__device__ inline _Float16 fabs(_Float16 x)
{
    return __builtin_fabsf16(x);
}

__device__ inline __bf16 fabs(__bf16 x)
{
    return static_cast<__bf16>(__builtin_fabsf(static_cast<float>(x)));
}

// --- isnan overloads ---

__device__ inline bool isnan(float x)
{
    return __builtin_isnan(x);
}

__device__ inline bool isnan(double x)
{
    return __builtin_isnan(x);
}

__device__ inline bool isnan(_Float16 x)
{
    return __builtin_isnan(x);
}

__device__ inline bool isnan(__bf16 x)
{
    return __builtin_isnan(x);
}

// --- isinf overloads ---

__device__ inline bool isinf(float x)
{
    return __builtin_isinf(x);
}

__device__ inline bool isinf(double x)
{
    return __builtin_isinf(x);
}

__device__ inline bool isinf(_Float16 x)
{
    return __builtin_isinf(x);
}

__device__ inline bool isinf(__bf16 x)
{
    return __builtin_isinf(x);
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

} // namespace gpu_ref
