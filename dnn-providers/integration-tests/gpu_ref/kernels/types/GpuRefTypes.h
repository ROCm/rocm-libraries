// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Self-contained device header for GPU reference kernels.
// No host includes allowed - this is compiled by HipRTC.
// COMPUTE_TYPE must be defined at compile time via -DCOMPUTE_TYPE=<type>.
// Convolution kernels also require -DX_TYPE=<type> -DW_TYPE=<type> -DY_TYPE=<type>.

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
// float and double overloads are provided by hiprtc_runtime.h;
// only _Float16 and __bf16 need custom overloads.

__device__ inline _Float16 fabs(_Float16 x)
{
    return __builtin_fabsf16(x);
}

__device__ inline __bf16 fabs(__bf16 x)
{
    return static_cast<__bf16>(__builtin_fabsf(static_cast<float>(x)));
}

// --- isnan overloads ---

__device__ inline bool isnan(_Float16 x)
{
    return __builtin_isnan(x);
}

__device__ inline bool isnan(__bf16 x)
{
    return __builtin_isnan(static_cast<float>(x));
}

// --- isinf overloads ---

__device__ inline bool isinf(_Float16 x)
{
    return __builtin_isinf(x);
}

__device__ inline bool isinf(__bf16 x)
{
    return __builtin_isinf(static_cast<float>(x));
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
