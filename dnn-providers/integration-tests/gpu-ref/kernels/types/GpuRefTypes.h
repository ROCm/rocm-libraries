// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Self-contained device header for GPU reference kernels.
// No host includes allowed - this is compiled by HipRTC.
// COMPUTE_TYPE must be defined at compile time via -DCOMPUTE_TYPE=<type>.
// Convolution kernels also require -DX_TYPE=<type> -DW_TYPE=<type> -DY_TYPE=<type>.

#pragma once

#include "GpuRefBatchnormArgs.h"
#include "GpuRefCommonArgs.h"
#include "GpuRefConvArgs.h"
#include "GpuRefLayernormArgs.h"
#include "GpuRefPointwiseArgs.h"
#include "GpuRefRMSNormArgs.h"

namespace gpu_ref
{

// --- FP8 E4M3 (OCP) device decode ---
// Byte-exact with hipdnn_data_sdk::types::fp8_e4m3 (OCP: 1 sign, 4 exp bias-7, 3 mantissa;
// no infinity; NaN == abs 0x7F). Implemented as a self-contained magnitude lookup so the
// kernel needs no <hip/hip_fp8.h> (unavailable under HipRTC's preinclude) and matches the
// host reference's dequantization exactly.
__device__ inline float fp8OcpE4m3ToFloat(unsigned char bits)
{
    // Magnitude by absolute bits [0, 0x7F). Index 0x7F is NaN (guarded below), so its slot
    // is never read and holds a placeholder.
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static constexpr float MAG[128]
        = {0.0f,        0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f,  0.009765625f,
           0.01171875f, 0.013671875f, 0.015625f,   0.017578125f, 0.01953125f, 0.021484375f,
           0.0234375f,  0.025390625f, 0.02734375f, 0.029296875f, 0.03125f,    0.03515625f,
           0.0390625f,  0.04296875f,  0.046875f,   0.05078125f,  0.0546875f,  0.05859375f,
           0.0625f,     0.0703125f,   0.078125f,   0.0859375f,   0.09375f,    0.1015625f,
           0.109375f,   0.1171875f,   0.125f,      0.140625f,    0.15625f,    0.171875f,
           0.1875f,     0.203125f,    0.21875f,    0.234375f,    0.25f,       0.28125f,
           0.3125f,     0.34375f,     0.375f,      0.40625f,     0.4375f,     0.46875f,
           0.5f,        0.5625f,      0.625f,      0.6875f,      0.75f,       0.8125f,
           0.875f,      0.9375f,      1.0f,        1.125f,       1.25f,       1.375f,
           1.5f,        1.625f,       1.75f,       1.875f,       2.0f,        2.25f,
           2.5f,        2.75f,        3.0f,        3.25f,        3.5f,        3.75f,
           4.0f,        4.5f,         5.0f,        5.5f,         6.0f,        6.5f,
           7.0f,        7.5f,         8.0f,        9.0f,         10.0f,       11.0f,
           12.0f,       13.0f,        14.0f,       15.0f,        16.0f,       18.0f,
           20.0f,       22.0f,        24.0f,       26.0f,        28.0f,       30.0f,
           32.0f,       36.0f,        40.0f,       44.0f,        48.0f,       52.0f,
           56.0f,       60.0f,        64.0f,       72.0f,        80.0f,       88.0f,
           96.0f,       104.0f,       112.0f,      120.0f,       128.0f,      144.0f,
           160.0f,      176.0f,       192.0f,      208.0f,       224.0f,      240.0f,
           256.0f,      288.0f,       320.0f,      352.0f,       384.0f,      416.0f,
           448.0f,      0.0f /* 0x7F: NaN, guarded */};
    const unsigned char absBits = bits & 0x7Fu;
    if(absBits == 0x7Fu)
    {
        const float qnan = __builtin_nanf("");
        return ((bits & 0x80u) != 0) ? -qnan : qnan;
    }
    const float mag = MAG[absBits];
    return ((bits & 0x80u) != 0) ? -mag : mag;
}

// Device fp8 E4M3 (OCP) storage type. One byte, decodes to float on read. HipRtcTypeName maps
// hipdnn_data_sdk::types::fp8_e4m3 to this name so the SDPA kernel loads fp8 Q/K/V via Q_TYPE.
// NOLINTNEXTLINE(readability-identifier-naming)
struct GpuRefFp8E4M3
{
    unsigned char data;
    __device__ operator float() const // NOLINT(google-explicit-constructor)
    {
        return fp8OcpE4m3ToFloat(data);
    }
};

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
