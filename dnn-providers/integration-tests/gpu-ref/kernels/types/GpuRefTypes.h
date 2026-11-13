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
#include "GpuRefMatmulArgs.h"
#include "GpuRefPointwiseArgs.h"
#include "GpuRefRMSNormArgs.h"
#include "GpuRefReductionArgs.h"
#include "GpuRefResampleArgs.h"

namespace gpu_ref
{

// --- FP8 E4M3 (OCP) device decode ---
// Bit-exact with hipdnn_data_sdk::types::fp8_e4m3 (bias 7, no infinity, NaN at abs 0x7F).
// A lookup table avoids <hip/hip_fp8.h>, which HipRTC's preinclude does not provide.
__device__ inline float fp8OcpE4m3ToFloat(unsigned char bits)
{
    // Magnitude by abs bits. Slot 0x7F is NaN and handled below, so its value is unused.
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

// --- FP8 E4M3 FNUZ device decode ---
// Bit-exact with hipdnn_data_sdk::types::fp8_e4m3_fnuz: bias 8, no infinity, no negative zero,
// the single NaN at 0x80. Normals are (8 + m) * 2^(e - 11), subnormals m * 2^-10.
__device__ inline float fp8FnuzE4m3ToFloat(unsigned char bits)
{
    if(bits == 0x80u)
    {
        return __builtin_nanf("");
    }
    const unsigned exponent = (bits >> 3) & 0xFu;
    const unsigned mantissa = bits & 0x7u;
    // 2^(e - 11) built from its float bits (biased exponent e - 11 + 127).
    const float mag = exponent == 0 ? static_cast<float>(mantissa) * 0.0009765625f
                                    : static_cast<float>(8u + mantissa)
                                          * __builtin_bit_cast(float, (exponent + 116u) << 23);
    return ((bits & 0x80u) != 0) ? -mag : mag;
}

// --- FP8 E5M2 device decode ---
// Bit-exact with hipdnn_data_sdk::types::fp8_e5m2 (OCP: bias 15, infinity at exponent 31 with a
// zero mantissa, NaN otherwise) and fp8_e5m2_fnuz (bias 16, no infinity or negative zero, the
// single NaN at 0x80). Normals are (4 + m) * 2^(e - bias - 2), subnormals m * 2^(-bias - 1).
__device__ inline float fp8OcpE5m2ToFloat(unsigned char bits)
{
    const unsigned exponent = (bits >> 2) & 0x1Fu;
    const unsigned mantissa = bits & 0x3u;
    const bool negative = (bits & 0x80u) != 0;
    if(exponent == 31)
    {
        const float special = mantissa == 0 ? __builtin_huge_valf() : __builtin_nanf("");
        return negative ? -special : special;
    }
    const float mag = exponent == 0 ? static_cast<float>(mantissa) * 0.0000152587890625f // 2^-16
                                    : static_cast<float>(4u + mantissa)
                                          * __builtin_bit_cast(float, (exponent + 110u) << 23);
    return negative ? -mag : mag;
}

__device__ inline float fp8FnuzE5m2ToFloat(unsigned char bits)
{
    if(bits == 0x80u)
    {
        return __builtin_nanf("");
    }
    const unsigned exponent = (bits >> 2) & 0x1Fu;
    const unsigned mantissa = bits & 0x3u;
    const float mag = exponent == 0 ? static_cast<float>(mantissa) * 0.00000762939453125f // 2^-17
                                    : static_cast<float>(4u + mantissa)
                                          * __builtin_bit_cast(float, (exponent + 109u) << 23);
    return ((bits & 0x80u) != 0) ? -mag : mag;
}

// The fp8 formats, numbered as SdpaFp8Format / SdpaFwdArgs::pFp8Format.
#define GPU_REF_FP8_E4M3 1
#define GPU_REF_FP8_E4M3_FNUZ 2
#define GPU_REF_FP8_E5M2 3
#define GPU_REF_FP8_E5M2_FNUZ 4

__device__ inline float fp8ToFloat(unsigned char bits, int format)
{
    switch(format)
    {
    case GPU_REF_FP8_E4M3_FNUZ:
        return fp8FnuzE4m3ToFloat(bits);
    case GPU_REF_FP8_E5M2:
        return fp8OcpE5m2ToFloat(bits);
    case GPU_REF_FP8_E5M2_FNUZ:
        return fp8FnuzE5m2ToFloat(bits);
    default:
        return fp8OcpE4m3ToFloat(bits);
    }
}

// --- FP8 device encode ---
// Bit-exact with hipdnn_data_sdk's encoders: round to nearest, ties to even, saturating (also for
// infinity), FNUZ formats have no negative zero and NaN at 0x80. Finite magnitudes increase with
// the code up to the format's largest finite code, so this searches the decoded values; a tie goes
// to the even code, which is round-half-to-even.
template <int FORMAT>
__device__ inline unsigned char fp8FromFloat(float x)
{
    constexpr bool FNUZ = FORMAT == GPU_REF_FP8_E4M3_FNUZ || FORMAT == GPU_REF_FP8_E5M2_FNUZ;
    // Largest finite code: E4M3 0x7E (0x7F is NaN), E5M2 0x7B (0x7C up are inf / NaN).
    constexpr unsigned MAX_CODE = FNUZ ? 0x7Fu : (FORMAT == GPU_REF_FP8_E5M2 ? 0x7Bu : 0x7Eu);
    const auto decode
        = [](unsigned code) { return fp8ToFloat(static_cast<unsigned char>(code), FORMAT); };
    const bool negative = __builtin_signbit(x) != 0;
    if(__builtin_isnan(x))
    {
        return FNUZ ? 0x80u : static_cast<unsigned char>(0x7Fu | (negative ? 0x80u : 0u));
    }
    const float a = __builtin_fabsf(x);
    unsigned code = MAX_CODE;
    if(a < decode(MAX_CODE))
    {
        unsigned lo = 0;
        unsigned hi = MAX_CODE; // decode(lo) <= a < decode(hi)
        while(hi - lo > 1)
        {
            const unsigned mid = (lo + hi) / 2;
            if(decode(mid) <= a)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        const float below = a - decode(lo);
        const float above = decode(hi) - a;
        code = below < above ? lo : (above < below ? hi : ((lo & 1u) == 0 ? lo : hi));
    }
    if(FNUZ && code == 0)
    {
        return 0;
    }
    return static_cast<unsigned char>(code | (negative ? 0x80u : 0u));
}

// One-byte fp8 storage that decodes to float on read and encodes on construction. HipRtcTypeName
// maps hipdnn_data_sdk's fp8 types to these, so SDPA kernels load and store fp8 tensors.
template <int FORMAT>
struct GpuRefFp8 // NOLINT(readability-identifier-naming)
{
    unsigned char data;
    GpuRefFp8() = default;
    __device__ explicit GpuRefFp8(float f)
        : data(fp8FromFloat<FORMAT>(f))
    {
    }
    __device__ operator float() const // NOLINT(google-explicit-constructor)
    {
        return fp8ToFloat(data, FORMAT);
    }
};
// NOLINTBEGIN(readability-identifier-naming)
using GpuRefFp8E4M3 = GpuRefFp8<GPU_REF_FP8_E4M3>;
using GpuRefFp8E4M3Fnuz = GpuRefFp8<GPU_REF_FP8_E4M3_FNUZ>;
using GpuRefFp8E5M2 = GpuRefFp8<GPU_REF_FP8_E5M2>;
using GpuRefFp8E5M2Fnuz = GpuRefFp8<GPU_REF_FP8_E5M2_FNUZ>;
// NOLINTEND(readability-identifier-naming)

// x rounded to an fp8 format and back (GPU_REF_FP8_*; 0 leaves x).
__device__ inline float roundToFp8(float x, int format)
{
    switch(format)
    {
    case GPU_REF_FP8_E4M3:
        return fp8ToFloat(fp8FromFloat<GPU_REF_FP8_E4M3>(x), format);
    case GPU_REF_FP8_E4M3_FNUZ:
        return fp8ToFloat(fp8FromFloat<GPU_REF_FP8_E4M3_FNUZ>(x), format);
    case GPU_REF_FP8_E5M2:
        return fp8ToFloat(fp8FromFloat<GPU_REF_FP8_E5M2>(x), format);
    case GPU_REF_FP8_E5M2_FNUZ:
        return fp8ToFloat(fp8FromFloat<GPU_REF_FP8_E5M2_FNUZ>(x), format);
    default:
        return x;
    }
}

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
__device__ inline T fromAccum(COMPUTE_TYPE x)
{
    return safeConvert<T>(x);
}

template <typename T>
__device__ inline T fromAccum(COMPUTE_TYPE x, T* /*tag*/)
{
    return fromAccum<T>(x);
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
