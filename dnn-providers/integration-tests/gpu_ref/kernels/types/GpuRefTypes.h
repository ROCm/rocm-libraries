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

// --- Software FP8 decoders ---
// HIPRTC compiles this header self-contained (no host includes), and clang has no
// builtin scalar fp8 type, so the GPU reference stores each fp8 input as a raw byte
// and decodes it to float in software. The decode reproduces the host data-SDK value
// for every finite pattern; random test inputs never produce inf/NaN, so those
// patterns are intentionally left to fall through the finite path.

namespace fp8_detail
{

// 2^e as a float for e within the normal float exponent range (true for every fp8
// scale). Composes the IEEE-754 bits directly, mirroring the union idiom used by
// truncateToTf32 below, so no device math-library symbol is required.
__device__ inline float pow2(int e)
{
    union
    {
        unsigned int u32;
        float f32;
    } cvt;
    cvt.u32 = static_cast<unsigned int>(127 + e) << 23U;
    return cvt.f32;
}

// Decode an fp8 byte given the format's exponent/mantissa widths and bias. Subnormals
// (exp == 0) and normals are handled uniformly as mantissa * 2^scale.
__device__ inline float decode(unsigned int bits, int expBits, int mantBits, int bias)
{
    const unsigned int signBit = 1U << (expBits + mantBits);
    const unsigned int expMask = ((1U << expBits) - 1U) << mantBits;
    const unsigned int mantMask = (1U << mantBits) - 1U;

    const unsigned int exp = (bits & expMask) >> mantBits;
    const unsigned int mant = bits & mantMask;

    float magnitude;
    if(exp == 0U)
    {
        // Subnormal: value = mant * 2^(1 - bias - mantBits).
        magnitude = static_cast<float>(mant) * pow2(1 - bias - mantBits);
    }
    else
    {
        // Normal: value = (2^mantBits + mant) * 2^(exp - bias - mantBits).
        magnitude = static_cast<float>((1U << mantBits) | mant)
                    * pow2(static_cast<int>(exp) - bias - mantBits);
    }
    return ((bits & signBit) != 0U) ? -magnitude : magnitude;
}

} // namespace fp8_detail

// OCP and FNUZ fp8 storage types. data holds the raw 8-bit pattern; operator float
// decodes it. Biases: e4m3 OCP = 7, e5m2 OCP = 15, e4m3 FNUZ = 8, e5m2 FNUZ = 16.
struct fp8_e4m3
{
    unsigned char data;
    __device__ inline operator float() const
    {
        return fp8_detail::decode(data, 4, 3, 7);
    }
};

struct fp8_e5m2
{
    unsigned char data;
    __device__ inline operator float() const
    {
        return fp8_detail::decode(data, 5, 2, 15);
    }
};

struct fp8_e4m3_fnuz
{
    unsigned char data;
    __device__ inline operator float() const
    {
        return fp8_detail::decode(data, 4, 3, 8);
    }
};

struct fp8_e5m2_fnuz
{
    unsigned char data;
    __device__ inline operator float() const
    {
        return fp8_detail::decode(data, 5, 2, 16);
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
