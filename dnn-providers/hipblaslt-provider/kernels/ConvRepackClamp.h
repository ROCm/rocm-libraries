// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The relu / clamp tail of a fused explicit-GEMM convolution epilogue (ClampArgs): y clamped in
// place, in its own type, after the GEMM that wrote it. Part of the repack program
// (ConvRepack.cpp), embedded for HIPRTC as its own literal.

#pragma once

#include "ConvRepackArgs.h"

/// The chain's own comparisons, so NaN stays NaN and a bound replaces every value at or past it.
/// Eight elements a thread (four for fp32) where the tensor allows 16-byte accesses; a run with
/// nothing to clamp is not stored.
template <typename T>
__device__ void repackClampInPlace(const ClampArgs& args)
{
    constexpr int N = 16 / sizeof(T);
    using Vec = unsigned int __attribute__((ext_vector_type(4)));
    const long long firstThread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long threadStride = static_cast<long long>(gridDim.x) * blockDim.x;
    auto* data = static_cast<T*>(args.dst);
    const T lower = static_cast<T>(args.lower);
    const T upper = static_cast<T>(args.upper);
    const auto clamp = [&](T& value) {
        const float v = static_cast<float>(value);
        const bool low = v <= args.lower;
        const bool high = v >= args.upper;
        value = low ? lower : (high ? upper : value);
        return low || high;
    };
    long long first = 0;
    if(reinterpret_cast<unsigned long long>(data) % 16 == 0)
    {
        const long long runs = args.total / N;
        auto* vectors = reinterpret_cast<Vec*>(data);
        for(long long i = firstThread; i < runs; i += threadStride)
        {
            Vec packed = vectors[i];
            T values[N];
            __builtin_memcpy(values, &packed, sizeof(packed));
            bool changed = false;
#pragma unroll
            for(int e = 0; e < N; ++e)
            {
                changed = clamp(values[e]) || changed;
            }
            if(changed)
            {
                __builtin_memcpy(&packed, values, sizeof(packed));
                vectors[i] = packed;
            }
        }
        first = runs * N;
    }
    for(long long i = first + firstThread; i < args.total; i += threadStride)
    {
        T value = data[i];
        if(clamp(value))
        {
            data[i] = value;
        }
    }
}

extern "C" __global__ void ClampFloat(ClampArgs args)
{
    repackClampInPlace<float>(args);
}

extern "C" __global__ void ClampHalf(ClampArgs args)
{
    repackClampInPlace<_Float16>(args);
}

extern "C" __global__ void ClampBfloat16(ClampArgs args)
{
    repackClampInPlace<__bf16>(args);
}
