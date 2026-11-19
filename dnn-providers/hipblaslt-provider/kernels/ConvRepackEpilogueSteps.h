// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The steps around a GEMM that applies a fused explicit-GEMM convolution epilogue itself: the
// broadcast bias matrix it adds as its C (BroadcastArgs), and the relu / clamp tail that clamps y
// in place, in its own type, after it (ClampArgs). Part of the repack program (ConvRepack.cpp),
// embedded for HIPRTC as its own literal.

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

/// Each thread stores 16 bytes of one column (the column's value repeated) where the rows and
/// the destination allow, else one element; one division a store finds the column.
template <typename Bits>
__device__ void repackBroadcast(const BroadcastArgs& args)
{
    constexpr int N = 16 / sizeof(Bits);
    using Vec = unsigned int __attribute__((ext_vector_type(4)));
    const long long firstThread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long threadStride = static_cast<long long>(gridDim.x) * blockDim.x;
    const auto* src = static_cast<const Bits*>(args.src);
    auto* dst = static_cast<Bits*>(args.dst);
    if(args.rows % N == 0 && reinterpret_cast<unsigned long long>(dst) % 16 == 0)
    {
        const long long perColumn = args.rows / N;
        const long long total = perColumn * args.channels;
        auto* vectors = reinterpret_cast<Vec*>(dst);
        for(long long i = firstThread; i < total; i += threadStride)
        {
            const Bits value = src[(i / perColumn) * args.srcStride];
            Bits values[N];
#pragma unroll
            for(int e = 0; e < N; ++e)
            {
                values[e] = value;
            }
            Vec packed;
            __builtin_memcpy(&packed, values, sizeof(packed));
            vectors[i] = packed;
        }
        return;
    }
    const long long total = args.rows * args.channels;
    for(long long i = firstThread; i < total; i += threadStride)
    {
        dst[i] = src[(i / args.rows) * args.srcStride];
    }
}

extern "C" __global__ void Broadcast2(BroadcastArgs args)
{
    repackBroadcast<unsigned short>(args);
}

extern "C" __global__ void Broadcast4(BroadcastArgs args)
{
    repackBroadcast<unsigned int>(args);
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
