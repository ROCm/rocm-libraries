// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Configuration.hpp"
#include "HipKernelMath.hpp"
#include "VectorTypes.hpp"

namespace hip_kernel_provider::batchnorm
{

namespace reduction
{

namespace detail
{

const unsigned long long FULL_MASK = 0xFFFFFFFFFFFFFFFFull;

template <int N>
struct log2_floor
{
    constexpr static int value = log2_floor<(N >> 1)>::value + 1;
};
template <>
struct log2_floor<1>
{
    constexpr static int value = 0;
};
template <int N>
constexpr static int log2_floor_v = log2_floor<N>::value;

template <int N>
struct log2_ceil
{
    constexpr static int value = log2_floor_v<N> + ((1 << log2_floor_v<N>) == N ? 0 : 1);
};
template <int N>
constexpr static int log2_ceil_v = log2_ceil<N>::value;

} // namespace detail

// NOTE: unlike MIOpen's fixed version (MIOpen PR #4411), the reduction loop below does not bound
// `red` against `size`, only against `SizeLclData`. This is safe only because every current
// variant-2 call site (BatchNormFwdTrainSpatial.cpp, BatchNormBwdSpatial.cpp) launches with
// blockDim.x * blockDim.y * blockDim.z == SizeLclData / (2 * xstride) exactly, i.e. `size` is
// always forced equal to the LDS array's full logical capacity, and the host-side plan builder
// (BatchnormCommon.hpp: getSpatialMultipleConfig/getLocalConfigChannelLast) always rounds
// ylocalsize/xlocalsize up to a power of two. So `size < SizeLclData/(2*xstride)` and
// non-power-of-two `size` are both unreachable today; if a future tuning path ever launches this
// kernel with a `size` smaller than the LDS capacity, this loop would sum in stale/uninitialized
// rows and this guard would need MIOpen's `ylid + red < size` bound added back.
template <typename FloatAccumC, typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2_2d(FloatAccumC& x,
                                               FloatAccumC& y,
                                               FloatAccum scale,
                                               FloatAccumC (&lcl_data)[SizeLclData],
                                               unsigned int xstride,
                                               unsigned int xlid,
                                               unsigned int ylid,
                                               unsigned int size)
{
    unsigned int offset1 = 2 * (xlid + ylid * xstride);
    // store the values by pairs (so the compiler will generate
    // one instruction to read/write them)
    lcl_data[offset1 + 0] = x;
    lcl_data[offset1 + 1] = y;

    __syncthreads();
    for(unsigned int red = (1 << detail::log2_ceil_v<SizeLclData>) >> 1; red > 0; red >>= 1)
    {
        unsigned int offset2 = offset1 + red * xstride * 2;
        if(ylid < red && offset2 < SizeLclData)
        {
            // make sure there is one read and one write
            x += lcl_data[offset2 + 0];
            y += lcl_data[offset2 + 1];
            lcl_data[offset1 + 0] = x;
            lcl_data[offset1 + 1] = y;
        }
        __syncthreads();
    }
    x = static_cast<FloatAccumC>(lcl_data[xlid * 2 + 0] * scale);
    y = static_cast<FloatAccumC>(lcl_data[xlid * 2 + 1] * scale);
}

template <typename FloatAccumC, typename FloatAccum, unsigned int BlockSize>
__forceinline__ __device__ void
    reduce2(FloatAccumC& x, FloatAccumC& y, FloatAccum scale, unsigned int lid)
{
    static_assert(BlockSize > 0, "BlockSize must be positive");

    if constexpr(BlockSize == 1)
    {
        x *= scale;
        y *= scale;
        return;
    }

    if constexpr(BlockSize % 64 == 0)
    {
        for(unsigned int d = warpSize / 2; d >= 1; d >>= 1)
        {
            x += __shfl_down_sync(detail::FULL_MASK, x, d);
            y += __shfl_down_sync(detail::FULL_MASK, y, d);
        }

        if(BlockSize <= static_cast<unsigned int>(warpSize))
        {
            x = __shfl_sync(detail::FULL_MASK, x, 0) * scale;
            y = __shfl_sync(detail::FULL_MASK, y, 0) * scale;
            return;
        }

        constexpr unsigned int max_warps = BlockSize / 32;
        __shared__ FloatAccum s_x[max_warps];
        __shared__ FloatAccum s_y[max_warps];

        const unsigned int lane = lid % static_cast<unsigned int>(warpSize);
        const unsigned int wid = lid / static_cast<unsigned int>(warpSize);
        const unsigned int num_warps = BlockSize / static_cast<unsigned int>(warpSize);

        if(lane == 0)
        {
            s_x[wid] = x;
            s_y[wid] = y;
        }
        __syncthreads();

        if(wid == 0)
        {
            x = FloatAccum{0};
            y = FloatAccum{0};
            for(unsigned int i = lane; i < num_warps; i += static_cast<unsigned int>(warpSize))
            {
                x += s_x[i];
                y += s_y[i];
            }
            for(unsigned int d = warpSize / 2; d >= 1; d >>= 1)
            {
                x += __shfl_down_sync(detail::FULL_MASK, x, d);
                y += __shfl_down_sync(detail::FULL_MASK, y, d);
            }
        }

        if(lid == 0)
        {
            s_x[0] = x * scale;
            s_y[0] = y * scale;
        }
        __syncthreads();
        x = s_x[0];
        y = s_y[0];
    }
    else
    {
        // Slow path, mainly for the unlikely case of a 32 thread block
        __shared__ FloatAccum s_x[BlockSize];
        __shared__ FloatAccum s_y[BlockSize];

        s_x[lid] = x;
        s_y[lid] = y;
        __syncthreads();

        if(lid < static_cast<unsigned int>(warpSize))
        {
            x = FloatAccum{0};
            y = FloatAccum{0};
            for(unsigned int i = lid; i < BlockSize; i += static_cast<unsigned int>(warpSize))
            {
                x += s_x[i];
                y += s_y[i];
            }
            for(unsigned int d = warpSize / 2; d >= 1; d >>= 1)
            {
                x += __shfl_down_sync(detail::FULL_MASK, x, d);
                y += __shfl_down_sync(detail::FULL_MASK, y, d);
            }
        }

        if(lid == 0)
        {
            s_x[0] = x * scale;
            s_y[0] = y * scale;
        }
        __syncthreads();
        x = s_x[0];
        y = s_y[0];
    }
}

// Each accumulator holds a mean and the sum of squared deviations from that mean (M2).
// Counts are scalar even when the samples are vectors of adjacent channels.
template <typename T>
struct Welford
{
    T mean;
    T m2;
    float count;

    __forceinline__ __device__ void merge(T otherMean, T otherM2, float otherCount)
    {
        if(otherCount == 0.0f)
        {
            return;
        }
        if(count == 0.0f)
        {
            mean = otherMean;
            m2 = otherM2;
            count = otherCount;
            return;
        }
        const T delta = otherMean - mean;
        const float totalCount = count + otherCount;
        const float weight = otherCount * __builtin_amdgcn_rcpf(totalCount);
        mean = fma(delta, cast<T>(weight), mean);
        m2 = fma(delta, delta * cast<T>(count * weight), m2 + otherM2);
        count = totalCount;
    }

    // Branchless merge for a non-empty power-of-two chunk (otherCount in {1, 2, 4}).
    // When count == 0 (mean == 0, m2 == 0), otherCount * rcpf(otherCount) == 1.0f exactly.
    __forceinline__ __device__ void mergeNonEmpty(T otherMean, T otherM2, float otherCount)
    {
        const T delta = otherMean - mean;
        const float totalCount = count + otherCount;
        const float weight = otherCount * __builtin_amdgcn_rcpf(totalCount);
        mean = fma(delta, cast<T>(weight), mean);
        m2 = fma(delta, delta * cast<T>(count * weight), m2 + otherM2);
        count = totalCount;
    }

    template <typename Sample>
    __forceinline__ __device__ void add(Sample sample)
    {
        if constexpr(mapped_vector_info<T>::size == mapped_vector_info<Sample>::size)
        {
            const T x = cast<T>(sample);
            count += 1.0f;
            const T delta = x - mean;
            mean = fma(delta, cast<T>(__builtin_amdgcn_rcpf(count)), mean);
            m2 = fma(delta, x - mean, m2);
        }
        else
        {
            static_assert(mapped_vector_info<T>::size == 1
                          && (mapped_vector_info<Sample>::size == 2
                              || mapped_vector_info<Sample>::size == 4));
            // Merge spatial samples from one vector load before updating the running statistics.
            if constexpr(mapped_vector_info<Sample>::size == 4)
            {
                add4(sample.x, sample.y, sample.z, sample.w);
            }
            else
            {
                const T dx = sample.y - sample.x;
                const T meanXY = fma(dx, cast<T>(0.5f), sample.x);
                const T m2XY = dx * (dx * cast<T>(0.5f));
                mergeNonEmpty(meanXY, m2XY, 2.0f);
            }
        }
    }

    template <typename Sample>
    __forceinline__ __device__ void add4(Sample s0, Sample s1, Sample s2, Sample s3)
    {
        if constexpr(mapped_vector_info<T>::size == mapped_vector_info<Sample>::size)
        {
            const T x0 = cast<T>(s0);
            const T x1 = cast<T>(s1);
            const T x2 = cast<T>(s2);
            const T x3 = cast<T>(s3);
            const T d01 = x1 - x0;
            const T m01 = fma(d01, cast<T>(0.5f), x0);
            const T m2_01 = d01 * (d01 * cast<T>(0.5f));
            const T d23 = x3 - x2;
            const T m23 = fma(d23, cast<T>(0.5f), x2);
            const T delta = m23 - m01;
            const T chunkMean = fma(delta, cast<T>(0.5f), m01);
            const T chunkM2 = fma(delta, delta, fma(d23, d23 * cast<T>(0.5f), m2_01));
            mergeNonEmpty(chunkMean, chunkM2, 4.0f);
        }
        else
        {
            add(s0);
            add(s1);
            add(s2);
            add(s3);
        }
    }
};

// Single-channel, single-workgroup variants: only one partial per wave goes to LDS.
template <typename T, unsigned int WaveCount = config::lds_gcn_size>
__forceinline__ __device__ void reduce_welford(Welford<T>& stats, unsigned int lid)
{
    __shared__ Welford<T> waves[WaveCount];

#pragma unroll
    for(unsigned int offset = warpSize >> 1; offset > 0; offset >>= 1)
    {
        const auto otherMean = __shfl_down_sync(detail::FULL_MASK, stats.mean, offset);
        const auto otherM2 = __shfl_down_sync(detail::FULL_MASK, stats.m2, offset);
        const auto otherCount = __shfl_down_sync(detail::FULL_MASK, stats.count, offset);
        stats.merge(otherMean, otherM2, otherCount);
    }

    const auto waveSize = static_cast<unsigned int>(warpSize);
    if(lid % waveSize == 0)
    {
        waves[lid / waveSize] = stats;
    }
    __syncthreads();

    for(unsigned int red = WaveCount >> 1; red > 0; red >>= 1)
    {
        if(lid < red)
        {
            const auto other = waves[lid + red];
            stats = waves[lid];
            stats.merge(other.mean, other.m2, other.count);
            waves[lid] = stats;
        }
        __syncthreads();
    }
    stats = waves[0];
}

// Variant 2 maps x lanes to independent channels. The launch dimensions are powers of two.
template <unsigned int XSize, unsigned int BlockSize, typename T>
__forceinline__ __device__ void
    reduce_welford_2d(Welford<T>& stats, unsigned int xlid, unsigned int yzlid)
{
    __shared__ Welford<T> partials[BlockSize];
    const unsigned int lid = xlid + yzlid * XSize;
    partials[lid] = stats;
    __syncthreads();

    for(unsigned int red = (BlockSize / XSize) >> 1; red > 0; red >>= 1)
    {
        if(yzlid < red)
        {
            const auto other = partials[lid + red * XSize];
            stats.merge(other.mean, other.m2, other.count);
            partials[lid] = stats;
        }
        __syncthreads();
    }
    stats = partials[xlid];
}

} // namespace reduction

} // namespace hip_kernel_provider::batchnorm
