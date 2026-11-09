// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Configuration.hpp"
#include "StaticUnroll.hpp"

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

template <typename FloatAccumC, typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2(FloatAccumC& x,
                                            FloatAccumC& y,
                                            FloatAccum scale,
                                            FloatAccumC (&lcl_data_x)[SizeLclData],
                                            FloatAccumC (&lcl_data_y)[SizeLclData],
                                            unsigned int lid)
{
    lcl_data_x[lid] = x;
    lcl_data_y[lid] = y;
    __syncthreads();
    for(unsigned int red = (1 << detail::log2_ceil_v<SizeLclData>) >> 1; red > 0; red >>= 1)
    {
        if(lid < red && lid + red < SizeLclData)
        {
            lcl_data_x[lid] += lcl_data_x[lid + red];
            lcl_data_y[lid] += lcl_data_y[lid + red];
        }
        __syncthreads();
    }

    x = lcl_data_x[0] * scale;
    y = lcl_data_y[0] * scale;
}

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

template <typename FloatAccum>
__forceinline__ __device__ void dpp_interleaved_reduction(FloatAccum& temp_sum1,
                                                          FloatAccum& temp_sum2)
{
    __asm__ volatile("s_nop 4\n"
                     "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:1 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:2 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                     "v_add_f32 %1 %1 %1 row_shr:4 bank_mask:0xe\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_shr:8 bank_mask:0xc\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                     "v_add_f32 %1 %1 %1 row_bcast:15 row_mask:0xa\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_bcast:31 row_mask:0xc\n"
                     "s_nop 0"
                     : "=v"(temp_sum1), "=v"(temp_sum2)
                     : "0"(temp_sum1), "1"(temp_sum2));
}

template <typename FloatAccumC, typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void gcn_reduce2(FloatAccumC& x,
                                            FloatAccumC& y,
                                            FloatAccum scale,
                                            FloatAccumC (&lcl_data_x)[SizeLclData],
                                            FloatAccumC (&lcl_data_y)[SizeLclData],
                                            unsigned int lid)
{
    const unsigned int ldsidx = lid >> 6;
    dpp_interleaved_reduction(x, y);
    // Last thread
    if((lid % 64) == 63)
    {
        lcl_data_x[ldsidx] = x;
        lcl_data_y[ldsidx] = y;
    }

    __syncthreads();

    x = y = 0;

    // This could be changed to clang loop unroll(full), because the size is small
    static_unroll_count<unsigned int, 0, SizeLclData, 1, 2>{[&](unsigned int i) {
        x += lcl_data_x[i];
        y += lcl_data_y[i];
    }};

    x *= scale;
    y *= scale;
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

// Portable (shuffle-only) counterpart of the per-wave step inside reduce2_welford. MIOpen also
// has a GFX9 DPP-asm variant of this step (restricted to GFX9 by MIOpen PR #12889), which is not
// ported here: this provider always builds with HIP_PLUGIN_USE_AMDGCN=0, so the DPP path would be
// dead code.
template <typename FloatAccum>
__forceinline__ __device__ void shfl_interleaved_reduction_welford(FloatAccum& temp_mean,
                                                                    FloatAccum& temp_var,
                                                                    FloatAccum& temp_count)
{
#pragma unroll
    for(unsigned int offset = warpSize >> 1; offset >= 1; offset >>= 1)
    {
        FloatAccum otherMean = __shfl_down_sync(detail::FULL_MASK, temp_mean, offset);
        FloatAccum delta = otherMean - temp_mean;
        FloatAccum n_a = temp_count;
        FloatAccum n_b = __shfl_down_sync(detail::FULL_MASK, temp_count, offset);
        FloatAccum n_new = n_a + n_b;
        FloatAccum n_rcp = n_new != 0.0f ? __builtin_amdgcn_rcpf(n_new) : 0.0f;
        temp_mean = (temp_mean * n_a + otherMean * n_b) * n_rcp;
        temp_var = temp_var + __shfl_down_sync(detail::FULL_MASK, temp_var, offset)
                   + (delta * delta * (n_a * n_b)) * n_rcp;
        temp_count = n_new;
    }
}

// Welford's online algorithm for mean/variance, reducing per-thread (mean, variance, count)
// triples across a workgroup. Replaces the naive E[x^2] - mean^2 accumulation (reduce2/gcn_reduce2)
// for cases where catastrophic cancellation is a problem, e.g. inputs with a large mean
// (MIOpen PR #5013). Unlike reduce2/gcn_reduce2, this only has the portable shuffle-based wave
// reduction: see the comment on shfl_interleaved_reduction_welford above for why.
template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void reduce2_welford(FloatAccum& mean,
                                                 FloatAccum& variance,
                                                 FloatAccum& count,
                                                 FloatAccum scale,
                                                 FloatAccum (&lcl_data_mean)[SizeLclData],
                                                 FloatAccum (&lcl_data_variance)[SizeLclData],
                                                 FloatAccum (&lcl_data_count)[SizeLclData],
                                                 unsigned int lid)
{
    const unsigned int ldsidx = lid >> 6;

    shfl_interleaved_reduction_welford<FloatAccum>(mean, variance, count);

    // The shuffle reduction produces the wave's result in lane 0.
    if((lid % 64) == 0)
    {
        lcl_data_mean[ldsidx] = mean;
        lcl_data_variance[ldsidx] = variance;
        lcl_data_count[ldsidx] = count;
    }

    __syncthreads();

    // The reduction here merges partitions together in a tree-like fashion. Otherwise,
    // precision is lost in both the mean and variance.
    for(unsigned int red = (1 << detail::log2_ceil_v<SizeLclData>) >> 1; red > 0; red >>= 1)
    {
        if(lid < red && lid + red < SizeLclData)
        {
            FloatAccum delta = lcl_data_mean[lid + red] - lcl_data_mean[lid];
            FloatAccum n_a = lcl_data_count[lid];
            FloatAccum n_b = lcl_data_count[lid + red];
            FloatAccum n_new = n_a + n_b;
            FloatAccum n_new_rcp = n_new != 0.0f ? __builtin_amdgcn_rcpf(n_new) : 0.f;
            lcl_data_mean[lid] =
                (lcl_data_mean[lid] * n_a + lcl_data_mean[lid + red] * n_b) * n_new_rcp;
            lcl_data_variance[lid] = lcl_data_variance[lid] + lcl_data_variance[lid + red]
                                     + delta * delta * (n_a * n_b * n_new_rcp);
            lcl_data_count[lid] = n_new;
        }
        __syncthreads();
    }

    mean = lcl_data_mean[0];
    variance = lcl_data_variance[0] * scale;
}

// Manually unrolled Welford update for 4 elements read as a vector, plus a merge step into the
// running (mean, variance, curN). Ported from MIOpen PR #5013.
template <typename FpPrecType>
__forceinline__ __device__ void
    welford_step_unroll4(typename mapped_vector_type<FpPrecType, 4>::type& read4Prec,
                         FpPrecType& mean,
                         FpPrecType& variance,
                         FpPrecType& curN)
{
    FpPrecType mean4 = read4Prec.x;
    FpPrecType oldMean4 = mean4;
    FpPrecType variance4 = hip_kernel_provider::cast<FpPrecType>(0.);

    mean4 += read4Prec.y;
    mean4 *= 0.5f;
    variance4 = fma(read4Prec.y - mean4, read4Prec.y - oldMean4, variance4);

    oldMean4 = mean4;
    mean4 = mean4 * 2.0f + read4Prec.z;
    mean4 *= 0.333333333f;
    variance4 = fma(read4Prec.z - mean4, read4Prec.z - oldMean4, variance4);

    oldMean4 = mean4;
    mean4 = mean4 * 3.0f + read4Prec.w;
    mean4 *= 0.25f;
    variance4 = fma(read4Prec.w - mean4, read4Prec.w - oldMean4, variance4);

    // Merge the local mean and variance with the currently computed ones.
    FpPrecType delta = mean4 - mean;
    mean = (mean4 * 4.0f + mean * curN) / (curN + 4.f);
    variance += variance4 + delta * delta * 4.f * curN / (curN + 4.f);
    curN += 4.f;
}

} // namespace reduction

} // namespace hip_kernel_provider::batchnorm
