// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_REDUCTION_FUNCTIONS_HPP
#define GUARD_REDUCTION_FUNCTIONS_HPP

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "configuration.hpp"
#include "static_unroll.hpp"

// NOTE: This header should be independent from batchnorm_functions.hpp
// Even in OpenCL implementation, these functions are only enabled under
// certain conditions. But now, these templates will not be compiled before
// calling them.
namespace miopen {
namespace reduction {

namespace detail {

const unsigned long long FULL_MASK = 0xFFFFFFFFFFFFFFFFull;

__forceinline__ __device__ unsigned int next_power_of_2(unsigned int n)
{
    unsigned int p = 1;
    while(p < n)
        p <<= 1;
    return p;
}

} // namespace detail

template <typename FloatAccum, unsigned int BlockSize>
__forceinline__ __device__ void
reduce2(FloatAccum& x, FloatAccum& y, FloatAccum scale, unsigned int lid)
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

        const unsigned int lane      = lid % static_cast<unsigned int>(warpSize);
        const unsigned int wid       = lid / static_cast<unsigned int>(warpSize);
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
    lcl_data[offset1 + 0] = static_cast<FloatAccumC>(x);
    lcl_data[offset1 + 1] = static_cast<FloatAccumC>(y);

    __syncthreads();

    const unsigned int red_start = detail::next_power_of_2(size) >> 1;

    for(unsigned int red = red_start; red > 0; red >>= 1)
    {
        unsigned int offset2 = offset1 + red * xstride * 2;
        if(ylid < red && ylid + red < size && offset2 + 1 < SizeLclData)
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

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void gcn_reduce2(FloatAccum& x,
                                            FloatAccum& y,
                                            FloatAccum scale,
                                            FloatAccum (&lcl_data_x)[SizeLclData],
                                            FloatAccum (&lcl_data_y)[SizeLclData],
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

    static_unroll_count<unsigned int, 0, SizeLclData, 1, 2>{[&](unsigned int i) {
        x += lcl_data_x[i];
        y += lcl_data_y[i];
    }};

    x *= scale;
    y *= scale;
}

} // namespace reduction
} // namespace miopen

#endif
