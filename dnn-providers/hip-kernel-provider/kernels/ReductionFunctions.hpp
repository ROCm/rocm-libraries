// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

namespace hip_kernel_plugin
{
namespace reduction
{

namespace detail
{
template <int N>
struct log2_floor
{
    static constexpr int value = log2_floor<(N >> 1)>::value + 1;
};
template <>
struct log2_floor<1>
{
    static constexpr int value = 0;
};
template <int N>
constexpr int log2_floor_v = log2_floor<N>::value;

template <int N>
struct log2_ceil
{
    static constexpr int value = log2_floor_v<N> + ((1 << log2_floor_v<N>) == N ? 0 : 1);
};
template <int N>
constexpr int log2_ceil_v = log2_ceil<N>::value;

} // namespace detail

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2(FloatAccum& x,
                                            FloatAccum& y,
                                            FloatAccum scale,
                                            FloatAccum (&lcl_data_x)[SizeLclData],
                                            FloatAccum (&lcl_data_y)[SizeLclData],
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
    lcl_data[offset1 + 0] = static_cast<FloatAccumC>(x);
    lcl_data[offset1 + 1] = static_cast<FloatAccumC>(y);

    __syncthreads();
    for(unsigned int red = (1 << detail::log2_ceil_v<SizeLclData>) >> 1; red > 0; red >>= 1)
    {
        unsigned int offset2 = offset1 + red * xstride * 2;
        if(ylid < red && offset2 < SizeLclData)
        {
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

} // namespace reduction
} // namespace hip_kernel_plugin
