// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/utility/type_convert.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// HIP bounds work items per grid dimension as well as blocks per dimension.
inline bool DepthwiseBf16LaunchSupported(long_index_t x,
                                        long_index_t y,
                                        long_index_t z,
                                        size_t shared_bytes)
{
    constexpr long_index_t MaxGridX = std::numeric_limits<uint32_t>::max() / 256;
    if(x <= 0 || x > MaxGridX || y <= 0 || z <= 0)
        return false;
    int device;
    hipDeviceProp_t props{};
    if(hipGetDevice(&device) != hipSuccess ||
       hipGetDeviceProperties(&props, device) != hipSuccess)
        return false;
    return x <= props.maxGridSize[0] && y <= props.maxGridSize[1] &&
           z <= props.maxGridSize[2] && props.maxThreadsPerBlock >= 256 &&
           props.maxThreadsDim[0] >= 256 && shared_bytes <= props.sharedMemPerBlock;
}

template <typename DeviceIndex>
struct DepthwiseBwdWeightBf16FinalizeParams
{
    bhalf_t* wei;
    const float* partial;
    DeviceIndex weight_count;
    DeviceIndex partial_splits;
};

// Sixteen adjacent weight lanes reduce sixteen disjoint split lanes, then Set dW.
template <typename DeviceIndex>
__global__ void kernel_grouped_conv2d_bwd_weight_depthwise_strip_finalize_bf16(
    DepthwiseBwdWeightBf16FinalizeParams<DeviceIndex> a)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx125__)
    constexpr index_t WeightLanes = 16;
    constexpr index_t SplitLanes  = 16;
    const index_t tid             = threadIdx.x;
    const index_t weight_lane     = tid % WeightLanes;
    const index_t split_lane      = tid / WeightLanes;
    const DeviceIndex weight = static_cast<DeviceIndex>(blockIdx.x) * WeightLanes + weight_lane;
    float sum = 0;
    if(weight < a.weight_count)
    {
        for(DeviceIndex s = split_lane; s < a.partial_splits; s += SplitLanes)
            sum += a.partial[s * a.weight_count + weight];
    }

    constexpr index_t Waves = 8;
    const float other = __builtin_bit_cast(
        float, __builtin_amdgcn_ds_bpermute(((tid % 32 ^ 16) << 2), __builtin_bit_cast(int, sum)));
    if(split_lane % 2 == 0)
        sum += other;
    __shared__ float reduction[Waves][WeightLanes];
    if(split_lane % 2 == 0)
        reduction[split_lane / 2][weight_lane] = sum;
    __syncthreads();
    if(split_lane == 0 && weight < a.weight_count)
    {
        float wave_sums[Waves];
#pragma unroll
        for(index_t w = 0; w < Waves; ++w)
            wave_sums[w] = reduction[w][weight_lane];
#pragma unroll
        for(index_t step = Waves / 2; step > 0; step /= 2)
#pragma unroll
            for(index_t w = 0; w < step; ++w)
                wave_sums[w] += wave_sums[w + step];
        a.wei[weight] = type_convert<bhalf_t>(wave_sums[0]);
    }
#else
    ignore = a;
#endif
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
