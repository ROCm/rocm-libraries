// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <type_traits>

#include "RMSNormCommon.hpp"

constexpr unsigned int LOCAL_SIZE = HIP_PLUGIN_RMSNORM_LOCAL_SIZE;
constexpr unsigned int INNER_SIZE = HIP_PLUGIN_RMSNORM_INNER_SIZE;
constexpr unsigned int STRIDE = HIP_PLUGIN_RMSNORM_STRIDE;

using InputType = HIP_PLUGIN_RMSNORM_INPUT_TYPE;
using OutputType = HIP_PLUGIN_RMSNORM_OUTPUT_TYPE;
using ScaleType = HIP_PLUGIN_RMSNORM_SCALE_TYPE;
using ComputeType = HIP_PLUGIN_RMSNORM_COMPUTE_TYPE;

extern "C" __global__ void RMSnormFwd(const InputType* __restrict__ x,
                                      const ScaleType* __restrict__ weight,
                                      const ScaleType* __restrict__ bias,
                                      OutputType* __restrict__ y,
                                      ComputeType* __restrict__ rstd,
                                      float eps)
{
    // ComputeType must be float to prevent precision loss
    static_assert(std::is_same<ComputeType, float>::value,
                  "ComputeType must be float for the RMSnormFwd kernel");

    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    float pvar = 0.0f;
    __shared__ float ltmp[LOCAL_SIZE];

    // reduce sum
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;
        float tmp = hip_kernel_provider::rmsnorm::to_float32<InputType>(x[idx]);
        pvar += tmp * tmp;
    }

    ltmp[lid] = pvar;
    __syncthreads();
    for(unsigned int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }

    pvar = ltmp[0] / INNER_SIZE;
    float prstd = rsqrtf(pvar + eps);

    if(lid == 0 && rstd)
    {
        rstd[gid] = prstd;
    }

    // forward calculation
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;
        float y_val = hip_kernel_provider::rmsnorm::to_float32<InputType>(x[idx]) * prstd
                      * hip_kernel_provider::rmsnorm::to_float32<ScaleType>(weight[i]);
        if(bias != nullptr)
        {
            y_val += hip_kernel_provider::rmsnorm::to_float32<ScaleType>(bias[i]);
        }
        y[idx] = hip_kernel_provider::rmsnorm::from_float32<OutputType>(y_val);
    }
}
