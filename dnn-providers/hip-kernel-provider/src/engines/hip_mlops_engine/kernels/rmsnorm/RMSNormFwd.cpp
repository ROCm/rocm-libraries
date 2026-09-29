// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipKernelActivation.hpp"
#include "VectorTypes.hpp"

constexpr unsigned int LOCAL_SIZE = HIP_PLUGIN_RMSNORM_LOCAL_SIZE;
constexpr unsigned int INNER_SIZE = HIP_PLUGIN_RMSNORM_INNER_SIZE;
constexpr unsigned int STRIDE = HIP_PLUGIN_RMSNORM_STRIDE;

using InputType = HIP_PLUGIN_RMSNORM_INPUT_TYPE;
using OutputType = HIP_PLUGIN_RMSNORM_OUTPUT_TYPE;
using ScaleType = HIP_PLUGIN_RMSNORM_SCALE_TYPE;
using ComputeType = HIP_PLUGIN_RMSNORM_COMPUTE_TYPE;

extern "C" __global__ void rmSnormFwd(const InputType* __restrict__ x,
                                      const ScaleType* __restrict__ scale,
                                      const ScaleType* __restrict__ bias,
                                      OutputType* __restrict__ y,
                                      ComputeType* __restrict__ rstd,
                                      float eps,
                                      ComputeType alpha,
                                      ComputeType beta)
{
    // ComputeType must be float to prevent precision loss
    static_assert(std::is_same<ComputeType, float>::value,
                  "ComputeType must be float for the RMSnormFwd kernel");

    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    float pvar = 0.0f;
    __shared__ float s_ltmp[LOCAL_SIZE];

    // reduce sum
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;
        float tmp = hip_kernel_provider::cast<float>(x[idx]);
        pvar += tmp * tmp;
    }

    s_ltmp[lid] = pvar;
    __syncthreads();
    for(unsigned int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            s_ltmp[lid] += s_ltmp[lid + i];
        }
        __syncthreads();
    }

    pvar = s_ltmp[0] / INNER_SIZE;
    float prstd = rsqrtf(pvar + eps);

    if(lid == 0 && rstd)
    {
        rstd[gid] = prstd;
    }

    // forward calculation
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;
        float yVal = hip_kernel_provider::cast<float>(x[idx]) * prstd
                      * hip_kernel_provider::cast<float>(scale[i]);
        if(bias != nullptr)
        {
            yVal += hip_kernel_provider::cast<float>(bias[i]);
        }
        yVal = hip_kernel_provider::applyActivation<
            float,
            hip_kernel_provider::ActivationMode{HIP_PLUGIN_RMSNORM_NRN_OP_ID}>(yVal, alpha, beta);
        y[idx] = hip_kernel_provider::cast<OutputType>(yVal);
    }
}
