// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "VectorTypes.hpp"

constexpr unsigned int LOCAL_SIZE = HIP_PLUGIN_RMSNORM_LOCAL_SIZE;
constexpr unsigned int INNER_SIZE = HIP_PLUGIN_RMSNORM_INNER_SIZE;
constexpr unsigned int OUTER_SIZE = HIP_PLUGIN_RMSNORM_OUTER_SIZE;
constexpr unsigned int STRIDE = HIP_PLUGIN_RMSNORM_STRIDE;

using XType = HIP_PLUGIN_RMSNORM_X_TYPE;
using DyType = HIP_PLUGIN_RMSNORM_DY_TYPE;
using DxType = HIP_PLUGIN_RMSNORM_DX_TYPE;
using ScaleType = HIP_PLUGIN_RMSNORM_SCALE_TYPE;
using ComputeType = HIP_PLUGIN_RMSNORM_COMPUTE_TYPE;

extern "C" __global__ void rmsNormBwdWeightBias(const DyType* __restrict__ dy,
                                                const XType* __restrict__ x,
                                                const ComputeType* __restrict__ rstd,
                                                ScaleType* __restrict__ dweight,
                                                ScaleType* __restrict__ dbias)
{
    static_assert(std::is_same_v<ComputeType, float>,
                  "ComputeType must be float for the rmsNormBwdWeightBias kernel");

    // NOLINTNEXTLINE(readability-static-accessed-through-instance)
    const unsigned int tidx = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(tidx >= INNER_SIZE)
    {
        return;
    }

    float sumDw = 0.0f;
    float sumDb = 0.0f;

    // backward weight calculation
    for(unsigned int o = 0; o < OUTER_SIZE; ++o)
    {
        for(unsigned int s = 0; s < STRIDE; ++s)
        {
            const size_t idx = o * INNER_SIZE * STRIDE + tidx * STRIDE + s;

            const float prstd = rstd[o * STRIDE + s];
            const auto pdy = hip_kernel_provider::cast<float>(dy[idx]);
            const auto px = hip_kernel_provider::cast<float>(x[idx]);

            sumDw += pdy * px * prstd;
            sumDb += pdy;
        }
    }

    dweight[tidx] = hip_kernel_provider::cast<ScaleType>(sumDw);
    if(dbias != nullptr)
    {
        dbias[tidx] = hip_kernel_provider::cast<ScaleType>(sumDb);
    }
}

extern "C" __global__ void rmsNormBwdData(const DyType* __restrict__ dy,
                                          const XType* __restrict__ x,
                                          const ScaleType* __restrict__ weight,
                                          const ComputeType* __restrict__ rstd,
                                          DxType* __restrict__ dx)
{
    static_assert(std::is_same_v<ComputeType, float>,
                  "ComputeType must be float for the rmsNormBwdData kernel");

    const unsigned int gid = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int lid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    __shared__ float s_ltmp[LOCAL_SIZE];
    float mean = 0.0f;

    // reduce sum
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        const size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        const auto pdy = hip_kernel_provider::cast<float>(dy[idx]);
        const auto px = hip_kernel_provider::cast<float>(x[idx]);
        const auto pw = hip_kernel_provider::cast<float>(weight[i]);

        mean += pdy * pw * px;
    }

    s_ltmp[lid] = mean;
    __syncthreads();

    for(unsigned int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            s_ltmp[lid] += s_ltmp[lid + i];
        }
        __syncthreads();
    }

    mean = s_ltmp[0] / INNER_SIZE;
    const float prstd = rstd[gid];

    // backward data calculation
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        const size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        const auto pdy = hip_kernel_provider::cast<float>(dy[idx]);
        const auto px = hip_kernel_provider::cast<float>(x[idx]);
        const auto pw = hip_kernel_provider::cast<float>(weight[i]);

        const float dxVal = (pdy * pw * prstd) - (mean * px * prstd * prstd * prstd);
        dx[idx] = hip_kernel_provider::cast<DxType>(dxVal);
    }
}
