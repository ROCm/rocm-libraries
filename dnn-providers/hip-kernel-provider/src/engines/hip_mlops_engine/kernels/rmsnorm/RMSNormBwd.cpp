// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "HipKernelActivation.hpp"
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
using YType = HIP_PLUGIN_RMSNORM_Y_TYPE;

extern "C" __global__ void RMSnormBwdScaleBias(const DyType* __restrict__ dy,
                                               const XType* __restrict__ x,
                                               const ComputeType* __restrict__ rstd,
                                               ScaleType* __restrict__ dscale,
                                               ScaleType* __restrict__ dbias,
                                               const YType* __restrict__ y,
                                               ComputeType alpha,
                                               ComputeType beta)
{
    static_assert(std::is_same<ComputeType, float>::value,
                  "ComputeType must be float for the RMSnormBwdScaleBias kernel");

    const unsigned int tidx = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(tidx >= INNER_SIZE)
    {
        return;
    }

    float sum_dscale = 0.0f;
    float sum_dbias = 0.0f;

    // backward scale calculation
    for(unsigned int o = 0; o < OUTER_SIZE; ++o)
    {
        for(unsigned int s = 0; s < STRIDE; ++s)
        {
            size_t idx = o * INNER_SIZE * STRIDE + tidx * STRIDE + s;

            float prstd = hip_kernel_provider::cast<float>(rstd[o * STRIDE + s]);
            float pdy = hip_kernel_provider::cast<float>(dy[idx]);
            float px = hip_kernel_provider::cast<float>(x[idx]);
            if constexpr(hip_kernel_provider::ActivationMode{HIP_PLUGIN_RMSNORM_NRN_OP_ID}
                         != hip_kernel_provider::ActivationMode::PASTHRU)
            {
                float py = hip_kernel_provider::cast<float>(y[idx]);
                pdy = hip_kernel_provider::applyActivationGradient<
                    float,
                    hip_kernel_provider::ActivationMode{HIP_PLUGIN_RMSNORM_NRN_OP_ID}>(
                    pdy, py, alpha, beta);
            }

            sum_dscale += pdy * px * prstd;
            sum_dbias += pdy;
        }
    }

    dscale[tidx] = hip_kernel_provider::cast<ScaleType>(sum_dscale);
    if(dbias)
    {
        dbias[tidx] = hip_kernel_provider::cast<ScaleType>(sum_dbias);
    }
}

extern "C" __global__ void RMSnormBwdData(const DyType* __restrict__ dy,
                                          const XType* __restrict__ x,
                                          const ScaleType* __restrict__ scale,
                                          const ComputeType* __restrict__ rstd,
                                          DxType* __restrict__ dx,
                                          const YType* __restrict__ y,
                                          ComputeType alpha,
                                          ComputeType beta)
{
    static_assert(std::is_same<ComputeType, float>::value,
                  "ComputeType must be float for the RMSnormBwdData kernel");

    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    __shared__ float ltmp[LOCAL_SIZE];
    float mean = 0.0f;

    // reduce sum
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float pdy = hip_kernel_provider::cast<float>(dy[idx]);
        float px = hip_kernel_provider::cast<float>(x[idx]);
        float pscale = hip_kernel_provider::cast<float>(scale[i]);
        if constexpr(hip_kernel_provider::ActivationMode{HIP_PLUGIN_RMSNORM_NRN_OP_ID}
                     != hip_kernel_provider::ActivationMode::PASTHRU)
        {
            float py = hip_kernel_provider::cast<float>(y[idx]);
            pdy = hip_kernel_provider::applyActivationGradient<float,
                                                               hip_kernel_provider::ActivationMode{
                                                                   HIP_PLUGIN_RMSNORM_NRN_OP_ID}>(
                pdy, py, alpha, beta);
        }

        mean += pdy * pscale * px;
    }

    ltmp[lid] = mean;
    __syncthreads();

    for(unsigned int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }

    mean = ltmp[0] / INNER_SIZE;
    float prstd = rstd[gid];

    // backward data calculation
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float pdy = hip_kernel_provider::cast<float>(dy[idx]);
        float px = hip_kernel_provider::cast<float>(x[idx]);
        float pscale = hip_kernel_provider::cast<float>(scale[i]);
        if constexpr(hip_kernel_provider::ActivationMode{HIP_PLUGIN_RMSNORM_NRN_OP_ID}
                     != hip_kernel_provider::ActivationMode::PASTHRU)
        {
            float py = hip_kernel_provider::cast<float>(y[idx]);
            pdy = hip_kernel_provider::applyActivationGradient<float,
                                                               hip_kernel_provider::ActivationMode{
                                                                   HIP_PLUGIN_RMSNORM_NRN_OP_ID}>(
                pdy, py, alpha, beta);
        }

        float dx_val = (pdy * pscale * prstd) - (mean * px * prstd * prstd * prstd);
        dx[idx] = hip_kernel_provider::cast<DxType>(dx_val);
    }
}
