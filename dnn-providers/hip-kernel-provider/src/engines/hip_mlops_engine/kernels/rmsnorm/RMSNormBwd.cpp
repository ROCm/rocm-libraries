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

extern "C" __global__ void rmsNormBwdScaleBias(const DyType* __restrict__ dy,
                                               const XType* __restrict__ x,
                                               const ComputeType* __restrict__ rstd,
                                               ScaleType* __restrict__ dscale,
                                               ScaleType* __restrict__ dbias,
                                               const YType* __restrict__ y,
                                               ComputeType alpha,
                                               ComputeType beta)
{
    static_assert(std::is_same_v<ComputeType, float>,
                  "ComputeType must be float for the rmsNormBwdScaleBias kernel");

    // NOLINTNEXTLINE(readability-static-accessed-through-instance)
    const unsigned int tidx = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(tidx >= INNER_SIZE)
    {
        return;
    }

    float sumDScale = 0.0f;
    float sumDBias = 0.0f;

    // backward scale calculation
    for(unsigned int o = 0; o < OUTER_SIZE; ++o)
    {
        for(unsigned int s = 0; s < STRIDE; ++s)
        {
            const size_t idx = o * INNER_SIZE * STRIDE + tidx * STRIDE + s;

            const auto prstd = hip_kernel_provider::cast<float>(rstd[o * STRIDE + s]);
            auto pdy = hip_kernel_provider::cast<float>(dy[idx]);
            const auto px = hip_kernel_provider::cast<float>(x[idx]);
            if constexpr(hip_kernel_provider::ActivationMode{HIP_PLUGIN_RMSNORM_NRN_OP_ID}
                         != hip_kernel_provider::ActivationMode::PASTHRU)
            {
                const auto py = hip_kernel_provider::cast<float>(y[idx]);
                pdy = hip_kernel_provider::applyActivationGradient<
                    float,
                    hip_kernel_provider::ActivationMode{HIP_PLUGIN_RMSNORM_NRN_OP_ID}>(
                    pdy, py, alpha, beta);
            }

            sumDScale += pdy * px * prstd;
            sumDBias += pdy;
        }
    }

    dscale[tidx] = hip_kernel_provider::cast<ScaleType>(sumDScale);
    if(dbias != nullptr)
    {
        dbias[tidx] = hip_kernel_provider::cast<ScaleType>(sumDBias);
    }
}

extern "C" __global__ void rmsNormBwdData(const DyType* __restrict__ dy,
                                          const XType* __restrict__ x,
                                          const ScaleType* __restrict__ scale,
                                          const ComputeType* __restrict__ rstd,
                                          DxType* __restrict__ dx,
                                          const YType* __restrict__ y,
                                          ComputeType alpha,
                                          ComputeType beta)
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

        auto pdy = hip_kernel_provider::cast<float>(dy[idx]);
        const auto px = hip_kernel_provider::cast<float>(x[idx]);
        const auto pscale = hip_kernel_provider::cast<float>(scale[i]);
        if constexpr(hip_kernel_provider::ActivationMode{HIP_PLUGIN_RMSNORM_NRN_OP_ID}
                     != hip_kernel_provider::ActivationMode::PASTHRU)
        {
            const auto py = hip_kernel_provider::cast<float>(y[idx]);
            pdy = hip_kernel_provider::applyActivationGradient<float,
                                                               hip_kernel_provider::ActivationMode{
                                                                   HIP_PLUGIN_RMSNORM_NRN_OP_ID}>(
                pdy, py, alpha, beta);
        }

        mean += pdy * pscale * px;
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

        auto pdy = hip_kernel_provider::cast<float>(dy[idx]);
        const auto px = hip_kernel_provider::cast<float>(x[idx]);
        const auto pscale = hip_kernel_provider::cast<float>(scale[i]);
        if constexpr(hip_kernel_provider::ActivationMode{HIP_PLUGIN_RMSNORM_NRN_OP_ID}
                     != hip_kernel_provider::ActivationMode::PASTHRU)
        {
            const auto py = hip_kernel_provider::cast<float>(y[idx]);
            pdy = hip_kernel_provider::applyActivationGradient<float,
                                                               hip_kernel_provider::ActivationMode{
                                                                   HIP_PLUGIN_RMSNORM_NRN_OP_ID}>(
                pdy, py, alpha, beta);
        }

        const float dxVal = (pdy * pscale * prstd) - (mean * px * prstd * prstd * prstd);
        dx[idx] = hip_kernel_provider::cast<DxType>(dxVal);
    }
}
