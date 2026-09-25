// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LayernormCommon.hpp"

constexpr unsigned int PARALLEL_SIZE = HIP_PLUGIN_LAYERNORM_PARALLEL_SIZE;

extern "C" __global__ void LayernormBwd(const OutputType* __restrict__ dy,
                                        const InputType* __restrict__ x,
                                        const ScaleBiasType* __restrict__ scale,
                                        const MeanInvVarianceType* __restrict__ mean,
                                        const MeanInvVarianceType* __restrict__ rstd,
                                        InputType* __restrict__ dx,
                                        float* __restrict__ workspace,
                                        const float eps)
{
    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    __shared__ float ltmp1[LOCAL_SIZE];
    __shared__ float ltmp2[LOCAL_SIZE];
    float pmean = mean ? hip_kernel_provider::cast<float>(mean[gid]) : 0.0f;
    float prstd = rstd ? hip_kernel_provider::cast<float>(rstd[gid]) : 0.0f;
    if(!mean || !rstd)
    {
        __shared__ unsigned int ltmp3[LOCAL_SIZE];
        float tmpPmean;
        float tmpPrstd;
        calculateMeanRstd(ltmp1, ltmp2, ltmp3, x, eps, lid, o, s, tmpPmean, tmpPrstd);
        __syncthreads();
        workspace[gid + 2 * PARALLEL_SIZE * INNER_SIZE] = tmpPmean;
        workspace[gid + OUTER_SIZE * STRIDE + 2 * PARALLEL_SIZE * INNER_SIZE] = tmpPrstd;
        if(!mean)
        {
            pmean = tmpPmean;
        }
        if(!rstd)
        {
            prstd = tmpPrstd;
        }
    }

    float sum_dy_scale = 0.0f;
    float sum_dy_scale_x = 0.0f;

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float pdy_pscale = hip_kernel_provider::cast<float>(dy[idx])
                           * hip_kernel_provider::cast<float>(scale[i]);

        sum_dy_scale += pdy_pscale;
        sum_dy_scale_x += pdy_pscale * hip_kernel_provider::cast<float>(x[idx]);
    }

    ltmp1[lid] = sum_dy_scale;
    ltmp2[lid] = sum_dy_scale_x;
    __syncthreads();
    for(unsigned int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp1[lid] += ltmp1[lid + i];
            ltmp2[lid] += ltmp2[lid + i];
        }
        __syncthreads();
    }

    sum_dy_scale = ltmp1[0];
    sum_dy_scale_x = ltmp2[0];
    constexpr float inverseInnerSize = 1.0f / INNER_SIZE;
    float a = prstd * prstd * prstd * inverseInnerSize * (sum_dy_scale_x - sum_dy_scale * pmean);
    float b = prstd * sum_dy_scale * inverseInnerSize - a * pmean;

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float pdy = hip_kernel_provider::cast<float>(dy[idx]);
        float pscale = hip_kernel_provider::cast<float>(scale[i]);

        float value = prstd * pdy * pscale - a * hip_kernel_provider::cast<float>(x[idx]) - b;
        dx[idx] = hip_kernel_provider::cast<InputType>(value);
    }
}

extern "C" __global__ void LayernormBwdScaleBias(const OutputType* __restrict__ dy,
                                                 const InputType* __restrict__ x,
                                                 const MeanInvVarianceType* __restrict__ mean,
                                                 const MeanInvVarianceType* __restrict__ rstd,
                                                 ScaleBiasType* __restrict__ dscale,
                                                 ScaleBiasType* __restrict__ dbias,
                                                 const float* __restrict__ workspace)
{
    const unsigned int gid = threadIdx.x + blockIdx.x * LOCAL_SIZE;
    if(gid >= INNER_SIZE)
    {
        return;
    }

    float sum_ds = 0.0f;
    float sum_db = 0.0f;

    for(unsigned int o = 0; o < OUTER_SIZE; ++o)
    {
        for(unsigned int s = 0; s < STRIDE; ++s)
        {
            size_t idx = o * INNER_SIZE * STRIDE + gid * STRIDE + s;

            float pmean = mean != nullptr
                              ? hip_kernel_provider::cast<float>(mean[o * STRIDE + s])
                              : workspace[o * STRIDE + s + 2 * PARALLEL_SIZE * INNER_SIZE];
            float prstd = rstd != nullptr ? hip_kernel_provider::cast<float>(rstd[o * STRIDE + s])
                                          : workspace[o * STRIDE + s + OUTER_SIZE * STRIDE
                                                      + 2 * PARALLEL_SIZE * INNER_SIZE];
            float pdy = hip_kernel_provider::cast<float>(dy[idx]);

            sum_ds += prstd * pdy * (hip_kernel_provider::cast<float>(x[idx]) - pmean);
            sum_db += pdy;
        }
    }

    dscale[gid] = hip_kernel_provider::cast<ScaleBiasType>(sum_ds);
    dbias[gid] = hip_kernel_provider::cast<ScaleBiasType>(sum_db);
}

extern "C" __global__ void
    LayernormBwdScaleBiasParallel(const OutputType* __restrict__ dy,
                                  const InputType* __restrict__ x,
                                  const MeanInvVarianceType* __restrict__ mean,
                                  const MeanInvVarianceType* __restrict__ rstd,
                                  float* __restrict__ workspace)
{
    const unsigned int gid = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(gid >= INNER_SIZE * PARALLEL_SIZE)
    {
        return;
    }

    unsigned int pid = gid / INNER_SIZE;
    unsigned int s_lid = (gid % INNER_SIZE) * STRIDE;

    float sum_ds = 0.0f;
    float sum_db = 0.0f;

    for(unsigned int i = pid; i < OUTER_SIZE * STRIDE; i += PARALLEL_SIZE)
    {
        unsigned int o = i / STRIDE;
        unsigned int s = i % STRIDE;
        size_t idx = o * INNER_SIZE * STRIDE + s_lid + s;

        float pmean = mean ? hip_kernel_provider::cast<float>(mean[i])
                           : workspace[i + 2 * PARALLEL_SIZE * INNER_SIZE];
        float prstd = rstd ? hip_kernel_provider::cast<float>(rstd[i])
                           : workspace[i + OUTER_SIZE * STRIDE + 2 * PARALLEL_SIZE * INNER_SIZE];
        float pdy = hip_kernel_provider::cast<float>(dy[idx]);

        sum_ds += pdy * prstd * (hip_kernel_provider::cast<float>(x[idx]) - pmean);
        sum_db += pdy;
    }

    workspace[gid] = sum_ds;
    workspace[gid + PARALLEL_SIZE * INNER_SIZE] = sum_db;
}

extern "C" __global__ void LayernormBwdScaleBiasReduceSum(const float* __restrict__ workspace,
                                                          ScaleBiasType* __restrict__ dscale,
                                                          ScaleBiasType* __restrict__ dbias)
{
    const unsigned int gid = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(gid >= INNER_SIZE)
    {
        return;
    }

    float sum_ds = 0;
    float sum_db = 0;

    for(unsigned int i = 0; i < PARALLEL_SIZE; ++i)
    {
        size_t idx = i * INNER_SIZE + gid;
        sum_ds += workspace[idx];
        sum_db += workspace[idx + PARALLEL_SIZE * INNER_SIZE];
    }

    dscale[gid] = hip_kernel_provider::cast<ScaleBiasType>(sum_ds);
    dbias[gid] = hip_kernel_provider::cast<ScaleBiasType>(sum_db);
}
