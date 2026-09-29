// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LayernormCommon.hpp"

constexpr unsigned int PARALLEL_SIZE = HIP_PLUGIN_LAYERNORM_PARALLEL_SIZE;

extern "C" __global__ void layernormBwd(const OutputType* __restrict__ dy,
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

    __shared__ float s_ltmp1[LOCAL_SIZE];
    __shared__ float s_ltmp2[LOCAL_SIZE];
    float pmean = mean ? hip_kernel_provider::cast<float>(mean[gid]) : 0.0f;
    float prstd = rstd ? hip_kernel_provider::cast<float>(rstd[gid]) : 0.0f;
    if(!mean || !rstd)
    {
        __shared__ unsigned int s_ltmp3[LOCAL_SIZE];
        float tmpPmean;
        float tmpPrstd;
        calculateMeanRstd(s_ltmp1, s_ltmp2, s_ltmp3, x, eps, lid, o, s, tmpPmean, tmpPrstd);
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

    float sumDyScale = 0.0f;
    float sumDyScaleX = 0.0f;

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float pdyPscale = hip_kernel_provider::cast<float>(dy[idx])
                           * hip_kernel_provider::cast<float>(scale[i]);

        sumDyScale += pdyPscale;
        sumDyScaleX += pdyPscale * hip_kernel_provider::cast<float>(x[idx]);
    }

    s_ltmp1[lid] = sumDyScale;
    s_ltmp2[lid] = sumDyScaleX;
    __syncthreads();
    for(unsigned int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            s_ltmp1[lid] += s_ltmp1[lid + i];
            s_ltmp2[lid] += s_ltmp2[lid + i];
        }
        __syncthreads();
    }

    sumDyScale = s_ltmp1[0];
    sumDyScaleX = s_ltmp2[0];
    constexpr float INVERSE_INNER_SIZE = 1.0f / INNER_SIZE;
    float a = prstd * prstd * prstd * INVERSE_INNER_SIZE * (sumDyScaleX - sumDyScale * pmean);
    float b = prstd * sumDyScale * INVERSE_INNER_SIZE - a * pmean;

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float pdy = hip_kernel_provider::cast<float>(dy[idx]);
        float pscale = hip_kernel_provider::cast<float>(scale[i]);

        float value = prstd * pdy * pscale - a * hip_kernel_provider::cast<float>(x[idx]) - b;
        dx[idx] = hip_kernel_provider::cast<InputType>(value);
    }
}

extern "C" __global__ void layernormBwdScaleBias(const OutputType* __restrict__ dy,
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

    float sumDs = 0.0f;
    float sumDb = 0.0f;

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

            sumDs += prstd * pdy * (hip_kernel_provider::cast<float>(x[idx]) - pmean);
            sumDb += pdy;
        }
    }

    dscale[gid] = hip_kernel_provider::cast<ScaleBiasType>(sumDs);
    dbias[gid] = hip_kernel_provider::cast<ScaleBiasType>(sumDb);
}

extern "C" __global__ void
    layernormBwdScaleBiasParallel(const OutputType* __restrict__ dy,
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
    unsigned int sLid = (gid % INNER_SIZE) * STRIDE;

    float sumDs = 0.0f;
    float sumDb = 0.0f;

    for(unsigned int i = pid; i < OUTER_SIZE * STRIDE; i += PARALLEL_SIZE)
    {
        unsigned int o = i / STRIDE;
        unsigned int s = i % STRIDE;
        size_t idx = o * INNER_SIZE * STRIDE + sLid + s;

        float pmean = mean ? hip_kernel_provider::cast<float>(mean[i])
                           : workspace[i + 2 * PARALLEL_SIZE * INNER_SIZE];
        float prstd = rstd ? hip_kernel_provider::cast<float>(rstd[i])
                           : workspace[i + OUTER_SIZE * STRIDE + 2 * PARALLEL_SIZE * INNER_SIZE];
        float pdy = hip_kernel_provider::cast<float>(dy[idx]);

        sumDs += pdy * prstd * (hip_kernel_provider::cast<float>(x[idx]) - pmean);
        sumDb += pdy;
    }

    workspace[gid] = sumDs;
    workspace[gid + PARALLEL_SIZE * INNER_SIZE] = sumDb;
}

extern "C" __global__ void layernormBwdScaleBiasReduceSum(const float* __restrict__ workspace,
                                                          ScaleBiasType* __restrict__ dscale,
                                                          ScaleBiasType* __restrict__ dbias)
{
    const unsigned int gid = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(gid >= INNER_SIZE)
    {
        return;
    }

    float sumDs = 0;
    float sumDb = 0;

    for(unsigned int i = 0; i < PARALLEL_SIZE; ++i)
    {
        size_t idx = i * INNER_SIZE + gid;
        sumDs += workspace[idx];
        sumDb += workspace[idx + PARALLEL_SIZE * INNER_SIZE];
    }

    dscale[gid] = hip_kernel_provider::cast<ScaleBiasType>(sumDs);
    dbias[gid] = hip_kernel_provider::cast<ScaleBiasType>(sumDb);
}
