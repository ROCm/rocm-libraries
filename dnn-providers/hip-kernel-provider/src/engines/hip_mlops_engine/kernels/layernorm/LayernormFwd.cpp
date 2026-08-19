// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "VectorTypes.hpp"

constexpr unsigned int LOCAL_SIZE = HIP_PLUGIN_LAYERNORM_LOCAL_SIZE;
constexpr unsigned int INNER_SIZE = HIP_PLUGIN_LAYERNORM_INNER_SIZE;
constexpr unsigned int OUTER_SIZE = HIP_PLUGIN_LAYERNORM_OUTER_SIZE;
constexpr unsigned int STRIDE = HIP_PLUGIN_LAYERNORM_STRIDE;

using InputType = HIP_PLUGIN_LAYERNORM_INPUT_TYPE;
using OutputType = HIP_PLUGIN_LAYERNORM_OUTPUT_TYPE;
using ScaleBiasType = HIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE;
using MeanInvVarianceType = HIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE;

extern "C" __global__ void LayernormFwd(const InputType* __restrict__ x,
                                        OutputType* __restrict__ y,
                                        const ScaleBiasType* __restrict__ weight,
                                        const ScaleBiasType* __restrict__ bias,
                                        MeanInvVarianceType* __restrict__ mean,
                                        MeanInvVarianceType* __restrict__ rstd,
                                        const float eps)
{
    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    float pmean = 0.0f;
    float pm2 = 0.0f;
    unsigned int pcount = 0;
    __shared__ float ltmp1[LOCAL_SIZE];
    __shared__ float ltmp2[LOCAL_SIZE];
    __shared__ unsigned int ltmp3[LOCAL_SIZE];

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t x_idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float px = hip_kernel_provider::cast<float>(x[x_idx]);
        ++pcount;
        float delta = px - pmean;
        pmean += delta / static_cast<float>(pcount);
        float delta2 = px - pmean;
        pm2 += delta * delta2;
    }

    ltmp1[lid] = pmean;
    ltmp2[lid] = pm2;
    ltmp3[lid] = pcount;
    __syncthreads();
    for(unsigned int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            float leftmean = ltmp1[lid];
            float rightmean = ltmp1[lid + i];
            unsigned int leftcount = ltmp3[lid];
            unsigned int rightcount = ltmp3[lid + i];
            unsigned int count = leftcount + rightcount;
            float delta = rightmean - leftmean;
            ltmp1[lid] = count > 0 ? (leftcount * leftmean + rightcount * rightmean) / count : 0.0f;
            ltmp2[lid] += ltmp2[lid + i]
                          + (count > 0 ? delta * delta * leftcount * rightcount / count : 0.0f);
            ltmp3[lid] = count;
        }
        __syncthreads();
    }
    pmean = ltmp1[0];
    float pvar = ltmp2[0] / ltmp3[0];
    float prstd = rsqrtf(pvar + eps);

    if(lid == 0)
    {
        if(mean)
        {
            mean[gid] = hip_kernel_provider::cast<MeanInvVarianceType>(pmean);
        }
        if(rstd)
        {
            rstd[gid] = hip_kernel_provider::cast<MeanInvVarianceType>(prstd);
        }
    }

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float pweight = weight ? hip_kernel_provider::cast<float>(weight[i]) : 1.0f;
        float pbias = bias ? hip_kernel_provider::cast<float>(bias[i]) : 0.0f;

        float val = (hip_kernel_provider::cast<float>(x[idx]) - pmean) * prstd * pweight + pbias;
        y[idx] = hip_kernel_provider::cast<OutputType>(val);
    }
}
