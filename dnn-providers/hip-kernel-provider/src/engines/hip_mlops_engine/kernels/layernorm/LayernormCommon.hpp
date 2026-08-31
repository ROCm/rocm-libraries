// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "VectorTypes.hpp"

constexpr unsigned int LOCAL_SIZE = HIP_PLUGIN_LAYERNORM_LOCAL_SIZE;
constexpr unsigned int INNER_SIZE = HIP_PLUGIN_LAYERNORM_INNER_SIZE;
constexpr unsigned int OUTER_SIZE = HIP_PLUGIN_LAYERNORM_OUTER_SIZE;
constexpr unsigned int STRIDE = HIP_PLUGIN_LAYERNORM_STRIDE;

using InputType = HIP_PLUGIN_LAYERNORM_INPUT_TYPE;
using OutputType = HIP_PLUGIN_LAYERNORM_OUTPUT_TYPE;
using ScaleBiasType = HIP_PLUGIN_LAYERNORM_SCALE_BIAS_TYPE;
using MeanInvVarianceType = HIP_PLUGIN_LAYERNORM_MEAN_INV_VARIANCE_TYPE;

__forceinline__ __device__ void calculateMeanRstd(__shared__ float ltmp1[LOCAL_SIZE],
                                                  __shared__ float ltmp2[LOCAL_SIZE],
                                                  __shared__ unsigned int ltmp3[LOCAL_SIZE],
                                                  const InputType* __restrict__ x,
                                                  const float eps,
                                                  const unsigned int lid,
                                                  const unsigned int o,
                                                  const unsigned int s,
                                                  float& pmean,
                                                  float& prstd)
{
    pmean = 0.0f;
    float pm2 = 0.0f;
    unsigned int pcount = 0;

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
    prstd = rsqrtf(pvar + eps);
}
