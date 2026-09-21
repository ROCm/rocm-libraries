// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "LayernormCommon.hpp"

extern "C" __global__ void LayernormFwd(const InputType* __restrict__ x,
                                        OutputType* __restrict__ y,
                                        const ScaleBiasType* __restrict__ scale,
                                        const ScaleBiasType* __restrict__ bias,
                                        MeanInvVarianceType* __restrict__ mean,
                                        MeanInvVarianceType* __restrict__ rstd,
                                        const float eps)
{
    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    __shared__ float ltmp1[LOCAL_SIZE];
    __shared__ float ltmp2[LOCAL_SIZE];
    __shared__ unsigned int ltmp3[LOCAL_SIZE];
    float pmean;
    float prstd;
    calculateMeanRstd(ltmp1, ltmp2, ltmp3, x, eps, lid, o, s, pmean, prstd);

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

        float pscale = scale ? hip_kernel_provider::cast<float>(scale[i]) : 1.0f;
        float pbias = bias ? hip_kernel_provider::cast<float>(bias[i]) : 0.0f;

        float val = (hip_kernel_provider::cast<float>(x[idx]) - pmean) * prstd * pscale + pbias;
        y[idx] = hip_kernel_provider::cast<OutputType>(val);
    }
}
