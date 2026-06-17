// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "FloatTypes.h"

extern "C" __global__ void LayernormFwd(const FLOAT* __restrict__ x,
                                        FLOAT* __restrict__ y,
                                        const FLOAT* __restrict__ weight,
                                        const FLOAT* __restrict__ bias,
                                        FLOAT* __restrict__ mean,
                                        FLOAT* __restrict__ rstd,
                                        const float eps)
{
    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    FLOAT_ACCUM pmean = CVT_FP32_2ACCUM(0.0f);
    FLOAT_ACCUM pm2 = CVT_FP32_2ACCUM(0.0f);
    unsigned int pcount = 0;
    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];
    __shared__ unsigned int ltmp3[LOCAL_SIZE];

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t x_idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        FLOAT_ACCUM px = CVT_FLOAT2ACCUM(x[x_idx]);
        ++pcount;
        FLOAT_ACCUM delta = px - pmean;
        pmean += delta / static_cast<FLOAT_ACCUM>(pcount);
        FLOAT_ACCUM delta2 = px - pmean;
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
            FLOAT_ACCUM leftmean = ltmp1[lid];
            FLOAT_ACCUM rightmean = ltmp1[lid + i];
            unsigned int leftcount = ltmp3[lid];
            unsigned int rightcount = ltmp3[lid + i];
            unsigned int count = leftcount + rightcount;
            FLOAT_ACCUM delta = rightmean - leftmean;
            ltmp1[lid] = count > 0 ? (leftcount * leftmean + rightcount * rightmean) / count
                                   : CVT_FP32_2ACCUM(0.0f);
            ltmp2[lid] += ltmp2[lid + i]
                          + (count > 0 ? delta * delta * leftcount * rightcount / count
                                       : CVT_FP32_2ACCUM(0.0f));
            ltmp3[lid] = count;
        }
        __syncthreads();
    }
    pmean = ltmp1[0];
    FLOAT_ACCUM pvar = ltmp2[0] / ltmp3[0];
    FLOAT_ACCUM prstd = rsqrt(pvar + CVT_FP32_2ACCUM(eps));

    if(lid == 0)
    {
        if(mean)
        {
            mean[gid] = CVT_ACCUM2FLOAT(pmean);
        }
        if(rstd)
        {
            rstd[gid] = CVT_ACCUM2FLOAT(prstd);
        }
    }

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        FLOAT_ACCUM pweight = weight ? CVT_FLOAT2ACCUM(weight[i]) : CVT_FP32_2ACCUM(1.0f);
        FLOAT_ACCUM pbias = bias ? CVT_FLOAT2ACCUM(bias[i]) : CVT_FP32_2ACCUM(0.0f);

        FLOAT_ACCUM val = (CVT_FLOAT2ACCUM(x[idx]) - pmean) * prstd * pweight + pbias;
        y[idx] = CVT_ACCUM2FLOAT(val);
    }
}
