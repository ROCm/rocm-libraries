// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference Layernorm forward kernel.
// Compiled via HipRTC with -DX_TYPE=<type> -DY_TYPE=<type> -DSCALE_BIAS_TYPE=<type>
// -DMEAN_RSTD_TYPE=<type> -DCOMPUTE_TYPE=<type> -DLOCAL_SIZE=<value>
// -DOUTER_SIZE=<value> -DINNER_SIZE=<value> -DSTRIDE=<value>. Each thread block computes
// one normalization group (outerSize) reducing over the innerSize elements in parallel
// across the threads to compute the mean and inverse variance and then normalizing the
// input elements in the group to produce the output elements.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void LayernormFwdRef(LayernormFwdArgs args)
{
    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* y = static_cast<Y_TYPE*>(args.y);
    auto* scale = static_cast<const SCALE_BIAS_TYPE*>(args.scale);
    auto* bias = static_cast<const SCALE_BIAS_TYPE*>(args.bias);
    auto* mean = static_cast<MEAN_RSTD_TYPE*>(args.mean);
    auto* rstd = static_cast<MEAN_RSTD_TYPE*>(args.rstd);
    auto epsilon = args.epsilon;

    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    auto pmean = static_cast<COMPUTE_TYPE>(0.0);
    auto pm2 = static_cast<COMPUTE_TYPE>(0.0);
    unsigned int pcount = 0;
    __shared__ COMPUTE_TYPE ltmp1[LOCAL_SIZE];
    __shared__ COMPUTE_TYPE ltmp2[LOCAL_SIZE];
    __shared__ unsigned int ltmp3[LOCAL_SIZE];

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t x_idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        auto px = static_cast<COMPUTE_TYPE>(x[x_idx]);
        ++pcount;
        auto delta = px - pmean;
        pmean += delta / static_cast<COMPUTE_TYPE>(pcount);
        auto delta2 = px - pmean;
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
            auto leftmean = ltmp1[lid];
            auto rightmean = ltmp1[lid + i];
            unsigned int leftcount = ltmp3[lid];
            unsigned int rightcount = ltmp3[lid + i];
            unsigned int count = leftcount + rightcount;
            auto delta = rightmean - leftmean;
            ltmp1[lid] = count > 0 ? (leftcount * leftmean + rightcount * rightmean) / count : 0.0f;
            ltmp2[lid] += ltmp2[lid + i]
                          + (count > 0 ? delta * delta * leftcount * rightcount / count : 0.0f);
            ltmp3[lid] = count;
        }
        __syncthreads();
    }
    pmean = ltmp1[0];
    auto pvar = ltmp2[0] / ltmp3[0];
    auto prstd = rsqrtf(pvar + epsilon);

    if(lid == 0)
    {
        if(mean)
        {
            mean[gid] = static_cast<MEAN_RSTD_TYPE>(pmean);
        }
        if(rstd)
        {
            rstd[gid] = static_cast<MEAN_RSTD_TYPE>(prstd);
        }
    }

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        auto pscale = scale ? static_cast<COMPUTE_TYPE>(scale[i]) : static_cast<COMPUTE_TYPE>(1.0);
        auto pbias = bias ? static_cast<COMPUTE_TYPE>(bias[i]) : static_cast<COMPUTE_TYPE>(0.0);

        auto val = (static_cast<COMPUTE_TYPE>(x[idx]) - pmean) * prstd * pscale + pbias;
        y[idx] = static_cast<Y_TYPE>(val);
    }
}
