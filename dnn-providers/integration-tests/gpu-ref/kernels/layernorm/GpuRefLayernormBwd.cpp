// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference Layernorm backward kernel.
// Compiled via HipRTC with -DX_TYPE=<type> -DY_TYPE=<type> -DSCALE_BIAS_TYPE=<type>
// -DMEAN_RSTD_TYPE=<type> -DCOMPUTE_TYPE=<type> -DLOCAL_SIZE=<value>
// -DOUTER_SIZE=<value> -DINNER_SIZE=<value> -DSTRIDE=<value>. In the first kernel, each
// thread block computes one normalization group (outerSize) reducing over the innerSize
// elements in parallel across the threads to compute the mean and inverse variance if
// necessary and then calculating the gradient of the x elements from the input and
// output elements. The second kernel reuses the mean and inverse variance to calculate
// the gradients of the scale and bias from the input and output elements.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void LayernormBwdRef(LayernormBwdArgs args)
{
    auto* dy = static_cast<const Y_TYPE*>(args.dy);
    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* scale = static_cast<const SCALE_BIAS_TYPE*>(args.scale);
    auto* mean = static_cast<const MEAN_RSTD_TYPE*>(args.mean);
    auto* rstd = static_cast<const MEAN_RSTD_TYPE*>(args.rstd);
    auto* dx = static_cast<X_TYPE*>(args.dx);
    auto* workspace = static_cast<COMPUTE_TYPE*>(args.workspace);
    auto epsilon = args.epsilon;

    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    __shared__ COMPUTE_TYPE ltmp1[LOCAL_SIZE];
    __shared__ COMPUTE_TYPE ltmp2[LOCAL_SIZE];
    auto pmean = mean ? static_cast<COMPUTE_TYPE>(mean[gid]) : static_cast<COMPUTE_TYPE>(0.0);
    auto prstd = rstd ? static_cast<COMPUTE_TYPE>(rstd[gid]) : static_cast<COMPUTE_TYPE>(0.0);
    if(!mean || !rstd)
    {
        __shared__ unsigned int ltmp3[LOCAL_SIZE];
        auto tmpmean = static_cast<COMPUTE_TYPE>(0.0);
        auto tmprstd = static_cast<COMPUTE_TYPE>(0.0);
        auto pm2 = static_cast<COMPUTE_TYPE>(0.0);
        unsigned int pcount = 0;
        for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
        {
            size_t x_idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

            auto px = static_cast<COMPUTE_TYPE>(x[x_idx]);
            ++pcount;
            auto delta = px - tmpmean;
            tmpmean += delta / static_cast<COMPUTE_TYPE>(pcount);
            auto delta2 = px - tmpmean;
            pm2 += delta * delta2;
        }

        ltmp1[lid] = tmpmean;
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
                ltmp1[lid]
                    = count > 0 ? (leftcount * leftmean + rightcount * rightmean) / count : 0.0f;
                ltmp2[lid] += ltmp2[lid + i]
                              + (count > 0 ? delta * delta * leftcount * rightcount / count : 0.0f);
                ltmp3[lid] = count;
            }
            __syncthreads();
        }
        tmpmean = ltmp1[0];
        auto pvar = ltmp2[0] / ltmp3[0];
        tmprstd = rsqrtf(pvar + epsilon);
        workspace[gid] = tmpmean;
        workspace[gid + OUTER_SIZE * STRIDE] = tmprstd;
        if(!mean)
        {
            pmean = tmpmean;
        }
        if(!rstd)
        {
            prstd = tmprstd;
        }
    }

    auto sum_dy_scale = static_cast<COMPUTE_TYPE>(0.0);
    auto sum_dy_scale_x = static_cast<COMPUTE_TYPE>(0.0);

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        auto pdy_pscale = static_cast<COMPUTE_TYPE>(dy[idx]) * static_cast<COMPUTE_TYPE>(scale[i]);

        sum_dy_scale += pdy_pscale;
        sum_dy_scale_x += pdy_pscale * static_cast<COMPUTE_TYPE>(x[idx]);
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
    constexpr COMPUTE_TYPE pscale = static_cast<COMPUTE_TYPE>(1.0 / INNER_SIZE);
    auto a = prstd * prstd * prstd * pscale * (sum_dy_scale_x - sum_dy_scale * pmean);
    auto b = prstd * sum_dy_scale * pscale - a * pmean;

    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        auto pdy = static_cast<COMPUTE_TYPE>(dy[idx]);
        auto pscale = static_cast<COMPUTE_TYPE>(scale[i]);

        auto value = prstd * pdy * pscale - a * static_cast<COMPUTE_TYPE>(x[idx]) - b;
        dx[idx] = static_cast<X_TYPE>(value);
    }
}

extern "C" __global__ void LayernormBwdWeightsRef(LayernormBwdWeightsArgs args)
{
    const unsigned int gid = threadIdx.x + blockIdx.x * LOCAL_SIZE;
    if(gid >= INNER_SIZE)
    {
        return;
    }

    auto* dy = static_cast<const Y_TYPE*>(args.dy);
    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* mean = static_cast<const MEAN_RSTD_TYPE*>(args.mean);
    auto* rstd = static_cast<const MEAN_RSTD_TYPE*>(args.rstd);
    auto* dscale = static_cast<SCALE_BIAS_TYPE*>(args.dscale);
    auto* dbias = static_cast<SCALE_BIAS_TYPE*>(args.dbias);
    auto* workspace = static_cast<const COMPUTE_TYPE*>(args.workspace);

    auto sum_ds = static_cast<COMPUTE_TYPE>(0.0);
    auto sum_db = static_cast<COMPUTE_TYPE>(0.0);

    for(unsigned int o = 0; o < OUTER_SIZE; ++o)
    {
        for(unsigned int s = 0; s < STRIDE; ++s)
        {
            unsigned int idx = o * INNER_SIZE * STRIDE + gid * STRIDE + s;

            auto pmean = mean != nullptr ? static_cast<COMPUTE_TYPE>(mean[o * STRIDE + s])
                                         : workspace[o * STRIDE + s];
            auto prstd = rstd != nullptr ? static_cast<COMPUTE_TYPE>(rstd[o * STRIDE + s])
                                         : workspace[o * STRIDE + s + OUTER_SIZE * STRIDE];
            auto pdy = static_cast<COMPUTE_TYPE>(dy[idx]);

            sum_ds += prstd * pdy * (static_cast<COMPUTE_TYPE>(x[idx]) - pmean);
            sum_db += pdy;
        }
    }

    dscale[gid] = static_cast<SCALE_BIAS_TYPE>(sum_ds);
    dbias[gid] = static_cast<SCALE_BIAS_TYPE>(sum_db);
}
