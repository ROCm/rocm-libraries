// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference Batchnorm forward training kernel.
// Compiled via HipRTC with -DINPUT_TYPE=<type> -DOUTPUT_TYPE=<type> -DSCALE_BIAS_TYPE=<type>
// -DMEAN_VAR_TYPE=<type> -DCOMPUTE_TYPE=<type> -DLOCAL_SIZE=<value> -DIS_CHANNEL_LAST_LAYOUT=<0|1>.
// Each thread block computes the mean and variance for one channel, reducing over the N*H*W
// elements in parallel across the threads.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void BatchnormFwdTrainRef(BatchnormFwdTrainArgs args)
{
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* scale = static_cast<const SCALE_BIAS_TYPE*>(args.scale);
    auto* bias = static_cast<const SCALE_BIAS_TYPE*>(args.bias);
    auto* output = static_cast<OUTPUT_TYPE*>(args.output);
    constexpr long long localSize = static_cast<long long>(LOCAL_SIZE);
    constexpr bool isChannelLastLayout = static_cast<bool>(IS_CHANNEL_LAST_LAYOUT);
    const auto chw = args.c * args.hw;
    const auto nhw = args.n * args.hw;

    COMPUTE_TYPE pvscale;
    COMPUTE_TYPE pvbias;
    COMPUTE_TYPE mean;
    COMPUTE_TYPE variance;
    COMPUTE_TYPE invVariance;

    __shared__ COMPUTE_TYPE lcl_bias;
    __shared__ COMPUTE_TYPE lcl_scale;
    __shared__ COMPUTE_TYPE lcl_reduce_sum[localSize];
    __shared__ COMPUTE_TYPE lcl_reduce_sqsum[localSize];

    long long index = 0;
    const long long lid = threadIdx.x;
    const long long grpid = blockIdx.x;

    if(lid == 0)
    {
        lcl_scale = toAccum(scale[grpid]);
        lcl_bias = toAccum(bias[grpid]);
    }

    __syncthreads();

    // Accumulate sum(x) and sum(x*x) over the N*H*W elements
    COMPUTE_TYPE local_sum = static_cast<COMPUTE_TYPE>(0);
    COMPUTE_TYPE local_sqsum = static_cast<COMPUTE_TYPE>(0);

    for(long long i = lid; i < nhw; i += localSize)
    {
        const long long nidx = i / args.hw;
        const long long hwidx = i - (nidx * args.hw);

        if constexpr(isChannelLastLayout)
        {
            index = nidx * chw + hwidx * args.c + grpid;
        }
        else
        {
            index = nidx * chw + grpid * args.hw + hwidx;
        }

        const COMPUTE_TYPE xval = toAccum(input[index]);
        local_sum += xval;
        local_sqsum += xval * xval;
    }

    lcl_reduce_sum[lid] = local_sum;
    lcl_reduce_sqsum[lid] = local_sqsum;
    __syncthreads();

    // Block reduction to compute the total sum and sum of squares for the channel
    for(long long s = localSize >> 1; s > 0; s >>= 1)
    {
        if(lid < s)
        {
            lcl_reduce_sum[lid] += lcl_reduce_sum[lid + s];
            lcl_reduce_sqsum[lid] += lcl_reduce_sqsum[lid + s];
        }
        __syncthreads();
    }

    const COMPUTE_TYPE invNhw = static_cast<COMPUTE_TYPE>(1.0) / static_cast<COMPUTE_TYPE>(nhw);

    mean = lcl_reduce_sum[0] * invNhw;
    variance = lcl_reduce_sqsum[0] * invNhw - mean * mean;
    if(variance < static_cast<COMPUTE_TYPE>(0))
    {
        variance = static_cast<COMPUTE_TYPE>(0);
    }

    invVariance = rsqrt(variance + toAccum(args.epsilon));
    pvscale = lcl_scale;
    pvbias = lcl_bias;
    __syncthreads();

    // Normalize each element, apply scale and bias, and write to output
    for(long long i = lid; i < nhw; i += localSize)
    {
        const long long nidx = i / args.hw;
        const long long hwidx = i - (nidx * args.hw);

        if constexpr(isChannelLastLayout)
        {
            index = nidx * chw + hwidx * args.c + grpid;
        }
        else
        {
            index = nidx * chw + grpid * args.hw + hwidx;
        }

        const COMPUTE_TYPE xval = toAccum(input[index]);
        COMPUTE_TYPE yval = (xval - mean) * invVariance;
        yval = pvscale * yval + pvbias;
        output[index] = fromAccum(yval);
    }

    if(lid == 0)
    {
        // Write save mean and save inverse variance if requested
        if(args.mean != nullptr && args.invVariance != nullptr)
        {
            auto* saveMean = static_cast<MEAN_VAR_TYPE*>(args.mean);
            saveMean[grpid] = fromAccum(mean);

            auto* saveInvVar = static_cast<MEAN_VAR_TYPE*>(args.invVariance);
            saveInvVar[grpid] = fromAccum(invVariance);
        }

        // Update running mean and variance if requested
        if(args.prevResultRunningMean != nullptr && args.prevResultRunningVariance != nullptr
           && args.nextResultRunningMean != nullptr && args.nextResultRunningVariance != nullptr)
        {
            auto* prevRunningMean = static_cast<const MEAN_VAR_TYPE*>(args.prevResultRunningMean);
            auto* prevRunningVariance
                = static_cast<const MEAN_VAR_TYPE*>(args.prevResultRunningVariance);
            auto* nextRunningMean = static_cast<MEAN_VAR_TYPE*>(args.nextResultRunningMean);
            auto* nextRunningVariance = static_cast<MEAN_VAR_TYPE*>(args.nextResultRunningVariance);

            const COMPUTE_TYPE prevRunMean = toAccum(prevRunningMean[grpid]);
            const COMPUTE_TYPE prevRunVar = toAccum(prevRunningVariance[grpid]);
            const COMPUTE_TYPE expAvgFactor = toAccum(args.momentum);

            const COMPUTE_TYPE nextRunMean
                = mean * expAvgFactor - expAvgFactor * prevRunMean + prevRunMean;

            // Bessel's correction for unbiased variance estimate
            const COMPUTE_TYPE adjustedVariance
                = (nhw == 1)
                      ? variance
                      : variance
                            * (static_cast<COMPUTE_TYPE>(nhw) / static_cast<COMPUTE_TYPE>(nhw - 1));

            const COMPUTE_TYPE nextRunVar
                = (static_cast<COMPUTE_TYPE>(1.0) - expAvgFactor) * prevRunVar
                  + expAvgFactor * adjustedVariance;

            nextRunningMean[grpid] = fromAccum(nextRunMean);
            nextRunningVariance[grpid] = fromAccum(nextRunVar);
        }
    }
}
