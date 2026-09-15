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
    const COMPUTE_TYPE invNhw = static_cast<COMPUTE_TYPE>(1.0) / static_cast<COMPUTE_TYPE>(nhw);

    COMPUTE_TYPE pvScale;
    COMPUTE_TYPE pvBias;
    COMPUTE_TYPE mean;
    COMPUTE_TYPE variance;
    COMPUTE_TYPE invVariance;

    __shared__ COMPUTE_TYPE lclBias;
    __shared__ COMPUTE_TYPE lclScale;
    __shared__ COMPUTE_TYPE lclReduceSum[localSize];
    __shared__ COMPUTE_TYPE lclReduceSqSum[localSize];

    long long index = 0;
    const long long lid = threadIdx.x;
    const long long grpid = blockIdx.x;

    // Load scale and bias for the channel into shared memory
    if(lid == 0)
    {
        lclScale = toAccum(scale[grpid]);
        lclBias = toAccum(bias[grpid]);
    }
    __syncthreads();

    // Pass 1: Accumulate sum(x) over the N*H*W elements
    COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
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

        const COMPUTE_TYPE xVal = toAccum(input[index]);
        sum += xVal;
    }
    lclReduceSum[lid] = sum;
    __syncthreads();

    // Pass 1: Block reduce sum(x) and compute mean for the channel
    for(long long s = localSize >> 1; s > 0; s >>= 1)
    {
        if(lid < s)
        {
            lclReduceSum[lid] += lclReduceSum[lid + s];
        }
        __syncthreads();
    }
    mean = lclReduceSum[0] * invNhw;

    // Pass 2: Accumulate sum((x - mean)^2) over the N*H*W elements
    COMPUTE_TYPE sqSum = static_cast<COMPUTE_TYPE>(0);
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

        const COMPUTE_TYPE xVal = toAccum(input[index]);
        sqSum += (xVal - mean) * (xVal - mean);
    }
    lclReduceSqSum[lid] = sqSum;
    __syncthreads();

    // Pass 2: Block reduce sum((x - mean)^2) and compute variance for the channel
    for(long long s = localSize >> 1; s > 0; s >>= 1)
    {
        if(lid < s)
        {
            lclReduceSqSum[lid] += lclReduceSqSum[lid + s];
        }
        __syncthreads();
    }
    variance = lclReduceSqSum[0] * invNhw;
    if(variance < static_cast<COMPUTE_TYPE>(0))
    {
        variance = static_cast<COMPUTE_TYPE>(0);
    }

    // Compute inverse variance and load scale and bias into registers
    invVariance = rsqrt(variance + toAccum(args.epsilon));
    pvScale = lclScale;
    pvBias = lclBias;
    __syncthreads();

    // Normalize each element, apply scale and bias and write to output
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

        const COMPUTE_TYPE xVal = toAccum(input[index]);
        COMPUTE_TYPE yVal = (xVal - mean) * invVariance;
        yVal = pvScale * yVal + pvBias;
        OUTPUT_TYPE* tag = nullptr;
        output[index] = fromAccum(yVal, tag);
    }

    if(lid == 0)
    {
        // Write save mean and save inverse variance if requested
        if(args.mean != nullptr && args.invVariance != nullptr)
        {
            MEAN_VAR_TYPE* tag = nullptr;

            auto* saveMean = static_cast<MEAN_VAR_TYPE*>(args.mean);
            saveMean[grpid] = fromAccum(mean, tag);

            auto* saveInvVar = static_cast<MEAN_VAR_TYPE*>(args.invVariance);
            saveInvVar[grpid] = fromAccum(invVariance, tag);
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

            MEAN_VAR_TYPE* tag = nullptr;

            nextRunningMean[grpid] = fromAccum(nextRunMean, tag);
            nextRunningVariance[grpid] = fromAccum(nextRunVar, tag);
        }
    }
}
