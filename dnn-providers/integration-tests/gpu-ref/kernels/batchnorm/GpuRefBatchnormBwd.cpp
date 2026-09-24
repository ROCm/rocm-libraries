// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference Batchnorm backward kernel.
// Compiled via HipRTC with all tensor and compute types supplied as preprocessor defines.
// Each block handles one channel and reduces over its N*spatial elements.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void BatchnormBwdRef(BatchnormBwdArgs args)
{
    auto* dy = static_cast<const GRAD_OUTPUT_TYPE*>(args.dy);
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* scale = static_cast<const SCALE_BIAS_TYPE*>(args.scale);
    auto* dx = static_cast<GRAD_INPUT_TYPE*>(args.dx);
    auto* dscale = static_cast<SCALE_BIAS_TYPE*>(args.dscale);
    auto* dbias = static_cast<SCALE_BIAS_TYPE*>(args.dbias);
    auto* savedMean = static_cast<const MEAN_VAR_TYPE*>(args.mean);
    auto* savedInvVariance = static_cast<const MEAN_VAR_TYPE*>(args.invVariance);

    constexpr long long localSize = static_cast<long long>(LOCAL_SIZE);
    constexpr bool isChannelLastLayout = static_cast<bool>(IS_CHANNEL_LAST_LAYOUT);
    const long long channel = static_cast<long long>(blockIdx.x);
    const long long lid = static_cast<long long>(threadIdx.x);
    const long long nhw = args.n * args.hw;
    const long long chw = args.c * args.hw;
    const COMPUTE_TYPE invNhw = static_cast<COMPUTE_TYPE>(1.0) / static_cast<COMPUTE_TYPE>(nhw);

    __shared__ COMPUTE_TYPE reduceA[localSize];
    __shared__ COMPUTE_TYPE reduceB[localSize];

    COMPUTE_TYPE channelMean;
    COMPUTE_TYPE channelInvVariance;
    if(savedMean == nullptr)
    {
        COMPUTE_TYPE sum = static_cast<COMPUTE_TYPE>(0);
        COMPUTE_TYPE squareSum = static_cast<COMPUTE_TYPE>(0);
        for(long long i = lid; i < nhw; i += localSize)
        {
            const long long nidx = i / args.hw;
            const long long hwidx = i - nidx * args.hw;
            const long long index = isChannelLastLayout ? nidx * chw + hwidx * args.c + channel
                                                        : nidx * chw + channel * args.hw + hwidx;
            const COMPUTE_TYPE xValue = toAccum(input[index]);
            sum = sum + xValue;
            squareSum = squareSum + xValue * xValue;
        }

        reduceA[lid] = sum;
        reduceB[lid] = squareSum;
        __syncthreads();
        for(long long offset = localSize >> 1; offset > 0; offset >>= 1)
        {
            if(lid < offset)
            {
                reduceA[lid] = reduceA[lid] + reduceA[lid + offset];
                reduceB[lid] = reduceB[lid] + reduceB[lid + offset];
            }
            __syncthreads();
        }

        channelMean = reduceA[0] * invNhw;
        COMPUTE_TYPE variance = reduceB[0] * invNhw - channelMean * channelMean;
        if(variance < static_cast<COMPUTE_TYPE>(0))
        {
            variance = static_cast<COMPUTE_TYPE>(0);
        }
        channelInvVariance = rsqrt(variance + toAccum(args.epsilon));
    }
    else
    {
        channelMean = toAccum(savedMean[channel]);
        channelInvVariance = toAccum(savedInvVariance[channel]);
    }

    COMPUTE_TYPE dotProduct = static_cast<COMPUTE_TYPE>(0);
    COMPUTE_TYPE sumDy = static_cast<COMPUTE_TYPE>(0);
    for(long long i = lid; i < nhw; i += localSize)
    {
        const long long nidx = i / args.hw;
        const long long hwidx = i - nidx * args.hw;
        const long long index = isChannelLastLayout ? nidx * chw + hwidx * args.c + channel
                                                    : nidx * chw + channel * args.hw + hwidx;
        const COMPUTE_TYPE xValue = toAccum(input[index]);
        const COMPUTE_TYPE dyValue = toAccum(dy[index]);
        const COMPUTE_TYPE xHat = (xValue - channelMean) * channelInvVariance;
        dotProduct = dotProduct + xHat * dyValue;
        sumDy = sumDy + dyValue;
    }

    reduceA[lid] = dotProduct;
    reduceB[lid] = sumDy;
    __syncthreads();
    for(long long offset = localSize >> 1; offset > 0; offset >>= 1)
    {
        if(lid < offset)
        {
            reduceA[lid] = reduceA[lid] + reduceA[lid + offset];
            reduceB[lid] = reduceB[lid] + reduceB[lid + offset];
        }
        __syncthreads();
    }

    const COMPUTE_TYPE channelScale = toAccum(scale[channel]);
    const COMPUTE_TYPE meanDyXhat = reduceA[0] * invNhw;
    const COMPUTE_TYPE meanDy = reduceB[0] * invNhw;
    const COMPUTE_TYPE scalarCoefficient = channelScale * channelInvVariance;

    if(lid == 0)
    {
        SCALE_BIAS_TYPE* tag = nullptr;
        dscale[channel] = fromAccum(reduceA[0], tag);
        dbias[channel] = fromAccum(reduceB[0], tag);
    }

    GRAD_INPUT_TYPE* tag = nullptr;
    for(long long i = lid; i < nhw; i += localSize)
    {
        const long long nidx = i / args.hw;
        const long long hwidx = i - nidx * args.hw;
        const long long index = isChannelLastLayout ? nidx * chw + hwidx * args.c + channel
                                                    : nidx * chw + channel * args.hw + hwidx;
        const COMPUTE_TYPE xValue = toAccum(input[index]);
        const COMPUTE_TYPE dyValue = toAccum(dy[index]);
        const COMPUTE_TYPE xHat = (xValue - channelMean) * channelInvVariance;
        const COMPUTE_TYPE dxValue = (dyValue - meanDy - xHat * meanDyXhat) * scalarCoefficient;
        dx[index] = fromAccum(dxValue, tag);
    }
}
