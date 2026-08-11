// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference RMSNorm backward kernel.
// Compiled via HipRTC with -DGRAD_OUTPUT_TYPE=<type> -DINPUT_TYPE=<type> -DSCALE_TYPE=<type>
// -DGRAD_INPUT_TYPE=<type> -DCOMPUTE_TYPE=<type> -DLOCAL_SIZE=<value>.
// In the RMSNormBwdData kernel, each thread block computes one normalization group
// reducing over the innerSize elements in parallel across the threads to compute the
// mean of dy * w * x and then computing the gradInput elements for the group using
// the mean and invRms.
// In the RMSNormBwdWeightBias kernel, each thread computes one element of the scale and bias
// gradients, accumulating contributions across all outer groups and strides for that element.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void RMSNormBwdDataRef(RMSNormBwdArgs args)
{
    auto* gradOutput = static_cast<const GRAD_OUTPUT_TYPE*>(args.gradOutput);
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* scale = static_cast<const SCALE_TYPE*>(args.scale);
    auto* rstd = static_cast<const COMPUTE_TYPE*>(args.invRms);
    auto* gradInput = static_cast<GRAD_INPUT_TYPE*>(args.gradInput);

    constexpr long long localSize = static_cast<long long>(LOCAL_SIZE);

    // Each block handles one normalization group
    const long long gid = static_cast<long long>(blockIdx.x);
    const long long lid = static_cast<long long>(threadIdx.x);

    const long long innerSize = args.innerSize;
    const long long stride = args.stride;

    // Compute the outer and stride indices for this group
    const long long o = gid / stride;
    const long long s = gid % stride;

    __shared__ COMPUTE_TYPE ltmp[localSize];
    COMPUTE_TYPE mean = static_cast<COMPUTE_TYPE>(0);

    // Each thread accumulates a partial sum of dy * w * x over the innerSize elements
    for(long long i = lid; i < innerSize; i += localSize)
    {
        long long idx = o * innerSize * stride + i * stride + s;

        COMPUTE_TYPE pdy = toAccum(gradOutput[idx]);
        COMPUTE_TYPE px = toAccum(input[idx]);
        COMPUTE_TYPE pw = toAccum(scale[i]);

        mean += pdy * pw * px;
    }

    // Block reduction to compute the group mean of dy * w * x
    ltmp[lid] = mean;
    __syncthreads();
    for(long long i = localSize >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }

    mean = ltmp[0] / static_cast<COMPUTE_TYPE>(innerSize);
    COMPUTE_TYPE prstd = rstd[gid];

    // Compute gradInput for each element using the group's invRms and mean
    GRAD_INPUT_TYPE* tag = nullptr;
    for(long long i = lid; i < innerSize; i += localSize)
    {
        long long idx = o * innerSize * stride + i * stride + s;

        COMPUTE_TYPE pdy = toAccum(gradOutput[idx]);
        COMPUTE_TYPE px = toAccum(input[idx]);
        COMPUTE_TYPE pw = toAccum(scale[i]);

        COMPUTE_TYPE dxVal = (pdy * pw * prstd) - (mean * px * prstd * prstd * prstd);
        gradInput[idx] = fromAccum(dxVal, tag);
    }
}

extern "C" __global__ void RMSNormBwdWeightBiasRef(RMSNormBwdArgs args)
{
    auto* gradOutput = static_cast<const GRAD_OUTPUT_TYPE*>(args.gradOutput);
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* rstd = static_cast<const COMPUTE_TYPE*>(args.invRms);
    auto* gradScale = static_cast<SCALE_TYPE*>(args.gradScale);
    auto* gradBias = static_cast<SCALE_TYPE*>(args.gradBias);

    constexpr long long localSize = static_cast<long long>(LOCAL_SIZE);

    // Each thread handles one element of the scale and bias gradients,
    // accumulating contributions across all outer groups and strides.
    const long long tidx
        = static_cast<long long>(threadIdx.x) + static_cast<long long>(blockIdx.x) * localSize;

    const long long innerSize = args.innerSize;
    const long long outerSize = args.outerSize;
    const long long stride = args.stride;

    if(tidx >= innerSize)
    {
        return;
    }

    COMPUTE_TYPE sumDw = static_cast<COMPUTE_TYPE>(0);
    COMPUTE_TYPE sumDb = static_cast<COMPUTE_TYPE>(0);

    // Accumulate gradScale and gradBias contributions across all outer groups and strides.
    for(long long o = 0; o < outerSize; ++o)
    {
        for(long long s = 0; s < stride; ++s)
        {
            long long idx = o * innerSize * stride + tidx * stride + s;

            COMPUTE_TYPE prstd = rstd[o * stride + s];
            COMPUTE_TYPE pdy = toAccum(gradOutput[idx]);
            COMPUTE_TYPE px = toAccum(input[idx]);

            sumDw += pdy * px * prstd;
            sumDb += pdy;
        }
    }

    SCALE_TYPE* tag = nullptr;
    gradScale[tidx] = fromAccum(sumDw, tag);
    if(gradBias != nullptr)
    {
        gradBias[tidx] = fromAccum(sumDb, tag);
    }
}
