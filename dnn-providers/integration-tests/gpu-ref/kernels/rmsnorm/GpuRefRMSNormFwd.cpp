// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference RMSNorm forward kernel.
// Compiled via HipRTC with -DINPUT_TYPE=<type> -DOUTPUT_TYPE=<type> -DSCALE_TYPE=<type>
// -DCOMPUTE_TYPE=<type> -DLOCAL_SIZE=<value> -DHAS_BIAS=<0|1>.
// Each thread block computes one normalization group (outerSize) reducing over the
// innerSize elements in parallel across the threads to compute the RMS statistic
// and then normalizing the input elements in the group to produce the output elements.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void RMSNormFwdRef(RMSNormFwdArgs args)
{
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* scale = static_cast<const SCALE_TYPE*>(args.scale);
    auto* output = static_cast<OUTPUT_TYPE*>(args.output);
    auto* rstd = static_cast<COMPUTE_TYPE*>(args.invRms);
    [[maybe_unused]] auto* bias = static_cast<const SCALE_TYPE*>(args.bias);

    constexpr long long localSize = static_cast<long long>(LOCAL_SIZE);

    // Each block handles one normalization group
    const long long gid = static_cast<long long>(blockIdx.x);
    const long long lid = static_cast<long long>(threadIdx.x);

    const long long innerSize = static_cast<long long>(args.innerSize);
    const long long stride = static_cast<long long>(args.stride);

    // Compute the outer and stride indices for this group
    const long long o = gid / stride;
    const long long s = gid % stride;

    COMPUTE_TYPE pvar = static_cast<COMPUTE_TYPE>(0);
    __shared__ COMPUTE_TYPE ltmp[localSize];

    // Each thread accumulates a partial sum of squares over the innerSize elements
    for(long long i = lid; i < innerSize; i += localSize)
    {
        long long idx = o * innerSize * stride + i * stride + s;
        COMPUTE_TYPE tmp = toAccum(input[idx]);
        pvar += tmp * tmp;
    }

    // Block reduction to compute the total sum of squares for the group
    ltmp[lid] = pvar;
    __syncthreads();
    for(long long i = localSize >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }

    pvar = ltmp[0] / static_cast<COMPUTE_TYPE>(innerSize);
    COMPUTE_TYPE prstd = static_cast<COMPUTE_TYPE>(rsqrt(static_cast<double>(pvar) + args.eps));

    // Thread 0 in the block writes the inverse RMS statistic for the group, if requested
    if(lid == 0 && rstd)
    {
        rstd[gid] = prstd;
    }

    // Normalize each element using the group's invRms (prstd), then apply scale and optional bias
    OUTPUT_TYPE* tag = nullptr;
    for(long long i = lid; i < innerSize; i += localSize)
    {
        long long idx = o * innerSize * stride + i * stride + s;
        COMPUTE_TYPE y_val = toAccum(input[idx]) * prstd * toAccum(scale[i]);
        if constexpr(HAS_BIAS)
        {
            y_val += toAccum(bias[i]);
        }
        output[idx] = fromAccum(y_val, tag);
    }
}
