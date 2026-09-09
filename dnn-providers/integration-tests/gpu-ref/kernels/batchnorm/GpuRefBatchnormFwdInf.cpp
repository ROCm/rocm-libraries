// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference Batchnorm forward inference kernels.
// Compiled via HipRTC with -DINPUT_TYPE=<type> -DOUTPUT_TYPE=<type> -DSCALE_BIAS_TYPE=<type>
// -DMEAN_VAR_TYPE=<type> -DCOMPUTE_TYPE=<type> -DLOCAL_SIZE_X=<value> -DLOCAL_SIZE_Y=<value>

// Provides two entry points:
//  * BatchnormFwdInfRef        - supplied a pre-computed inverse variance.
//  * BatchnormFwdInfWithVarRef - supplied the raw variance and an epsilon, computing the
//                                inverse variance as 1/sqrt(variance + epsilon).

// 3D execution grid where the X dimension iterates along the tensor channel axis and
// Y dimension iterates along the tensor spatial axis. The Z local size is always 1 but
// the size of the Z grid dimension is min(batchSize, maxGridSizeToFillTheGPU), so
// each thread will loop over the remaining batches with stride of gridDim.z if necessary.

#include "GpuRefTypes.h"

using namespace gpu_ref;

__device__ __forceinline__ void batchnormFwdInfImpl(long long tidx,
                                                    long long tidy,
                                                    long long tidz,
                                                    const BatchnormFwdInfCommonArgs& args,
                                                    COMPUTE_TYPE compInvVar)
{
    const long long& batchSize = args.batchSize;
    const long long& cStride = args.cStride;
    const long long& hwStride = args.hwStride;
    const long long& batchStride = args.batchStride;

    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* scale = static_cast<const SCALE_BIAS_TYPE*>(args.scale);
    auto* bias = static_cast<const SCALE_BIAS_TYPE*>(args.bias);
    auto* estMean = static_cast<const MEAN_VAR_TYPE*>(args.estMean);
    auto* output = static_cast<OUTPUT_TYPE*>(args.output);

    auto compMean = static_cast<COMPUTE_TYPE>(estMean[tidx]);
    auto compScale = static_cast<COMPUTE_TYPE>(scale[tidx]);
    auto compBias = static_cast<COMPUTE_TYPE>(bias[tidx]);

    for(long long n = blockIdx.z; n < batchSize; n += gridDim.z)
    {
        const long long batchIndex = (n * batchStride) + (tidx * cStride) + (tidy * hwStride);
        COMPUTE_TYPE value = static_cast<COMPUTE_TYPE>(input[batchIndex]);
        COMPUTE_TYPE inhat = (value - compMean) * compInvVar;
        inhat = compScale * inhat + compBias;
        output[batchIndex] = static_cast<OUTPUT_TYPE>(inhat);
    }
}

extern "C" __global__ void BatchnormFwdInfRef(BatchnormFwdInfArgs args)
{
    const long long tidx = blockIdx.x * LOCAL_SIZE_X + threadIdx.x;
    const long long tidy = blockIdx.y * LOCAL_SIZE_Y + threadIdx.y;
    const long long tidz = blockIdx.z;

    // skip execution for out-of-bound threads
    if(tidx >= args.common.c || tidy >= args.common.hw || tidz >= args.common.batchSize)
    {
        return;
    }

    auto* invVar = static_cast<const MEAN_VAR_TYPE*>(args.invVar);
    auto compInvVar = static_cast<COMPUTE_TYPE>(invVar[tidx]);
    batchnormFwdInfImpl(tidx, tidy, tidz, args.common, compInvVar);
}

namespace gpu_ref::detail
{
__forceinline__ __device__ double rsqrt(double x)
{
    return ::rsqrt(x);
}
__forceinline__ __device__ float rsqrt(float x)
{
    return rsqrtf(x);
}
__forceinline__ __device__ _Float16 rsqrt(_Float16 x)
{
    return __ocml_rsqrt_f16(x);
}
__forceinline__ __device__ __bf16 rsqrt(__bf16 x)
{
    return static_cast<__bf16>(rsqrtf(static_cast<float>(x)));
}
} // namespace gpu_ref::detail

extern "C" __global__ void BatchnormFwdInfWithVarRef(BatchnormFwdInfWithVarArgs args)
{
    const long long tidx = blockIdx.x * LOCAL_SIZE_X + threadIdx.x;
    const long long tidy = blockIdx.y * LOCAL_SIZE_Y + threadIdx.y;
    const long long tidz = blockIdx.z;

    // skip execution for out-of-bound threads
    if(tidx >= args.common.c || tidy >= args.common.hw || tidz >= args.common.batchSize)
    {
        return;
    }

    // Compute inverse variance = 1 / sqrt(variance + epsilon)
    auto compEpsilon = static_cast<COMPUTE_TYPE>(args.epsilon);
    auto* estVar = static_cast<const MEAN_VAR_TYPE*>(args.estVar);
    auto compVar = static_cast<COMPUTE_TYPE>(estVar[tidx]);
    auto compInvVar = gpu_ref::detail::rsqrt(compVar + compEpsilon);
    batchnormFwdInfImpl(tidx, tidy, tidz, args.common, compInvVar);
}
