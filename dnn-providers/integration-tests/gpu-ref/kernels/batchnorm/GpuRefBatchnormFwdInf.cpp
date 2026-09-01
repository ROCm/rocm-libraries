// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU reference Batchnorm forward inference kernel.
// Compiled via HipRTC with -DINPUT_TYPE=<type> -DOUTPUT_TYPE=<type> -DSCALE_BIAS_TYPE=<type>
// -DMEAN_VAR_TYPE=<type> -DCOMPUTE_TYPE=<type> -DLOCAL_SIZE_X=<value> -DLOCAL_SIZE_Y=<value>

// 3D execution grid where the X dimension iterates along the tensor channel axis and
// Y dimension iterates along the tensor spatial axis. The Z local size is always 1 but
// the size of the Z grid dimension is min(batchSize, maxGridSizeToFillTheGPU), so
// each thread will loop over the remaining batches with stride of gridDim.z if necessary.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void BatchnormFwdInfRef(BatchnormFwdArgs args)
{
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* scale = static_cast<const SCALE_BIAS_TYPE*>(args.scale);
    auto* bias = static_cast<const SCALE_BIAS_TYPE*>(args.bias);
    auto* estMean = static_cast<const MEAN_VAR_TYPE*>(args.estMean);
    auto* invVar = static_cast<const MEAN_VAR_TYPE*>(args.invVar);
    auto* output = static_cast<OUTPUT_TYPE*>(args.output);

    const long long tidx = blockIdx.x * LOCAL_SIZE_X + threadIdx.x;
    const long long tidy = blockIdx.y * LOCAL_SIZE_Y + threadIdx.y;
    const long long tidz = blockIdx.z;

    const long long c = static_cast<long long>(args.c);
    const long long hw = static_cast<long long>(args.hw);
    const long long batchSize = static_cast<long long>(args.batchSize);
    const long long cStride = static_cast<long long>(args.cStride);
    const long long hwStride = static_cast<long long>(args.hwStride);
    const long long batchStride = static_cast<long long>(args.batchStride);

    // skip execution for out-of-bound threads
    if(tidx >= c || tidy >= hw || tidz >= batchSize)
    {
        return;
    }

    auto compMean = static_cast<COMPUTE_TYPE>(estMean[tidx]);
    auto compVar = static_cast<COMPUTE_TYPE>(invVar[tidx]);
    auto compScale = static_cast<COMPUTE_TYPE>(scale[tidx]);
    auto compBias = static_cast<COMPUTE_TYPE>(bias[tidx]);

    for(long long n = blockIdx.z; n < batchSize; n += gridDim.z)
    {
        const long long batchIndex = (n * batchStride) + (tidx * cStride) + (tidy * hwStride);
        COMPUTE_TYPE value = static_cast<COMPUTE_TYPE>(input[batchIndex]);
        COMPUTE_TYPE inhat = (value - compMean) * compVar;
        inhat = compScale * inhat + compBias;
        output[batchIndex] = static_cast<OUTPUT_TYPE>(inhat);
    }
}
