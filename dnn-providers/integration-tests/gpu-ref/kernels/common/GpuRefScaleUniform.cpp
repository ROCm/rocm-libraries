// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Kernel to scale uniform random values to a specified range.
// Compiled via HipRTC with -DSOURCE_TYPE=<type> -DTARGET_TYPE=<type>.
// For same-type in-place scaling: SOURCE_TYPE == TARGET_TYPE, src == dst.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void ScaleUniform(ScaleUniformArgs args)
{
    auto* src = static_cast<const SOURCE_TYPE*>(args.src);
    auto* dst = static_cast<TARGET_TYPE*>(args.dst);

    const long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                          + static_cast<long long>(threadIdx.x);

    if(idx >= args.count)
    {
        return;
    }

    const auto val = static_cast<double>(src[idx]);
    dst[idx] = safeConvert<TARGET_TYPE>(args.minValue + val * (args.maxValue - args.minValue));
}
