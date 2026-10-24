// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// fp8 dequantize for the SDPA forward reference (CpuFpReferenceSdpa::dequantize on the device):
// out[b, h, s, d] = float(in[b, h, s, d]) * descale(b, h / headsPerKv), out packed.
// Compiled via HipRTC with -DIN_TYPE=<type> -DCOMPUTE_TYPE=float. One thread per element.

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void sdpaDequantRef(SdpaDequantArgs args)
{
    const auto* in = static_cast<const IN_TYPE*>(args.in);
    const long long total = args.dims[0] * args.dims[1] * args.dims[2] * args.dims[3];
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= total)
    {
        return;
    }
    long long tmp = idx;
    const long long d = tmp % args.dims[3];
    tmp /= args.dims[3];
    const long long s = tmp % args.dims[2];
    tmp /= args.dims[2];
    const long long h = tmp % args.dims[1];
    const long long b = tmp / args.dims[1];

    const float factor = args.descale == nullptr
                             ? 1.0f
                             : args.descale[b * args.descaleBatchStride
                                            + (h / args.headsPerKv) * args.descaleHeadStride];
    const float x = toAccum(
        in[b * args.inStr.s[0] + h * args.inStr.s[1] + s * args.inStr.s[2] + d * args.inStr.s[3]]);
    args.out[idx] = x * factor;
}
