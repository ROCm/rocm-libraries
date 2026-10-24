// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// fp8 layers around the SDPA forward reference (the device side of CpuFpReferenceSdpa::forward's
// fp8 wrappers). One thread per element. Compiled via HipRTC with -DCOMPUTE_TYPE=float and
// -DIN_TYPE=<type> for sdpaDequantRef or -DO_TYPE=<type> for sdpaScaleOutRef.
//   sdpaDequantRef:  out[b, h, s, d] = float(in[b, h, s, d]) * descale(b, h / headsPerKv), packed.
//   sdpaScaleOutRef: out[b, h, s, d] = O_TYPE(in * scale) from a packed float O; amax gets
//                    max |in| (cuDNN amax_o).

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"

using namespace gpu_ref;

#ifdef IN_TYPE
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
#endif

#ifdef O_TYPE
extern "C" __global__ void sdpaScaleOutRef(SdpaScaleOutArgs args)
{
    auto* out = static_cast<O_TYPE*>(args.out);
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

    const float x = args.in[idx];
    if(args.amax != nullptr)
    {
        // |x| >= 0, so its float bits order like unsigned integers.
        atomicMax(reinterpret_cast<unsigned int*>(args.amax),
                  __builtin_bit_cast(unsigned int, __builtin_fabsf(x)));
    }
    O_TYPE* tag = nullptr;
    out[b * args.outStr.s[0] + h * args.outStr.s[1] + s * args.outStr.s[2] + d * args.outStr.s[3]]
        = fromAccum(static_cast<COMPUTE_TYPE>(x * args.scale), tag);
}
#endif
