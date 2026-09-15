// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference convolution forward kernels.
// Compiled via HipRTC with -DX_TYPE=<type> -DW_TYPE=<type> -DY_TYPE=<type> -DCOMPUTE_TYPE=<type>.
// One thread per output element. Uses stride-based indexing to handle any layout.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void convFwdRef1d(ConvFwdArgs1d args)
{
    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* w = static_cast<const W_TYPE*>(args.w);
    auto* y = static_cast<Y_TYPE*>(args.y);

    long long totalOutputElements = args.N * args.K * args.Wo;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Decompose linear index into (n, k, wo)
    long long wo = idx % args.Wo;
    long long tmp = idx / args.Wo;
    long long k = tmp % args.K;
    long long n = tmp / args.K;

    // Group parameters
    long long cPerGroup = args.C / args.groups;
    long long g = k * args.groups / args.K;
    long long baseInputChannel = g * cPerGroup;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);

    for(long long c = 0; c < cPerGroup; ++c)
    {
        long long xChannel = baseInputChannel + c;

        for(long long kw = 0; kw < args.Kw; ++kw)
        {
            long long wi = wo * args.strideW + kw * args.dilW - args.padW;
            if(wi < 0 || wi >= args.Wi)
            {
                continue;
            }

            long long xIdx = n * args.xStr.s[0] + xChannel * args.xStr.s[1] + wi * args.xStr.s[2];
            long long wIdx = k * args.wStr.s[0] + c * args.wStr.s[1] + kw * args.wStr.s[2];

#ifdef USE_TF32
            float xf = truncateToTf32(static_cast<float>(toAccum(x[xIdx])));
            float wf = truncateToTf32(static_cast<float>(toAccum(w[wIdx])));
            acc += static_cast<COMPUTE_TYPE>(xf) * static_cast<COMPUTE_TYPE>(wf);
#else
            acc += toAccum(x[xIdx]) * toAccum(w[wIdx]);
#endif
        }
    }

    long long yIdx = n * args.yStr.s[0] + k * args.yStr.s[1] + wo * args.yStr.s[2];
    Y_TYPE* tag = nullptr;

    if(args.beta == 0.0)
    {
        y[yIdx] = fromAccum(args.alpha * acc, tag);
    }
    else
    {
        y[yIdx] = fromAccum(args.alpha * acc + args.beta * toAccum(y[yIdx]), tag);
    }
}

extern "C" __global__ void convFwdRef2d(ConvFwdArgs2d args)
{
    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* w = static_cast<const W_TYPE*>(args.w);
    auto* y = static_cast<Y_TYPE*>(args.y);

    long long totalOutputElements = args.N * args.K * args.Ho * args.Wo;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Decompose linear index into (n, k, ho, wo)
    long long wo = idx % args.Wo;
    long long tmp = idx / args.Wo;
    long long ho = tmp % args.Ho;
    tmp = tmp / args.Ho;
    long long k = tmp % args.K;
    long long n = tmp / args.K;

    // Group parameters
    long long cPerGroup = args.C / args.groups;
    long long g = k * args.groups / args.K;
    long long baseInputChannel = g * cPerGroup;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);

    for(long long c = 0; c < cPerGroup; ++c)
    {
        long long xChannel = baseInputChannel + c;

        for(long long kh = 0; kh < args.Kh; ++kh)
        {
            long long hi = ho * args.strideH + kh * args.dilH - args.padH;
            if(hi < 0 || hi >= args.Hi)
            {
                continue;
            }

            for(long long kw = 0; kw < args.Kw; ++kw)
            {
                long long wi = wo * args.strideW + kw * args.dilW - args.padW;
                if(wi < 0 || wi >= args.Wi)
                {
                    continue;
                }

                long long xIdx = n * args.xStr.s[0] + xChannel * args.xStr.s[1]
                                 + hi * args.xStr.s[2] + wi * args.xStr.s[3];
                long long wIdx = k * args.wStr.s[0] + c * args.wStr.s[1] + kh * args.wStr.s[2]
                                 + kw * args.wStr.s[3];

#ifdef USE_TF32
                float xf = truncateToTf32(static_cast<float>(toAccum(x[xIdx])));
                float wf = truncateToTf32(static_cast<float>(toAccum(w[wIdx])));
                acc += static_cast<COMPUTE_TYPE>(xf) * static_cast<COMPUTE_TYPE>(wf);
#else
                acc += toAccum(x[xIdx]) * toAccum(w[wIdx]);
#endif
            }
        }
    }

    long long yIdx
        = n * args.yStr.s[0] + k * args.yStr.s[1] + ho * args.yStr.s[2] + wo * args.yStr.s[3];
    Y_TYPE* tag = nullptr;

    if(args.beta == 0.0)
    {
        y[yIdx] = fromAccum(args.alpha * acc, tag);
    }
    else
    {
        y[yIdx] = fromAccum(args.alpha * acc + args.beta * toAccum(y[yIdx]), tag);
    }
}

extern "C" __global__ void convFwdRef3d(ConvFwdArgs3d args)
{
    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* w = static_cast<const W_TYPE*>(args.w);
    auto* y = static_cast<Y_TYPE*>(args.y);

    long long totalOutputElements = args.N * args.K * args.Do * args.Ho * args.Wo;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Decompose linear index into (n, k, do_, ho, wo)
    long long wo = idx % args.Wo;
    long long tmp = idx / args.Wo;
    long long ho = tmp % args.Ho;
    tmp = tmp / args.Ho;
    long long do_ = tmp % args.Do;
    tmp = tmp / args.Do;
    long long k = tmp % args.K;
    long long n = tmp / args.K;

    // Group parameters
    long long cPerGroup = args.C / args.groups;
    long long g = k * args.groups / args.K;
    long long baseInputChannel = g * cPerGroup;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);

    for(long long c = 0; c < cPerGroup; ++c)
    {
        long long xChannel = baseInputChannel + c;

        for(long long kd = 0; kd < args.Kd; ++kd)
        {
            long long di = do_ * args.strideD + kd * args.dilD - args.padD;
            if(di < 0 || di >= args.Di)
            {
                continue;
            }

            for(long long kh = 0; kh < args.Kh; ++kh)
            {
                long long hi = ho * args.strideH + kh * args.dilH - args.padH;
                if(hi < 0 || hi >= args.Hi)
                {
                    continue;
                }

                for(long long kw = 0; kw < args.Kw; ++kw)
                {
                    long long wi = wo * args.strideW + kw * args.dilW - args.padW;
                    if(wi < 0 || wi >= args.Wi)
                    {
                        continue;
                    }

                    long long xIdx = n * args.xStr.s[0] + xChannel * args.xStr.s[1]
                                     + di * args.xStr.s[2] + hi * args.xStr.s[3]
                                     + wi * args.xStr.s[4];
                    long long wIdx = k * args.wStr.s[0] + c * args.wStr.s[1] + kd * args.wStr.s[2]
                                     + kh * args.wStr.s[3] + kw * args.wStr.s[4];

#ifdef USE_TF32
                    float xf = truncateToTf32(static_cast<float>(toAccum(x[xIdx])));
                    float wf = truncateToTf32(static_cast<float>(toAccum(w[wIdx])));
                    acc += static_cast<COMPUTE_TYPE>(xf) * static_cast<COMPUTE_TYPE>(wf);
#else
                    acc += toAccum(x[xIdx]) * toAccum(w[wIdx]);
#endif
                }
            }
        }
    }

    long long yIdx = n * args.yStr.s[0] + k * args.yStr.s[1] + do_ * args.yStr.s[2]
                     + ho * args.yStr.s[3] + wo * args.yStr.s[4];
    Y_TYPE* tag = nullptr;

    if(args.beta == 0.0)
    {
        y[yIdx] = fromAccum(args.alpha * acc, tag);
    }
    else
    {
        y[yIdx] = fromAccum(args.alpha * acc + args.beta * toAccum(y[yIdx]), tag);
    }
}
