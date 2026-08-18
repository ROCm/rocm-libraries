// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference convolution backward weight (wgrad) kernels.
// Compiled via HipRTC with -DX_TYPE=<type> -DW_TYPE=<type> -DY_TYPE=<type> -DCOMPUTE_TYPE=<type>.
// One thread per weight gradient element. Uses stride-based indexing to handle any layout.
// W_TYPE is the dw (weight gradient) type, X_TYPE is the input type, Y_TYPE is the dy type.

#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void convWrwRef1d(ConvWrwArgs1d args)
{
    auto* dw = static_cast<W_TYPE*>(args.dw);
    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* dy = static_cast<const Y_TYPE*>(args.dy);

    long long cPerGroup = args.C / args.groups;
    long long kPerGroup = args.K / args.groups;
    long long totalElements = args.K * cPerGroup * args.Kw;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (k, localC, kw)
    long long kw = idx % args.Kw;
    long long tmp = idx / args.Kw;
    long long localC = tmp % cPerGroup;
    long long k = tmp / cPerGroup;

    // Group parameters
    long long g = k / kPerGroup;
    long long c = g * cPerGroup + localC;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);

    for(long long n = 0; n < args.N; ++n)
    {
        for(long long wo = 0; wo < args.Wo; ++wo)
        {
            long long xi = wo * args.strideW + kw * args.dilW - args.padW;
            if(xi < 0 || xi >= args.Wi)
            {
                continue;
            }

            long long dyIdx = n * args.dyStr.s[0] + k * args.dyStr.s[1] + wo * args.dyStr.s[2];
            long long xIdx = n * args.xStr.s[0] + c * args.xStr.s[1] + xi * args.xStr.s[2];

            acc += toAccum(dy[dyIdx]) * toAccum(x[xIdx]);
        }
    }

    long long dwIdx = k * args.dwStr.s[0] + localC * args.dwStr.s[1] + kw * args.dwStr.s[2];
    W_TYPE* tag = nullptr;

    if(args.beta == 0.0)
    {
        dw[dwIdx] = fromAccum(args.alpha * acc, tag);
    }
    else
    {
        dw[dwIdx] = fromAccum(args.alpha * acc + args.beta * toAccum(dw[dwIdx]), tag);
    }
}

extern "C" __global__ void convWrwRef2d(ConvWrwArgs2d args)
{
    auto* dw = static_cast<W_TYPE*>(args.dw);
    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* dy = static_cast<const Y_TYPE*>(args.dy);

    long long cPerGroup = args.C / args.groups;
    long long kPerGroup = args.K / args.groups;
    long long totalElements = args.K * cPerGroup * args.Kh * args.Kw;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (k, localC, kh, kw)
    long long kw = idx % args.Kw;
    long long tmp = idx / args.Kw;
    long long kh = tmp % args.Kh;
    tmp = tmp / args.Kh;
    long long localC = tmp % cPerGroup;
    long long k = tmp / cPerGroup;

    // Group parameters
    long long g = k / kPerGroup;
    long long c = g * cPerGroup + localC;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);

    for(long long n = 0; n < args.N; ++n)
    {
        for(long long ho = 0; ho < args.Ho; ++ho)
        {
            long long hi = ho * args.strideH + kh * args.dilH - args.padH;
            if(hi < 0 || hi >= args.Hi)
            {
                continue;
            }

            for(long long wo = 0; wo < args.Wo; ++wo)
            {
                long long wi = wo * args.strideW + kw * args.dilW - args.padW;
                if(wi < 0 || wi >= args.Wi)
                {
                    continue;
                }

                long long dyIdx = n * args.dyStr.s[0] + k * args.dyStr.s[1] + ho * args.dyStr.s[2]
                                  + wo * args.dyStr.s[3];
                long long xIdx = n * args.xStr.s[0] + c * args.xStr.s[1] + hi * args.xStr.s[2]
                                 + wi * args.xStr.s[3];

                acc += toAccum(dy[dyIdx]) * toAccum(x[xIdx]);
            }
        }
    }

    long long dwIdx = k * args.dwStr.s[0] + localC * args.dwStr.s[1] + kh * args.dwStr.s[2]
                      + kw * args.dwStr.s[3];
    W_TYPE* tag = nullptr;

    if(args.beta == 0.0)
    {
        dw[dwIdx] = fromAccum(args.alpha * acc, tag);
    }
    else
    {
        dw[dwIdx] = fromAccum(args.alpha * acc + args.beta * toAccum(dw[dwIdx]), tag);
    }
}

extern "C" __global__ void convWrwRef3d(ConvWrwArgs3d args)
{
    auto* dw = static_cast<W_TYPE*>(args.dw);
    auto* x = static_cast<const X_TYPE*>(args.x);
    auto* dy = static_cast<const Y_TYPE*>(args.dy);

    long long cPerGroup = args.C / args.groups;
    long long kPerGroup = args.K / args.groups;
    long long totalElements = args.K * cPerGroup * args.Kd * args.Kh * args.Kw;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (k, localC, kd, kh, kw)
    long long kw = idx % args.Kw;
    long long tmp = idx / args.Kw;
    long long kh = tmp % args.Kh;
    tmp = tmp / args.Kh;
    long long kd = tmp % args.Kd;
    tmp = tmp / args.Kd;
    long long localC = tmp % cPerGroup;
    long long k = tmp / cPerGroup;

    // Group parameters
    long long g = k / kPerGroup;
    long long c = g * cPerGroup + localC;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);

    for(long long n = 0; n < args.N; ++n)
    {
        for(long long do_ = 0; do_ < args.Do; ++do_)
        {
            long long di = do_ * args.strideD + kd * args.dilD - args.padD;
            if(di < 0 || di >= args.Di)
            {
                continue;
            }

            for(long long ho = 0; ho < args.Ho; ++ho)
            {
                long long hi = ho * args.strideH + kh * args.dilH - args.padH;
                if(hi < 0 || hi >= args.Hi)
                {
                    continue;
                }

                for(long long wo = 0; wo < args.Wo; ++wo)
                {
                    long long wi = wo * args.strideW + kw * args.dilW - args.padW;
                    if(wi < 0 || wi >= args.Wi)
                    {
                        continue;
                    }

                    long long dyIdx = n * args.dyStr.s[0] + k * args.dyStr.s[1]
                                      + do_ * args.dyStr.s[2] + ho * args.dyStr.s[3]
                                      + wo * args.dyStr.s[4];
                    long long xIdx = n * args.xStr.s[0] + c * args.xStr.s[1] + di * args.xStr.s[2]
                                     + hi * args.xStr.s[3] + wi * args.xStr.s[4];

                    acc += toAccum(dy[dyIdx]) * toAccum(x[xIdx]);
                }
            }
        }
    }

    long long dwIdx = k * args.dwStr.s[0] + localC * args.dwStr.s[1] + kd * args.dwStr.s[2]
                      + kh * args.dwStr.s[3] + kw * args.dwStr.s[4];
    W_TYPE* tag = nullptr;

    if(args.beta == 0.0)
    {
        dw[dwIdx] = fromAccum(args.alpha * acc, tag);
    }
    else
    {
        dw[dwIdx] = fromAccum(args.alpha * acc + args.beta * toAccum(dw[dwIdx]), tag);
    }
}
