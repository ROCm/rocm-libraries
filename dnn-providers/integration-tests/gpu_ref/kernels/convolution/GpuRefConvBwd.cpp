// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference convolution backward data (dgrad) kernels.
// Compiled via HipRTC with -DX_TYPE=<type> -DW_TYPE=<type> -DY_TYPE=<type> -DCOMPUTE_TYPE=<type>.
// One thread per input gradient element. Uses stride-based indexing to handle any layout.
// X_TYPE is the dx (input gradient) type, Y_TYPE is the dy (output gradient) type.

#include "GpuRefTypes.h"

extern "C" __global__ void convBwdRef1d(ConvBwdArgs1d args)
{
    auto* dx = static_cast<X_TYPE*>(args.dx);
    auto* w = static_cast<const W_TYPE*>(args.w);
    auto* dy = static_cast<const Y_TYPE*>(args.dy);

    long long totalElements = args.N * args.C * args.Wi;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (n, c, wi)
    long long wi = idx % args.Wi;
    long long tmp = idx / args.Wi;
    long long c = tmp % args.C;
    long long n = tmp / args.C;

    // Group parameters
    long long kPerGroup = args.K / args.groups;
    long long cPerGroup = args.C / args.groups;
    long long g = c * args.groups / args.C;
    long long localC = c - g * cPerGroup;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);

    for(long long kw = 0; kw < args.Kw; ++kw)
    {
        long long tmpW = wi + args.padW - kw * args.dilW;
        if(tmpW % args.strideW != 0)
        {
            continue;
        }
        long long wo = tmpW / args.strideW;
        if(wo < 0 || wo >= args.Wo)
        {
            continue;
        }

        for(long long k = 0; k < kPerGroup; ++k)
        {
            long long kIdx = g * kPerGroup + k;

            long long dyIdx = n * args.dyStr.s[0] + kIdx * args.dyStr.s[1] + wo * args.dyStr.s[2];
            long long wIdx = kIdx * args.wStr.s[0] + localC * args.wStr.s[1] + kw * args.wStr.s[2];

            acc += toAccum(dy[dyIdx]) * toAccum(w[wIdx]);
        }
    }

    long long dxIdx = n * args.dxStr.s[0] + c * args.dxStr.s[1] + wi * args.dxStr.s[2];
    X_TYPE* tag = nullptr;

    if(args.beta == 0.0)
    {
        dx[dxIdx] = fromAccum(args.alpha * acc, tag);
    }
    else
    {
        dx[dxIdx] = fromAccum(args.alpha * acc + args.beta * toAccum(dx[dxIdx]), tag);
    }
}

extern "C" __global__ void convBwdRef2d(ConvBwdArgs2d args)
{
    auto* dx = static_cast<X_TYPE*>(args.dx);
    auto* w = static_cast<const W_TYPE*>(args.w);
    auto* dy = static_cast<const Y_TYPE*>(args.dy);

    long long totalElements = args.N * args.C * args.Hi * args.Wi;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (n, c, hi, wi)
    long long wi = idx % args.Wi;
    long long tmp = idx / args.Wi;
    long long hi = tmp % args.Hi;
    tmp = tmp / args.Hi;
    long long c = tmp % args.C;
    long long n = tmp / args.C;

    // Group parameters
    long long kPerGroup = args.K / args.groups;
    long long cPerGroup = args.C / args.groups;
    long long g = c * args.groups / args.C;
    long long localC = c - g * cPerGroup;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);

    for(long long kh = 0; kh < args.Kh; ++kh)
    {
        long long tmpH = hi + args.padH - kh * args.dilH;
        if(tmpH % args.strideH != 0)
        {
            continue;
        }
        long long ho = tmpH / args.strideH;
        if(ho < 0 || ho >= args.Ho)
        {
            continue;
        }

        for(long long kw = 0; kw < args.Kw; ++kw)
        {
            long long tmpW = wi + args.padW - kw * args.dilW;
            if(tmpW % args.strideW != 0)
            {
                continue;
            }
            long long wo = tmpW / args.strideW;
            if(wo < 0 || wo >= args.Wo)
            {
                continue;
            }

            for(long long k = 0; k < kPerGroup; ++k)
            {
                long long kIdx = g * kPerGroup + k;

                long long dyIdx = n * args.dyStr.s[0] + kIdx * args.dyStr.s[1]
                                  + ho * args.dyStr.s[2] + wo * args.dyStr.s[3];
                long long wIdx = kIdx * args.wStr.s[0] + localC * args.wStr.s[1]
                                 + kh * args.wStr.s[2] + kw * args.wStr.s[3];

                acc += toAccum(dy[dyIdx]) * toAccum(w[wIdx]);
            }
        }
    }

    long long dxIdx
        = n * args.dxStr.s[0] + c * args.dxStr.s[1] + hi * args.dxStr.s[2] + wi * args.dxStr.s[3];
    X_TYPE* tag = nullptr;

    if(args.beta == 0.0)
    {
        dx[dxIdx] = fromAccum(args.alpha * acc, tag);
    }
    else
    {
        dx[dxIdx] = fromAccum(args.alpha * acc + args.beta * toAccum(dx[dxIdx]), tag);
    }
}

extern "C" __global__ void convBwdRef3d(ConvBwdArgs3d args)
{
    auto* dx = static_cast<X_TYPE*>(args.dx);
    auto* w = static_cast<const W_TYPE*>(args.w);
    auto* dy = static_cast<const Y_TYPE*>(args.dy);

    long long totalElements = args.N * args.C * args.Di * args.Hi * args.Wi;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (n, c, di, hi, wi)
    long long wi = idx % args.Wi;
    long long tmp = idx / args.Wi;
    long long hi = tmp % args.Hi;
    tmp = tmp / args.Hi;
    long long di = tmp % args.Di;
    tmp = tmp / args.Di;
    long long c = tmp % args.C;
    long long n = tmp / args.C;

    // Group parameters
    long long kPerGroup = args.K / args.groups;
    long long cPerGroup = args.C / args.groups;
    long long g = c * args.groups / args.C;
    long long localC = c - g * cPerGroup;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);

    for(long long kd = 0; kd < args.Kd; ++kd)
    {
        long long tmpD = di + args.padD - kd * args.dilD;
        if(tmpD % args.strideD != 0)
        {
            continue;
        }
        long long do_ = tmpD / args.strideD;
        if(do_ < 0 || do_ >= args.Do)
        {
            continue;
        }

        for(long long kh = 0; kh < args.Kh; ++kh)
        {
            long long tmpH = hi + args.padH - kh * args.dilH;
            if(tmpH % args.strideH != 0)
            {
                continue;
            }
            long long ho = tmpH / args.strideH;
            if(ho < 0 || ho >= args.Ho)
            {
                continue;
            }

            for(long long kw = 0; kw < args.Kw; ++kw)
            {
                long long tmpW = wi + args.padW - kw * args.dilW;
                if(tmpW % args.strideW != 0)
                {
                    continue;
                }
                long long wo = tmpW / args.strideW;
                if(wo < 0 || wo >= args.Wo)
                {
                    continue;
                }

                for(long long k = 0; k < kPerGroup; ++k)
                {
                    long long kIdx = g * kPerGroup + k;

                    long long dyIdx = n * args.dyStr.s[0] + kIdx * args.dyStr.s[1]
                                      + do_ * args.dyStr.s[2] + ho * args.dyStr.s[3]
                                      + wo * args.dyStr.s[4];
                    long long wIdx = kIdx * args.wStr.s[0] + localC * args.wStr.s[1]
                                     + kd * args.wStr.s[2] + kh * args.wStr.s[3]
                                     + kw * args.wStr.s[4];

                    acc += toAccum(dy[dyIdx]) * toAccum(w[wIdx]);
                }
            }
        }
    }

    long long dxIdx = n * args.dxStr.s[0] + c * args.dxStr.s[1] + di * args.dxStr.s[2]
                      + hi * args.dxStr.s[3] + wi * args.dxStr.s[4];
    X_TYPE* tag = nullptr;

    if(args.beta == 0.0)
    {
        dx[dxIdx] = fromAccum(args.alpha * acc, tag);
    }
    else
    {
        dx[dxIdx] = fromAccum(args.alpha * acc + args.beta * toAccum(dx[dxIdx]), tag);
    }
}
