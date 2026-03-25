// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference convolution forward kernels.
// Compiled via HipRTC with -DSRC_TYPE=<type> -DDST_TYPE=<type> -DACC_TYPE=<type>.
// One thread per output element. Uses stride-based indexing to handle any layout.

#include "GpuRefTypes.h"

extern "C" __global__ void convFwdRef2d(const SRC_TYPE* __restrict__ x,
                                        const SRC_TYPE* __restrict__ w,
                                        DST_TYPE* __restrict__ y,
                                        Strides4 xStr,
                                        Strides4 wStr,
                                        Strides4 yStr,
                                        long long N,
                                        long long C,
                                        long long Hi,
                                        long long Wi,
                                        long long K,
                                        long long Ho,
                                        long long Wo,
                                        long long Kh,
                                        long long Kw,
                                        long long strideH,
                                        long long strideW,
                                        long long dilH,
                                        long long dilW,
                                        long long padH,
                                        long long padW,
                                        long long groups,
                                        double alpha,
                                        double beta)
{
    long long totalOutputElements = N * K * Ho * Wo;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Decompose linear index into (n, k, ho, wo)
    long long wo = idx % Wo;
    long long tmp = idx / Wo;
    long long ho = tmp % Ho;
    tmp = tmp / Ho;
    long long k = tmp % K;
    long long n = tmp / K;

    // Group parameters
    long long cPerGroup = C / groups;
    long long kPerGroup = K / groups;
    long long g = k / kPerGroup;
    long long baseInputChannel = g * cPerGroup;

    ACC_TYPE acc = static_cast<ACC_TYPE>(0);

    for(long long c = 0; c < cPerGroup; ++c)
    {
        long long xChannel = baseInputChannel + c;

        for(long long kh = 0; kh < Kh; ++kh)
        {
            long long hi = ho * strideH + kh * dilH - padH;
            if(hi < 0 || hi >= Hi)
            {
                continue;
            }

            for(long long kw = 0; kw < Kw; ++kw)
            {
                long long wi = wo * strideW + kw * dilW - padW;
                if(wi < 0 || wi >= Wi)
                {
                    continue;
                }

                long long xIdx
                    = n * xStr.s[0] + xChannel * xStr.s[1] + hi * xStr.s[2] + wi * xStr.s[3];
                long long wIdx = k * wStr.s[0] + c * wStr.s[1] + kh * wStr.s[2] + kw * wStr.s[3];

#ifdef USE_TF32
                float xf = truncateToTf32(static_cast<float>(toAccum(x[xIdx])));
                float wf = truncateToTf32(static_cast<float>(toAccum(w[wIdx])));
                acc += static_cast<ACC_TYPE>(xf) * static_cast<ACC_TYPE>(wf);
#else
                acc += toAccum(x[xIdx]) * toAccum(w[wIdx]);
#endif
            }
        }
    }

    long long yIdx = n * yStr.s[0] + k * yStr.s[1] + ho * yStr.s[2] + wo * yStr.s[3];
    DST_TYPE* tag = nullptr;

    if(beta == 0.0)
    {
        y[yIdx] = fromAccum(alpha * acc, tag);
    }
    else
    {
        y[yIdx] = fromAccum(alpha * acc + beta * toAccum(y[yIdx]), tag);
    }
}

extern "C" __global__ void convFwdRef3d(const SRC_TYPE* __restrict__ x,
                                        const SRC_TYPE* __restrict__ w,
                                        DST_TYPE* __restrict__ y,
                                        Strides5 xStr,
                                        Strides5 wStr,
                                        Strides5 yStr,
                                        long long N,
                                        long long C,
                                        long long Di,
                                        long long Hi,
                                        long long Wi,
                                        long long K,
                                        long long Do,
                                        long long Ho,
                                        long long Wo,
                                        long long Kd,
                                        long long Kh,
                                        long long Kw,
                                        long long strideD,
                                        long long strideH,
                                        long long strideW,
                                        long long dilD,
                                        long long dilH,
                                        long long dilW,
                                        long long padD,
                                        long long padH,
                                        long long padW,
                                        long long groups,
                                        double alpha,
                                        double beta)
{
    long long totalOutputElements = N * K * Do * Ho * Wo;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Decompose linear index into (n, k, do_, ho, wo)
    long long wo = idx % Wo;
    long long tmp = idx / Wo;
    long long ho = tmp % Ho;
    tmp = tmp / Ho;
    long long do_ = tmp % Do;
    tmp = tmp / Do;
    long long k = tmp % K;
    long long n = tmp / K;

    // Group parameters
    long long cPerGroup = C / groups;
    long long kPerGroup = K / groups;
    long long g = k / kPerGroup;
    long long baseInputChannel = g * cPerGroup;

    ACC_TYPE acc = static_cast<ACC_TYPE>(0);

    for(long long c = 0; c < cPerGroup; ++c)
    {
        long long xChannel = baseInputChannel + c;

        for(long long kd = 0; kd < Kd; ++kd)
        {
            long long di = do_ * strideD + kd * dilD - padD;
            if(di < 0 || di >= Di)
            {
                continue;
            }

            for(long long kh = 0; kh < Kh; ++kh)
            {
                long long hi = ho * strideH + kh * dilH - padH;
                if(hi < 0 || hi >= Hi)
                {
                    continue;
                }

                for(long long kw = 0; kw < Kw; ++kw)
                {
                    long long wi = wo * strideW + kw * dilW - padW;
                    if(wi < 0 || wi >= Wi)
                    {
                        continue;
                    }

                    long long xIdx = n * xStr.s[0] + xChannel * xStr.s[1] + di * xStr.s[2]
                                     + hi * xStr.s[3] + wi * xStr.s[4];
                    long long wIdx = k * wStr.s[0] + c * wStr.s[1] + kd * wStr.s[2] + kh * wStr.s[3]
                                     + kw * wStr.s[4];

#ifdef USE_TF32
                    float xf = truncateToTf32(static_cast<float>(toAccum(x[xIdx])));
                    float wf = truncateToTf32(static_cast<float>(toAccum(w[wIdx])));
                    acc += static_cast<ACC_TYPE>(xf) * static_cast<ACC_TYPE>(wf);
#else
                    acc += toAccum(x[xIdx]) * toAccum(w[wIdx]);
#endif
                }
            }
        }
    }

    long long yIdx
        = n * yStr.s[0] + k * yStr.s[1] + do_ * yStr.s[2] + ho * yStr.s[3] + wo * yStr.s[4];
    DST_TYPE* tag = nullptr;

    if(beta == 0.0)
    {
        y[yIdx] = fromAccum(alpha * acc, tag);
    }
    else
    {
        y[yIdx] = fromAccum(alpha * acc + beta * toAccum(y[yIdx]), tag);
    }
}
