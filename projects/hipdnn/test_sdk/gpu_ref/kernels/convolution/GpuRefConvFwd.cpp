// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference convolution forward kernel.
// Compiled via HipRTC with -DDATA_TYPE=<type>.
// One thread per output element. Naive but correct.

#include "GpuRefTypes.h"

extern "C" __global__ void convFwdRef(const DATA_TYPE* __restrict__ x,
                                      const DATA_TYPE* __restrict__ w,
                                      DATA_TYPE* __restrict__ y,
                                      long long N,
                                      long long Cin,
                                      long long Hin,
                                      long long Win,
                                      long long Cout,
                                      long long Hout,
                                      long long Wout,
                                      long long Kh,
                                      long long Kw,
                                      long long strideH,
                                      long long strideW,
                                      long long dilationH,
                                      long long dilationW,
                                      long long padH,
                                      long long padW,
                                      long long groups)
{
    long long totalOutputElements = N * Cout * Hout * Wout;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Decompose linear index into (n, cout, hout, wout) for NCHW layout
    long long wout = idx % Wout;
    long long tmp = idx / Wout;
    long long hout = tmp % Hout;
    tmp = tmp / Hout;
    long long cout = tmp % Cout;
    long long n = tmp / Cout;

    // Determine group parameters
    long long channelsPerGroup = Cin / groups;
    long long outputChannelsPerGroup = Cout / groups;
    long long g = cout / outputChannelsPerGroup;
    long long baseInputChannel = g * channelsPerGroup;

    // Weight layout: [Cout, Cin/groups, Kh, Kw]
    long long wChannels = channelsPerGroup;

    AccumType acc = static_cast<AccumType>(0);

    for(long long c = 0; c < channelsPerGroup; ++c)
    {
        long long xChannel = baseInputChannel + c;

        for(long long kh = 0; kh < Kh; ++kh)
        {
            long long hi = hout * strideH + kh * dilationH - padH;
            if(hi < 0 || hi >= Hin)
            {
                continue;
            }

            for(long long kw = 0; kw < Kw; ++kw)
            {
                long long wi = wout * strideW + kw * dilationW - padW;
                if(wi < 0 || wi >= Win)
                {
                    continue;
                }

                // x index: [n, xChannel, hi, wi] in NCHW
                long long xIdx = ((n * Cin + xChannel) * Hin + hi) * Win + wi;

                // w index: [cout, c, kh, kw] where c is within group
                long long wIdx = ((cout * wChannels + c) * Kh + kh) * Kw + kw;

                acc += toFloat(x[xIdx]) * toFloat(w[wIdx]);
            }
        }
    }

    // y index: [n, cout, hout, wout] in NCHW
    long long yIdx = ((n * Cout + cout) * Hout + hout) * Wout + wout;
    DATA_TYPE* tag = nullptr;
    y[yIdx] = fromFloat(acc, tag);
}
