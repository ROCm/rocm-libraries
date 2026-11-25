// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ConvWrwArgs.h"

// Complete runtime-geometry loop 10. FP32 values are staged without rounding; both operands
// are widened before multiplication. FP64 partials and the final FP64 fold round only at dw.
// Separate entry points/argument blocks keep all existing FP32 partial kernels unchanged.
static_assert(CONV_K3_LOOP == 10 && CONV_K3_LAYOUT < 2, "K3 accurate: packed CF or CL");
static_assert((CONV_K3_BK == 8 || CONV_K3_BK == 16) && (CONV_K3_BC == 8 || CONV_K3_BC == 16)
                  && (CONV_K3_BQ == 32 || CONV_K3_BQ == 64),
              "K3 accurate: bounded tile");

extern "C" __global__ void __launch_bounds__(CONV_K3_BK* CONV_K3_BC)
    ConvWrwAccurate(K3AccurateArgs args)
{
    constexpr int BK = CONV_K3_BK, BC = CONV_K3_BC, BQ = CONV_K3_BQ;
    constexpr int THREADS = BK * BC;
    const K3Args& a = args.geometry;
    const int tid = threadIdx.x;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    const int ct = (cg - 1) / BC + 1, kt = (kg - 1) / BK + 1;
    long long tile = blockIdx.x;
    const long long k0 = (tile % kt) * BK;
    tile /= kt;
    const long long c0 = (tile % ct) * BC;
    tile /= ct;
    const int taps = a.ft * a.fr * a.fs;
    const int tap = static_cast<int>(tile % taps), g = static_cast<int>(tile / taps);
    const int fs = tap % a.fs, fr = tap / a.fs % a.fr, ft = tap / a.fs / a.fr;
    const long long positions = static_cast<long long>(a.n) * a.dout * a.ho * a.wo;
    const long long begin = static_cast<long long>(blockIdx.z) * a.stepsPerSplit;
    const long long limit = begin + a.stepsPerSplit;
    const long long end = limit < a.steps ? limit : a.steps;
    const float* x = static_cast<const float*>(a.x);
    const float* dy = static_cast<const float*>(a.dy);
    // Position-contiguous padded LDS rows avoid bank aliasing for adjacent channel owners.
    __shared__ float xs[BC][BQ + 1];
    __shared__ float ys[BK][BQ + 1];
    double acc = 0.0;
    for(long long step = begin; step < end; ++step)
    {
        // Map lanes to the contiguous activation axis: positions for CF, channels for CL.
        // Every tile slot is overwritten, including channel/position tails and borders.
        for(int i = tid; i < (BK + BC) * BQ; i += THREADS)
        {
            const bool loadY = i < BK * BQ;
            const int local = loadY ? i : i - BK * BQ;
            const int channels = loadY ? BK : BC;
            const int ch = CONV_K3_LAYOUT == 0 ? local / BQ : local % channels;
            const int r = CONV_K3_LAYOUT == 0 ? local % BQ : local / channels;
            const long long q = step * BQ + r;
            const long long channel = (loadY ? k0 : c0) + ch;
            float value = 0.0f;
            if(q < positions && channel < (loadY ? kg : cg))
            {
                long long p = q;
                const int ow = static_cast<int>(p % a.wo);
                p /= a.wo;
                const int oh = static_cast<int>(p % a.ho);
                p /= a.ho;
                const int od = static_cast<int>(p % a.dout);
                const long long n = p / a.dout;
                if(loadY)
                {
                    const long long offset = n * a.ys[0]
                                             + (g * static_cast<long long>(kg) + channel) * a.ys[1]
                                             + od * a.ys[2] + oh * a.ys[3] + ow * a.ys[4];
                    value = dy[offset];
                }
                else
                {
                    const long long id = static_cast<long long>(od) * a.sd
                                         + static_cast<long long>(ft) * a.dd - a.pd;
                    const long long ih = static_cast<long long>(oh) * a.sh
                                         + static_cast<long long>(fr) * a.dh - a.ph;
                    const long long iw = static_cast<long long>(ow) * a.sw
                                         + static_cast<long long>(fs) * a.dw - a.pw;
                    // Form signed element offsets only after the spatial mask passes. The
                    // host checked tensor byte spans, so every in-range sum fits int64.
                    if(id >= 0 && id < a.di && ih >= 0 && ih < a.hi && iw >= 0 && iw < a.wi)
                    {
                        const long long offset
                            = n * a.xs[0] + (g * static_cast<long long>(cg) + channel) * a.xs[1]
                              + id * a.xs[2] + ih * a.xs[3] + iw * a.xs[4];
                        value = x[offset];
                    }
                }
            }
            if(loadY)
                ys[ch][r] = value;
            else
                xs[ch][r] = value;
        }
        __syncthreads();
        const int k = tid / BC, c = tid % BC;
        // Do not multiply artificial position tails (in particular, no zero * infinity).
        const long long remaining = positions - step * BQ;
        const int count = remaining < BQ ? static_cast<int>(remaining) : BQ;
        for(int r = 0; r < count; ++r)
            acc = __builtin_fma(static_cast<double>(ys[k][r]), static_cast<double>(xs[c][r]), acc);
        __syncthreads();
    }
    const long long k = k0 + tid / BC, c = c0 + tid % BC;
    if(k < kg && c < cg)
    {
        const long long elem = ((static_cast<long long>(g) * taps + tap) * kg + k) * cg + c;
        args.partial[static_cast<long long>(blockIdx.z) * a.partialElems + elem] = acc;
    }
}

extern "C" __global__ void __launch_bounds__(256) ConvWrwAccurateReduce(K3AccurateReduceArgs args)
{
    const K3ReduceArgs& a = args.geometry;
    // Decode contiguous output storage, not the partial layout: writes stay coalesced for
    // either packed filter layout. True convolution reflects all taps (flattened reversal).
    const long long output = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if(output >= a.elems)
        return;
    const long long k = output / (static_cast<long long>(a.cg) * a.taps);
    const long long within = output % (static_cast<long long>(a.cg) * a.taps);
    const int c = a.wc == 1 ? static_cast<int>(within % a.cg) : static_cast<int>(within / a.taps);
    int tap = a.wc == 1 ? static_cast<int>(within / a.cg) : static_cast<int>(within % a.taps);
    if(a.flip)
        tap = a.taps - 1 - tap;
    const long long elem = ((k / a.kg * a.taps + tap) * a.kg + k % a.kg) * a.cg + c;
    double sum = 0.0;
    for(int split = 0; split < a.splits; ++split)
        sum += args.partial[static_cast<long long>(split) * a.partialElems + elem];
    static_cast<float*>(a.dw)[output] = static_cast<float>(sum);
}
