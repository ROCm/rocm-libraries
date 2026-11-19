// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ConvWrwCore.h"

// gfx1250 packed channels-last, Kg <= 4: each thread owns one input channel and
// accumulates every output channel in FP32. No LDS, matrix padding, or atomics.
// Four adjacent positions use four independent FP32 chains per output channel.
// All four x/dy operands are loaded before their FMAs; dy addresses are uniform
// within a wave, with no explicit lane-to-SGPR transfers on the reduction path.
// BQ defines the same flattened reduction steps/split boundaries as loop 11.
template <class T, int BC, int BQ>
__device__ inline K3_LOOP_INLINE void k3WrwSmallK(const K3Args& a)
{
    static_assert(BC == 64 || BC == 128 || BC == 256, "K3 small K: whole wave32 blocks");
    static_assert(BQ > 0, "K3 small K: positive reduction tile");
    const int tid = threadIdx.x;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    const int ct = (cg - 1) / BC + 1;
    long long block = blockIdx.x;
    const long long c = (block % ct) * BC + tid;
    // No barriers or lane broadcasts: threads owning no valid C channel can leave.
    if(c >= cg)
        return;
    block /= ct;
    const int taps = a.ft * a.fr * a.fs;
    const int tap = static_cast<int>(block % taps), g = static_cast<int>(block / taps);
    const int fs = tap % a.fs, fr = tap / a.fs % a.fr, ft = tap / a.fs / a.fr;

    const long long positions = static_cast<long long>(a.n) * a.dout * a.ho * a.wo;
    const long long begin = static_cast<long long>(blockIdx.z) * a.stepsPerSplit;
    const long long limit = begin + a.stepsPerSplit;
    const long long end = limit < a.steps ? limit : a.steps;
    const long long qBegin = begin * BQ;
    const long long qLimit = end * BQ;
    const long long qEnd = qLimit < positions ? qLimit : positions;
    float acc[4][4]{};

    if(qBegin < qEnd)
    {
        // Decode only the first position of this split. Thereafter carry the
        // coordinates and element offsets across rows, planes and images.
        long long p = qBegin;
        int ow = static_cast<int>(p % a.wo);
        p /= a.wo;
        int oh = static_cast<int>(p % a.ho);
        p /= a.ho;
        int od = static_cast<int>(p % a.dout);
        const long long n = p / a.dout;
        const long long xd = static_cast<long long>(ft) * a.dd - a.pd;
        const long long xh = static_cast<long long>(fr) * a.dh - a.ph;
        const long long xw = static_cast<long long>(fs) * a.dw - a.pw;
        long long id = static_cast<long long>(od) * a.sd + xd;
        long long ih = static_cast<long long>(oh) * a.sh + xh;
        long long iw = static_cast<long long>(ow) * a.sw + xw;
        // Padding/stride can put masked coordinates far outside the tensor.
        // Carry offsets modulo 2^64 to avoid signed overflow there; when the
        // spatial/channel masks pass, the residue is the actual in-range offset.
        // This path is admitted only for packed channels-last tensors: channel stride is
        // one and adjacent dy positions are exactly total-K elements apart.
        const unsigned long long xs0 = a.xs[0], xs1 = 1, xs2 = a.xs[2], xs3 = a.xs[3],
                                 xs4 = static_cast<unsigned>(a.c);
        const unsigned long long ys1 = 1, ys4 = static_cast<unsigned>(a.k);
        unsigned long long xo = static_cast<unsigned long long>(n) * xs0
                                + (static_cast<unsigned long long>(g) * cg + c) * xs1
                                + static_cast<unsigned long long>(id) * xs2
                                + static_cast<unsigned long long>(ih) * xs3
                                + static_cast<unsigned long long>(iw) * xs4;
        unsigned long long yo = static_cast<unsigned long long>(qBegin) * ys4
                                + static_cast<unsigned long long>(g) * kg;
        const unsigned long long xStepW = a.sw * xs4;
        const unsigned long long xStepH = a.sh * xs3;
        const unsigned long long xStepD = a.sd * xs2;
        const unsigned long long xCarryH = xStepH - a.wo * xStepW;
        const unsigned long long xCarryD = xStepD - a.ho * xStepH;
        const unsigned long long xCarryN = xs0 - a.dout * xStepD;
        const T* x = static_cast<const T*>(a.x);
        const T* dy = static_cast<const T*>(a.dy);
        for(long long q = qBegin; q < qEnd;)
        {
            int consumed = 1;
            // Subtraction-form bounds avoid overflowing q/ow near their limits.
            // The four-position path never crosses a row or touches padded x.
            if(qEnd - q >= 4 && a.wo - ow >= 4 && id >= 0 && id < a.di && ih >= 0 && ih < a.hi
               && iw >= 0 && static_cast<long long>(a.wi) - iw > 3LL * a.sw)
            {
                float xv[4], yv[4][4];
#pragma unroll
                for(int r = 0; r < 4; ++r)
                    xv[r] = K1Elem<T>::load(x[xo + r * xStepW]);
#pragma unroll
                for(int k = 0; k < 4; ++k)
                    if(k < kg)
                    {
#pragma unroll
                        for(int r = 0; r < 4; ++r)
                            yv[k][r] = K1Elem<T>::load(dy[yo + k * ys1 + r * ys4]);
                    }
#pragma unroll
                for(int k = 0; k < 4; ++k)
                    if(k < kg)
                    {
#pragma unroll
                        for(int r = 0; r < 4; ++r)
                            acc[k][r] = __builtin_fmaf(yv[k][r], xv[r], acc[k][r]);
                    }
                consumed = 4;
            }
            else
            {
                // Padding contributes zero x, but still loads dy and executes
                // the FMA so 0 * NaN/Inf propagates as in the generic matrix loop.
                float xv = 0.0f;
                if(id >= 0 && id < a.di && ih >= 0 && ih < a.hi && iw >= 0 && iw < a.wi)
                    xv = K1Elem<T>::load(x[xo]);
                float yv[4];
#pragma unroll
                for(int k = 0; k < 4; ++k)
                    if(k < kg)
                        yv[k] = K1Elem<T>::load(dy[yo + k * ys1]);
#pragma unroll
                for(int k = 0; k < 4; ++k)
                    if(k < kg)
                        acc[k][0] = __builtin_fmaf(yv[k], xv, acc[k][0]);
            }

            q += consumed;
            xo += consumed * xStepW;
            yo += consumed * ys4;
            iw += static_cast<long long>(consumed) * a.sw;
            ow += consumed;
            if(ow == a.wo)
            {
                ow = 0;
                iw = xw;
                xo += xCarryH;
                ih += a.sh;
                if(++oh == a.ho)
                {
                    oh = 0;
                    ih = xh;
                    xo += xCarryD;
                    id += a.sd;
                    if(++od == a.dout)
                    {
                        od = 0;
                        id = xd;
                        // This carry advances n without re-decoding it.
                        xo += xCarryN;
                    }
                }
            }
        }
    }

    const K3Out out = k3OutOf<T>(a, g, taps, blockIdx.z);
    k3WithOut<T>(a, out, [&](auto* base) {
        typedef typename K3PointeeOf<decltype(base)>::type E;
        if(c < cg)
        {
#pragma unroll
            for(int k = 0; k < 4; ++k)
                if(k < kg)
                {
                    // Fixed fold order; all chains and partials remain FP32.
                    const float sum = ((acc[k][0] + acc[k][1]) + acc[k][2]) + acc[k][3];
                    base[tap * out.ts + k * out.ks + c] = K1Elem<E>::store(sum);
                }
        }
    });
}
