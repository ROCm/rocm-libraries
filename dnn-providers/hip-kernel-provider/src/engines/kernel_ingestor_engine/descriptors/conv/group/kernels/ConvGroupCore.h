// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Small-group convolution cores (K4): groups of cg <= 16 channels each way, run per 16-channel
// block on a 16x16 matrix-core atom (the K1 atoms) with the block-diagonal expanded filter, so a
// block's A fragment (16 output pixels x 16 channels) feeds 16 / cg whole groups at once.
#pragma once

#include "ConvGroupArgs.h"
#include "ConvIgemmCore.h"

// Matrix-core kernel. A workgroup of 4 waves stages the input halo of TH output rows x 16 WP
// output columns x CB channel blocks in LDS (16-byte loads along the channels; rows padded by 16
// bytes so the 16 lanes of a fragment read sit on different banks), then the waves take (row,
// channel block) items, each WP fragments of 16 output columns, accumulating every tap's A
// fragment (from LDS) against that tap's B fragments (the block's expanded filter, in registers).
// Grid: (blocks of CB channel blocks) x (column tiles) x (row tiles) x images, flattened.
template <class T, class Atom, int R, int S, int ST, int TH, int WP, int CB>
__device__ void k4ConvMma(const K4Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = 4 * WAVE;
    constexpr int TW = 16 * WP;
    constexpr int CH = 16 * CB;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int PITCH = CH + VEC; // elements a staged pixel takes
    constexpr int HR = (TH - 1) * ST + R, HC = (TW - 1) * ST + S;
    constexpr int TAPS = R * S;
    constexpr int KC = 16 / Atom::KA; // atom k-steps per tap
    static_assert(Atom::DIM == 16 && 16 % Atom::KA == 0, "16x16 atoms with k-steps dividing 16");
    typedef typename K1Raw<16>::type Raw;
    __shared__ __attribute__((aligned(16))) T halo[HR * HC * PITCH];

    const int blocks = a.c / CH;
    long long b = blockIdx.x;
    const int cbt = static_cast<int>(b % blocks);
    b /= blocks;
    const int tws = (a.wo + TW - 1) / TW;
    const int twi = static_cast<int>(b % tws);
    b /= tws;
    const int ths = (a.ho + TH - 1) / TH;
    const int thi = static_cast<int>(b % ths);
    const long long img = b / ths;
    const int oh0 = thi * TH, ow0 = twi * TW;
    const int ih0 = oh0 * ST - a.ph, iw0 = ow0 * ST - a.pw;
    const T* in = static_cast<const T*>(a.in);
    const int tid = threadIdx.x;
    constexpr int VP = CH / VEC; // vectors a staged pixel takes
    for(int i = tid; i < HR * HC * VP; i += THREADS)
    {
        const int v = i % VP, p = i / VP, hc = p % HC, hr = p / HC;
        const int ih = ih0 + hr, iw = iw0 + hc;
        // Clamped address, then zero: no per-load branch.
        const bool ok = static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi)
                        && static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi);
        const long long at = ((img * a.hi + (ok ? ih : 0)) * a.wi + (ok ? iw : 0)) * a.c
                             + static_cast<long long>(cbt) * CH + v * VEC;
        Raw val = *reinterpret_cast<const Raw*>(in + at);
        if(!ok)
            val = Raw{};
        *reinterpret_cast<Raw*>(halo + p * PITCH + v * VEC) = val;
    }
    __syncthreads();

    const int lane = tid % WAVE, wave = tid / WAVE;
    const int row = lane % 16;
    const int kb = Atom::template kBase<Atom::KPL>(lane);
    const T* w = static_cast<const T*>(a.w);
    T* out = static_cast<T*>(a.out);
    for(int item = wave; item < TH * CB; item += 4)
    {
        const int r = item / CB, cbi = item % CB;
        const int oh = oh0 + r;
        if(oh >= a.ho)
            continue;
        const int cb = cbt * CB + cbi;
        typename Atom::frag bf[TAPS][KC];
#pragma unroll
        for(int t = 0; t < TAPS; ++t)
#pragma unroll
            for(int q = 0; q < KC; ++q)
                bf[t][q]
                    = k1Frag<Atom>(w + ((static_cast<long long>(cb) * TAPS + t) * 16 + row) * 16
                                   + q * Atom::KA + kb);
        typename Atom::acc acc[WP];
#pragma unroll
        for(int p = 0; p < WP; ++p)
            acc[p] = typename Atom::acc{};
#pragma unroll
        for(int fr = 0; fr < R; ++fr)
#pragma unroll
            for(int fs = 0; fs < S; ++fs)
#pragma unroll
                for(int p = 0; p < WP; ++p)
                {
                    const int hc = (p * 16 + row) * ST + fs, hr = r * ST + fr;
                    const T* src = halo + (hr * HC + hc) * PITCH + cbi * 16 + kb;
#pragma unroll
                    for(int q = 0; q < KC; ++q)
                        acc[p] = Atom::mma(
                            k1Frag<Atom>(src + q * Atom::KA), bf[fr * S + fs][q], acc[p]);
                }
#pragma unroll
        for(int p = 0; p < WP; ++p)
#pragma unroll
            for(int e = 0; e < Atom::ACC; ++e)
            {
                const int ow = ow0 + p * 16 + Atom::dRow(lane, e);
                if(ow < a.wo)
                    out[((img * a.ho + oh) * a.wo + ow) * a.c + cb * 16 + lane % 16]
                        = K1Elem<T>::store(acc[p][e]);
            }
    }
}

// The filter expansion (K4ExpandArgs), one element per thread, grid-strided.
__device__ inline void k4Expand(const K4ExpandArgs& a)
{
    const int taps = a.r * a.s;
    const long long total = static_cast<long long>(a.c / 16) * taps * 256;
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += step)
    {
        const int kk = static_cast<int>(i % 16);
        const int nn = static_cast<int>((i / 16) % 16);
        const int t = static_cast<int>((i / 256) % taps);
        const int cb = static_cast<int>(i / (256LL * taps));
        const int n = cb * 16 + nn, k = cb * 16 + kk;
        int fr = t / a.s, fs = t % a.s;
        // Backward data reflects the tap; true convolution reflects it (again).
        if(a.backward != a.reflect)
        {
            fr = a.r - 1 - fr;
            fs = a.s - 1 - fs;
        }
        // The graph filter's output channel and its group-local input channel.
        const int oc = a.backward ? k : n, ic = a.backward ? n : k;
        const bool own = oc / a.cg == ic / a.cg;
        const long long src = oc * a.sK + (ic % a.cg) * a.sC + fr * a.sR + fs * a.sS;
        if(a.elementBytes == 4)
            static_cast<unsigned*>(a.dst)[i] = own ? static_cast<const unsigned*>(a.src)[src] : 0u;
        else
            static_cast<unsigned short*>(a.dst)[i]
                = own ? static_cast<const unsigned short*>(a.src)[src]
                      : static_cast<unsigned short>(0);
    }
}
