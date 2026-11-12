// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Small-group convolution cores (K4): groups of cg <= 16 channels each way, run per 16-channel
// block on a 16x16 matrix-core atom (the K1 atoms) with the block-diagonal expanded filter, so a
// block's A fragment (16 output pixels x 16 channels) feeds 16 / cg whole groups at once.
// Backward weights (k4WrwMma, k4WrwFma, k4WrwReduce) keep the diagonal blocks of the same
// block products, the reduction over dy positions.
#pragma once

#include "ConvGroupArgs.h"
#include "ConvIgemmCore.h"

// Stages depth plane id's input halo of a workgroup in LDS: HR x HC pixels from (ih0, iw0), CH
// channels from channel tile cbt, each pixel PITCH elements, zero outside the input. Channels-last
// (CF 0): 16-byte loads along the channels. Channels-first (CF 1): a thread gathers 16 bytes of
// channels of one pixel (one load a channel, consecutive threads consecutive columns, so each
// load is a coalesced row segment) and stores them as one vector.
template <class T, int CF, int HR, int HC, int CH, int PITCH, int THREADS>
__device__ __forceinline__ void k4StageHalo(const K4Args& a,
                                            const T* in,
                                            T* halo,
                                            long long img,
                                            int id,
                                            int cbt,
                                            int ih0,
                                            int iw0,
                                            int tid)
{
    typedef typename K1Raw<16>::type Raw;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int VP = CH / VEC; // vectors a staged pixel takes
    if constexpr(CF != 0)
    {
        const long long planes = static_cast<long long>(a.di) * a.hi * a.wi;
        const T* src = in + (img * a.c + static_cast<long long>(cbt) * CH) * planes
                       + static_cast<long long>(id) * a.hi * a.wi;
        for(int i = tid; i < VP * HR * HC; i += THREADS)
        {
            const int hc = i % HC, hr = (i / HC) % HR, v = i / (HC * HR);
            const int ih = ih0 + hr, iw = iw0 + hc;
            const bool ok = static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi)
                            && static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi);
            const T* at = src + v * VEC * planes + (ok ? ih : 0) * a.wi + (ok ? iw : 0);
            union
            {
                Raw raw;
                T e[VEC];
            } val;
#pragma unroll
            for(int j = 0; j < VEC; ++j)
                val.e[j] = at[j * planes];
            if(!ok)
                val.raw = Raw{};
            *reinterpret_cast<Raw*>(halo + (hr * HC + hc) * PITCH + v * VEC) = val.raw;
        }
    }
    else
    {
        const long long plane = img * a.di + id;
        for(int i = tid; i < HR * HC * VP; i += THREADS)
        {
            const int v = i % VP, p = i / VP, hc = p % HC, hr = p / HC;
            const int ih = ih0 + hr, iw = iw0 + hc;
            // Clamped address, then zero: no per-load branch.
            const bool ok = static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi)
                            && static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi);
            const long long at = ((plane * a.hi + (ok ? ih : 0)) * a.wi + (ok ? iw : 0)) * a.c
                                 + static_cast<long long>(cbt) * CH + v * VEC;
            Raw val = *reinterpret_cast<const Raw*>(in + at);
            if(!ok)
                val = Raw{};
            *reinterpret_cast<Raw*>(halo + p * PITCH + v * VEC) = val;
        }
    }
}

// Writes a workgroup's output tile, staged in LDS as [row][channel][column] (rows of TW + 1), to
// packed channels-first out: one row of output columns at a time, consecutive threads
// consecutive columns.
template <class T, int TH, int TW, int CH, int THREADS>
__device__ __forceinline__ void k4StoreTileChannelsFirst(const K4Args& a,
                                                         T* out,
                                                         const T* tile,
                                                         long long img,
                                                         int od,
                                                         int cbt,
                                                         int oh0,
                                                         int ow0,
                                                         int tid)
{
    constexpr int OP = TW + 1;
    const long long outPlanes = static_cast<long long>(a.outD) * a.outH * a.outW;
    T* dst = out + (img * a.c + static_cast<long long>(cbt) * CH) * outPlanes
             + static_cast<long long>(od * a.osd + a.ood) * a.outH * a.outW;
    for(int i = tid; i < TH * CH * TW; i += THREADS)
    {
        const int ow = i % TW, ch = (i / TW) % CH, r = i / (TW * CH);
        const int oh = oh0 + r;
        if(oh < a.ho && ow0 + ow < a.wo)
            dst[ch * outPlanes + static_cast<long long>(oh * a.osh + a.ooh) * a.outW
                + (ow0 + ow) * a.osw + a.oow]
                = tile[(r * CH + ch) * OP + ow];
    }
}

// A halo column's slot: at stride 2 a halo row holds its even columns first, then its odd ones, so
// the 16 lanes of a fragment read (columns 2 apart) read consecutive pixels.
template <int ST, int HC>
__device__ __forceinline__ int k4HaloColumn(int hc)
{
    if constexpr(ST == 2)
        return (hc & 1) * ((HC + 1) / 2) + (hc >> 1);
    else
        return hc;
}

// Matrix-core kernel. A workgroup of 4 waves owns TH output rows x 16 WP output columns x CB
// channel blocks of one output plane. Per depth tap it stages that input plane's halo in LDS
// (16-byte loads along the channels; pixels padded by 16 bytes so the 16 lanes of a fragment read
// sit on different banks), then each wave runs its (row, channel block) items, WP fragments of 16
// output columns each, accumulating every (row, column) tap's A fragment (from LDS) against that
// tap's B fragments (the block's expanded filter). Grid: (blocks of CB channel blocks) x (column
// tiles) x (row tiles) x output planes x images, flattened.
// CF: in and out are packed channels-first (N C D H W) and read and written in place: the halo is
// gathered along its rows (consecutive threads, consecutive columns of one channel), and the
// accumulators go out through LDS so each row of output columns is written contiguously.
template <class T, class Atom, int FT, int R, int S, int ST, int TH, int WP, int CB, int CF = 0>
__device__ void k4ConvMma(const K4Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = 4 * WAVE;
    constexpr int TW = 16 * WP;
    constexpr int CH = 16 * CB;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int PITCH = CH + VEC; // elements a staged pixel takes
    constexpr int HR = (TH - 1) * ST + R, HC = (TW - 1) * ST + S;
    constexpr int TAPS = FT * R * S;
    constexpr int KC = 16 / Atom::KA; // atom k-steps per tap
    constexpr int ITEMS = TH * CB / 4; // (row, channel block) items a wave
    static_assert(Atom::DIM == 16 && 16 % Atom::KA == 0, "16x16 atoms with k-steps dividing 16");
    static_assert(TH * CB % 4 == 0, "whole items a wave");
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
    b /= ths;
    const int od = static_cast<int>(b % a.dout);
    const long long img = b / a.dout;
    const int oh0 = thi * TH, ow0 = twi * TW;
    const int ih0 = oh0 * ST - a.ph, iw0 = ow0 * ST - a.pw;
    const T* in = static_cast<const T*>(a.in);
    const T* w = static_cast<const T*>(a.w);
    T* out = static_cast<T*>(a.out);
    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int row = lane % 16;
    const int kb = Atom::template kBase<Atom::KPL>(lane);
    constexpr int VP = CH / VEC; // vectors a staged pixel takes

    typename Atom::acc acc[ITEMS][WP];
#pragma unroll
    for(int it = 0; it < ITEMS; ++it)
#pragma unroll
        for(int p = 0; p < WP; ++p)
            acc[it][p] = typename Atom::acc{};

    for(int ft = 0; ft < FT; ++ft)
    {
        const int id = od * a.sd - a.pd + ft;
        // A plane past the input contributes nothing (uniform across the workgroup).
        if(static_cast<unsigned>(id) >= static_cast<unsigned>(a.di))
            continue;
        if(ft > 0)
            __syncthreads(); // the previous plane's reads are done
        const long long plane = img * a.di + id;
        if constexpr(CF != 0)
        {
            const long long planes = static_cast<long long>(a.di) * a.hi * a.wi;
            const T* src = in + (img * a.c + static_cast<long long>(cbt) * CH) * planes
                           + static_cast<long long>(id) * a.hi * a.wi;
            // A thread gathers VEC channels of one halo pixel (one load a channel, consecutive
            // threads consecutive columns) and stores them as one 16-byte vector.
            for(int i = tid; i < VP * HR * HC; i += THREADS)
            {
                const int hc = i % HC, hr = (i / HC) % HR, v = i / (HC * HR);
                const int ih = ih0 + hr, iw = iw0 + hc;
                const bool ok = static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi)
                                && static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi);
                const T* at = src + v * VEC * planes + (ok ? ih : 0) * a.wi + (ok ? iw : 0);
                union
                {
                    Raw raw;
                    T e[VEC];
                } val;
#pragma unroll
                for(int j = 0; j < VEC; ++j)
                    val.e[j] = at[j * planes];
                if(!ok)
                    val.raw = Raw{};
                if constexpr(ST == 2)
                    *reinterpret_cast<Raw*>(halo + (hr * HC + k4HaloColumn<ST, HC>(hc)) * PITCH
                                            + v * VEC)
                        = val.raw;
                else
                    *reinterpret_cast<Raw*>(halo + (hr * HC + hc) * PITCH + v * VEC) = val.raw;
            }
        }
        else
        {
            for(int i = tid; i < HR * HC * VP; i += THREADS)
            {
                const int v = i % VP, p = i / VP, hc = p % HC, hr = p / HC;
                const int ih = ih0 + hr, iw = iw0 + hc;
                // Clamped address, then zero: no per-load branch.
                const bool ok = static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi)
                                && static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi);
                const long long at = ((plane * a.hi + (ok ? ih : 0)) * a.wi + (ok ? iw : 0)) * a.c
                                     + static_cast<long long>(cbt) * CH + v * VEC;
                Raw val = *reinterpret_cast<const Raw*>(in + at);
                if(!ok)
                    val = Raw{};
                if constexpr(ST == 2)
                    *reinterpret_cast<Raw*>(halo + (hr * HC + k4HaloColumn<ST, HC>(hc)) * PITCH
                                            + v * VEC)
                        = val;
                else
                    *reinterpret_cast<Raw*>(halo + p * PITCH + v * VEC) = val;
            }
        }
        __syncthreads();
#pragma unroll
        for(int it = 0; it < ITEMS; ++it)
        {
            const int item = wave + 4 * it;
            const int r = item / CB, cbi = item % CB;
            if(oh0 + r >= a.ho)
                continue;
            const int cb = cbt * CB + cbi;
#pragma unroll
            for(int fr = 0; fr < R; ++fr)
#pragma unroll
                for(int fs = 0; fs < S; ++fs)
                {
                    const int tap = (ft * R + fr) * S + fs;
                    typename Atom::frag bf[KC];
#pragma unroll
                    for(int q = 0; q < KC; ++q)
                        bf[q] = k1Frag<Atom>(
                            w + ((static_cast<long long>(cb) * TAPS + tap) * 16 + row) * 16
                            + q * Atom::KA + kb);
#pragma unroll
                    for(int p = 0; p < WP; ++p)
                    {
                        const int hc = (p * 16 + row) * ST + fs, hr = r * ST + fr;
                        const T* src
                            = halo + (hr * HC + k4HaloColumn<ST, HC>(hc)) * PITCH + cbi * 16 + kb;
#pragma unroll
                        for(int q = 0; q < KC; ++q)
                            acc[it][p]
                                = Atom::mma(k1Frag<Atom>(src + q * Atom::KA), bf[q], acc[it][p]);
                    }
                }
        }
    }

    const long long outPlane = img * a.outD + od * a.osd + a.ood;
    if constexpr(CF != 0)
    {
        // Through LDS: [row][channel][column] (a column pad), then rows of columns out.
        constexpr int OP = TW + 1;
        static_assert(TH * CH * OP <= HR * HC * PITCH, "the output tile fits the halo");
        __syncthreads();
#pragma unroll
        for(int it = 0; it < ITEMS; ++it)
        {
            const int item = wave + 4 * it;
            const int r = item / CB, cbi = item % CB;
            if(oh0 + r >= a.ho)
                continue;
#pragma unroll
            for(int p = 0; p < WP; ++p)
#pragma unroll
                for(int e = 0; e < Atom::ACC; ++e)
                    halo[(r * CH + cbi * 16 + lane % 16) * OP + p * 16 + Atom::dRow(lane, e)]
                        = K1Elem<T>::store(acc[it][p][e]);
        }
        __syncthreads();
        const long long outPlanes = static_cast<long long>(a.outD) * a.outH * a.outW;
        T* dst = out + (img * a.c + static_cast<long long>(cbt) * CH) * outPlanes
                 + static_cast<long long>(od * a.osd + a.ood) * a.outH * a.outW;
        for(int i = tid; i < TH * CH * TW; i += THREADS)
        {
            const int ow = i % TW, ch = (i / TW) % CH, r = i / (TW * CH);
            const int oh = oh0 + r;
            if(oh < a.ho && ow0 + ow < a.wo)
                dst[ch * outPlanes + static_cast<long long>(oh * a.osh + a.ooh) * a.outW
                    + (ow0 + ow) * a.osw + a.oow]
                    = halo[(r * CH + ch) * OP + ow];
        }
        return;
    }
#pragma unroll
    for(int it = 0; it < ITEMS; ++it)
    {
        const int item = wave + 4 * it;
        const int r = item / CB, cbi = item % CB;
        const int oh = oh0 + r;
        if(oh >= a.ho)
            continue;
        const int cb = cbt * CB + cbi;
#pragma unroll
        for(int p = 0; p < WP; ++p)
#pragma unroll
            for(int e = 0; e < Atom::ACC; ++e)
            {
                const int ow = ow0 + p * 16 + Atom::dRow(lane, e);
                if(ow < a.wo)
                    out[((outPlane * a.outH + oh * a.osh + a.ooh) * a.outW + ow * a.osw + a.oow)
                            * a.c
                        + cb * 16 + lane % 16]
                        = K1Elem<T>::store(acc[it][p][e]);
            }
    }
}

// Phase-fused backward data at stride 2 (rows and columns): one launch writes every dx pixel of a
// depth phase. A workgroup owns 2 TH dx rows x 2 TW dx columns (TW = 16 WP) x CB channel blocks:
// TH x TW positions j of each of the four (row, column) phases, dx[2 j + p]. It stages the dy halo
// once (unit gather stride; R x S are the compact taps a phase reaches, the halo one row and one
// column wider for the phases' differing paddings), runs the four phases one after another
// against their expanded filters (w + phase a.wps), parks each phase's results in an LDS output
// tile, and writes the tile out whole: whole pixels (channels-last) or whole rows of columns
// (channels-first), every dx line written once by one workgroup. Same grid as k4ConvMma over the
// phase-0 positions (a.ho x a.wo).
template <class T, class Atom, int FT, int R, int S, int TH, int WP, int CB, int CF = 0>
__device__ void k4ConvMmaPhased(const K4Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = 4 * WAVE;
    constexpr int TW = 16 * WP;
    constexpr int CH = 16 * CB;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int PITCH = CH + VEC;
    constexpr int HR = TH + R, HC = TW + S; // one more row and column than a phase needs
    constexpr int TAPS = FT * R * S;
    constexpr int KC = 16 / Atom::KA;
    constexpr int ITEMS = TH * CB / 4;
    // The output tile: channels-last [row][column][channel] (pixels padded like the halo),
    // channels-first [row][channel][column] (rows padded by one).
    constexpr int OUT = CF != 0 ? 2 * TH * CH * (2 * TW + 1) : 2 * TH * 2 * TW * PITCH;
    static_assert(Atom::DIM == 16 && 16 % Atom::KA == 0, "16x16 atoms with k-steps dividing 16");
    static_assert(TH * CB % 4 == 0, "whole items a wave");
    typedef typename K1Raw<16>::type Raw;
    __shared__ __attribute__((aligned(16))) T halo[HR * HC * PITCH];
    __shared__ __attribute__((aligned(16))) T tile[OUT];

    const int blocks = a.c / CH;
    long long b = blockIdx.x;
    const int cbt = static_cast<int>(b % blocks);
    b /= blocks;
    const int tws = (a.wo + TW - 1) / TW;
    const int twi = static_cast<int>(b % tws);
    b /= tws;
    const int ths = (a.ho + TH - 1) / TH;
    const int thi = static_cast<int>(b % ths);
    b /= ths;
    const int od = static_cast<int>(b % a.dout);
    const long long img = b / a.dout;
    const int j0 = thi * TH, k0 = twi * TW; // the tile's first phase row and column
    const int padH = a.fph[0] > a.fph[1] ? a.fph[0] : a.fph[1];
    const int padW = a.fpw[0] > a.fpw[1] ? a.fpw[0] : a.fpw[1];
    const int ih0 = j0 - padH, iw0 = k0 - padW;
    const T* in = static_cast<const T*>(a.in);
    const T* w = static_cast<const T*>(a.w);
    T* out = static_cast<T*>(a.out);
    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int row = lane % 16;
    const int kb = Atom::template kBase<Atom::KPL>(lane);

    for(int ph = 0; ph < 4; ++ph)
    {
        const int pr = ph / 2, pc = ph % 2;
        const int ro = padH - a.fph[pr], co = padW - a.fpw[pc]; // the phase's halo offsets
        const T* wp = w + ph * a.wps;
        typename Atom::acc acc[ITEMS][WP];
#pragma unroll
        for(int it = 0; it < ITEMS; ++it)
#pragma unroll
            for(int p = 0; p < WP; ++p)
                acc[it][p] = typename Atom::acc{};
        for(int ft = 0; ft < FT; ++ft)
        {
            const int id = od - a.pd + ft;
            if(static_cast<unsigned>(id) >= static_cast<unsigned>(a.di))
                continue;
            // Restage only when the plane changes (2-D and the first phase stage once).
            if(FT > 1 || ph == 0)
            {
                __syncthreads();
                k4StageHalo<T, CF, HR, HC, CH, PITCH, THREADS>(
                    a, in, halo, img, id, cbt, ih0, iw0, tid);
                __syncthreads();
            }
#pragma unroll
            for(int it = 0; it < ITEMS; ++it)
            {
                const int item = wave + 4 * it;
                const int r = item / CB, cbi = item % CB;
                const int cb = cbt * CB + cbi;
#pragma unroll
                for(int fr = 0; fr < R; ++fr)
#pragma unroll
                    for(int fs = 0; fs < S; ++fs)
                    {
                        const int tap = (ft * R + fr) * S + fs;
                        typename Atom::frag bf[KC];
#pragma unroll
                        for(int q = 0; q < KC; ++q)
                            bf[q] = k1Frag<Atom>(
                                wp + ((static_cast<long long>(cb) * TAPS + tap) * 16 + row) * 16
                                + q * Atom::KA + kb);
#pragma unroll
                        for(int p = 0; p < WP; ++p)
                        {
                            const int hc = p * 16 + row + fs + co, hr = r + fr + ro;
                            const T* src = halo + (hr * HC + hc) * PITCH + cbi * 16 + kb;
#pragma unroll
                            for(int q = 0; q < KC; ++q)
                                acc[it][p] = Atom::mma(
                                    k1Frag<Atom>(src + q * Atom::KA), bf[q], acc[it][p]);
                        }
                    }
            }
        }
        // Park the phase's results: tile row 2 r + pr, column 2 (fragment column) + pc.
#pragma unroll
        for(int it = 0; it < ITEMS; ++it)
        {
            const int item = wave + 4 * it;
            const int r = item / CB, cbi = item % CB;
#pragma unroll
            for(int p = 0; p < WP; ++p)
#pragma unroll
                for(int e = 0; e < Atom::ACC; ++e)
                {
                    const int r2 = 2 * r + pr, c2 = 2 * (p * 16 + Atom::dRow(lane, e)) + pc;
                    const int ch = cbi * 16 + lane % 16;
                    tile[CF != 0 ? (r2 * CH + ch) * (2 * TW + 1) + c2
                                 : (r2 * 2 * TW + c2) * PITCH + ch]
                        = K1Elem<T>::store(acc[it][p][e]);
                }
        }
    }
    __syncthreads();
    const int x0 = 2 * j0, y0 = 2 * k0; // the tile's first dx row and column
    if constexpr(CF != 0)
    {
        const long long outPlanes = static_cast<long long>(a.outD) * a.outH * a.outW;
        T* dst = out + (img * a.c + static_cast<long long>(cbt) * CH) * outPlanes
                 + static_cast<long long>(od * a.osd + a.ood) * a.outH * a.outW;
        for(int i = tid; i < 2 * TH * CH * 2 * TW; i += THREADS)
        {
            const int c2 = i % (2 * TW), ch = (i / (2 * TW)) % CH, r2 = i / (2 * TW * CH);
            if(x0 + r2 < a.outH && y0 + c2 < a.outW)
                dst[ch * outPlanes + static_cast<long long>(x0 + r2) * a.outW + y0 + c2]
                    = tile[(r2 * CH + ch) * (2 * TW + 1) + c2];
        }
    }
    else
    {
        constexpr int VP = CH / VEC;
        const long long outPlane = img * a.outD + od * a.osd + a.ood;
        for(int i = tid; i < 2 * TH * 2 * TW * VP; i += THREADS)
        {
            const int v = i % VP, px = i / VP, c2 = px % (2 * TW), r2 = px / (2 * TW);
            if(x0 + r2 < a.outH && y0 + c2 < a.outW)
                *reinterpret_cast<Raw*>(out
                                        + ((outPlane * a.outH + x0 + r2) * a.outW + y0 + c2) * a.c
                                        + static_cast<long long>(cbt) * CH + v * VEC)
                    = *reinterpret_cast<const Raw*>(tile + px * PITCH + v * VEC);
        }
    }
}

// FMA kernel (no matrix cores: fp32 on RDNA, and any element type anywhere): the same grid,
// halo and expanded filter as the matrix-core kernel, without the block-diagonal zeros. 256
// threads own TH output rows x TW columns x CB channel blocks of one output plane. A thread owns
// OC output channels of one group (chunk q of the tile's CH / OC) at PX consecutive output
// columns of one row; output pixels run fastest across threads, so a wave shares one chunk and
// reads its group-diagonal filter with wave-uniform (scalar) loads. Per depth tap the workgroup
// stages the plane's halo in LDS; per (row, column) tap a thread holds its OC x CG filter values
// and accumulates its PX pixels' CG group inputs. CG is the graph's channels a group.
template <class T, int FT, int R, int S, int ST, int CG, int OC, int TH, int PX, int CB, int CF = 0>
__device__ void k4ConvFma(const K4Args& a)
{
    constexpr int THREADS = 256;
    constexpr int CH = 16 * CB;
    constexpr int NQ = CH / OC; // channel chunks a workgroup
    constexpr int NPIX = THREADS / NQ; // pixel groups (a chunk's threads)
    constexpr int CGS = NPIX / TH; // column groups a row
    constexpr int TW = CGS * PX;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int PITCH = CH + VEC;
    constexpr int HR = (TH - 1) * ST + R, HC = (TW - 1) * ST + S;
    constexpr int TAPS = FT * R * S;
    // A chunk's output channels: GS of one group, NSB such sub-blocks (several groups when
    // OC > CG).
    constexpr int GS = OC <= CG ? OC : CG;
    constexpr int NSB = OC / GS;
    static_assert(16 % CG == 0 && 16 % OC == 0 && (CG % OC == 0 || OC % CG == 0), "whole groups");
    static_assert(NPIX % 64 == 0 && NPIX % TH == 0, "a wave within one chunk, whole rows");
    static_assert(TH * CH * (TW + 1) <= HR * HC * PITCH, "the output tile fits the halo");
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
    b /= ths;
    const int od = static_cast<int>(b % a.dout);
    const long long img = b / a.dout;
    const int oh0 = thi * TH, ow0 = twi * TW;
    const int ih0 = oh0 * ST - a.ph, iw0 = ow0 * ST - a.pw;
    const T* in = static_cast<const T*>(a.in);
    const T* w = static_cast<const T*>(a.w);
    T* out = static_cast<T*>(a.out);
    const int tid = threadIdx.x;
    // The chunk is wave-uniform (NPIX is a multiple of every wave size).
    const int q = __builtin_amdgcn_readfirstlane(tid / NPIX);
    const int pg = tid % NPIX;
    const int r = pg / CGS, col0 = (pg % CGS) * PX;
    const int ch0 = q * OC; // the thread's first output channel in the tile
    const int n0 = ch0 % 16; // ... in its 16-channel block
    // The chunk's block of the expanded filter: w16[block][tap][n][k].
    const T* wq = w + static_cast<long long>(cbt * CB + ch0 / 16) * TAPS * 256;

    float acc[PX][OC];
#pragma unroll
    for(int p = 0; p < PX; ++p)
#pragma unroll
        for(int o = 0; o < OC; ++o)
            acc[p][o] = 0.0f;

    for(int ft = 0; ft < FT; ++ft)
    {
        const int id = od * a.sd - a.pd + ft;
        if(static_cast<unsigned>(id) >= static_cast<unsigned>(a.di))
            continue;
        if(ft > 0)
            __syncthreads();
        k4StageHalo<T, CF, HR, HC, CH, PITCH, THREADS>(a, in, halo, img, id, cbt, ih0, iw0, tid);
        __syncthreads();
#pragma unroll
        for(int fr = 0; fr < R; ++fr)
#pragma unroll
            for(int fs = 0; fs < S; ++fs)
            {
                const T* wt = wq + ((ft * R + fr) * S + fs) * 256 + n0 * 16;
                // wr[o][i]: output o's weight of its group's input i.
                float wr[OC][CG];
#pragma unroll
                for(int o = 0; o < OC; ++o)
#pragma unroll
                    for(int i = 0; i < CG; ++i)
                        wr[o][i] = K1Elem<T>::load(wt[o * 16 + (n0 + o) / CG * CG + i]);
#pragma unroll
                for(int p = 0; p < PX; ++p)
                {
                    const int hr = r * ST + fr, hc = (col0 + p) * ST + fs;
#pragma unroll
                    for(int sb = 0; sb < NSB; ++sb)
                    {
                        const T* src = halo + (hr * HC + hc) * PITCH + (ch0 + sb * GS) / CG * CG;
#pragma unroll
                        for(int i = 0; i < CG; ++i)
                        {
                            const float x = K1Elem<T>::load(src[i]);
#pragma unroll
                            for(int o = 0; o < GS; ++o)
                                acc[p][sb * GS + o]
                                    = fmaf(x, wr[sb * GS + o][i], acc[p][sb * GS + o]);
                        }
                    }
                }
            }
    }

    const int oh = oh0 + r;
    if constexpr(CF != 0)
    {
        constexpr int OP = TW + 1;
        __syncthreads();
#pragma unroll
        for(int p = 0; p < PX; ++p)
#pragma unroll
            for(int o = 0; o < OC; ++o)
                halo[(r * CH + ch0 + o) * OP + col0 + p] = K1Elem<T>::store(acc[p][o]);
        __syncthreads();
        k4StoreTileChannelsFirst<T, TH, TW, CH, THREADS>(a, out, halo, img, od, cbt, oh0, ow0, tid);
        return;
    }
    if(oh >= a.ho)
        return;
    const long long outRow
        = (img * a.outD + od * a.osd + a.ood) * a.outH + static_cast<long long>(oh) * a.osh + a.ooh;
#pragma unroll
    for(int p = 0; p < PX; ++p)
    {
        const int ow = ow0 + col0 + p;
        if(ow < a.wo)
        {
            T* dst = out + (outRow * a.outW + ow * a.osw + a.oow) * a.c
                     + static_cast<long long>(cbt) * CH + ch0;
#pragma unroll
            for(int o = 0; o < OC; ++o)
                dst[o] = K1Elem<T>::store(acc[p][o]);
        }
    }
}

// The wide-group expansion (a.nbi > 0): w16[group][out block][tap][in block][n][k], n and k the
// group-local output and input channels (ob 16 + n, ib 16 + k) of the gather, zero past cg.
__device__ inline void k4ExpandWide(const K4ExpandArgs& a, int taps)
{
    const int nb = a.nbi;
    const long long total = static_cast<long long>(a.c / a.cg) * nb * taps * nb * 256;
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += step)
    {
        const int kk = static_cast<int>(i % 16);
        const int nn = static_cast<int>((i / 16) % 16);
        const int ib = static_cast<int>((i / 256) % nb);
        const int tap = static_cast<int>((i / (256LL * nb)) % taps);
        const int ob = static_cast<int>((i / (256LL * nb * taps)) % nb);
        const int g = static_cast<int>(i / (256LL * nb * taps * nb));
        const int n = ob * 16 + nn, k = ib * 16 + kk;
        const int v = tap / (a.rc * a.sc), q = (tap / a.sc) % a.rc, u = tap % a.sc;
        int ft = a.tBase + a.tStep * v, fr = a.rBase + a.rStep * q, fs = a.sBase + a.sStep * u;
        const bool inside = ft >= 0 && ft < a.t && fr >= 0 && fr < a.r && fs >= 0 && fs < a.s
                            && n < a.cg && k < a.cg;
        if(a.reflect)
        {
            ft = a.t - 1 - ft;
            fr = a.r - 1 - fr;
            fs = a.s - 1 - fs;
        }
        // The graph filter's output channel and its group-local input channel.
        const int oc = g * a.cg + (a.swap ? k : n), ic = a.swap ? n : k;
        const long long src
            = inside ? oc * a.sK + ic * a.sC + ft * a.sT + fr * a.sR + fs * a.sS : 0;
        if(a.elementBytes == 4)
            static_cast<unsigned*>(a.dst)[i]
                = inside ? static_cast<const unsigned*>(a.src)[src] : 0u;
        else
            static_cast<unsigned short*>(a.dst)[i]
                = inside ? static_cast<const unsigned short*>(a.src)[src]
                         : static_cast<unsigned short>(0);
    }
}

// Wide-group matrix-core kernel (16 < cg <= 64): each group's cg channels padded to NBI 16-channel
// blocks. A workgroup of 4 waves owns TH output rows x 16 WP columns x NBO of a group's output
// blocks; it stages the group's input halo (NBI blocks) in LDS per depth tap and each wave
// accumulates its (row, output block) items over every tap and input block, B fragments from the
// dense per-group expansion. Grid: (groups x output-block tiles) x column tiles x row tiles x
// output planes x images. Channels need not be multiples of 16; cg is a multiple of the 16-byte
// vector (a.cg runtime).
// Three-bf16 (P 3, T bf16, TO float): in and w each hold three parts (a.xps and a.wpt elements
// apart); each depth tap stages and multiplies the parts one after another into the same
// accumulators, and the output is written in TO.
template <class T,
          class Atom,
          int FT,
          int R,
          int S,
          int ST,
          int TH,
          int WP,
          int NBI,
          int NBO,
          int CF = 0,
          class TO = T,
          int P = 1>
__device__ void k4ConvMmaWide(const K4Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = 4 * WAVE;
    constexpr int TW = 16 * WP;
    constexpr int CH = 16 * NBI; // the group's padded input channels
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int VP = CH / VEC;
    constexpr int PITCH = CH + VEC;
    constexpr int HR = (TH - 1) * ST + R, HC = (TW - 1) * ST + S;
    constexpr int TAPS = FT * R * S;
    constexpr int KC = 16 / Atom::KA;
    constexpr int RPW = TH / 4; // output rows a wave
    constexpr int OCH = 16 * NBO; // output channels a workgroup
    static_assert(Atom::DIM == 16 && 16 % Atom::KA == 0, "16x16 atoms with k-steps dividing 16");
    static_assert(TH % 4 == 0 && NBI % NBO == 0, "whole rows a wave, whole tiles a group");
    static_assert(CF == 0 || TH * OCH * (TW + 1) * sizeof(TO) <= HR * HC * PITCH * sizeof(T),
                  "output tile fits the halo");
    typedef typename K1Raw<16>::type Raw;
    __shared__ __attribute__((aligned(16))) T halo[HR * HC * PITCH];

    const int groups = a.c / a.cg;
    constexpr int OT = NBI / NBO; // output tiles a group
    long long b = blockIdx.x;
    const int ot = static_cast<int>(b % OT);
    b /= OT;
    const int g = static_cast<int>(b % groups);
    b /= groups;
    const int tws = (a.wo + TW - 1) / TW;
    const int twi = static_cast<int>(b % tws);
    b /= tws;
    const int ths = (a.ho + TH - 1) / TH;
    const int thi = static_cast<int>(b % ths);
    b /= ths;
    const int od = static_cast<int>(b % a.dout);
    const long long img = b / a.dout;
    const int oh0 = thi * TH, ow0 = twi * TW;
    const int ih0 = oh0 * ST - a.ph, iw0 = ow0 * ST - a.pw;
    const T* in0 = static_cast<const T*>(a.in);
    const T* w0 = static_cast<const T*>(a.w)
                  + static_cast<long long>(g * NBI + ot * NBO) * TAPS * NBI * 256;
    TO* out = static_cast<TO*>(a.out);
    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int row = lane % 16;
    const int kb = Atom::template kBase<Atom::KPL>(lane);
    const int c0 = g * a.cg; // the group's first channel

    // A wave's register tile: RPW rows x WP 16-column fragments x NBO output blocks; each A
    // fragment serves NBO multiplies, each B fragment RPW x WP.
    typename Atom::acc acc[RPW][WP][NBO];
#pragma unroll
    for(int rr = 0; rr < RPW; ++rr)
#pragma unroll
        for(int p = 0; p < WP; ++p)
#pragma unroll
            for(int ob = 0; ob < NBO; ++ob)
                acc[rr][p][ob] = typename Atom::acc{};

    // Depth taps, each its P parts one after another.
    for(int step = 0; step < FT * P; ++step)
    {
        const int ft = step / P, part = step % P;
        const int id = od * a.sd - a.pd + ft;
        if(static_cast<unsigned>(id) >= static_cast<unsigned>(a.di))
            continue;
        if(ft > 0 || part > 0)
            __syncthreads();
        const T* in = P > 1 ? in0 + part * a.xps : in0;
        const T* w = P > 1 ? w0 + part * a.wpt : w0;
        for(int i = tid; i < HR * HC * VP; i += THREADS)
        {
            int v, hc, hr;
            if constexpr(CF != 0)
            {
                hc = i % HC;
                hr = (i / HC) % HR;
                v = i / (HC * HR);
            }
            else
            {
                v = i % VP;
                const int p = i / VP;
                hc = p % HC;
                hr = p / HC;
            }
            const int ih = ih0 + hr, iw = iw0 + hc;
            const bool ok = static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi)
                            && static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi)
                            && v * VEC < a.cg;
            const int sih = ok ? ih : 0, siw = ok ? iw : 0, sv = ok ? v : 0;
            Raw val;
            if constexpr(CF != 0)
            {
                // VEC channel planes of one pixel, one load each (channels past cg read as 0).
                const long long planes = static_cast<long long>(a.di) * a.hi * a.wi;
                const T* at = in + (img * a.c + c0 + sv * VEC) * planes
                              + (static_cast<long long>(id) * a.hi + sih) * a.wi + siw;
                union
                {
                    Raw raw;
                    T e[VEC];
                } u;
#pragma unroll
                for(int j = 0; j < VEC; ++j)
                    u.e[j] = sv * VEC + j < a.cg ? at[j * planes] : T{};
                val = u.raw;
            }
            else
            {
                val = *reinterpret_cast<const Raw*>(
                    in + (((img * a.di + id) * a.hi + sih) * a.wi + siw) * a.c + c0 + sv * VEC);
            }
            if(!ok)
                val = Raw{};
            *reinterpret_cast<Raw*>(halo + (hr * HC + k4HaloColumn<ST, HC>(hc)) * PITCH + v * VEC)
                = val;
        }
        __syncthreads();
#pragma unroll
        for(int fr = 0; fr < R; ++fr)
#pragma unroll
            for(int fs = 0; fs < S; ++fs)
            {
                const int tap = (ft * R + fr) * S + fs;
#pragma unroll
                for(int ib = 0; ib < NBI; ++ib)
                {
                    typename Atom::frag bf[NBO][KC];
#pragma unroll
                    for(int ob = 0; ob < NBO; ++ob)
#pragma unroll
                        for(int q = 0; q < KC; ++q)
                            bf[ob][q] = k1Frag<Atom>(
                                w
                                + (((static_cast<long long>(ob) * TAPS + tap) * NBI + ib) * 16
                                   + row)
                                      * 16
                                + q * Atom::KA + kb);
#pragma unroll
                    for(int rr = 0; rr < RPW; ++rr)
                    {
                        const int r = wave + 4 * rr;
#pragma unroll
                        for(int p = 0; p < WP; ++p)
                        {
                            const int hc = (p * 16 + row) * ST + fs, hr = r * ST + fr;
                            const T* src = halo + (hr * HC + k4HaloColumn<ST, HC>(hc)) * PITCH
                                           + ib * 16 + kb;
#pragma unroll
                            for(int q = 0; q < KC; ++q)
                            {
                                const auto af = k1Frag<Atom>(src + q * Atom::KA);
#pragma unroll
                                for(int ob = 0; ob < NBO; ++ob)
                                    acc[rr][p][ob] = Atom::mma(af, bf[ob][q], acc[rr][p][ob]);
                            }
                        }
                    }
                }
            }
    }

    const long long outPlane = img * a.outD + od * a.osd + a.ood;
    if constexpr(CF != 0)
    {
        constexpr int OP = TW + 1;
        TO* tile = reinterpret_cast<TO*>(halo);
        __syncthreads();
#pragma unroll
        for(int rr = 0; rr < RPW; ++rr)
#pragma unroll
            for(int ob = 0; ob < NBO; ++ob)
#pragma unroll
                for(int p = 0; p < WP; ++p)
#pragma unroll
                    for(int e = 0; e < Atom::ACC; ++e)
                        tile[((wave + 4 * rr) * OCH + ob * 16 + lane % 16) * OP + p * 16
                             + Atom::dRow(lane, e)]
                            = K1Elem<TO>::store(acc[rr][p][ob][e]);
        __syncthreads();
        const long long outPlanes = static_cast<long long>(a.outD) * a.outH * a.outW;
        for(int i = tid; i < TH * OCH * TW; i += THREADS)
        {
            const int ow = i % TW, ch = (i / TW) % OCH, r = i / (TW * OCH);
            const int oh = oh0 + r, gch = ot * OCH + ch;
            if(oh < a.ho && ow0 + ow < a.wo && gch < a.cg)
                out[(img * a.c + c0 + gch) * outPlanes
                    + (static_cast<long long>(od * a.osd + a.ood) * a.outH + oh * a.osh + a.ooh)
                          * a.outW
                    + (ow0 + ow) * a.osw + a.oow]
                    = tile[(r * OCH + ch) * OP + ow];
        }
        return;
    }
#pragma unroll
    for(int rr = 0; rr < RPW; ++rr)
#pragma unroll
        for(int ob = 0; ob < NBO; ++ob)
        {
            const int oh = oh0 + wave + 4 * rr;
            const int gch = (ot * NBO + ob) * 16 + lane % 16;
            if(oh >= a.ho || gch >= a.cg)
                continue;
#pragma unroll
            for(int p = 0; p < WP; ++p)
#pragma unroll
                for(int e = 0; e < Atom::ACC; ++e)
                {
                    const int ow = ow0 + p * 16 + Atom::dRow(lane, e);
                    if(ow < a.wo)
                        out[((outPlane * a.outH + oh * a.osh + a.ooh) * a.outW + ow * a.osw + a.oow)
                                * a.c
                            + c0 + gch]
                            = K1Elem<TO>::store(acc[rr][p][ob][e]);
                }
        }
}

// Phase-fused backward data at stride 2 for wide groups: k4ConvMmaPhased's four row and column
// phases into an LDS output tile, with k4ConvMmaWide's group padding and register tile. A
// workgroup owns 2 TH x 2 TW dx pixels (TW = 16 WP) x NBO of a group's output blocks.
template <class T, class Atom, int FT, int R, int S, int TH, int WP, int NBI, int NBO, int CF = 0>
__device__ void k4ConvMmaWidePhased(const K4Args& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = 4 * WAVE;
    constexpr int TW = 16 * WP;
    constexpr int CH = 16 * NBI;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int VP = CH / VEC;
    constexpr int PITCH = CH + VEC;
    constexpr int HR = TH + R, HC = TW + S;
    constexpr int TAPS = FT * R * S;
    constexpr int KC = 16 / Atom::KA;
    constexpr int RPW = TH / 4;
    constexpr int OCH = 16 * NBO;
    constexpr int OPITCH = OCH + VEC;
    constexpr int OUT = CF != 0 ? 2 * TH * OCH * (2 * TW + 1) : 2 * TH * 2 * TW * OPITCH;
    static_assert(Atom::DIM == 16 && 16 % Atom::KA == 0, "16x16 atoms with k-steps dividing 16");
    static_assert(TH % 4 == 0 && NBI % NBO == 0, "whole rows a wave, whole tiles a group");
    typedef typename K1Raw<16>::type Raw;
    __shared__ __attribute__((aligned(16))) T halo[HR * HC * PITCH];
    __shared__ __attribute__((aligned(16))) T tile[OUT];

    const int groups = a.c / a.cg;
    constexpr int OT = NBI / NBO;
    long long b = blockIdx.x;
    const int ot = static_cast<int>(b % OT);
    b /= OT;
    const int g = static_cast<int>(b % groups);
    b /= groups;
    const int tws = (a.wo + TW - 1) / TW;
    const int twi = static_cast<int>(b % tws);
    b /= tws;
    const int ths = (a.ho + TH - 1) / TH;
    const int thi = static_cast<int>(b % ths);
    b /= ths;
    const int od = static_cast<int>(b % a.dout);
    const long long img = b / a.dout;
    const int j0 = thi * TH, k0 = twi * TW;
    const int padH = a.fph[0] > a.fph[1] ? a.fph[0] : a.fph[1];
    const int padW = a.fpw[0] > a.fpw[1] ? a.fpw[0] : a.fpw[1];
    const int ih0 = j0 - padH, iw0 = k0 - padW;
    const T* in = static_cast<const T*>(a.in);
    T* out = static_cast<T*>(a.out);
    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int row = lane % 16;
    const int kb = Atom::template kBase<Atom::KPL>(lane);
    const int c0 = g * a.cg;

    for(int ph = 0; ph < 4; ++ph)
    {
        const int pr = ph / 2, pc = ph % 2;
        const int ro = padH - a.fph[pr], co = padW - a.fpw[pc];
        const T* w = static_cast<const T*>(a.w) + ph * a.wps
                     + static_cast<long long>(g * NBI + ot * NBO) * TAPS * NBI * 256;
        typename Atom::acc acc[RPW][WP][NBO];
#pragma unroll
        for(int rr = 0; rr < RPW; ++rr)
#pragma unroll
            for(int p = 0; p < WP; ++p)
#pragma unroll
                for(int ob = 0; ob < NBO; ++ob)
                    acc[rr][p][ob] = typename Atom::acc{};
        for(int ft = 0; ft < FT; ++ft)
        {
            const int id = od - a.pd + ft;
            if(static_cast<unsigned>(id) >= static_cast<unsigned>(a.di))
                continue;
            if(FT > 1 || ph == 0)
            {
                __syncthreads();
                for(int i = tid; i < HR * HC * VP; i += THREADS)
                {
                    int v, hc, hr;
                    if constexpr(CF != 0)
                    {
                        hc = i % HC;
                        hr = (i / HC) % HR;
                        v = i / (HC * HR);
                    }
                    else
                    {
                        v = i % VP;
                        const int p = i / VP;
                        hc = p % HC;
                        hr = p / HC;
                    }
                    const int ih = ih0 + hr, iw = iw0 + hc;
                    const bool ok = static_cast<unsigned>(ih) < static_cast<unsigned>(a.hi)
                                    && static_cast<unsigned>(iw) < static_cast<unsigned>(a.wi)
                                    && v * VEC < a.cg;
                    const int sih = ok ? ih : 0, siw = ok ? iw : 0, sv = ok ? v : 0;
                    Raw val;
                    if constexpr(CF != 0)
                    {
                        const long long planes = static_cast<long long>(a.di) * a.hi * a.wi;
                        const T* at = in + (img * a.c + c0 + sv * VEC) * planes
                                      + (static_cast<long long>(id) * a.hi + sih) * a.wi + siw;
                        union
                        {
                            Raw raw;
                            T e[VEC];
                        } u;
#pragma unroll
                        for(int j = 0; j < VEC; ++j)
                            u.e[j] = sv * VEC + j < a.cg ? at[j * planes] : T{};
                        val = u.raw;
                    }
                    else
                    {
                        val = *reinterpret_cast<const Raw*>(
                            in + (((img * a.di + id) * a.hi + sih) * a.wi + siw) * a.c + c0
                            + sv * VEC);
                    }
                    if(!ok)
                        val = Raw{};
                    *reinterpret_cast<Raw*>(halo + (hr * HC + hc) * PITCH + v * VEC) = val;
                }
                __syncthreads();
            }
#pragma unroll
            for(int fr = 0; fr < R; ++fr)
#pragma unroll
                for(int fs = 0; fs < S; ++fs)
                {
                    const int tap = (ft * R + fr) * S + fs;
#pragma unroll
                    for(int ib = 0; ib < NBI; ++ib)
                    {
                        typename Atom::frag bf[NBO][KC];
#pragma unroll
                        for(int ob = 0; ob < NBO; ++ob)
#pragma unroll
                            for(int q = 0; q < KC; ++q)
                                bf[ob][q] = k1Frag<Atom>(
                                    w
                                    + (((static_cast<long long>(ob) * TAPS + tap) * NBI + ib) * 16
                                       + row)
                                          * 16
                                    + q * Atom::KA + kb);
#pragma unroll
                        for(int rr = 0; rr < RPW; ++rr)
                        {
                            const int r = wave + 4 * rr;
#pragma unroll
                            for(int p = 0; p < WP; ++p)
                            {
                                const int hc = p * 16 + row + fs + co, hr = r + fr + ro;
                                const T* src = halo + (hr * HC + hc) * PITCH + ib * 16 + kb;
#pragma unroll
                                for(int q = 0; q < KC; ++q)
                                {
                                    const auto af = k1Frag<Atom>(src + q * Atom::KA);
#pragma unroll
                                    for(int ob = 0; ob < NBO; ++ob)
                                        acc[rr][p][ob] = Atom::mma(af, bf[ob][q], acc[rr][p][ob]);
                                }
                            }
                        }
                    }
                }
        }
        // Park the phase: tile row 2 r + pr, column 2 (fragment column) + pc.
#pragma unroll
        for(int rr = 0; rr < RPW; ++rr)
#pragma unroll
            for(int ob = 0; ob < NBO; ++ob)
#pragma unroll
                for(int p = 0; p < WP; ++p)
#pragma unroll
                    for(int e = 0; e < Atom::ACC; ++e)
                    {
                        const int r2 = 2 * (wave + 4 * rr) + pr;
                        const int c2 = 2 * (p * 16 + Atom::dRow(lane, e)) + pc;
                        const int ch = ob * 16 + lane % 16;
                        tile[CF != 0 ? (r2 * OCH + ch) * (2 * TW + 1) + c2
                                     : (r2 * 2 * TW + c2) * OPITCH + ch]
                            = K1Elem<T>::store(acc[rr][p][ob][e]);
                    }
    }
    __syncthreads();
    const int x0 = 2 * j0, y0 = 2 * k0;
    if constexpr(CF != 0)
    {
        const long long outPlanes = static_cast<long long>(a.outD) * a.outH * a.outW;
        for(int i = tid; i < 2 * TH * OCH * 2 * TW; i += THREADS)
        {
            const int c2 = i % (2 * TW), ch = (i / (2 * TW)) % OCH, r2 = i / (2 * TW * OCH);
            const int gch = ot * OCH + ch;
            if(x0 + r2 < a.outH && y0 + c2 < a.outW && gch < a.cg)
                out[(img * a.c + c0 + gch) * outPlanes
                    + (static_cast<long long>(od * a.osd + a.ood) * a.outH + x0 + r2) * a.outW + y0
                    + c2]
                    = tile[(r2 * OCH + ch) * (2 * TW + 1) + c2];
        }
    }
    else
    {
        // Whole 16-byte vectors of a pixel's channels (cg is a multiple of the vector).
        constexpr int OVP = OCH / VEC;
        const long long outPlane = img * a.outD + od * a.osd + a.ood;
        for(int i = tid; i < 2 * TH * 2 * TW * OVP; i += THREADS)
        {
            const int v = i % OVP, px = i / OVP, c2 = px % (2 * TW), r2 = px / (2 * TW);
            const int gch = ot * OCH + v * VEC;
            if(x0 + r2 < a.outH && y0 + c2 < a.outW && gch < a.cg)
                *reinterpret_cast<Raw*>(
                    out + ((outPlane * a.outH + x0 + r2) * a.outW + y0 + c2) * a.c + c0 + gch)
                    = *reinterpret_cast<const Raw*>(tile + px * OPITCH + v * VEC);
        }
    }
}

// The filter expansion (K4ExpandArgs), one element per thread, grid-strided.
__device__ inline void k4Expand(const K4ExpandArgs& a)
{
    const int taps = a.tc * a.rc * a.sc;
    if(a.nbi > 0)
    {
        k4ExpandWide(a, taps);
        return;
    }
    const long long total = static_cast<long long>(a.c / 16) * taps * 256;
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += step)
    {
        const int kk = static_cast<int>(i % 16);
        const int nn = static_cast<int>((i / 16) % 16);
        const int tap = static_cast<int>((i / 256) % taps);
        const int cb = static_cast<int>(i / (256LL * taps));
        const int n = cb * 16 + nn, k = cb * 16 + kk;
        const int v = tap / (a.rc * a.sc), q = (tap / a.sc) % a.rc, u = tap % a.sc;
        int ft = a.tBase + a.tStep * v, fr = a.rBase + a.rStep * q, fs = a.sBase + a.sStep * u;
        const bool inside = ft >= 0 && ft < a.t && fr >= 0 && fr < a.r && fs >= 0 && fs < a.s;
        if(a.reflect)
        {
            ft = a.t - 1 - ft;
            fr = a.r - 1 - fr;
            fs = a.s - 1 - fs;
        }
        // The graph filter's output channel and its group-local input channel.
        const int oc = a.swap ? k : n, ic = a.swap ? n : k;
        const bool own = inside && oc / a.cg == ic / a.cg;
        const long long src
            = own ? oc * a.sK + (ic % a.cg) * a.sC + ft * a.sT + fr * a.sR + fs * a.sS : 0;
        if(a.elementBytes == 4)
            static_cast<unsigned*>(a.dst)[i] = own ? static_cast<const unsigned*>(a.src)[src] : 0u;
        else
            static_cast<unsigned short*>(a.dst)[i]
                = own ? static_cast<const unsigned short*>(a.src)[src]
                      : static_cast<unsigned short>(0);
    }
}

// ---------------------------------------------------------------------------------------------
// Backward weights (CONV_GROUP_WRW, K4WrwArgs).

// One staging of NR x NC pixels of plane `plane` of image `img` (a tensor of `planes` x h x w
// pixels and c channels, packed channels-first (CF) or channels-last), from row r0 and column
// c0, CH channels from ch0, zero outside the plane: load() issues all of a thread's loads into
// registers (straight-line, so none waits on another), store() puts them in LDS [pixel][PITCH];
// a tile's loads are issued before the barrier that ends the previous tile's multiplies.
// Channels-last: 16-byte loads along the channels. Channels-first: a thread gathers VEC channels
// of one pixel (one load a channel, consecutive threads consecutive columns) and stores them as
// one vector.
// MASKC: channels at or past cmax (counted from ch0) read as zero, never loaded from (wide
// groups padded to whole blocks).
template <class T, int CF, int NR, int NC, int CH, int PITCH, int THREADS, bool MASKC = false>
struct K4WrwStaging
{
    typedef typename K1Raw<16>::type Raw;
    static constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    static constexpr int VP = CH / VEC; // vectors a pixel
    static constexpr int N = VP * NR * NC;
    static constexpr int IT = (N + THREADS - 1) / THREADS;
    Raw v[IT];

    // The LDS element offset of unit i.
    __device__ static int slot(int i)
    {
        if constexpr(CF != 0)
            return (i % (NC * NR)) * PITCH + i / (NC * NR) * VEC;
        else
            return (i / VP) * PITCH + i % VP * VEC;
    }

    __device__ void load(const T* src,
                         long long img,
                         int plane,
                         int planes,
                         int h,
                         int w,
                         int c,
                         int ch0,
                         int r0,
                         int c0,
                         int tid,
                         int cmax = CH)
    {
        const long long hw = static_cast<long long>(h) * w;
#pragma unroll
        for(int it = 0; it < IT; ++it)
        {
            // Straight-line: a unit past N loads unit 0 (stored by no one), so every load issues
            // before the first use.
            const bool live = N % THREADS == 0 || tid + it * THREADS < N;
            const int i = live ? tid + it * THREADS : 0;
            int pc, pr, vv;
            if constexpr(CF != 0)
            {
                pc = i % NC;
                pr = (i / NC) % NR;
                vv = i / (NC * NR);
            }
            else
            {
                vv = i % VP;
                pc = (i / VP) % NC;
                pr = i / VP / NC;
            }
            const int ih = r0 + pr, iw = c0 + pc;
            // Clamped address, then zero: no per-load branch.
            const bool ok = static_cast<unsigned>(ih) < static_cast<unsigned>(h)
                            && static_cast<unsigned>(iw) < static_cast<unsigned>(w);
            const int sh = ok ? ih : 0, sw = ok ? iw : 0;
            if constexpr(CF != 0)
            {
                const long long image = static_cast<long long>(planes) * hw;
                const T* at = src + (img * c + ch0) * image + plane * hw
                              + static_cast<long long>(sh) * w + sw;
                union
                {
                    Raw raw;
                    T e[VEC];
                } val;
#pragma unroll
                for(int j = 0; j < VEC; ++j)
                {
                    if constexpr(MASKC)
                    {
                        const bool in = vv * VEC + j < cmax;
                        const T e = at[(in ? vv * VEC + j : 0) * image];
                        val.e[j] = in ? e : T{};
                    }
                    else
                    {
                        val.e[j] = at[(vv * VEC + j) * image];
                    }
                }
                v[it] = ok ? val.raw : Raw{};
            }
            else
            {
                const bool in = !MASKC || vv * VEC < cmax; // whole vectors: cmax a multiple
                const long long at
                    = (((img * planes + plane) * h + sh) * w + sw) * c + ch0 + (in ? vv * VEC : 0);
                const Raw val = *reinterpret_cast<const Raw*>(src + at);
                v[it] = ok && in ? val : Raw{};
            }
        }
    }

    __device__ void store(T* dst, int tid) const
    {
#pragma unroll
        for(int it = 0; it < IT; ++it)
        {
            const int i = tid + it * THREADS;
            if(N % THREADS != 0 && i >= N)
                continue;
            *reinterpret_cast<Raw*>(dst + slot(i)) = v[it];
        }
    }
};

// Element i of a fragment (a scalar fragment when the atom takes one element a lane).
template <class Atom, class T>
__device__ __forceinline__ void k4FragSet(typename Atom::frag& f, int i, T v)
{
    if constexpr(Atom::KPL == 1)
        f = __builtin_bit_cast(typename Atom::elem, v);
    else
        f[i] = __builtin_bit_cast(typename Atom::elem, v);
}

// The LDS reads of a wrw fragment from a channels-last [pixel][channel] tile: a lane's KPL
// positions of its channel. Element by element (NP = KPL reads of one element); on gfx950 for
// 16-bit atoms of 4 or 8 positions a lane, by the transpose read ds_read_b64_tr_b16 (NP = KPL / 4
// reads): in each 16-lane group, source lane s reads 4 channels 4 (s % 4) .. at the group's
// position s / 4 and the read hands lane j channel j at the 4 positions. Every lane supplies its
// own address, so the positions may be any pixels (flattened chunks, the zero pixel).
#if defined(__gfx950__)
#define K4_WRW_TR 1
#else
#define K4_WRW_TR 0
#endif
template <class Atom, class T>
struct K4WrwFrag
{
    static constexpr int KPL = Atom::KPL;
    static constexpr bool TR = K4_WRW_TR && sizeof(T) == 2 && (KPL == 4 || KPL == 8);
    static constexpr int NP = TR ? KPL / 4 : KPL; // reads (and positions) a lane addresses
    // The fragment position (0 .. KPL - 1 past the lane's kBase) read i addresses.
    __device__ static int pos(int lane, int i)
    {
        return TR ? 4 * i + (lane % 16) / 4 : i;
    }
    // The lane's channel within its 16-channel block.
    __device__ static int chan(int lane)
    {
        return TR ? 4 * (lane % 4) : lane % 16;
    }
    // The fragment from element offsets off[i] (pixel offset + block base + chan).
    __device__ static typename Atom::frag read(const T* sm, const int (&off)[NP])
    {
        typename Atom::frag f;
        if constexpr(TR)
        {
#if defined(__gfx950__)
            typedef short s4 __attribute__((ext_vector_type(4)));
            typedef __attribute__((address_space(3))) s4 lds_s4;
            s4 v[NP];
#pragma unroll
            for(int i = 0; i < NP; ++i)
                v[i] = __builtin_amdgcn_ds_read_tr16_b64_v4i16((lds_s4*)(sm + off[i]));
            if constexpr(NP == 1)
                f = __builtin_bit_cast(typename Atom::frag, v[0]);
            else
                f = __builtin_bit_cast(typename Atom::frag,
                                       __builtin_shufflevector(v[0], v[1], 0, 1, 2, 3, 4, 5, 6, 7));
#endif
        }
        else
        {
#pragma unroll
            for(int i = 0; i < NP; ++i)
                k4FragSet<Atom>(f, i, sm[off[i]]);
        }
        return f;
    }
};

// Matrix-core backward weights. Per 16-channel block the block-diagonal product
//   D[k][c] += A[k][q] B[c][q], A = dy, B = x at the tap's shift, q the dy positions,
// keeps the 16 / cg diagonal cg x cg blocks (whole groups) of each 16 x 16 result. A workgroup
// of 4 waves owns CB channel blocks, one depth tap (ft) and a split's run of tiles; per tile it
// stages the TH x TW dy positions and the x halo they reach ((TH - 1) ST + R rows x (TW - 1) ST
// + S columns) channels-last in LDS, then multiplies chunks of KA positions (the tile's positions
// flattened, so a chunk may span rows): a lane reads its KPL positions element by element, the x
// address of each its position's halo pixel plus the tap's offset. Positions past the tile's
// valid rows and columns read a zero pixel in both operands. The tile's global loads are issued
// before the barrier that ends the previous tile's multiplies. Waves: CB >= 4, CB / 4 blocks a
// wave; else 4 / CB waves a block, each taking every (4 / CB)-th chunk, their sums added in wave
// order through LDS at the end. The workgroup writes its split's partial slot.
template <class T, class Atom, int R, int S, int ST, int TH, int TW, int CB, int CF>
__device__ void k4WrwMma(const K4WrwArgs& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = 4 * WAVE;
    constexpr int CH = 16 * CB;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int PITCH = CH + VEC;
    constexpr int HR = (TH - 1) * ST + R, HC = (TW - 1) * ST + S;
    constexpr int P = TH * TW;
    constexpr int KA = Atom::KA, KPL = Atom::KPL;
    constexpr int NCH = (P + KA - 1) / KA; // chunks a tile
    constexpr int TAPS = R * S;
    constexpr int WPB = CB >= 4 ? 1 : 4 / CB; // waves a block
    constexpr int BPW = CB >= 4 ? CB / 4 : 1; // blocks a wave
    constexpr int XPIX = HR * HC; // the halo's pixels; then the dy tile's, then one zero pixel
    constexpr int ZERO = (XPIX + P) * PITCH;
    // The waves' sums of a block (WPB > 1), one wave a block at a time, TPR taps a round: as many
    // as the staging's LDS holds (at least one).
    constexpr int STAGE_BYTES = (XPIX + P + 1) * PITCH * static_cast<int>(sizeof(T));
    constexpr int TAP_BYTES = CB * Atom::ACC * WAVE * 4;
    constexpr int TPR = STAGE_BYTES / TAP_BYTES >= TAPS ? TAPS
                        : STAGE_BYTES / TAP_BYTES > 0   ? STAGE_BYTES / TAP_BYTES
                                                        : 1;
    constexpr int RED = WPB > 1 ? TPR * TAP_BYTES : 0;
    constexpr int SM = (STAGE_BYTES > RED ? STAGE_BYTES : RED) / static_cast<int>(sizeof(T));
    static_assert(Atom::DIM == 16, "16x16 atoms");
    static_assert(CB == 1 || CB == 2 || CB % 4 == 0, "whole blocks a wave or waves a block");
    __shared__ __attribute__((aligned(16))) T sm[SM];

    const int cTiles = a.c / CH;
    long long b = blockIdx.x;
    const int cbt = static_cast<int>(b % cTiles);
    b /= cTiles;
    const int ft = static_cast<int>(b % a.ft);
    const long long split = b / a.ft;
    const long long begin = split * a.tiles / a.splits;
    const long long end = (split + 1) * a.tiles / a.splits;
    const T* x = static_cast<const T*>(a.x);
    const T* dy = static_cast<const T*>(a.dy);
    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int c16 = lane % 16;
    const int kb = Atom::template kBase<KPL>(lane);
    const int sub = wave % WPB;
    const int blk0 = wave / WPB * BPW; // the wave's first block within the tile

    for(int i = tid; i < PITCH; i += THREADS)
        sm[ZERO + i] = T{};

    typename Atom::acc acc[BPW][TAPS];
#pragma unroll
    for(int bb = 0; bb < BPW; ++bb)
#pragma unroll
        for(int tap = 0; tap < TAPS; ++tap)
            acc[bb][tap] = typename Atom::acc{};

    // The first tile's coordinates; later tiles step them (no divisions a tile).
    int twi, thi, od;
    long long img;
    {
        long long t = begin;
        twi = static_cast<int>(t % a.tilesW);
        t /= a.tilesW;
        thi = static_cast<int>(t % a.tilesH);
        t /= a.tilesH;
        od = static_cast<int>(t % a.dout);
        img = t / a.dout;
    }
    K4WrwStaging<T, CF, HR, HC, CH, PITCH, THREADS> xs;
    K4WrwStaging<T, CF, TH, TW, CH, PITCH, THREADS> ys;
    for(long long ti = begin; ti < end; ++ti)
    {
        const int id = od * a.sd - a.pd + ft;
        const int oh0 = thi * TH, ow0 = twi * TW;
        const long long tImg = img;
        const int tOd = od;
        if(++twi == a.tilesW)
        {
            twi = 0;
            if(++thi == a.tilesH)
            {
                thi = 0;
                if(++od == a.dout)
                {
                    od = 0;
                    ++img;
                }
            }
        }
        // A plane past the input contributes nothing (uniform across the workgroup).
        if(static_cast<unsigned>(id) >= static_cast<unsigned>(a.di))
            continue;
        // All of the tile's loads, then its LDS stores.
        xs.load(
            x, tImg, id, a.di, a.hi, a.wi, a.c, cbt * CH, oh0 * ST - a.ph, ow0 * ST - a.pw, tid);
        ys.load(dy, tImg, tOd, a.dout, a.ho, a.wo, a.c, cbt * CH, oh0, ow0, tid);
        __syncthreads(); // the previous tile's reads are done
        xs.store(sm, tid);
        ys.store(sm + XPIX * PITCH, tid);
        __syncthreads();
        const int vh = a.ho - oh0, vw = a.wo - ow0; // the tile's valid rows and columns
#pragma unroll
        for(int jj = 0; jj < (NCH + WPB - 1) / WPB; ++jj)
        {
            const int j = jj * WPB + sub;
            if(NCH % WPB != 0 && j >= NCH)
                break;
            typedef K4WrwFrag<Atom, T> F;
            int dyo[F::NP], xo[F::NP];
            bool ok[F::NP];
#pragma unroll
            for(int i = 0; i < F::NP; ++i)
            {
                const int p = j * KA + kb + F::pos(lane, i);
                const int r = p / TW, q = p % TW;
                ok[i] = p < P && r < vh && q < vw;
                dyo[i] = ok[i] ? (XPIX + p) * PITCH : ZERO;
                xo[i] = (r * ST * HC + q * ST) * PITCH;
            }
#pragma unroll
            for(int bb = 0; bb < BPW; ++bb)
            {
                const int cbo = (blk0 + bb) * 16 + F::chan(lane);
                int o[F::NP];
#pragma unroll
                for(int i = 0; i < F::NP; ++i)
                    o[i] = dyo[i] + cbo;
                const typename Atom::frag af = F::read(sm, o);
#pragma unroll
                for(int fr = 0; fr < R; ++fr)
#pragma unroll
                    for(int fs = 0; fs < S; ++fs)
                    {
                        const int tapo = (fr * HC + fs) * PITCH;
#pragma unroll
                        for(int i = 0; i < F::NP; ++i)
                            o[i] = (ok[i] ? xo[i] + tapo : ZERO) + cbo;
                        const typename Atom::frag bf = F::read(sm, o);
                        acc[bb][fr * S + fs] = Atom::mma(af, bf, acc[bb][fr * S + fs]);
                    }
            }
        }
    }

    if constexpr(WPB > 1)
    {
        // Wave sub 0 of each block adds the others' sums in wave order, TPR taps a round.
        float* red = reinterpret_cast<float*>(sm);
        const int blk = wave / WPB;
#pragma unroll
        for(int t0 = 0; t0 < TAPS; t0 += TPR)
#pragma unroll
            for(int s = 1; s < WPB; ++s)
            {
                __syncthreads();
                if(sub == s)
#pragma unroll
                    for(int t = 0; t < TPR && t0 + t < TAPS; ++t)
#pragma unroll
                        for(int e = 0; e < Atom::ACC; ++e)
                            red[((t * CB + blk) * Atom::ACC + e) * WAVE + lane] = acc[0][t0 + t][e];
                __syncthreads();
                if(sub == 0)
#pragma unroll
                    for(int t = 0; t < TPR && t0 + t < TAPS; ++t)
#pragma unroll
                        for(int e = 0; e < Atom::ACC; ++e)
                            acc[0][t0 + t][e]
                                += red[((t * CB + blk) * Atom::ACC + e) * WAVE + lane];
            }
        if(sub != 0)
            return;
    }
    // The diagonal (in-group) entries into the split's slot: [g][k][c][tap].
    const int taps = a.ft * TAPS;
    float* out = a.partial + split * a.partialElems;
#pragma unroll
    for(int bb = 0; bb < BPW; ++bb)
    {
        const int ch = cbt * CH + (blk0 + bb) * 16; // the block's first channel
#pragma unroll
        for(int tap = 0; tap < TAPS; ++tap)
#pragma unroll
            for(int e = 0; e < Atom::ACC; ++e)
            {
                const int k = Atom::dRow(lane, e);
                if(k / a.cg == c16 / a.cg)
                {
                    const int g = (ch + k) / a.cg;
                    out[((static_cast<long long>(g) * a.cg + k % a.cg) * a.cg + c16 % a.cg) * taps
                        + ft * TAPS + tap]
                        = acc[bb][tap][e];
                }
            }
    }
}

// Wide-group matrix-core backward weights (16 < cg <= 64, cg a multiple of 8; the catalog ships
// NBI 2, groups of 24 and 32): a group's cg
// channels padded to NBI 16-channel blocks (staged zero past cg). A workgroup owns one group's
// NBO dy blocks (output tile ot of NBI / NBO) against all NBI x blocks, one depth tap and a
// split's run of tiles; wave w multiplies x block w % NBI against the NBO dy blocks (the tap's B
// fragment serves all NBO), every (NBI < 4: 4 / NBI)-th chunk, the waves of an x block added in
// wave order through LDS at the end (NBI 3: the fourth wave idles). Dense per group: every entry
// of a block product within cg is a dw entry. Same tiles, flattened chunks and zero pixel as
// k4WrwMma; partials [split][g][k][c][tap].
template <class T, class Atom, int R, int S, int ST, int TH, int TW, int NBI, int NBO, int CF>
__device__ void k4WrwMmaWide(const K4WrwArgs& a)
{
    constexpr int WAVE = Atom::WAVE;
    constexpr int THREADS = 4 * WAVE;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int XCH = 16 * NBI, YCH = 16 * NBO;
    constexpr int XPITCH = XCH + VEC, YPITCH = YCH + VEC;
    constexpr int HR = (TH - 1) * ST + R, HC = (TW - 1) * ST + S;
    constexpr int P = TH * TW;
    constexpr int KA = Atom::KA, KPL = Atom::KPL;
    constexpr int NCH = (P + KA - 1) / KA;
    constexpr int TAPS = R * S;
    constexpr int WPB = NBI >= 4 ? 1 : (NBI == 3 ? 1 : 4 / NBI); // waves an x block
    constexpr int XOFF = 0, YOFF = HR * HC * XPITCH; // element offsets of the x halo, dy tile
    constexpr int ZERO = YOFF + P * YPITCH; // one zero pixel (XPITCH wide)
    // The x blocks' waves' sums (WPB > 1), TPR taps a round: as many as the staging's LDS holds.
    constexpr int STAGE_BYTES = (ZERO + XPITCH) * static_cast<int>(sizeof(T));
    constexpr int TAP_BYTES = NBI * NBO * Atom::ACC * WAVE * 4;
    constexpr int TPR = STAGE_BYTES / TAP_BYTES >= TAPS ? TAPS
                        : STAGE_BYTES / TAP_BYTES > 0   ? STAGE_BYTES / TAP_BYTES
                                                        : 1;
    constexpr int RED = WPB > 1 ? TPR * TAP_BYTES : 0;
    constexpr int SM = (STAGE_BYTES > RED ? STAGE_BYTES : RED) / static_cast<int>(sizeof(T));
    static_assert(Atom::DIM == 16 && NBI >= 2 && NBI <= 4 && NBI % NBO == 0, "wide tiles");
    __shared__ __attribute__((aligned(16))) T sm[SM];

    const int groups = a.c / a.cg;
    constexpr int OT = NBI / NBO;
    long long b = blockIdx.x;
    const int ot = static_cast<int>(b % OT);
    b /= OT;
    const int g = static_cast<int>(b % groups);
    b /= groups;
    const int ft = static_cast<int>(b % a.ft);
    const long long split = b / a.ft;
    const long long begin = split * a.tiles / a.splits;
    const long long end = (split + 1) * a.tiles / a.splits;
    const T* x = static_cast<const T*>(a.x);
    const T* dy = static_cast<const T*>(a.dy);
    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int c16 = lane % 16;
    const int kb = Atom::template kBase<KPL>(lane);
    const int ib = wave % NBI, sub = wave / NBI; // the wave's x block; its chunk subset
    const bool active = NBI != 3 || wave < 3;
    const int c0 = g * a.cg; // the group's first channel
    const int k0 = ot * YCH; // the tile's first dy channel within the group

    for(int i = tid; i < XPITCH; i += THREADS)
        sm[ZERO + i] = T{};

    typename Atom::acc acc[NBO][TAPS];
#pragma unroll
    for(int ob = 0; ob < NBO; ++ob)
#pragma unroll
        for(int tap = 0; tap < TAPS; ++tap)
            acc[ob][tap] = typename Atom::acc{};

    int twi, thi, od;
    long long img;
    {
        long long t = begin;
        twi = static_cast<int>(t % a.tilesW);
        t /= a.tilesW;
        thi = static_cast<int>(t % a.tilesH);
        t /= a.tilesH;
        od = static_cast<int>(t % a.dout);
        img = t / a.dout;
    }
    K4WrwStaging<T, CF, HR, HC, XCH, XPITCH, THREADS, true> xs;
    K4WrwStaging<T, CF, TH, TW, YCH, YPITCH, THREADS, true> ys;
    for(long long ti = begin; ti < end; ++ti)
    {
        const int id = od * a.sd - a.pd + ft;
        const int oh0 = thi * TH, ow0 = twi * TW;
        const long long tImg = img;
        const int tOd = od;
        if(++twi == a.tilesW)
        {
            twi = 0;
            if(++thi == a.tilesH)
            {
                thi = 0;
                if(++od == a.dout)
                {
                    od = 0;
                    ++img;
                }
            }
        }
        if(static_cast<unsigned>(id) >= static_cast<unsigned>(a.di))
            continue;
        xs.load(
            x, tImg, id, a.di, a.hi, a.wi, a.c, c0, oh0 * ST - a.ph, ow0 * ST - a.pw, tid, a.cg);
        ys.load(dy, tImg, tOd, a.dout, a.ho, a.wo, a.c, c0 + k0, oh0, ow0, tid, a.cg - k0);
        __syncthreads();
        xs.store(sm + XOFF, tid);
        ys.store(sm + YOFF, tid);
        __syncthreads();
        if(!active)
            continue;
        const int vh = a.ho - oh0, vw = a.wo - ow0;
#pragma unroll
        for(int jj = 0; jj < (NCH + WPB - 1) / WPB; ++jj)
        {
            const int j = jj * WPB + sub;
            if(NCH % WPB != 0 && j >= NCH)
                break;
            typedef K4WrwFrag<Atom, T> F;
            int dyo[F::NP], xo[F::NP];
            bool ok[F::NP];
#pragma unroll
            for(int i = 0; i < F::NP; ++i)
            {
                const int p = j * KA + kb + F::pos(lane, i);
                const int r = p / TW, q = p % TW;
                ok[i] = p < P && r < vh && q < vw;
                dyo[i] = ok[i] ? YOFF + p * YPITCH : ZERO;
                xo[i] = (r * ST * HC + q * ST) * XPITCH;
            }
            const int ch = F::chan(lane);
            int o[F::NP];
            typename Atom::frag af[NBO];
#pragma unroll
            for(int ob = 0; ob < NBO; ++ob)
            {
#pragma unroll
                for(int i = 0; i < F::NP; ++i)
                    o[i] = dyo[i] + ob * 16 + ch;
                af[ob] = F::read(sm, o);
            }
#pragma unroll
            for(int fr = 0; fr < R; ++fr)
#pragma unroll
                for(int fs = 0; fs < S; ++fs)
                {
                    const int tapo = (fr * HC + fs) * XPITCH;
#pragma unroll
                    for(int i = 0; i < F::NP; ++i)
                        o[i] = (ok[i] ? xo[i] + tapo : ZERO) + ib * 16 + ch;
                    const typename Atom::frag bf = F::read(sm, o);
#pragma unroll
                    for(int ob = 0; ob < NBO; ++ob)
                        acc[ob][fr * S + fs] = Atom::mma(af[ob], bf, acc[ob][fr * S + fs]);
                }
        }
    }

    if constexpr(WPB > 1)
    {
        float* red = reinterpret_cast<float*>(sm);
#pragma unroll
        for(int t0 = 0; t0 < TAPS; t0 += TPR)
#pragma unroll
            for(int s = 1; s < WPB; ++s)
            {
                __syncthreads();
                if(sub == s)
#pragma unroll
                    for(int t = 0; t < TPR && t0 + t < TAPS; ++t)
#pragma unroll
                        for(int ob = 0; ob < NBO; ++ob)
#pragma unroll
                            for(int e = 0; e < Atom::ACC; ++e)
                                red[(((t * NBI + ib) * NBO + ob) * Atom::ACC + e) * WAVE + lane]
                                    = acc[ob][t0 + t][e];
                __syncthreads();
                if(sub == 0)
#pragma unroll
                    for(int t = 0; t < TPR && t0 + t < TAPS; ++t)
#pragma unroll
                        for(int ob = 0; ob < NBO; ++ob)
#pragma unroll
                            for(int e = 0; e < Atom::ACC; ++e)
                                acc[ob][t0 + t][e]
                                    += red[(((t * NBI + ib) * NBO + ob) * Atom::ACC + e) * WAVE
                                           + lane];
            }
        if(sub != 0)
            return;
    }
    if(!active)
        return;
    const int taps = a.ft * TAPS;
    const int c = ib * 16 + c16;
    float* out
        = a.partial + split * a.partialElems + static_cast<long long>(g) * a.cg * a.cg * taps;
#pragma unroll
    for(int ob = 0; ob < NBO; ++ob)
#pragma unroll
        for(int tap = 0; tap < TAPS; ++tap)
#pragma unroll
            for(int e = 0; e < Atom::ACC; ++e)
            {
                const int k = k0 + ob * 16 + Atom::dRow(lane, e);
                if(k < a.cg && c < a.cg)
                    out[(static_cast<long long>(k) * a.cg + c) * taps + ft * TAPS + tap]
                        = acc[ob][tap][e];
            }
}

// FMA backward weights (no matrix cores: fp32 on RDNA, and any element type anywhere): the same
// grid, tiles, staging and partials as k4WrwMma, without the block-diagonal zeros. 256 threads; a
// unit is one (output channel k, input channel c) pair of a group among the tile's CB blocks (16
// CB x CG units, CG the graph's channels a group), owning all R x S taps; 256 / units threads
// share a unit, each taking every (256 / units)-th row segment of PX dy positions of a tile (a
// segment within one dy row, TW a multiple of PX) and holding the x row window the segment's
// taps reach in registers. The threads of a unit are added in order through LDS at the end.
template <class T, int R, int S, int ST, int TH, int TW, int CB, int CG, int PX, int CF>
__device__ void k4WrwFma(const K4WrwArgs& a)
{
    constexpr int THREADS = 256;
    constexpr int CH = 16 * CB;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    constexpr int PITCH = CH + VEC;
    constexpr int HR = (TH - 1) * ST + R, HC = (TW - 1) * ST + S;
    constexpr int P = TH * TW;
    constexpr int TAPS = R * S;
    constexpr int UNITS = CH * CG;
    constexpr int SUBS = THREADS / UNITS; // threads a unit
    constexpr int SEGS = TH * (TW / PX); // row segments a tile
    constexpr int WIN = (PX - 1) * ST + S; // x columns a segment's taps reach
    constexpr int XPIX = HR * HC;
    constexpr int ZERO = (XPIX + P) * PITCH;
    constexpr int RED = SUBS > 1 ? THREADS * TAPS : 0;
    constexpr int STAGE_BYTES = (XPIX + P + 1) * PITCH * static_cast<int>(sizeof(T));
    constexpr int SM
        = (STAGE_BYTES > RED * 4 ? STAGE_BYTES : RED * 4) / static_cast<int>(sizeof(T));
    static_assert(16 % CG == 0 && THREADS % UNITS == 0, "whole groups, whole units a block");
    static_assert(TW % PX == 0, "whole segments a row");
    __shared__ __attribute__((aligned(16))) T sm[SM];

    const int cTiles = a.c / CH;
    long long b = blockIdx.x;
    const int cbt = static_cast<int>(b % cTiles);
    b /= cTiles;
    const int ft = static_cast<int>(b % a.ft);
    const long long split = b / a.ft;
    const long long begin = split * a.tiles / a.splits;
    const long long end = (split + 1) * a.tiles / a.splits;
    const T* x = static_cast<const T*>(a.x);
    const T* dy = static_cast<const T*>(a.dy);
    const int tid = threadIdx.x;
    const int unit = tid % UNITS, sub = tid / UNITS;
    const int k = unit / CG, c = unit % CG; // the tile's channel k; c within k's group
    const int xc = k / CG * CG + c; // the tile's x channel

    for(int i = tid; i < PITCH; i += THREADS)
        sm[ZERO + i] = T{};

    float acc[TAPS];
#pragma unroll
    for(int tap = 0; tap < TAPS; ++tap)
        acc[tap] = 0.0f;

    int twi, thi, od;
    long long img;
    {
        long long t = begin;
        twi = static_cast<int>(t % a.tilesW);
        t /= a.tilesW;
        thi = static_cast<int>(t % a.tilesH);
        t /= a.tilesH;
        od = static_cast<int>(t % a.dout);
        img = t / a.dout;
    }
    K4WrwStaging<T, CF, HR, HC, CH, PITCH, THREADS> xs;
    K4WrwStaging<T, CF, TH, TW, CH, PITCH, THREADS> ys;
    for(long long ti = begin; ti < end; ++ti)
    {
        const int id = od * a.sd - a.pd + ft;
        const int oh0 = thi * TH, ow0 = twi * TW;
        const long long tImg = img;
        const int tOd = od;
        if(++twi == a.tilesW)
        {
            twi = 0;
            if(++thi == a.tilesH)
            {
                thi = 0;
                if(++od == a.dout)
                {
                    od = 0;
                    ++img;
                }
            }
        }
        if(static_cast<unsigned>(id) >= static_cast<unsigned>(a.di))
            continue;
        xs.load(
            x, tImg, id, a.di, a.hi, a.wi, a.c, cbt * CH, oh0 * ST - a.ph, ow0 * ST - a.pw, tid);
        ys.load(dy, tImg, tOd, a.dout, a.ho, a.wo, a.c, cbt * CH, oh0, ow0, tid);
        __syncthreads();
        xs.store(sm, tid);
        ys.store(sm + XPIX * PITCH, tid);
        __syncthreads();
        const int vh = a.ho - oh0, vw = a.wo - ow0;
        for(int sg = sub; sg < SEGS; sg += SUBS)
        {
            const int r = sg / (TW / PX), q0 = sg % (TW / PX) * PX;
            if(r >= vh || q0 >= vw)
                continue;
            // Positions past the valid columns read the zero pixel's dy and skip their taps.
            float d[PX];
            bool ok[PX];
#pragma unroll
            for(int p = 0; p < PX; ++p)
            {
                ok[p] = q0 + p < vw;
                d[p] = K1Elem<T>::load(sm[(ok[p] ? (XPIX + r * TW + q0 + p) * PITCH : ZERO) + k]);
            }
#pragma unroll
            for(int fr = 0; fr < R; ++fr)
            {
                const T* row = sm + ((r * ST + fr) * HC + q0 * ST) * PITCH + xc;
                float xv[WIN];
#pragma unroll
                for(int j = 0; j < WIN; ++j)
                    xv[j] = K1Elem<T>::load(row[j * PITCH]);
#pragma unroll
                for(int p = 0; p < PX; ++p)
#pragma unroll
                    for(int fs = 0; fs < S; ++fs)
                        acc[fr * S + fs] = ok[p] ? fmaf(d[p], xv[p * ST + fs], acc[fr * S + fs])
                                                 : acc[fr * S + fs];
            }
        }
    }

    if constexpr(SUBS > 1)
    {
        // Thread sub 0 of each unit adds the others' sums in order.
        __syncthreads();
        float* red = reinterpret_cast<float*>(sm);
#pragma unroll
        for(int tap = 0; tap < TAPS; ++tap)
            red[tap * THREADS + tid] = acc[tap];
        __syncthreads();
        if(sub != 0)
            return;
        for(int s = 1; s < SUBS; ++s)
#pragma unroll
            for(int tap = 0; tap < TAPS; ++tap)
                acc[tap] += red[tap * THREADS + s * UNITS + unit];
    }
    const int taps = a.ft * TAPS;
    const int g = (cbt * CH + k) / a.cg;
    float* out = a.partial + split * a.partialElems
                 + ((static_cast<long long>(g) * a.cg + k % a.cg) * a.cg + c) * taps + ft * TAPS;
#pragma unroll
    for(int tap = 0; tap < TAPS; ++tap)
        out[tap] = acc[tap];
}

// The partials' fixed-order sum (K4WrwReduceArgs): a block's 256 threads are 256 / G dw
// elements (in the partials' [g][k][c][tap] order) by G slot groups, G the slot count rounded up
// to a power of two, at most K4_WRW_REDUCE_GROUPS; group j adds slots j, j + G, ... in order,
// and the group sums are added in group order through LDS. The association depends only on the
// slot count, so the result is bitwise reproducible.
template <class T>
__device__ inline void k4WrwReduce(const K4WrwReduceArgs& a)
{
    __shared__ float part[256];
    const int G = k4WrwReduceGroups(a.slots), E = 256 / G;
    const int lane = threadIdx.x % E, grp = threadIdx.x / E;
    const long long i = static_cast<long long>(blockIdx.x) * E + lane;
    const bool live = i < a.partialElems;
    float sum = 0.0f;
    if(live)
        for(int sp = grp; sp < a.slots; sp += G)
            sum += a.partial[sp * a.partialElems + i];
    part[threadIdx.x] = sum;
    __syncthreads();
    if(grp != 0 || !live)
        return;
    for(int j = 1; j < G; ++j)
        sum += part[j * E + lane];
    const int taps = a.t * a.r * a.s;
    int tap = static_cast<int>(i % taps);
    long long rest = i / taps;
    const int c = static_cast<int>(rest % a.cg);
    const long long k = rest / a.cg; // over all groups
    if(a.reflect)
        tap = taps - 1 - tap;
    const int ts = tap % a.s, tr = (tap / a.s) % a.r, tt = tap / (a.s * a.r);
    static_cast<T*>(a.dw)[k * a.sK + c * a.sC + tt * a.sT + tr * a.sR + ts * a.sS]
        = K1Elem<T>::store(sum);
}
