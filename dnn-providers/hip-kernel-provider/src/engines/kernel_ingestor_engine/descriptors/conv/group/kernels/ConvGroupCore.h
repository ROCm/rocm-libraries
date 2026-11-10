// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Small-group convolution cores (K4): groups of cg <= 16 channels each way, run per 16-channel
// block on a 16x16 matrix-core atom (the K1 atoms) with the block-diagonal expanded filter, so a
// block's A fragment (16 output pixels x 16 channels) feeds 16 / cg whole groups at once.
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

// The filter expansion (K4ExpandArgs), one element per thread, grid-strided.
__device__ inline void k4Expand(const K4ExpandArgs& a)
{
    const int taps = a.tc * a.rc * a.sc;
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
