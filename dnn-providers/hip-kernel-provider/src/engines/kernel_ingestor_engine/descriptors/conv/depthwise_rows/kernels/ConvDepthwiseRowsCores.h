// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Channel-row depthwise convolution cores for the CONV_DEPTHWISE_ROWS engines.
//
// Every block works inside one channel row or plane: the plane (or row) index comes from
// blockIdx alone, so the channel, and with it the filter, is uniform across the block and the
// filter taps are scalar loads held in SGPRs. Shapes, padding, stride and dilation are runtime
// arguments (ConvDepthwiseValuArgs.h); only the filter extent may be compile-time.
//
//   rowsFwdNchw          forward plane tile, channels-first. The block stages its input halo in
//                        LDS as fp32 (coalesced along the contiguous width, padding resolved as
//                        zeros) and each thread computes an RH x RW register tile. With a
//                        dilation d > 1 the thread's outputs step along the dilation grid
//                        (x0, x0 + d, ...), so they share their inputs: a row of the patch is
//                        RW + KW - 1 LDS reads, not RW * KW. Any stride, dilation and padding.
//   rowsBwdPhase         backward data of a 1-D (or row-separable 2-D) strided depthwise
//                        convolution, i.e. a depthwise ConvTranspose1d, by stride phases: with
//                        i + pad = s * q + r, dx[i] = sum_m w[r + s * m] * dy[q - m]. A thread
//                        owns Q values of q and writes their s * Q consecutive dx elements; no
//                        multiply is spent on a skipped tap and no index needs a modulo. The
//                        block stages its dy span once in LDS. Channels-first or channels-last
//                        (rows of one channel, position stride c).
//   rowsBwdPhaseLanes    the same phases, channels-last, lanes over (position, channel) with the
//                        channel fastest, so every dy load and dx store is contiguous across the
//                        wave whatever the channel count; the filter is per lane.
//
// Plain vector-ALU FMA with an fp32 accumulator; no wave-size intrinsics (wave32 and wave64).
#pragma once

// Embedded headers are keyed by bare file name (src/cmake/KernelEmbedding.cmake).
#include "ConvDepthwiseValuArgs.h"

namespace conv_depthwise_rows
{

/// Loads one element as fp32. 16-bit elements are read as a full-width (zero-extended) 16-bit
/// load: a d16 load keeps the destination's high half, which chains every load in a loop on
/// the one before it on gfx11.
template <typename T>
__device__ inline float rowsLoad(const T* p)
{
    if constexpr(sizeof(T) == 2)
    {
        const unsigned short bits = *reinterpret_cast<const unsigned short*>(p);
        return static_cast<float>(__builtin_bit_cast(T, bits));
    }
    else
    {
        return static_cast<float>(*p);
    }
}

/// @p v clamped into [0, extent): out-of-range reads load an in-range element and select 0
/// after, so a guarded load needs no branch (and no wait of its own).
__device__ inline int rowsClamp(int v, int extent)
{
    return v < 0 ? 0 : (v >= extent ? extent - 1 : v);
}

/// A band's filter tap as fp32. On gfx9, an fp16 tap converted here is kept an opaque fp32 VGPR:
/// an LLVM 24 build for gfx942 fails instruction selection ("Cannot select: f32 = Constant<-1>",
/// an LLVM ERROR that aborts the process) for the fp16 5x5 bands with four or eight rows per
/// thread when it folds the taps' conversions into mixed-precision FMAs. Elsewhere, and for the
/// other variants (which that build compiles), the tap stays a scalar register.
template <typename T>
__device__ inline float rowsBandTap(const T* p)
{
    float v = rowsLoad(p);
#if defined(__GFX9__)
    if constexpr(sizeof(T) == 2 && !__is_same(T, __bf16))
    {
        asm volatile("" : "+v"(v));
    }
#endif
    return v;
}

template <typename T>
__device__ inline T rowsFromF32(float v)
{
    return static_cast<T>(v);
}

// ---- forward plane tile ----------------------------------------------------------------

/// Accumulates one thread's RH x RW tile from the staged halo, compile-time filter.
/// SEGH: rows (sh == 1, row step == dh) -- the patch rows lie on the dilation grid and are read
/// once each. SEGW: the same along the width (sw == 1, column step == dw).
template <int KH, int KW, int RH, int RW, bool SEGH, bool SEGW>
__device__ inline void rowsAccumulate(const float* tile,
                                      int ldw,
                                      int y0,
                                      int ys,
                                      int x0,
                                      int xs,
                                      int sh,
                                      int sw,
                                      int dh,
                                      int dw,
                                      const float (&w)[KH][KW],
                                      float (&acc)[RH][RW])
{
    if constexpr(SEGH)
    {
#pragma unroll
        for(int pi = 0; pi < RH + KH - 1; ++pi)
        {
            const float* line = tile + (y0 + pi * dh) * ldw;
            if constexpr(SEGW)
            {
                float seg[RW + KW - 1];
#pragma unroll
                for(int e = 0; e < RW + KW - 1; ++e)
                {
                    seg[e] = line[x0 + e * dw];
                }
#pragma unroll
                for(int i = 0; i < RH; ++i)
                {
                    const int ky = pi - i;
                    if(ky < 0 || ky >= KH)
                    {
                        continue;
                    }
#pragma unroll
                    for(int kx = 0; kx < KW; ++kx)
                    {
#pragma unroll
                        for(int j = 0; j < RW; ++j)
                        {
                            acc[i][j] += w[ky][kx] * seg[j + kx];
                        }
                    }
                }
            }
            else
            {
#pragma unroll
                for(int i = 0; i < RH; ++i)
                {
                    const int ky = pi - i;
                    if(ky < 0 || ky >= KH)
                    {
                        continue;
                    }
#pragma unroll
                    for(int kx = 0; kx < KW; ++kx)
                    {
#pragma unroll
                        for(int j = 0; j < RW; ++j)
                        {
                            acc[i][j] += w[ky][kx] * line[(x0 + j * xs) * sw + kx * dw];
                        }
                    }
                }
            }
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < RH; ++i)
        {
#pragma unroll
            for(int ky = 0; ky < KH; ++ky)
            {
                const float* line = tile + ((y0 + i * ys) * sh + ky * dh) * ldw;
                if constexpr(SEGW)
                {
                    float seg[RW + KW - 1];
#pragma unroll
                    for(int e = 0; e < RW + KW - 1; ++e)
                    {
                        seg[e] = line[x0 + e * dw];
                    }
#pragma unroll
                    for(int kx = 0; kx < KW; ++kx)
                    {
#pragma unroll
                        for(int j = 0; j < RW; ++j)
                        {
                            acc[i][j] += w[ky][kx] * seg[j + kx];
                        }
                    }
                }
                else
                {
#pragma unroll
                    for(int kx = 0; kx < KW; ++kx)
                    {
#pragma unroll
                        for(int j = 0; j < RW; ++j)
                        {
                            acc[i][j] += w[ky][kx] * line[(x0 + j * xs) * sw + kx * dw];
                        }
                    }
                }
            }
        }
    }
}

/// Rows of the input halo a forward tile of @p bh x @p bw outputs reads.
__host__ __device__ inline int rowsHaloExtent(int tileOut, int stride, int filter, int dilation)
{
    return (tileOut - 1) * stride + (filter - 1) * dilation + 1;
}

/// Forward depthwise convolution, channels-first, one plane per block row of the grid.
/// Block TX x TY threads, tile (TY * RH) x (TX * RW) outputs; KH = KW = 0 reads the filter
/// extent at runtime. The staged halo must fit LDS floats (the host gates it).
template <typename T, int KH, int KW, int TX, int TY, int RH, int RW, int LDS, bool FLIP>
__device__ inline void rowsFwdNchw(const T* __restrict__ in,
                                   const T* __restrict__ wt,
                                   T* __restrict__ out,
                                   const ConvDepthwiseValuArgs& a)
{
    constexpr int THREADS = TX * TY;
    constexpr int BH = TY * RH;
    constexpr int BW = TX * RW;
    __shared__ float tile[LDS];

    const int kh = KH > 0 ? KH : a.kh;
    const int kw = KW > 0 ? KW : a.kw;
    const int lh = rowsHaloExtent(BH, a.sh, kh, a.dh);
    const int lw = rowsHaloExtent(BW, a.sw, kw, a.dw);
    const int tilesW = (a.wo + BW - 1) / BW;
    const int tilesH = (a.ho + BH - 1) / BH;
    const int tilesPerPlane = tilesW * tilesH;

    const int blk = static_cast<int>(blockIdx.x);
    const int plane = blk / tilesPerPlane;
    const int t = blk - plane * tilesPerPlane;
    const int th = t / tilesW;
    const int tw = t - th * tilesW;
    const int c = plane % a.c;
    const int oy0 = th * BH;
    const int ox0 = tw * BW;
    const int iy0 = oy0 * a.sh - a.ph;
    const int ix0 = ox0 * a.sw - a.pw;
    const T* src = in + static_cast<long>(plane) * a.hi * a.wi;
    const int tid = static_cast<int>(threadIdx.x);

    if(lh == 1)
    {
        const bool rowIn = iy0 >= 0 && iy0 < a.hi;
        const T* line = src + static_cast<long>(rowsClamp(iy0, a.hi)) * a.wi;
        for(int col = tid; col < lw; col += THREADS)
        {
            const int ix = ix0 + col;
            const float v = rowsLoad(&line[rowsClamp(ix, a.wi)]);
            tile[col] = (rowIn && ix >= 0 && ix < a.wi) ? v : 0.0f;
        }
    }
    else
    {
        const int total = lh * lw;
        for(int idx = tid; idx < total; idx += THREADS)
        {
            const int row
                = static_cast<int>(static_cast<unsigned>(idx) / static_cast<unsigned>(lw));
            const int col = idx - row * lw;
            const int iy = iy0 + row;
            const int ix = ix0 + col;
            const float v = rowsLoad(
                &src[static_cast<long>(rowsClamp(iy, a.hi)) * a.wi + rowsClamp(ix, a.wi)]);
            tile[idx] = (iy >= 0 && iy < a.hi && ix >= 0 && ix < a.wi) ? v : 0.0f;
        }
    }
    __syncthreads();

    const int tx = tid % TX;
    const int ty = tid / TX;
    // Lanes on the dilation grid when the thread count along the axis is a multiple of the
    // dilation: a thread's outputs are d apart and share their taps' inputs.
    const bool gridW = a.sw == 1 && a.dw > 1 && TX % a.dw == 0;
    const bool gridH = a.sh == 1 && a.dh > 1 && TY % a.dh == 0;
    const int x0 = gridW ? (tx / a.dw) * RW * a.dw + tx % a.dw : tx * RW;
    const int xs = gridW ? a.dw : 1;
    const int y0 = gridH ? (ty / a.dh) * RH * a.dh + ty % a.dh : ty * RH;
    const int ys = gridH ? a.dh : 1;

    float acc[RH][RW];
#pragma unroll
    for(int i = 0; i < RH; ++i)
    {
#pragma unroll
        for(int j = 0; j < RW; ++j)
        {
            acc[i][j] = 0.0f;
        }
    }

    if constexpr(KH > 0 && KW > 0)
    {
        float w[KH][KW];
        const T* wc = wt + static_cast<long>(c) * KH * KW;
#pragma unroll
        for(int ky = 0; ky < KH; ++ky)
        {
#pragma unroll
            for(int kx = 0; kx < KW; ++kx)
            {
                w[ky][kx] = rowsLoad(&wc[FLIP ? (KH - 1 - ky) * KW + (KW - 1 - kx) : ky * KW + kx]);
            }
        }
        const bool segW = a.sw == 1 && xs == a.dw;
        const bool segH = a.sh == 1 && ys == a.dh;
        if(segH && segW)
        {
            rowsAccumulate<KH, KW, RH, RW, true, true>(
                tile, lw, y0, ys, x0, xs, a.sh, a.sw, a.dh, a.dw, w, acc);
        }
        else if(segH)
        {
            rowsAccumulate<KH, KW, RH, RW, true, false>(
                tile, lw, y0, ys, x0, xs, a.sh, a.sw, a.dh, a.dw, w, acc);
        }
        else if(segW)
        {
            rowsAccumulate<KH, KW, RH, RW, false, true>(
                tile, lw, y0, ys, x0, xs, a.sh, a.sw, a.dh, a.dw, w, acc);
        }
        else
        {
            rowsAccumulate<KH, KW, RH, RW, false, false>(
                tile, lw, y0, ys, x0, xs, a.sh, a.sw, a.dh, a.dw, w, acc);
        }
    }
    else
    {
        const T* wc = wt + static_cast<long>(c) * kh * kw;
        for(int ky = 0; ky < kh; ++ky)
        {
            for(int kx = 0; kx < kw; ++kx)
            {
                const float wv
                    = rowsLoad(&wc[FLIP ? (kh - 1 - ky) * kw + (kw - 1 - kx) : ky * kw + kx]);
#pragma unroll
                for(int i = 0; i < RH; ++i)
                {
                    const float* line = tile + ((y0 + i * ys) * a.sh + ky * a.dh) * lw;
#pragma unroll
                    for(int j = 0; j < RW; ++j)
                    {
                        acc[i][j] += wv * line[(x0 + j * xs) * a.sw + kx * a.dw];
                    }
                }
            }
        }
    }

    T* dst = out + static_cast<long>(plane) * a.ho * a.wo;
#pragma unroll
    for(int i = 0; i < RH; ++i)
    {
        const int oy = oy0 + y0 + i * ys;
        if(oy >= a.ho)
        {
            continue;
        }
#pragma unroll
        for(int j = 0; j < RW; ++j)
        {
            const int ox = ox0 + x0 + j * xs;
            if(ox < a.wo)
            {
                dst[static_cast<long>(oy) * a.wo + ox] = rowsFromF32<T>(acc[i][j]);
            }
        }
    }
}

/// Threads a band block launches: one per (row group, column) unit rounded up to whole 64-lane
/// groups, at most @p maxThreads.
__host__ __device__ inline int rowsBandThreads(int groups, int wo, int maxThreads)
{
    const long units = static_cast<long>(groups) * wo;
    const long rounded = (units + 63) / 64 * 64;
    return static_cast<int>(rounded < maxThreads ? rounded : maxThreads);
}

/// Forward depthwise convolution, channels-first, one band of BH output rows of one plane per
/// block, every column: no tile edge along the width, so a plane width that is not a multiple
/// of a tile (65, 33, 28) costs nothing. The halo band is staged in LDS as fp32; the block's
/// threads walk (row group, column) units with the column fastest, each computing RH rows of one
/// column (on the dilation grid when the group count is a multiple of dh). The host launches
/// bandThreads() threads, at most MAXT: the unit count rounded up to whole waves, so a band
/// that fits one round of units has no idle tail round.
template <typename T, int KH, int KW, int BH, int RH, int MAXT, int LDS, bool FLIP>
__device__ inline void rowsFwdBandNchw(const T* __restrict__ in,
                                       const T* __restrict__ wt,
                                       T* __restrict__ out,
                                       const ConvDepthwiseValuArgs& a)
{
    static_assert(BH % RH == 0, "the row groups must divide the band");
    constexpr int G = BH / RH;
    __shared__ float tile[LDS];

    const int kh = KH > 0 ? KH : a.kh;
    const int kw = KW > 0 ? KW : a.kw;
    const int lh = rowsHaloExtent(BH, a.sh, kh, a.dh);
    const int lw = rowsHaloExtent(a.wo, a.sw, kw, a.dw);
    const int bands = (a.ho + BH - 1) / BH;
    const int blk = static_cast<int>(blockIdx.x);
    const int plane = blk / bands;
    const int band = blk - plane * bands;
    const int c = plane % a.c;
    const int oy0 = band * BH;
    const int iy0 = oy0 * a.sh - a.ph;
    const int ix0 = -a.pw;
    const T* src = in + static_cast<long>(plane) * a.hi * a.wi;
    const int tid = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);

    const int total = lh * lw;
    for(int idx = tid; idx < total; idx += threads)
    {
        const int row = static_cast<int>(static_cast<unsigned>(idx) / static_cast<unsigned>(lw));
        const int col = idx - row * lw;
        const int iy = iy0 + row;
        const int ix = ix0 + col;
        const float v
            = rowsLoad(&src[static_cast<long>(rowsClamp(iy, a.hi)) * a.wi + rowsClamp(ix, a.wi)]);
        tile[idx] = (iy >= 0 && iy < a.hi && ix >= 0 && ix < a.wi) ? v : 0.0f;
    }
    __syncthreads();

    const bool gridH = a.sh == 1 && a.dh > 1 && G % a.dh == 0;
    const int ys = gridH ? a.dh : 1;
    const bool segH = a.sh == 1 && ys == a.dh;
    const int units = G * a.wo;
    const T* wc = wt + static_cast<long>(c) * kh * kw;
    T* dst = out + static_cast<long>(plane) * a.ho * a.wo;

    float w[KH > 0 ? KH : 1][KW > 0 ? KW : 1];
    if constexpr(KH > 0 && KW > 0)
    {
#pragma unroll
        for(int ky = 0; ky < KH; ++ky)
        {
#pragma unroll
            for(int kx = 0; kx < KW; ++kx)
            {
                w[ky][kx]
                    = rowsBandTap(&wc[FLIP ? (KH - 1 - ky) * KW + (KW - 1 - kx) : ky * KW + kx]);
            }
        }
    }

    for(int u = tid; u < units; u += threads)
    {
        const int g = static_cast<int>(static_cast<unsigned>(u) / static_cast<unsigned>(a.wo));
        const int x = u - g * a.wo;
        const int y0 = gridH ? (g / a.dh) * RH * a.dh + g % a.dh : g * RH;
        const float* colBase = tile + x * a.sw;
        float acc[RH];
#pragma unroll
        for(int i = 0; i < RH; ++i)
        {
            acc[i] = 0.0f;
        }
        if constexpr(KH > 0 && KW > 0)
        {
            if(segH)
            {
#pragma unroll
                for(int pi = 0; pi < RH + KH - 1; ++pi)
                {
                    const float* line = colBase + (y0 + pi * a.dh) * lw;
                    float v[KW];
#pragma unroll
                    for(int kx = 0; kx < KW; ++kx)
                    {
                        v[kx] = line[kx * a.dw];
                    }
#pragma unroll
                    for(int i = 0; i < RH; ++i)
                    {
                        const int ky = pi - i;
                        if(ky < 0 || ky >= KH)
                        {
                            continue;
                        }
#pragma unroll
                        for(int kx = 0; kx < KW; ++kx)
                        {
                            acc[i] += w[ky][kx] * v[kx];
                        }
                    }
                }
            }
            else
            {
#pragma unroll
                for(int i = 0; i < RH; ++i)
                {
#pragma unroll
                    for(int ky = 0; ky < KH; ++ky)
                    {
                        const float* line = colBase + ((y0 + i * ys) * a.sh + ky * a.dh) * lw;
#pragma unroll
                        for(int kx = 0; kx < KW; ++kx)
                        {
                            acc[i] += w[ky][kx] * line[kx * a.dw];
                        }
                    }
                }
            }
        }
        else
        {
            for(int ky = 0; ky < kh; ++ky)
            {
                for(int kx = 0; kx < kw; ++kx)
                {
                    const float wv = rowsBandTap(
                        &wc[FLIP ? (kh - 1 - ky) * kw + (kw - 1 - kx) : ky * kw + kx]);
#pragma unroll
                    for(int i = 0; i < RH; ++i)
                    {
                        acc[i] += wv * colBase[((y0 + i * ys) * a.sh + ky * a.dh) * lw + kx * a.dw];
                    }
                }
            }
        }
#pragma unroll
        for(int i = 0; i < RH; ++i)
        {
            const int oy = oy0 + y0 + i * ys;
            if(oy < a.ho)
            {
                dst[static_cast<long>(oy) * a.wo + x] = rowsFromF32<T>(acc[i]);
            }
        }
    }
}

// ---- backward data by stride phases ---------------------------------------------------------
//
// Gather terms, as v5_gather_bwd: in = dy with extents (hi, wi), out = dx with (ho, wo), and the
// forward convolution's padding pw and stride sw (== S). The height axis is the identity (filter
// height 1, stride 1, no padding), so every (image, channel, row) is an independent 1-D problem.

/// The first phase index q and the count of them the dx row needs.
__host__ __device__ inline void rowsPhaseRange(int dxLen, int pad, int stride, int& qmin, int& numQ)
{
    qmin = pad / stride;
    numQ = (dxLen - 1 + pad) / stride - qmin + 1;
}

template <typename T, int K, int S, int Q, int THREADS, int MMAX, bool FLIP, int LAYOUT>
__device__ inline void rowsBwdPhase(const T* __restrict__ dy,
                                    const T* __restrict__ wt,
                                    T* __restrict__ dx,
                                    const ConvDepthwiseValuArgs& g)
{
    constexpr int MC = K > 0 ? (K + S - 1) / S : MMAX;
    constexpr int QB = THREADS * Q;
    __shared__ float span[QB + MC - 1];

    const int kw = K > 0 ? K : g.kw;
    const int m = (kw + S - 1) / S;
    const int ly = g.wi;
    const int lx = g.wo;
    const int pad = g.pw;
    int qmin = 0;
    int numQ = 0;
    rowsPhaseRange(lx, pad, S, qmin, numQ);
    const int qBlocks = (numQ + QB - 1) / QB;

    const int blk = static_cast<int>(blockIdx.x);
    const int row = blk / qBlocks;
    const int qb = blk - row * qBlocks;
    int c = 0;
    long ybase = 0;
    long xbase = 0;
    long ps = 1;
    if constexpr(LAYOUT == 1)
    {
        c = (row / g.ho) % g.c;
        ybase = static_cast<long>(row) * ly;
        xbase = static_cast<long>(row) * lx;
    }
    else
    {
        c = row % g.c;
        const long nh = row / g.c;
        ybase = nh * ly * g.c + c;
        xbase = nh * lx * g.c + c;
        ps = g.c;
    }

    const int qb0 = qmin + qb * QB;
    const int o0 = qb0 - (m - 1);
    const int tid = static_cast<int>(threadIdx.x);
    for(int idx = tid; idx < QB + m - 1; idx += THREADS)
    {
        const int o = o0 + idx;
        const float v = rowsLoad(&dy[ybase + static_cast<long>(rowsClamp(o, ly)) * ps]);
        span[idx] = (o >= 0 && o < ly) ? v : 0.0f;
    }
    __syncthreads();

    const int q0 = qb0 + tid * Q;
    if(q0 >= qmin + numQ)
    {
        return;
    }
    float acc[Q][S];
#pragma unroll
    for(int j = 0; j < Q; ++j)
    {
#pragma unroll
        for(int r = 0; r < S; ++r)
        {
            acc[j][r] = 0.0f;
        }
    }
    const T* wc = wt + static_cast<long>(c) * kw;
    if constexpr(K > 0)
    {
        float w[K];
#pragma unroll
        for(int k = 0; k < K; ++k)
        {
            w[k] = rowsLoad(&wc[FLIP ? K - 1 - k : k]);
        }
        float win[Q + MC - 1];
#pragma unroll
        for(int e = 0; e < Q + MC - 1; ++e)
        {
            win[e] = span[tid * Q + e];
        }
#pragma unroll
        for(int r = 0; r < S; ++r)
        {
#pragma unroll
            for(int mm = 0; mm < MC; ++mm)
            {
                const int k = r + S * mm;
                if(k >= K)
                {
                    continue;
                }
#pragma unroll
                for(int j = 0; j < Q; ++j)
                {
                    acc[j][r] += w[k] * win[j - mm + MC - 1];
                }
            }
        }
    }
    else
    {
        const float* base = span + tid * Q + m - 1;
#pragma unroll
        for(int r = 0; r < S; ++r)
        {
            for(int k = r, mm = 0; k < kw; k += S, ++mm)
            {
                const float wv = rowsLoad(&wc[FLIP ? kw - 1 - k : k]);
#pragma unroll
                for(int j = 0; j < Q; ++j)
                {
                    acc[j][r] += wv * base[j - mm];
                }
            }
        }
    }

#pragma unroll
    for(int j = 0; j < Q; ++j)
    {
        const int q = q0 + j;
#pragma unroll
        for(int r = 0; r < S; ++r)
        {
            const int i = S * q + r - pad;
            if(i >= 0 && i < lx)
            {
                dx[xbase + static_cast<long>(i) * ps] = rowsFromF32<T>(acc[j][r]);
            }
        }
    }
}

template <typename T, int K, int S, bool FLIP>
__device__ inline void rowsBwdPhaseLanes(const T* __restrict__ dy,
                                         const T* __restrict__ wt,
                                         T* __restrict__ dx,
                                         const ConvDepthwiseValuArgs& g)
{
    const int kw = K > 0 ? K : g.kw;
    const int ly = g.wi;
    const int lx = g.wo;
    const int pad = g.pw;
    const int ch = g.c;
    int qmin = 0;
    int numQ = 0;
    rowsPhaseRange(lx, pad, S, qmin, numQ);
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int c = static_cast<int>(t % ch);
    const long rest = t / ch;
    const int qi = static_cast<int>(rest % numQ);
    const long nh = rest / numQ;
    if(nh >= static_cast<long>(g.n) * g.ho)
    {
        return;
    }
    const int q = qmin + qi;
    const long ybase = nh * ly * ch + c;
    const long xbase = nh * lx * ch + c;
    const T* wc = wt + static_cast<long>(c) * kw;

    float acc[S];
#pragma unroll
    for(int r = 0; r < S; ++r)
    {
        acc[r] = 0.0f;
    }
    if constexpr(K > 0)
    {
        constexpr int MC = (K + S - 1) / S;
        float d[MC];
#pragma unroll
        for(int mm = 0; mm < MC; ++mm)
        {
            const int o = q - mm;
            const float v = rowsLoad(&dy[ybase + static_cast<long>(rowsClamp(o, ly)) * ch]);
            d[mm] = (o >= 0 && o < ly) ? v : 0.0f;
        }
#pragma unroll
        for(int r = 0; r < S; ++r)
        {
#pragma unroll
            for(int mm = 0; mm < MC; ++mm)
            {
                const int k = r + S * mm;
                if(k < K)
                {
                    acc[r] += rowsLoad(&wc[FLIP ? K - 1 - k : k]) * d[mm];
                }
            }
        }
    }
    else
    {
        for(int mm = 0; mm * S < kw; ++mm)
        {
            const int o = q - mm;
            const float v = rowsLoad(&dy[ybase + static_cast<long>(rowsClamp(o, ly)) * ch]);
            const float dv = (o >= 0 && o < ly) ? v : 0.0f;
#pragma unroll
            for(int r = 0; r < S; ++r)
            {
                const int k = r + S * mm;
                if(k < kw)
                {
                    acc[r] += rowsLoad(&wc[FLIP ? kw - 1 - k : k]) * dv;
                }
            }
        }
    }
#pragma unroll
    for(int r = 0; r < S; ++r)
    {
        const int i = S * q + r - pad;
        if(i >= 0 && i < lx)
        {
            dx[xbase + static_cast<long>(i) * ch] = rowsFromF32<T>(acc[r]);
        }
    }
}

} // namespace conv_depthwise_rows
