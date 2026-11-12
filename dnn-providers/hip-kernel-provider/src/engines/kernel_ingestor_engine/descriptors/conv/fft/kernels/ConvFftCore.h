// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The FFT convolution kernels' building blocks: complex arithmetic, a register FFT of
// compile-time length, the block-level 2-D real FFT of a batch of tiles (and its inverse)
// through LDS, the loaders and storers of the four transform kernels, and the two
// frequency-domain products (batched complex GEMMs). Every transform and product is fp32,
// whatever the tensors' element type. Builtin types only (HIPRTC embedded include).
//
// Design (cuDNN FFT_TILING-style overlap-save, written from the published descriptions; no
// code taken from any FFT library): each output tile of oh x ow points is the circular
// cross-correlation of a th x tw input window with the zero-padded, dilated filter, valid on
// the first oh x ow points because oh = th - spanH + 1. A real 2-D tile is transformed as
// th x (tw/2 + 1) bins: rows in pairs (two real rows as one complex FFT, split by symmetry),
// then complex FFTs down the columns. The product of window and filter spectra is a batched
// complex matrix product over channels, one per (group, bin). No atomics anywhere: every sum
// has one owner and a fixed order.
#pragma once

#include "ConvFftArgs.h"
#include "ConvFftTwiddles.h"

typedef ConvFftComplex cfx;

__device__ __forceinline__ cfx cfxMake(float x, float y)
{
    cfx r;
    r.x = x;
    r.y = y;
    return r;
}
__device__ __forceinline__ cfx cfxAdd(cfx a, cfx b)
{
    return cfxMake(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ cfx cfxSub(cfx a, cfx b)
{
    return cfxMake(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ cfx cfxMul(cfx a, cfx b)
{
    return cfxMake(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ __forceinline__ cfx cfxConj(cfx a)
{
    return cfxMake(a.x, -a.y);
}
/// acc += a * b
__device__ __forceinline__ void cfxMac(cfx& acc, cfx a, cfx b)
{
    acc.x = __builtin_fmaf(a.x, b.x, acc.x);
    acc.y = __builtin_fmaf(a.x, b.y, acc.y);
    acc.x = __builtin_fmaf(-a.y, b.y, acc.x);
    acc.y = __builtin_fmaf(a.y, b.x, acc.y);
}

// ---- element types: fp32, fp16, bf16 (as its 16 bits) ----------------------------------------

template <class T>
struct FftElem;
template <>
struct FftElem<float>
{
    __device__ static float load(float v)
    {
        return v;
    }
    __device__ static float store(float v)
    {
        return v;
    }
};
template <>
struct FftElem<_Float16>
{
    __device__ static float load(_Float16 v)
    {
        return static_cast<float>(v);
    }
    __device__ static _Float16 store(float v)
    {
        return static_cast<_Float16>(v);
    }
};
template <>
struct FftElem<unsigned short>
{
    __device__ static float load(unsigned short v)
    {
        return __uint_as_float(static_cast<unsigned int>(v) << 16);
    }
    __device__ static unsigned short store(float v)
    {
        const unsigned int u = __float_as_uint(v);
        if((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0)
        {
            return static_cast<unsigned short>((u >> 16) | 0x40u); // quiet NaN
        }
        const unsigned int rounded = u + 0x7fffu + ((u >> 16) & 1u); // nearest even
        return static_cast<unsigned short>(rounded >> 16);
    }
};

// ---- register FFT --------------------------------------------------------------------------

constexpr int fftBitReverse(int i, int length)
{
    int r = 0;
    for(int b = 1; b < length; b <<= 1)
    {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

/// In-place radix-2 FFT of v[0, L), unnormalized: forward exp(-2 pi i jk / L), inverse
/// exp(+2 pi i jk / L). L is a power of two up to CONV_FFT_TW_N; every index is a compile-time
/// constant once unrolled, so v stays in registers.
template <int L, bool INV>
__device__ __forceinline__ void fftReg(cfx* v)
{
#pragma unroll
    for(int i = 0; i < L; ++i)
    {
        const int j = fftBitReverse(i, L);
        if(j > i)
        {
            const cfx t = v[i];
            v[i] = v[j];
            v[j] = t;
        }
    }
#pragma unroll
    for(int len = 2; len <= L; len <<= 1)
    {
#pragma unroll
        for(int i = 0; i < L; i += len)
        {
#pragma unroll
            for(int j = 0; j < len / 2; ++j)
            {
                const cfx u = v[i + j];
                const cfx o = v[i + j + len / 2];
                cfx x;
                if(j == 0)
                {
                    x = o;
                }
                else if(4 * j == len)
                {
                    // exp(-+ i pi / 2): forward -i, inverse +i
                    x = INV ? cfxMake(-o.y, o.x) : cfxMake(o.y, -o.x);
                }
                else
                {
                    const ConvFftTw t = fft_tw[j * (CONV_FFT_TW_N / len)];
                    x = cfxMul(cfxMake(t.c, INV ? t.s : -t.s), o);
                }
                v[i + j] = cfxAdd(u, x);
                v[i + j + len / 2] = cfxSub(u, x);
            }
        }
    }
}

// ---- block-level 2-D real FFT of PB tiles ------------------------------------------------------
//
// spec (LDS) holds PB tiles of F = TH * (TW / 2 + 1) bins, bin f = row * (TW / 2 + 1) + column.
// The real tiles pass through the same LDS (row stride TW + 1 floats, against bank conflicts):
// the forward transform loads them cooperatively (consecutive threads, consecutive points of a
// row: coalesced for a unit w stride) and the inverse stores them the same way. Real rows go
// through the FFT in pairs: two rows of one tile (TH > 1) or the single rows of two tiles
// (TH == 1, PB even); every pair has its own thread (PAIRS <= the block's threads), so a pass
// reads all its rows before any thread writes over them. Loader::value(t, row, i) gives point
// (row, i) of tile t; Storer::value(t, row, i, v) takes it back (unscaled).

template <int TH, int TW, int PB>
struct FftShape
{
    static constexpr int HW = TW / 2 + 1;
    static constexpr int F = TH * HW;
    static constexpr int PAIRS = TH == 1 ? PB / 2 : PB * TH / 2;
    static constexpr int RS = TW + 1; ///< real row stride (floats)
};

template <int TH, int PB>
__device__ __forceinline__ void fftPair(int q, int& ia, int& ra, int& ib, int& rb)
{
    if(TH == 1)
    {
        ia = 2 * q;
        ib = 2 * q + 1;
        ra = 0;
        rb = 0;
    }
    else
    {
        ia = q / (TH / 2);
        ib = ia;
        ra = 2 * (q % (TH / 2));
        rb = ra + 1;
    }
}

template <int TH, int TW, int PB>
__device__ __forceinline__ void fftColumns(cfx* spec, bool inverse)
{
    using S = FftShape<TH, TW, PB>;
    for(int q = threadIdx.x; q < PB * S::HW; q += blockDim.x)
    {
        cfx* col = spec + (q / S::HW) * S::F + q % S::HW;
        cfx v[TH];
#pragma unroll
        for(int r = 0; r < TH; ++r)
        {
            v[r] = col[r * S::HW];
        }
        if(inverse)
        {
            fftReg<TH, true>(v);
        }
        else
        {
            fftReg<TH, false>(v);
        }
#pragma unroll
        for(int r = 0; r < TH; ++r)
        {
            col[r * S::HW] = v[r];
        }
    }
}

template <int TH, int TW, int PB, class Loader>
__device__ void fft2dForward(cfx* spec, const Loader& loader)
{
    using S = FftShape<TH, TW, PB>;
    float* real = reinterpret_cast<float*>(spec);
    loader.template fill<TH, TW, PB, S::RS>(real);
    __syncthreads();
    // Rows at or past loader.liveRows() are zero in every tile (a filter's span, a dy window's
    // height): their spectra are zero, so only the live row pairs run, packed onto the first
    // threads.
    const int livePairs = TH == 1 ? 1 : (loader.liveRows() + 1) / 2;
    const int activePairs = TH == 1 ? S::PAIRS : PB * livePairs;
    const int q = threadIdx.x;
    const bool active = q < activePairs;
    int ia = 0, ra = 0, ib = 0, rb = 0;
    cfx z[TW];
    if(active)
    {
        if(TH == 1)
        {
            fftPair<TH, PB>(q, ia, ra, ib, rb);
        }
        else
        {
            ia = q / livePairs;
            ib = ia;
            ra = 2 * (q % livePairs);
            rb = ra + 1;
        }
        const float* a = real + (ia * TH + ra) * S::RS;
        const float* b = real + (ib * TH + rb) * S::RS;
#pragma unroll
        for(int i = 0; i < TW; ++i)
        {
            z[i] = cfxMake(a[i], b[i]);
        }
    }
    __syncthreads();
    if(active)
    {
        fftReg<TW, false>(z);
        cfx* sa = spec + ia * S::F + ra * S::HW;
        cfx* sb = spec + ib * S::F + rb * S::HW;
#pragma unroll
        for(int k = 0; k < S::HW; ++k)
        {
            const cfx zk = z[k];
            const cfx zm = cfxConj(z[(TW - k) % TW]);
            const cfx s = cfxAdd(zk, zm);
            const cfx d = cfxSub(zk, zm);
            sa[k] = cfxMake(0.5f * s.x, 0.5f * s.y);
            sb[k] = cfxMake(0.5f * d.y, -0.5f * d.x);
        }
    }
    if(TH > 1)
    {
        const int deadRows = TH - 2 * livePairs;
        for(int idx = threadIdx.x; idx < PB * deadRows * S::HW; idx += blockDim.x)
        {
            const int t = idx / (deadRows * S::HW);
            const int row = 2 * livePairs + (idx / S::HW) % deadRows;
            spec[t * S::F + row * S::HW + idx % S::HW] = cfxMake(0.0f, 0.0f);
        }
        __syncthreads();
        fftColumns<TH, TW, PB>(spec, false);
    }
    __syncthreads();
}

template <int TH, int TW, int PB, class Storer>
__device__ void fft2dInverse(cfx* spec, const Storer& storer)
{
    using S = FftShape<TH, TW, PB>;
    if(TH > 1)
    {
        fftColumns<TH, TW, PB>(spec, true);
        __syncthreads();
    }
    const int q = threadIdx.x;
    const bool active = q < S::PAIRS;
    int ia = 0, ra = 0, ib = 0, rb = 0;
    cfx z[TW];
    if(active)
    {
        fftPair<TH, PB>(q, ia, ra, ib, rb);
        const cfx* sa = spec + ia * S::F + ra * S::HW;
        const cfx* sb = spec + ib * S::F + rb * S::HW;
        // Z = A + iB over the full length, from the half spectra of the two real rows (A, B
        // Hermitian: the bins past TW / 2 are conjugates). Bins 0 and TW / 2 are real.
        z[0] = cfxMake(sa[0].x, sb[0].x);
        z[TW / 2] = cfxMake(sa[TW / 2].x, sb[TW / 2].x);
#pragma unroll
        for(int k = 1; k < TW / 2; ++k)
        {
            const cfx a = sa[k];
            const cfx b = sb[k];
            z[k] = cfxMake(a.x - b.y, a.y + b.x);
            z[TW - k] = cfxMake(a.x + b.y, b.x - a.y);
        }
    }
    __syncthreads();
    float* real = reinterpret_cast<float*>(spec);
    if(active)
    {
        fftReg<TW, true>(z);
        float* a = real + (ia * TH + ra) * S::RS;
        float* b = real + (ib * TH + rb) * S::RS;
#pragma unroll
        for(int i = 0; i < TW; ++i)
        {
            a[i] = z[i].x;
            b[i] = z[i].y;
        }
    }
    __syncthreads();
    storer.template store<TH, TW, PB, S::RS>(real);
}

// ---- loaders and storers ---------------------------------------------------------------------

/// A tile's place: its image and its window origin.
struct FftTile
{
    int image;
    int h0;
    int w0;
    int live;
};

__device__ __forceinline__ FftTile fftTileOf(const ConvFftArgs& a, int local)
{
    FftTile t;
    const int p = a.p0 + local;
    t.live = local < a.pc ? 1 : 0;
    const int per = a.tilesH * a.tilesW;
    t.image = t.live ? p / per : 0;
    const int rem = t.live ? p % per : 0;
    t.h0 = (rem / a.tilesW) * a.oh;
    t.w0 = (rem % a.tilesW) * a.ow;
    return t;
}

/// Fills tiles[0, PB) with the block's tiles (pBlock ..; PB may exceed the threads) and waits.
template <int PB>
__device__ __forceinline__ void fftLoadTiles(FftTile* tiles, const ConvFftArgs& a, int pBlock)
{
    for(int t = threadIdx.x; t < PB; t += blockDim.x)
    {
        tiles[t] = fftTileOf(a, pBlock + t);
    }
    __syncthreads();
}

/// Data windows: point (row, i) of tile t is src (image, channel, h0 - padH + row, w0 - padW +
/// i), read when row < validH and i < validW, zero elsewhere. The source fields are held by value
/// so one kernel can read two sources (backward weights' depthwise path).
template <class T>
struct FftDataLoader
{
    const T* src;
    long long chanOff;
    long long sN;
    long long sH;
    long long sW;
    int srcH;
    int srcW;
    int padH;
    int padW;
    int validH;
    int validW;
    const FftTile* tiles;

    __device__ int liveRows() const
    {
        return validH;
    }

    /// The block's real tiles, cooperatively (consecutive threads, consecutive points of a row).
    template <int TH, int TW, int PB, int RS>
    __device__ void fill(float* real) const
    {
        for(int idx = threadIdx.x; idx < PB * TH * TW; idx += blockDim.x)
        {
            const int t = idx / (TH * TW);
            const int r = (idx / TW) % TH;
            const int i = idx % TW;
            real[(t * TH + r) * RS + i] = value(t, r, i);
        }
    }

    __device__ float value(int t, int row, int i) const
    {
        const FftTile tile = tiles[t];
        const int h = tile.h0 - padH + row;
        const int w = tile.w0 - padW + i;
        if(!tile.live || row >= validH || i >= validW || h < 0 || h >= srcH || w < 0 || w >= srcW)
        {
            return 0.0f;
        }
        return FftElem<T>::load(
            src[chanOff + static_cast<long long>(tile.image) * sN + static_cast<long long>(h) * sH
                + static_cast<long long>(w) * sW]);
    }
};

/// The loader of the launch's main source (src) for channel @p channel (group-major).
template <class T>
__device__ __forceinline__ FftDataLoader<T>
    fftSrcLoader(const ConvFftArgs& a, long long channel, const FftTile* tiles)
{
    return FftDataLoader<T>{static_cast<const T*>(a.src),
                            channel * a.srcStr[1],
                            a.srcStr[0],
                            a.srcStr[2],
                            a.srcStr[3],
                            a.srcH,
                            a.srcW,
                            a.padH,
                            a.padW,
                            a.validH,
                            a.validW,
                            tiles};
}

/// The loader of the second source (src2: the zero-padded dy windows, no padding).
template <class T>
__device__ __forceinline__ FftDataLoader<T>
    fftSrc2Loader(const ConvFftArgs& a, long long channel, const FftTile* tiles)
{
    return FftDataLoader<T>{static_cast<const T*>(a.src2),
                            channel * a.src2Str[1],
                            a.src2Str[0],
                            a.src2Str[2],
                            a.src2Str[3],
                            a.src2H,
                            a.src2W,
                            0,
                            0,
                            a.valid2H,
                            a.valid2W,
                            tiles};
}

/// Filter taps: instance t is (a, b0 + t); the tile holds tap (r, s) at (r * dilH, s * dilW),
/// reflected when flip. Filled as zeros, then the taps scattered (r s of them, not th tw).
template <class T>
struct FftFilterLoader
{
    const T* w;
    const ConvFftArgs* a;
    int g;
    int ia;
    int b0;

    __device__ int liveRows() const
    {
        return (a->r - 1) * a->dilH + 1;
    }

    template <int TH, int TW, int PB, int RS>
    __device__ void fill(float* real) const
    {
        for(int idx = threadIdx.x; idx < PB * TH * RS; idx += blockDim.x)
        {
            real[idx] = 0.0f;
        }
        __syncthreads();
        const int taps = a->r * a->s;
        for(int idx = threadIdx.x; idx < PB * taps; idx += blockDim.x)
        {
            const int t = idx / taps;
            const int r = (idx % taps) / a->s;
            const int s = idx % a->s;
            const int b = b0 + t;
            if(b >= a->cb)
            {
                continue;
            }
            const int k = a->filtMode == 0 ? g * a->cb + b : g * a->ca + ia;
            const int c = a->filtMode == 0 ? ia : b;
            const int sr = a->flip ? a->r - 1 - r : r;
            const int ss = a->flip ? a->s - 1 - s : s;
            real[(t * TH + r * a->dilH) * RS + s * a->dilW]
                = FftElem<T>::load(w[static_cast<long long>(k) * a->filtStr[0]
                                     + static_cast<long long>(c) * a->filtStr[1]
                                     + static_cast<long long>(sr) * a->filtStr[2]
                                     + static_cast<long long>(ss) * a->filtStr[3]]);
        }
    }
};

/// Output tiles: point (row, i) of tile t is (h0 + row, w0 + i), kept when row < oh, i < ow
/// and inside dst.
template <class T, int TH, int TW>
struct FftOutStorer
{
    T* dst;
    long long chanOff;
    const ConvFftArgs* a;
    const FftTile* tiles;

    template <int TH_, int TW_, int PB, int RS>
    __device__ void store(const float* real) const
    {
        for(int idx = threadIdx.x; idx < PB * TH * TW; idx += blockDim.x)
        {
            const int t = idx / (TH * TW);
            const int r = (idx / TW) % TH;
            const int i = idx % TW;
            value(t, r, i, real[(t * TH + r) * RS + i]);
        }
    }

    __device__ void value(int t, int row, int i, float v) const
    {
        const FftTile tile = tiles[t];
        const int h = tile.h0 + row;
        const int w = tile.w0 + i;
        if(!tile.live || row >= a->oh || i >= a->ow || h >= a->dstH || w >= a->dstW)
        {
            return;
        }
        constexpr float scale = 1.0f / static_cast<float>(TH * TW);
        dst[chanOff + static_cast<long long>(tile.image) * a->dstStr[0]
            + static_cast<long long>(h) * a->dstStr[2] + static_cast<long long>(w) * a->dstStr[3]]
            = FftElem<T>::store(v * scale);
    }
};

/// Weight gradient: tap (r, s) of instance t (x channel c0 + t) is lag (r * dilH, s * dilW),
/// written reflected when flip. Gathered tap by tap (r s of them, not th tw).
template <class T, int TH, int TW>
struct FftWgradStorer
{
    T* dst;
    const ConvFftArgs* a;
    int k;
    int c0;

    template <int TH_, int TW_, int PB, int RS>
    __device__ void store(const float* real) const
    {
        constexpr float scale = 1.0f / static_cast<float>(TH * TW);
        const int taps = a->r * a->s;
        for(int idx = threadIdx.x; idx < PB * taps; idx += blockDim.x)
        {
            const int t = idx / taps;
            const int r = (idx % taps) / a->s;
            const int s = idx % a->s;
            const int c = c0 + t;
            if(c >= a->ca)
            {
                continue;
            }
            const int dr = a->flip ? a->r - 1 - r : r;
            const int ds = a->flip ? a->s - 1 - s : s;
            dst[static_cast<long long>(k) * a->dstStr[0] + static_cast<long long>(c) * a->dstStr[1]
                + static_cast<long long>(dr) * a->dstStr[2]
                + static_cast<long long>(ds) * a->dstStr[3]]
                = FftElem<T>::store(real[(t * TH + r * a->dilH) * RS + s * a->dilW] * scale);
        }
    }
};

// ---- the kernels' bodies -----------------------------------------------------------------------

/// Data transform: block = PB tiles of one (group, channel); spectra to wsOut[g][F][chans][pc].
template <class T, int TH, int TW, int PB>
__device__ void fftDataTransform(const ConvFftArgs& a)
{
    constexpr int F = FftShape<TH, TW, PB>::F;
    __shared__ cfx spec[PB * F];
    __shared__ FftTile tiles[PB];
    const int blocksP = (a.pcGrid + PB - 1) / PB; // the grid is sized for a full chunk
    const int gc = blockIdx.x / blocksP; // g * chans + channel
    const int pBlock = (blockIdx.x % blocksP) * PB;
    const int g = gc / a.chans;
    fftLoadTiles<PB>(tiles, a, pBlock);
    fft2dForward<TH, TW, PB>(spec, fftSrcLoader<T>(a, gc, tiles));
    ConvFftComplex* out = a.wsOut;
    const int ch = gc % a.chans;
    for(int q = threadIdx.x; q < F * PB; q += blockDim.x)
    {
        const int f = q / PB;
        const int t = q % PB;
        if(pBlock + t < a.pc)
        {
            out[((static_cast<long long>(g) * F + f) * a.chans + ch) * a.pc + pBlock + t]
                = spec[t * F + f];
        }
    }
}

/// Filter transform: block = PB filters (g, a, b0 .. b0 + PB); conj spectra to
/// wsOut[g][F][ca][cb] (the correlation's filter factor).
template <class T, int TH, int TW, int PB>
__device__ void fftFilterTransform(const ConvFftArgs& a)
{
    constexpr int F = FftShape<TH, TW, PB>::F;
    __shared__ cfx spec[PB * F];
    const int blocksB = (a.cb + PB - 1) / PB;
    const int ga = blockIdx.x / blocksB;
    const int b0 = (blockIdx.x % blocksB) * PB;
    const int g = ga / a.ca;
    const int ia = ga % a.ca;
    FftFilterLoader<T> loader{static_cast<const T*>(a.filt), &a, g, ia, b0};
    fft2dForward<TH, TW, PB>(spec, loader);
    for(int q = threadIdx.x; q < F * PB; q += blockDim.x)
    {
        const int f = q / PB;
        const int t = q % PB;
        if(b0 + t < a.cb)
        {
            a.wsOut[((static_cast<long long>(g) * F + f) * a.ca + ia) * a.cb + b0 + t]
                = cfxConj(spec[t * F + f]);
        }
    }
}

/// Inverse to the output: block = PB tiles of one (group, output channel b).
template <class T, int TH, int TW, int PB>
__device__ void fftOutputInverse(const ConvFftArgs& a)
{
    constexpr int F = FftShape<TH, TW, PB>::F;
    __shared__ cfx spec[PB * F];
    __shared__ FftTile tiles[PB];
    const int blocksP = (a.pcGrid + PB - 1) / PB; // the grid is sized for a full chunk
    const int gb = blockIdx.x / blocksP;
    const int pBlock = (blockIdx.x % blocksP) * PB;
    const int g = gb / a.cb;
    const int b = gb % a.cb;
    for(int t = threadIdx.x; t < PB; t += blockDim.x)
    {
        tiles[t] = fftTileOf(a, pBlock + t);
    }
    for(int q = threadIdx.x; q < F * PB; q += blockDim.x)
    {
        const int f = q / PB;
        const int t = q % PB;
        spec[t * F + f]
            = pBlock + t < a.pc
                  ? a.wsY[((static_cast<long long>(g) * F + f) * a.cb + b) * a.pc + pBlock + t]
                  : cfxMake(0.0f, 0.0f);
    }
    __syncthreads();
    FftOutStorer<T, TH, TW> storer{
        static_cast<T*>(a.dst), static_cast<long long>(gb) * a.dstStr[1], &a, tiles};
    fft2dInverse<TH, TW, PB>(spec, storer);
}

/// Inverse to dw: block = PB x channels (c0 ..) of one (group, dy channel k).
template <class T, int TH, int TW, int PB>
__device__ void fftWgradInverse(const ConvFftArgs& a)
{
    constexpr int F = FftShape<TH, TW, PB>::F;
    __shared__ cfx spec[PB * F];
    const int blocksC = (a.ca + PB - 1) / PB;
    const int gk = blockIdx.x / blocksC;
    const int c0 = (blockIdx.x % blocksC) * PB;
    const int g = gk / a.cb;
    const int k = gk % a.cb;
    for(int q = threadIdx.x; q < F * PB; q += blockDim.x)
    {
        const int f = q / PB;
        const int t = q % PB;
        spec[t * F + f]
            = c0 + t < a.ca
                  ? a.wsY[((static_cast<long long>(g) * F + f) * a.cb + k) * a.ca + c0 + t]
                  : cfxMake(0.0f, 0.0f);
    }
    __syncthreads();
    FftWgradStorer<T, TH, TW> storer{static_cast<T*>(a.dst), &a, gk, c0};
    fft2dInverse<TH, TW, PB>(spec, storer);
}

/// Depthwise forward / backward data (one input and one output channel a group, so no product
/// over channels): block = PB tiles of channel g; forward transform, multiply by the filter
/// spectrum (wsB[g][F], conjugated by the filter transform), inverse to the output tiles.
template <class T, int TH, int TW, int PB>
__device__ void fftDepthwise(const ConvFftArgs& a)
{
    constexpr int F = FftShape<TH, TW, PB>::F;
    __shared__ cfx spec[PB * F];
    __shared__ FftTile tiles[PB];
    const int blocksP = (a.pcGrid + PB - 1) / PB; // the grid is sized for a full chunk
    const int g = blockIdx.x / blocksP;
    const int pBlock = (blockIdx.x % blocksP) * PB;
    fftLoadTiles<PB>(tiles, a, pBlock);
    fft2dForward<TH, TW, PB>(spec, fftSrcLoader<T>(a, g, tiles));
    const cfx* filter = a.wsB + static_cast<long long>(g) * F;
    for(int q = threadIdx.x; q < F * PB; q += blockDim.x)
    {
        const int f = q / PB;
        const int t = q % PB;
        spec[t * F + f] = cfxMul(spec[t * F + f], filter[f]);
    }
    __syncthreads();
    FftOutStorer<T, TH, TW> storer{
        static_cast<T*>(a.dst), static_cast<long long>(g) * a.dstStr[1], &a, tiles};
    fft2dInverse<TH, TW, PB>(spec, storer);
}

/// Depthwise backward weights, partial sums: block = (channel g, partition of tilesPerPart
/// tiles). Per PB tiles: x windows (src) and zero-padded dy windows (src2) to spectra, then
/// conj(DY) X summed per bin in registers (thread tid owns bins tid, tid + TB, ...), in tile
/// order; the partition's sums go to wsOut[part][g][F].
template <class T, int TH, int TW, int PB, int TB>
__device__ void fftDepthwiseWgrad(const ConvFftArgs& a)
{
    constexpr int F = FftShape<TH, TW, PB>::F;
    constexpr int PER = (F + TB - 1) / TB;
    __shared__ cfx specX[PB * F];
    __shared__ cfx specD[PB * F];
    __shared__ FftTile tiles[PB];
    const int g = blockIdx.x / a.parts;
    const int part = blockIdx.x % a.parts;
    const int first = part * a.tilesPerPart;
    const int last = first + a.tilesPerPart < a.pc ? first + a.tilesPerPart : a.pc;
    cfx acc[PER];
#pragma unroll
    for(int i = 0; i < PER; ++i)
    {
        acc[i] = cfxMake(0.0f, 0.0f);
    }
    for(int pBlock = first; pBlock < last; pBlock += PB)
    {
        fftLoadTiles<PB>(tiles, a, pBlock);
        fft2dForward<TH, TW, PB>(specX, fftSrcLoader<T>(a, g, tiles));
        fft2dForward<TH, TW, PB>(specD, fftSrc2Loader<T>(a, g, tiles));
#pragma unroll
        for(int i = 0; i < PER; ++i)
        {
            const int f = static_cast<int>(threadIdx.x) + i * TB;
            if(f < F)
            {
                for(int t = 0; t < PB && pBlock + t < last; ++t)
                {
                    cfxMac(acc[i], cfxConj(specD[t * F + f]), specX[t * F + f]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int i = 0; i < PER; ++i)
    {
        const int f = static_cast<int>(threadIdx.x) + i * TB;
        if(f < F)
        {
            a.wsOut[(static_cast<long long>(part) * a.groups + g) * F + f] = acc[i];
        }
    }
}

/// Depthwise backward weights, fold: wsY[g][F] = the partitions' sums in partition order.
template <int TH, int TW>
__device__ void fftDepthwiseFold(const ConvFftArgs& a)
{
    constexpr int F = TH * (TW / 2 + 1);
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(a.groups) * F;
    if(i >= total)
    {
        return;
    }
    cfx sum = cfxMake(0.0f, 0.0f);
    for(int part = 0; part < a.parts; ++part)
    {
        sum = cfxAdd(sum, a.wsA[part * total + i]);
    }
    a.wsY[i] = sum;
}

/// Forward / backward-data product: Y[g][f][b][p] = sum_a A[g][f][a][p] * B[g][f][a][b] (B
/// already conjugated). Block tile BM tiles x BN output channels, TM x 4 a thread (TM 4 or 8; BM
/// = 256 / (BN / 4) x TM), 256 threads; the next BK channels are fetched into
/// registers while the current ones are multiplied. grid.x = (groups * F) * tilesN * tilesM.
template <int BN, int TMV>
struct FftProductTile
{
    static constexpr int TM = TMV;
    static constexpr int TN = 4;
    static constexpr int TXM = 256 / (BN / TN);
    static constexpr int BM = TXM * TM;
};
template <int BN, int TMV>
__device__ void fftProduct(const ConvFftArgs& a)
{
    constexpr int TM = FftProductTile<BN, TMV>::TM;
    constexpr int TN = FftProductTile<BN, TMV>::TN;
    constexpr int TXM = FftProductTile<BN, TMV>::TXM;
    constexpr int BM = FftProductTile<BN, TMV>::BM;
    constexpr int BK = BN == 16 ? 8 : 16;
    constexpr int LA = BK * BM / 256;
    constexpr int LB = (BK * BN + 255) / 256;
    __shared__ cfx sa[BK][BM];
    __shared__ cfx sb[BK][BN];
    const int tilesM = (a.pcGrid + BM - 1) / BM; // the grid is sized for a full chunk
    const int tilesN = (a.cb + BN - 1) / BN;
    int bid = blockIdx.x;
    const int tm = bid % tilesM;
    bid /= tilesM;
    const int tn = bid % tilesN;
    const long long gf = bid / tilesN;
    const int m0 = tm * BM;
    const int n0 = tn * BN;
    const cfx* A = a.wsA + gf * a.ca * a.pc;
    const cfx* B = a.wsB + gf * a.ca * a.cb;
    cfx* Y = a.wsY + gf * a.cb * a.pc;
    const int tid = threadIdx.x;
    const int tx = tid % TXM;
    const int ty = tid / TXM;
    cfx ra[LA], rb[LB];
    const auto fetch = [&](int k0) {
#pragma unroll
        for(int j = 0; j < LA; ++j)
        {
            const int idx = tid + 256 * j;
            const int kk = idx / BM;
            const int m = idx % BM;
            ra[j] = (k0 + kk < a.ca && m0 + m < a.pc)
                        ? A[static_cast<long long>(k0 + kk) * a.pc + m0 + m]
                        : cfxMake(0.0f, 0.0f);
        }
#pragma unroll
        for(int j = 0; j < LB; ++j)
        {
            const int idx = tid + 256 * j;
            const int kk = idx / BN;
            const int nn = idx % BN;
            rb[j] = (idx < BK * BN && k0 + kk < a.ca && n0 + nn < a.cb)
                        ? B[static_cast<long long>(k0 + kk) * a.cb + n0 + nn]
                        : cfxMake(0.0f, 0.0f);
        }
    };
    cfx acc[TM][TN];
#pragma unroll
    for(int i = 0; i < TM; ++i)
#pragma unroll
        for(int j = 0; j < TN; ++j)
            acc[i][j] = cfxMake(0.0f, 0.0f);
    fetch(0);
    for(int k0 = 0; k0 < a.ca; k0 += BK)
    {
#pragma unroll
        for(int j = 0; j < LA; ++j)
        {
            const int idx = tid + 256 * j;
            sa[idx / BM][idx % BM] = ra[j];
        }
#pragma unroll
        for(int j = 0; j < LB; ++j)
        {
            const int idx = tid + 256 * j;
            if(idx < BK * BN)
            {
                sb[idx / BN][idx % BN] = rb[j];
            }
        }
        __syncthreads();
        if(k0 + BK < a.ca)
        {
            fetch(k0 + BK);
        }
#pragma unroll
        for(int kk = 0; kk < BK; ++kk)
        {
            cfx va[TM], vb[TN];
#pragma unroll
            for(int i = 0; i < TM; ++i)
            {
                va[i] = sa[kk][tx * TM + i];
            }
#pragma unroll
            for(int j = 0; j < TN; ++j)
            {
                vb[j] = sb[kk][ty * TN + j];
            }
#pragma unroll
            for(int i = 0; i < TM; ++i)
#pragma unroll
                for(int j = 0; j < TN; ++j)
                    cfxMac(acc[i][j], va[i], vb[j]);
        }
        __syncthreads();
    }
#pragma unroll
    for(int j = 0; j < TN; ++j)
    {
        const int nn = n0 + ty * TN + j;
        if(nn >= a.cb)
        {
            continue;
        }
#pragma unroll
        for(int i = 0; i < TM; ++i)
        {
            const int m = m0 + tx * TM + i;
            if(m < a.pc)
            {
                Y[static_cast<long long>(nn) * a.pc + m] = acc[i][j];
            }
        }
    }
}

/// Backward-weights product: W[g][f][k][c] (+)= sum_p conj(D[g][f][k][p]) * A[g][f][c][p]
/// (A = x spectra, D = dy spectra in wsB). Block tile BM x channels by BN dy channels.
template <int BN>
__device__ void fftWgradProduct(const ConvFftArgs& a)
{
    constexpr int BM = 4096 / BN;
    constexpr int BK = 16;
    constexpr int TXM = BM / 4;
    __shared__ cfx sa[BK][BM + 1];
    __shared__ cfx sd[BK][BN + 1];
    const int tilesM = (a.ca + BM - 1) / BM;
    const int tilesN = (a.cb + BN - 1) / BN;
    int bid = blockIdx.x;
    const int tm = bid % tilesM;
    bid /= tilesM;
    const int tn = bid % tilesN;
    const long long gf = bid / tilesN;
    const int m0 = tm * BM;
    const int n0 = tn * BN;
    const cfx* A = a.wsA + gf * a.ca * a.pc;
    const cfx* D = a.wsB + gf * a.cb * a.pc;
    cfx* Y = a.wsY + gf * a.cb * a.ca;
    const int tx = threadIdx.x % TXM;
    const int ty = threadIdx.x / TXM;
    cfx acc[4][4];
#pragma unroll
    for(int i = 0; i < 4; ++i)
#pragma unroll
        for(int j = 0; j < 4; ++j)
            acc[i][j] = cfxMake(0.0f, 0.0f);
    for(int k0 = 0; k0 < a.pc; k0 += BK)
    {
        for(int idx = threadIdx.x; idx < BK * BM; idx += 256)
        {
            const int m = idx / BK;
            const int kk = idx % BK;
            sa[kk][m] = (k0 + kk < a.pc && m0 + m < a.ca)
                            ? A[static_cast<long long>(m0 + m) * a.pc + k0 + kk]
                            : cfxMake(0.0f, 0.0f);
        }
        for(int idx = threadIdx.x; idx < BK * BN; idx += 256)
        {
            const int nn = idx / BK;
            const int kk = idx % BK;
            sd[kk][nn] = (k0 + kk < a.pc && n0 + nn < a.cb)
                             ? cfxConj(D[static_cast<long long>(n0 + nn) * a.pc + k0 + kk])
                             : cfxMake(0.0f, 0.0f);
        }
        __syncthreads();
#pragma unroll
        for(int kk = 0; kk < BK; ++kk)
        {
            cfx va[4], vd[4];
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                va[i] = sa[kk][tx * 4 + i];
                vd[i] = sd[kk][ty * 4 + i];
            }
#pragma unroll
            for(int i = 0; i < 4; ++i)
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    cfxMac(acc[i][j], vd[j], va[i]);
        }
        __syncthreads();
    }
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        const int nn = n0 + ty * 4 + j;
        if(nn >= a.cb)
        {
            continue;
        }
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const int m = m0 + tx * 4 + i;
            if(m < a.ca)
            {
                cfx* out = Y + static_cast<long long>(nn) * a.ca + m;
                *out = a.accumulate ? cfxAdd(*out, acc[i][j]) : acc[i][j];
            }
        }
    }
}
