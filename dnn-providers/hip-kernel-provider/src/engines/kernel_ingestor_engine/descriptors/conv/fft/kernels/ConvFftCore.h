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
// Real rows go through the FFT in pairs: two rows of one tile (TH > 1) or the single rows of
// two tiles (TH == 1, PB even). Loader::row(ia, ra, ib, rb, z) fills z[i] = (a_i, b_i) with
// row ra of tile ia and row rb of tile ib; Storer::row(...) takes them back (unscaled).

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

template <int TH, int TW, int PB, class Loader>
__device__ void fft2dForward(cfx* spec, const Loader& loader)
{
    constexpr int HW = TW / 2 + 1;
    constexpr int F = TH * HW;
    constexpr int PAIRS = TH == 1 ? PB / 2 : PB * TH / 2;
    for(int q = threadIdx.x; q < PAIRS; q += blockDim.x)
    {
        int ia, ra, ib, rb;
        fftPair<TH, PB>(q, ia, ra, ib, rb);
        cfx z[TW];
        loader.row(ia, ra, ib, rb, z);
        fftReg<TW, false>(z);
        cfx* sa = spec + ia * F + ra * HW;
        cfx* sb = spec + ib * F + rb * HW;
#pragma unroll
        for(int k = 0; k < HW; ++k)
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
        __syncthreads();
        for(int q = threadIdx.x; q < PB * HW; q += blockDim.x)
        {
            cfx* col = spec + (q / HW) * F + q % HW;
            cfx v[TH];
#pragma unroll
            for(int r = 0; r < TH; ++r)
            {
                v[r] = col[r * HW];
            }
            fftReg<TH, false>(v);
#pragma unroll
            for(int r = 0; r < TH; ++r)
            {
                col[r * HW] = v[r];
            }
        }
    }
    __syncthreads();
}

template <int TH, int TW, int PB, class Storer>
__device__ void fft2dInverse(cfx* spec, const Storer& storer)
{
    constexpr int HW = TW / 2 + 1;
    constexpr int F = TH * HW;
    constexpr int PAIRS = TH == 1 ? PB / 2 : PB * TH / 2;
    if(TH > 1)
    {
        for(int q = threadIdx.x; q < PB * HW; q += blockDim.x)
        {
            cfx* col = spec + (q / HW) * F + q % HW;
            cfx v[TH];
#pragma unroll
            for(int r = 0; r < TH; ++r)
            {
                v[r] = col[r * HW];
            }
            fftReg<TH, true>(v);
#pragma unroll
            for(int r = 0; r < TH; ++r)
            {
                col[r * HW] = v[r];
            }
        }
        __syncthreads();
    }
    for(int q = threadIdx.x; q < PAIRS; q += blockDim.x)
    {
        int ia, ra, ib, rb;
        fftPair<TH, PB>(q, ia, ra, ib, rb);
        const cfx* sa = spec + ia * F + ra * HW;
        const cfx* sb = spec + ib * F + rb * HW;
        cfx z[TW];
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
        fftReg<TW, true>(z);
        storer.row(ia, ra, ib, rb, z);
    }
}

// ---- loaders and storers ---------------------------------------------------------------------

/// A tile's place: its image and its window origin.
struct FftTile
{
    int image;
    int h0;
    int w0;
    bool live;
};

__device__ __forceinline__ FftTile fftTileOf(const ConvFftArgs& a, int local)
{
    FftTile t;
    const int p = a.p0 + local;
    t.live = local < a.pc;
    const int per = a.tilesH * a.tilesW;
    t.image = t.live ? p / per : 0;
    const int rem = t.live ? p % per : 0;
    t.h0 = (rem / a.tilesW) * a.oh;
    t.w0 = (rem % a.tilesW) * a.ow;
    return t;
}

/// Data windows: rows of src (n, channel, h, w), window origin (h0 - padH, w0 - padW), the
/// first validH x validW points of the window, zero elsewhere. The source fields are held by
/// value so one kernel can read two sources (backward weights' depthwise path); the tiles come
/// from the launch's arguments.
template <class T, int TW, int PB>
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
    const ConvFftArgs* a;
    int pBlock;

    __device__ void one(int inst, int row, cfx* z, bool imag) const
    {
        const FftTile t = fftTileOf(*a, pBlock + inst);
        const int h = t.h0 - padH + row;
        const bool rowOk = t.live && row < validH && h >= 0 && h < srcH;
        const T* base
            = src + chanOff + static_cast<long long>(t.image) * sN + static_cast<long long>(h) * sH;
        const int w0 = t.w0 - padW;
#pragma unroll
        for(int i = 0; i < TW; ++i)
        {
            const int w = w0 + i;
            const float v = (rowOk && i < validW && w >= 0 && w < srcW)
                                ? FftElem<T>::load(base[static_cast<long long>(w) * sW])
                                : 0.0f;
            if(imag)
            {
                z[i].y = v;
            }
            else
            {
                z[i].x = v;
            }
        }
    }

    __device__ void row(int ia, int ra, int ib, int rb, cfx* z) const
    {
        one(ia, ra, z, false);
        one(ib, rb, z, true);
    }
};

/// The loader of the launch's main source (src) for channel @p channel (group-major).
template <class T, int TW, int PB>
__device__ __forceinline__ FftDataLoader<T, TW, PB>
    fftSrcLoader(const ConvFftArgs& a, long long channel, int pBlock)
{
    return FftDataLoader<T, TW, PB>{static_cast<const T*>(a.src),
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
                                    &a,
                                    pBlock};
}

/// The loader of the second source (src2: the zero-padded dy windows, no padding).
template <class T, int TW, int PB>
__device__ __forceinline__ FftDataLoader<T, TW, PB>
    fftSrc2Loader(const ConvFftArgs& a, long long channel, int pBlock)
{
    return FftDataLoader<T, TW, PB>{static_cast<const T*>(a.src2),
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
                                    &a,
                                    pBlock};
}

/// Filter taps: instance t is (a, b0 + t); the tile holds tap (r, s) at (r * dilH, s * dilW),
/// reflected when flip.
template <class T, int TW, int PB>
struct FftFilterLoader
{
    const T* w;
    const ConvFftArgs* a;
    int g;
    int ia;
    int b0;

    __device__ void one(int inst, int row, cfx* z, bool imag) const
    {
        const int b = b0 + inst;
        const bool live = b < a->cb && row % a->dilH == 0 && row / a->dilH < a->r;
        int r = row / a->dilH;
        if(a->flip)
        {
            r = a->r - 1 - r;
        }
        const int k = a->filtMode == 0 ? g * a->cb + b : g * a->ca + ia;
        const int c = a->filtMode == 0 ? ia : b;
        const T* base = w + static_cast<long long>(k) * a->filtStr[0]
                        + static_cast<long long>(c) * a->filtStr[1]
                        + static_cast<long long>(r) * a->filtStr[2];
#pragma unroll
        for(int i = 0; i < TW; ++i)
        {
            int s = i / a->dilW;
            const bool ok = live && i % a->dilW == 0 && s < a->s;
            if(a->flip)
            {
                s = a->s - 1 - s;
            }
            const float v
                = ok ? FftElem<T>::load(base[static_cast<long long>(s) * a->filtStr[3]]) : 0.0f;
            if(imag)
            {
                z[i].y = v;
            }
            else
            {
                z[i].x = v;
            }
        }
    }

    __device__ void row(int ia_, int ra, int ib, int rb, cfx* z) const
    {
        one(ia_, ra, z, false);
        one(ib, rb, z, true);
    }
};

/// Output tiles: point (row, i) of tile t is (h0 + row, w0 + i), kept when row < oh, i < ow
/// and inside dst.
template <class T, int TH, int TW, int PB>
struct FftOutStorer
{
    T* dst;
    long long chanOff;
    const ConvFftArgs* a;
    int pBlock;

    __device__ void one(int inst, int row, const cfx* z, bool imag) const
    {
        const FftTile t = fftTileOf(*a, pBlock + inst);
        const int h = t.h0 + row;
        if(!t.live || row >= a->oh || h >= a->dstH)
        {
            return;
        }
        constexpr float scale = 1.0f / static_cast<float>(TH * TW);
        T* base = dst + chanOff + static_cast<long long>(t.image) * a->dstStr[0]
                  + static_cast<long long>(h) * a->dstStr[2];
#pragma unroll
        for(int i = 0; i < TW; ++i)
        {
            const int w = t.w0 + i;
            if(i < a->ow && w < a->dstW)
            {
                base[static_cast<long long>(w) * a->dstStr[3]]
                    = FftElem<T>::store((imag ? z[i].y : z[i].x) * scale);
            }
        }
    }

    __device__ void row(int ia, int ra, int ib, int rb, const cfx* z) const
    {
        one(ia, ra, z, false);
        one(ib, rb, z, true);
    }
};

/// Weight gradient: lag (row, i) of instance t (x channel c0 + t) is tap (row / dilH, i / dilW)
/// when both divide, reflected when flip.
template <class T, int TH, int TW, int PB>
struct FftWgradStorer
{
    T* dst;
    const ConvFftArgs* a;
    int k;
    int c0;

    __device__ void one(int inst, int row, const cfx* z, bool imag) const
    {
        const int c = c0 + inst;
        if(c >= a->ca || row % a->dilH != 0 || row / a->dilH >= a->r)
        {
            return;
        }
        int r = row / a->dilH;
        if(a->flip)
        {
            r = a->r - 1 - r;
        }
        constexpr float scale = 1.0f / static_cast<float>(TH * TW);
        T* base = dst + static_cast<long long>(k) * a->dstStr[0]
                  + static_cast<long long>(c) * a->dstStr[1]
                  + static_cast<long long>(r) * a->dstStr[2];
#pragma unroll
        for(int i = 0; i < TW; ++i)
        {
            if(i % a->dilW == 0 && i / a->dilW < a->s)
            {
                const int s = a->flip ? a->s - 1 - i / a->dilW : i / a->dilW;
                base[static_cast<long long>(s) * a->dstStr[3]]
                    = FftElem<T>::store((imag ? z[i].y : z[i].x) * scale);
            }
        }
    }

    __device__ void row(int ia, int ra, int ib, int rb, const cfx* z) const
    {
        one(ia, ra, z, false);
        one(ib, rb, z, true);
    }
};

// ---- the kernels' bodies -----------------------------------------------------------------------

/// Data transform: block = PB tiles of one (group, channel); spectra to wsOut[g][F][chans][pc].
template <class T, int TH, int TW, int PB>
__device__ void fftDataTransform(const ConvFftArgs& a)
{
    constexpr int F = TH * (TW / 2 + 1);
    __shared__ cfx spec[PB * F];
    const int blocksP = (a.pc + PB - 1) / PB;
    const int gc = blockIdx.x / blocksP; // g * chans + channel
    const int pBlock = (blockIdx.x % blocksP) * PB;
    const int g = gc / a.chans;
    fft2dForward<TH, TW, PB>(spec, fftSrcLoader<T, TW, PB>(a, gc, pBlock));
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
    constexpr int F = TH * (TW / 2 + 1);
    __shared__ cfx spec[PB * F];
    const int blocksB = (a.cb + PB - 1) / PB;
    const int ga = blockIdx.x / blocksB;
    const int b0 = (blockIdx.x % blocksB) * PB;
    const int g = ga / a.ca;
    const int ia = ga % a.ca;
    FftFilterLoader<T, TW, PB> loader{static_cast<const T*>(a.filt), &a, g, ia, b0};
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
    constexpr int F = TH * (TW / 2 + 1);
    __shared__ cfx spec[PB * F];
    const int blocksP = (a.pc + PB - 1) / PB;
    const int gb = blockIdx.x / blocksP;
    const int pBlock = (blockIdx.x % blocksP) * PB;
    const int g = gb / a.cb;
    const int b = gb % a.cb;
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
    FftOutStorer<T, TH, TW, PB> storer{
        static_cast<T*>(a.dst), static_cast<long long>(gb) * a.dstStr[1], &a, pBlock};
    fft2dInverse<TH, TW, PB>(spec, storer);
}

/// Inverse to dw: block = PB x channels (c0 ..) of one (group, dy channel k).
template <class T, int TH, int TW, int PB>
__device__ void fftWgradInverse(const ConvFftArgs& a)
{
    constexpr int F = TH * (TW / 2 + 1);
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
    FftWgradStorer<T, TH, TW, PB> storer{static_cast<T*>(a.dst), &a, gk, c0};
    fft2dInverse<TH, TW, PB>(spec, storer);
}

/// Depthwise forward / backward data (one input and one output channel a group, so no product
/// over channels): block = PB tiles of channel g; forward transform, multiply by the filter
/// spectrum (wsB[g][F], conjugated by the filter transform), inverse to the output tiles.
template <class T, int TH, int TW, int PB>
__device__ void fftDepthwise(const ConvFftArgs& a)
{
    constexpr int F = TH * (TW / 2 + 1);
    __shared__ cfx spec[PB * F];
    const int blocksP = (a.pc + PB - 1) / PB;
    const int g = blockIdx.x / blocksP;
    const int pBlock = (blockIdx.x % blocksP) * PB;
    fft2dForward<TH, TW, PB>(spec, fftSrcLoader<T, TW, PB>(a, g, pBlock));
    const cfx* filter = a.wsB + static_cast<long long>(g) * F;
    for(int q = threadIdx.x; q < F * PB; q += blockDim.x)
    {
        const int f = q / PB;
        const int t = q % PB;
        spec[t * F + f] = cfxMul(spec[t * F + f], filter[f]);
    }
    __syncthreads();
    FftOutStorer<T, TH, TW, PB> storer{
        static_cast<T*>(a.dst), static_cast<long long>(g) * a.dstStr[1], &a, pBlock};
    fft2dInverse<TH, TW, PB>(spec, storer);
}

/// Depthwise backward weights, partial sums: block = (channel g, partition of tilesPerPart
/// tiles). Per PB tiles: x windows (src) and zero-padded dy windows (src2) to spectra, then
/// conj(DY) X summed per bin in registers (thread tid owns bins tid, tid + TB, ...), in tile
/// order; the partition's sums go to wsOut[part][g][F].
template <class T, int TH, int TW, int PB, int TB>
__device__ void fftDepthwiseWgrad(const ConvFftArgs& a)
{
    constexpr int F = TH * (TW / 2 + 1);
    constexpr int PER = (F + TB - 1) / TB;
    __shared__ cfx specX[PB * F];
    __shared__ cfx specD[PB * F];
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
        fft2dForward<TH, TW, PB>(specX, fftSrcLoader<T, TW, PB>(a, g, pBlock));
        fft2dForward<TH, TW, PB>(specD, fftSrc2Loader<T, TW, PB>(a, g, pBlock));
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
/// already conjugated). Block tile BM tiles x BN output channels, 4 x 4 a thread, 256 threads.
/// grid.x = (groups * F) * tilesN * tilesM.
template <int BN>
__device__ void fftProduct(const ConvFftArgs& a, int bins)
{
    constexpr int BM = 4096 / BN;
    constexpr int BK = 8;
    constexpr int TXM = BM / 4;
    __shared__ cfx sa[BK][BM];
    __shared__ cfx sb[BK][BN];
    const int tilesM = (a.pc + BM - 1) / BM;
    const int tilesN = (a.cb + BN - 1) / BN;
    int bid = blockIdx.x;
    const int tm = bid % tilesM;
    bid /= tilesM;
    const int tn = bid % tilesN;
    const long long gf = bid / tilesN;
    (void)bins;
    const int m0 = tm * BM;
    const int n0 = tn * BN;
    const cfx* A = a.wsA + gf * a.ca * a.pc;
    const cfx* B = a.wsB + gf * a.ca * a.cb;
    cfx* Y = a.wsY + gf * a.cb * a.pc;
    const int tx = threadIdx.x % TXM;
    const int ty = threadIdx.x / TXM;
    cfx acc[4][4];
#pragma unroll
    for(int i = 0; i < 4; ++i)
#pragma unroll
        for(int j = 0; j < 4; ++j)
            acc[i][j] = cfxMake(0.0f, 0.0f);
    for(int k0 = 0; k0 < a.ca; k0 += BK)
    {
        for(int idx = threadIdx.x; idx < BK * BM; idx += 256)
        {
            const int kk = idx / BM;
            const int m = idx % BM;
            sa[kk][m] = (k0 + kk < a.ca && m0 + m < a.pc)
                            ? A[static_cast<long long>(k0 + kk) * a.pc + m0 + m]
                            : cfxMake(0.0f, 0.0f);
        }
        for(int idx = threadIdx.x; idx < BK * BN; idx += 256)
        {
            const int kk = idx / BN;
            const int nn = idx % BN;
            sb[kk][nn] = (k0 + kk < a.ca && n0 + nn < a.cb)
                             ? B[static_cast<long long>(k0 + kk) * a.cb + n0 + nn]
                             : cfxMake(0.0f, 0.0f);
        }
        __syncthreads();
#pragma unroll
        for(int kk = 0; kk < BK; ++kk)
        {
            cfx va[4], vb[4];
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                va[i] = sa[kk][tx * 4 + i];
                vb[i] = sb[kk][ty * 4 + i];
            }
#pragma unroll
            for(int i = 0; i < 4; ++i)
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    cfxMac(acc[i][j], va[i], vb[j]);
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
