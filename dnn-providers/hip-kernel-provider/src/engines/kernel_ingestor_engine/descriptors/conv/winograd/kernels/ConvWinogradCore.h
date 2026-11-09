// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Winograd convolution core: the F(2,3) / F(4,3) transforms and the multipass engine F(m,3), 3x3 stride 1 (dilation 1), forward and backward data.
// HIPRTC-compatible: no standard headers.
//
// Every launch is written for a "forward-form" problem: in (N, CI, HI, WI) -> out (N, CO, HO, WO),
// 3x3 correlation with top/left padding (ph, pw), groups. HO/WO are explicit, so any padding
// (including negative, i.e. cropping) works: reads outside the input are zero.
//   forward:       in = x,  out = y,  CI = C, CO = K, pad = (p_h, p_w).
//   backward data: in = dy, out = dx, CI = K, CO = C, pad = (2 - p_h, 2 - p_w); the filter is
//                  rotated 180 degrees and its K/C roles swapped inside the filter transform's
//                  address walk (no pre-pass).
// Layouts: tensors NCHW (filter KCRS) or NHWC (filter KRSC); only the loaders/stores differ.
// The workspace layout is the same for both.
//
// Passes (P = (m+2)^2 Winograd positions, T = N * ceil(HO/m) * ceil(WO/m) tiles,
// cgi = CI/G, cgo = CO/G; Op = operand type = the I/O dtype, rounded once after the fp32 transform):
//   1. filter transform  w -> U[G][P][cgo][cgi]  (Op)
//   2. input transform   in -> V[G][P][cgi][T]   (Op)
//   3. batched GEMM      M[G][P][cgo][T] (fp32) = U[g][p] (cgo x cgi) * V[g][p] (cgi x T),
//                        batch = G * P, row-major, fp32 accumulate:
//                          A = U: lda = cgi, strideA = cgo * cgi
//                          B = V: ldb = T,   strideB = cgi * T
//                          C = M: ldc = T,   strideC = cgo * T
//                        Column-major BLAS (rocBLAS/hipBLASLt) sees the same memory as
//                        M^T = V^T * U^T: gemm(N, N, m = T, n = cgo, k = cgi,
//                        A = V (lda T), B = U (ldb cgi), C = M (ldc T)) with no transposes.
//   4. output transform  M -> out
// All transform arithmetic is fp32; the GEMM accumulates in fp32.
#pragma once

#include "ConvWinogradArgs.h"

typedef _Float16 k5_half;
typedef unsigned short k5_bf16; // bf16 bits

// ---------------------------------------------------------------------------------------------
// Element conversion (same rounding as K1)

__device__ inline float k5BfToFloat(unsigned short b)
{
    return __uint_as_float(static_cast<unsigned>(b) << 16);
}
__device__ inline unsigned short k5FloatToBf(float v)
{
    unsigned u = __float_as_uint(v);
    if((u & 0x7f800000u) == 0x7f800000u)
        return static_cast<unsigned short>((u >> 16) | ((u & 0xffffu) != 0 ? 0x40u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<unsigned short>(u >> 16);
}

template <class T>
struct K5Elem;
template <>
struct K5Elem<float>
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
struct K5Elem<k5_half>
{
    __device__ static float load(k5_half v)
    {
        return static_cast<float>(v);
    }
    __device__ static k5_half store(float v)
    {
        return static_cast<k5_half>(v);
    }
};
template <>
struct K5Elem<k5_bf16>
{
    __device__ static float load(k5_bf16 v)
    {
        return k5BfToFloat(v);
    }
    __device__ static k5_bf16 store(float v)
    {
        return k5FloatToBf(v);
    }
};

// ---------------------------------------------------------------------------------------------
// 1-D transforms. bt: B^T d (A -> A), g: G w (3 -> A), at: A^T m (A -> M). Strided in and out so
// the same code runs the row and the column pass.

template <int M>
struct K5Wino;

template <>
struct K5Wino<2> // F(2,3), Lavin
{
    static constexpr int A = 4;
    __device__ static void bt(const float* d, int ds, float* o, int os)
    {
        const float d0 = d[0], d1 = d[ds], d2 = d[2 * ds], d3 = d[3 * ds];
        o[0] = d0 - d2;
        o[os] = d1 + d2;
        o[2 * os] = d2 - d1;
        o[3 * os] = d1 - d3;
    }
    __device__ static void g(const float* w, int ws, float* o, int os)
    {
        const float w0 = w[0], w1 = w[ws], w2 = w[2 * ws];
        const float t = 0.5f * (w0 + w2);
        o[0] = w0;
        o[os] = t + 0.5f * w1;
        o[2 * os] = t - 0.5f * w1;
        o[3 * os] = w2;
    }
    __device__ static void at(const float* m, int ms, float* o, int os)
    {
        const float m0 = m[0], m1 = m[ms], m2 = m[2 * ms], m3 = m[3 * ms];
        o[0] = m0 + m1 + m2;
        o[os] = m1 - m2 - m3;
    }
};

template <>
struct K5Wino<4> // F(4,3), Lavin
{
    static constexpr int A = 6;
    __device__ static void bt(const float* d, int ds, float* o, int os)
    {
        const float d0 = d[0], d1 = d[ds], d2 = d[2 * ds], d3 = d[3 * ds], d4 = d[4 * ds],
                    d5 = d[5 * ds];
        o[0] = 4.0f * d0 - 5.0f * d2 + d4;
        o[os] = -4.0f * (d1 + d2) + d3 + d4;
        o[2 * os] = 4.0f * (d1 - d2) - d3 + d4;
        o[3 * os] = 2.0f * (d3 - d1) - d2 + d4;
        o[4 * os] = 2.0f * (d1 - d3) - d2 + d4;
        o[5 * os] = 4.0f * d1 - 5.0f * d3 + d5;
    }
    __device__ static void g(const float* w, int ws, float* o, int os)
    {
        const float w0 = w[0], w1 = w[ws], w2 = w[2 * ws];
        o[0] = 0.25f * w0;
        o[os] = (-1.0f / 6.0f) * (w0 + w1 + w2);
        o[2 * os] = (-1.0f / 6.0f) * (w0 - w1 + w2);
        const float e = (1.0f / 24.0f) * w0 + (1.0f / 6.0f) * w2;
        o[3 * os] = e + (1.0f / 12.0f) * w1;
        o[4 * os] = e - (1.0f / 12.0f) * w1;
        o[5 * os] = w2;
    }
    __device__ static void at(const float* m, int ms, float* o, int os)
    {
        const float m0 = m[0], m1 = m[ms], m2 = m[2 * ms], m3 = m[3 * ms], m4 = m[4 * ms],
                    m5 = m[5 * ms];
        const float s12 = m1 + m2, d12 = m1 - m2, s34 = m3 + m4, d34 = m3 - m4;
        o[0] = m0 + s12 + s34;
        o[os] = d12 + 2.0f * d34;
        o[2 * os] = s12 + 4.0f * s34;
        o[3 * os] = d12 + 8.0f * d34 + m5;
    }
};

template <>
struct K5Wino<3> // F(3,2), for the stride-2 phase sub-filters. B^T is F(2,3)'s: the Toom-Cook
    // form's last rows of B^T and G are negated together, which leaves the products
{
    static constexpr int A = 4;
    __device__ static void bt(const float* d, int ds, float* o, int os)
    {
        K5Wino<2>::bt(d, ds, o, os);
    }
    __device__ static void g(const float* w, int ws, float* o, int os)
    {
        const float w0 = w[0], w1 = w[ws];
        o[0] = w0;
        o[os] = 0.5f * (w0 + w1);
        o[2 * os] = 0.5f * (w0 - w1);
        o[3 * os] = -w1;
    }
    __device__ static void at(const float* m, int ms, float* o, int os)
    {
        const float m0 = m[0], m1 = m[ms], m2 = m[2 * ms], m3 = m[3 * ms];
        o[0] = m0 + m1 + m2;
        o[os] = m1 - m2;
        o[2 * os] = m1 + m2 + m3;
    }
};

/// Filter taps of a tile: F(3,2) runs the 2x2 stride-2 phase sub-filters, the others 3x3.
template <int M>
struct K5Taps
{
    static constexpr int R = M == 3 ? 2 : 3;
};

/// G f G^T of one R x R filter (row-major) into the A x A Winograd domain.
template <int M>
__device__ inline void k5FilterTile(const float* f, float* u)
{
    constexpr int A = K5Wino<M>::A, R = K5Taps<M>::R;
    float t[A * R];
#pragma unroll
    for(int c = 0; c < R; ++c)
        K5Wino<M>::g(f + c, R, t + c, R);
#pragma unroll
    for(int p = 0; p < A; ++p)
        K5Wino<M>::g(t + p * R, 1, u + p * A, 1);
}

/// The forward-form R x R filter of (out channel o, in channel i), both within group g, from the
/// graph's K x C/G x 3 x 3 filter: rotated and K/C-swapped for backward data, reflected for true
/// convolution, and cut to its phase sub-filter at stride 2 (taps past the 3x3 are zero).
template <class T, int LAYOUT, int DIR, int R>
__device__ inline void k5GatherFilter(const K5Args& a, int g, int o, int i, float* f)
{
    const int cgi = a.ci / a.groups, cgo = a.co / a.groups;
    const T* w = static_cast<const T*>(a.w);
    int kf, cf, cgf;
    if(a.mode == 0)
    {
        kf = DIR == K5_FORWARD ? g * cgo + o : g * cgi + i;
        cf = DIR == K5_FORWARD ? i : o;
        cgf = DIR == K5_FORWARD ? cgi : cgo;
    }
    else if(a.mode == 1)
    {
        kf = g * cgo + o;
        cf = i >> 2;
        cgf = cgi >> 2;
    }
    else
    {
        kf = g * cgi + i;
        cf = o >> 2;
        cgf = cgo >> 2;
    }
    const bool rot = (DIR == K5_BACKWARD_DATA) != (a.flip != 0);
    const int phase = a.mode == 1 ? (i & 3) : (o & 3);
#pragma unroll
    for(int t = 0; t < R; ++t)
    {
#pragma unroll
        for(int u = 0; u < R; ++u)
        {
            int r, s;
            if(a.mode == 0)
            {
                r = rot ? 2 - t : t;
                s = rot ? 2 - u : u;
            }
            else if(a.mode == 1)
            {
                r = 2 * t + (phase >> 1);
                s = 2 * u + (phase & 1);
            }
            else
            {
                r = (phase >> 1) + 2 - 2 * t;
                s = (phase & 1) + 2 - 2 * u;
            }
            float v = 0.0f;
            if(r <= 2 && s <= 2)
            {
                if(a.mode != 0 && a.flip != 0)
                {
                    r = 2 - r;
                    s = 2 - s;
                }
                const long long off
                    = LAYOUT == K5_NCHW ? ((static_cast<long long>(kf) * cgf + cf) * 3 + r) * 3 + s
                                        : ((static_cast<long long>(kf) * 3 + r) * 3 + s) * cgf + cf;
                v = K5Elem<T>::load(w[off]);
            }
            f[t * R + u] = v;
        }
    }
}

/// Element offset of the forward-form input at virtual channel cc (over all groups) and virtual
/// patch row / column (tile origin plus patch index, before padding), or -1 for padding.
template <int LAYOUT>
__device__ inline long long k5InOffset(const K5Args& a, int img, int cc, int vr, int vc)
{
    int ch = cc, channels = a.ci, h = vr - a.ph, w = vc - a.pw, hs = a.hi, ws = a.wi;
    if(a.mode == 1)
    {
        const int phase = cc & 3;
        ch = cc >> 2;
        channels = a.ci >> 2;
        h = 2 * vr + (phase >> 1) - a.xph;
        w = 2 * vc + (phase & 1) - a.xpw;
        hs = a.xh;
        ws = a.xw;
    }
    if(h < 0 || h >= hs || w < 0 || w >= ws)
        return -1;
    return LAYOUT == K5_NCHW ? ((static_cast<long long>(img) * channels + ch) * hs + h) * ws + w
                             : ((static_cast<long long>(img) * hs + h) * ws + w) * channels + ch;
}

// ---------------------------------------------------------------------------------------------
// Pass 1: filter transform. U[g][p][o][i] = (G w G^T)[p], w = the forward-form 3x3 for out
// channel o, in channel i. A block covers a K5_FT x K5_FT tile of (filter K, filter C) of one group:
// the raw 3x3s are loaded in memory order into LDS, then each thread transforms one pair and
// the stores run along i (K5_FT = 16: one pair per thread). Backward data swaps the roles (o = filter C, i = filter K) and reads
// the taps rotated by 180 degrees. Grid: (ceil(cgi/K5_FT), ceil(cgo/K5_FT), G), 256 threads.

#define K5_FT 16

template <class T, class Op, int M, int LAYOUT, int DIR>
__device__ void k5FilterTransform(const K5Args& a)
{
    constexpr int A = K5Wino<M>::A, P = A * A, FT = K5_FT;
    const int cgi = a.ci / a.groups, cgo = a.co / a.groups;
    if constexpr(M == 3)
    {
        // Stride-2 phase sub-filters: one (o, i) pair per thread, i fastest.
        const int i = static_cast<int>(blockIdx.x) * FT + static_cast<int>(threadIdx.x) % FT;
        const int o = static_cast<int>(blockIdx.y) * FT + static_cast<int>(threadIdx.x) / FT;
        const int g = blockIdx.z;
        if(i >= cgi || o >= cgo)
            return;
        float f[4], u[P];
        k5GatherFilter<T, LAYOUT, DIR, 2>(a, g, o, i, f);
        k5FilterTile<3>(f, u);
        Op* U = static_cast<Op*>(a.u);
        const long long ps = static_cast<long long>(cgo) * cgi;
        const long long base
            = static_cast<long long>(g) * P * ps + static_cast<long long>(o) * cgi + i;
#pragma unroll
        for(int p = 0; p < P; ++p)
            U[base + p * ps] = K5Elem<Op>::store(u[p]);
        return;
    }
    const int kgf = DIR == K5_FORWARD ? cgo : cgi; // filter K per group
    const int cgf = DIR == K5_FORWARD ? cgi : cgo; // filter C per group
    const int g = blockIdx.z;
    // blockIdx.x walks i (the U-contiguous index), so neighbouring blocks fill the same lines.
    const int c0 = (DIR == K5_FORWARD ? blockIdx.x : blockIdx.y) * FT;
    const int k0 = (DIR == K5_FORWARD ? blockIdx.y : blockIdx.x) * FT;
    const int tid = threadIdx.x;
    const T* w = static_cast<const T*>(a.w);
    __shared__ float fl[9 * FT * (FT + 1)]; // [rs][kl][cl]

    for(int e = tid; e < 9 * FT * FT; e += 256)
    {
        int kl, cl, rs;
        if constexpr(LAYOUT == K5_NCHW) // (kf, cf, rs): rs fastest
        {
            rs = e % 9;
            cl = (e / 9) % FT;
            kl = e / (9 * FT);
        }
        else // (kf, rs, cf): cf fastest
        {
            cl = e % FT;
            rs = (e / FT) % 9;
            kl = e / (9 * FT);
        }
        const int kf = k0 + kl, cf = c0 + cl;
        float val = 0.0f;
        if(kf < kgf && cf < cgf)
        {
            const long long kk = static_cast<long long>(g) * kgf + kf;
            const long long off
                = LAYOUT == K5_NCHW ? (kk * cgf + cf) * 9 + rs : (kk * 9 + rs) * cgf + cf;
            val = K5Elem<T>::load(w[off]);
        }
        fl[(rs * FT + kl) * (FT + 1) + cl] = val;
    }
    __syncthreads();

    Op* U = static_cast<Op*>(a.u);
    const long long ps = static_cast<long long>(cgo) * cgi;
    for(int e = tid; e < FT * FT; e += 256)
    {
        // Store order: i fastest. Forward: i = filter C; backward: i = filter K.
        const int fast = e % FT, slow = e / FT;
        const int kl = DIR == K5_FORWARD ? slow : fast;
        const int cl = DIR == K5_FORWARD ? fast : slow;
        const int kf = k0 + kl, cf = c0 + cl;
        if(kf >= kgf || cf >= cgf)
            continue;
        // Backward data rotates the filter, true convolution reflects it: together they cancel.
        const bool rot = (DIR == K5_BACKWARD_DATA) != (a.flip != 0);
        float f[9];
#pragma unroll
        for(int q = 0; q < 9; ++q)
            f[q] = fl[((rot ? 8 - q : q) * FT + kl) * (FT + 1) + cl];
        float t[A * 3], u[P];
#pragma unroll
        for(int c = 0; c < 3; ++c) // columns: t[:, c] = G f[:, c]
            K5Wino<M>::g(f + c, 3, t + c, 3);
#pragma unroll
        for(int p = 0; p < A; ++p) // rows: u[p, :] = G t[p, :]
            K5Wino<M>::g(t + p * 3, 1, u + p * A, 1);
        const int o = DIR == K5_FORWARD ? kf : cf, i = DIR == K5_FORWARD ? cf : kf;
        const long long base
            = static_cast<long long>(g) * P * ps + static_cast<long long>(o) * cgi + i;
#pragma unroll
        for(int p = 0; p < P; ++p)
            U[base + p * ps] = K5Elem<Op>::store(u[p]);
    }
}

// ---------------------------------------------------------------------------------------------
// Pass 2: input transform. A block covers TB tiles x CB channels of one group. NCHW computes
// tile-fast and writes V directly (rows of TB tiles); NHWC computes channel-fast (coalesced
// channel loads), stages through LDS and writes tile-fast.

template <int LAYOUT>
struct K5XformBlock
{
    static constexpr int THREADS = 256;
    static constexpr int TB = LAYOUT == K5_NCHW ? 64 : 16;
    static constexpr int CB = THREADS / TB;
};

template <class T, class Op, int M, int LAYOUT>
__device__ void k5InputTransform(const K5Args& a)
{
    constexpr int A = K5Wino<M>::A, P = A * A;
    typedef K5XformBlock<LAYOUT> B;
    const int cgi = a.ci / a.groups;
    const int g = blockIdx.z;
    const int t0 = blockIdx.x * B::TB, c0 = blockIdx.y * B::CB;
    const int tid = threadIdx.x;
    const int tl = LAYOUT == K5_NCHW ? tid % B::TB : tid / B::CB;
    const int cl = LAYOUT == K5_NCHW ? tid / B::TB : tid % B::CB;
    const int t = t0 + tl, c = c0 + cl;
    const bool valid = t < a.tiles && c < cgi;

    float d[P];
#pragma unroll
    for(int p = 0; p < P; ++p)
        d[p] = 0.0f;
    if(valid)
    {
        const int tw = t % a.tw, r = t / a.tw;
        const int th = r % a.th, img = r / a.th;
        const T* in = static_cast<const T*>(a.in);
        const int cc = g * cgi + c;
#pragma unroll
        for(int i = 0; i < A; ++i)
        {
#pragma unroll
            for(int j = 0; j < A; ++j)
            {
                const long long off = k5InOffset<LAYOUT>(a, img, cc, th * M + i, tw * M + j);
                d[i * A + j] = off >= 0 ? K5Elem<T>::load(in[off]) : 0.0f;
            }
        }
        float tmp[P];
#pragma unroll
        for(int j = 0; j < A; ++j) // columns
            K5Wino<M>::bt(d + j, A, tmp + j, A);
#pragma unroll
        for(int i = 0; i < A; ++i) // rows
            K5Wino<M>::bt(tmp + i * A, 1, d + i * A, 1);
    }

    Op* V = static_cast<Op*>(a.v);
    const long long ps = static_cast<long long>(cgi) * a.tiles;
    const long long gbase = static_cast<long long>(g) * P * ps;
    if constexpr(LAYOUT == K5_NCHW)
    {
        if(valid)
        {
#pragma unroll
            for(int p = 0; p < P; ++p)
                V[gbase + p * ps + static_cast<long long>(c) * a.tiles + t]
                    = K5Elem<Op>::store(d[p]);
        }
    }
    else
    {
        __shared__ float s[P * B::CB * (B::TB + 1)];
#pragma unroll
        for(int p = 0; p < P; ++p)
            s[(p * B::CB + cl) * (B::TB + 1) + tl] = d[p];
        __syncthreads();
        const int tl2 = tid % B::TB, cl2 = tid / B::TB;
        const int t2 = t0 + tl2, c2 = c0 + cl2;
        if(t2 < a.tiles && c2 < cgi)
        {
#pragma unroll
            for(int p = 0; p < P; ++p)
                V[gbase + p * ps + static_cast<long long>(c2) * a.tiles + t2]
                    = K5Elem<Op>::store(s[(p * B::CB + cl2) * (B::TB + 1) + tl2]);
        }
    }
}

// Direct-conv parity: an output whose 3x3 window lies entirely in the padding is exactly 0 (under
// F(4,3) the transforms would leave ~1e-7 of the tile's other values there).
__device__ inline bool k5Live(const K5Args& a, int oh, int ow)
{
    if(a.mode == 1)
        return 2 * oh - a.xph + 2 >= 0 && 2 * oh - a.xph < a.xh && 2 * ow - a.xpw + 2 >= 0
               && 2 * ow - a.xpw < a.xw;
    return oh - a.ph + 2 >= 0 && oh - a.ph < a.hi && ow - a.pw + 2 >= 0 && ow - a.pw < a.wi;
}

/// Stride-2 backward data: whether dx at (h, w) has any dy element under the filter.
__device__ inline bool k5LiveBwd2(const K5Args& a, int h, int w)
{
    const int h0 = (h + a.xph - 1) >> 1, h1 = (h + a.xph) >> 1;
    const int w0 = (w + a.xpw - 1) >> 1, w1 = (w + a.xpw) >> 1;
    return (h0 > 0 ? h0 : 0) <= (h1 < a.hi - 1 ? h1 : a.hi - 1)
           && (w0 > 0 ? w0 : 0) <= (w1 < a.wi - 1 ? w1 : a.wi - 1);
}

/// Stores one forward-form output (virtual channel kk over all groups, virtual position oh < ho,
/// ow < wo): in place (stride 1, stride-2 forward) or phase-scattered into dx (stride-2
/// backward data), with outputs whose window is all padding exactly 0.
template <class T, int LAYOUT>
__device__ inline void k5StoreOut(const K5Args& a, T* out, int img, int kk, int oh, int ow, float v)
{
    int ch = kk, channels = a.co, h = oh, w = ow, hs = a.ho, ws = a.wo;
    bool live;
    if(a.mode == 2)
    {
        const int phase = kk & 3;
        ch = kk >> 2;
        channels = a.co >> 2;
        h = 2 * oh + (phase >> 1) - a.xph;
        w = 2 * ow + (phase & 1) - a.xpw;
        hs = a.xh;
        ws = a.xw;
        if(h < 0 || h >= hs || w < 0 || w >= ws)
            return;
        live = k5LiveBwd2(a, h, w);
    }
    else
        live = k5Live(a, oh, ow);
    const long long off = LAYOUT == K5_NCHW
                              ? ((static_cast<long long>(img) * channels + ch) * hs + h) * ws + w
                              : ((static_cast<long long>(img) * hs + h) * ws + w) * channels + ch;
    out[off] = K5Elem<T>::store(live ? v : 0.0f);
}

// ---------------------------------------------------------------------------------------------
// Pass 4: output transform. A block covers TB tiles x KB out channels of one group; M is read
// tile-fast. NCHW writes rows of m outputs per thread; NHWC stages through LDS and writes
// channel-fast.

template <class T, int M, int LAYOUT>
__device__ void k5OutputTransform(const K5Args& a)
{
    constexpr int A = K5Wino<M>::A, P = A * A;
    typedef K5XformBlock<LAYOUT> B;
    const int cgo = a.co / a.groups;
    const int g = blockIdx.z;
    const int t0 = blockIdx.x * B::TB, k0 = blockIdx.y * B::CB;
    const int tid = threadIdx.x;
    const int tl = tid % B::TB, kl = tid / B::TB;
    const int t = t0 + tl, k = k0 + kl;
    const long long ps = static_cast<long long>(cgo) * a.tiles;
    const long long gbase = static_cast<long long>(g) * P * ps;
    T* out = static_cast<T*>(a.out);

    float y[M * M];
#pragma unroll
    for(int q = 0; q < M * M; ++q)
        y[q] = 0.0f;
    if(t < a.tiles && k < cgo)
    {
        float mm[P], tmp[M * A];
#pragma unroll
        for(int p = 0; p < P; ++p)
            mm[p] = a.mw[gbase + p * ps + static_cast<long long>(k) * a.tiles + t];
#pragma unroll
        for(int j = 0; j < A; ++j) // columns: tmp[:, j] = A^T mm[:, j]
            K5Wino<M>::at(mm + j, A, tmp + j, A);
#pragma unroll
        for(int i = 0; i < M; ++i) // rows
            K5Wino<M>::at(tmp + i * A, 1, y + i * M, 1);
        if constexpr(LAYOUT == K5_NCHW)
        {
            const int tw = t % a.tw, r = t / a.tw;
            const int th = r % a.th, img = r / a.th;
            const int kk = g * cgo + k;
#pragma unroll
            for(int i = 0; i < M; ++i)
            {
                const int oh = th * M + i;
                if(oh >= a.ho)
                    continue;
#pragma unroll
                for(int j = 0; j < M; ++j)
                {
                    const int ow = tw * M + j;
                    if(ow < a.wo)
                        k5StoreOut<T, K5_NCHW>(a, out, img, kk, oh, ow, y[i * M + j]);
                }
            }
        }
    }
    if constexpr(LAYOUT == K5_NHWC)
    {
        __shared__ float s[M * M * B::CB * (B::TB + 1)];
#pragma unroll
        for(int q = 0; q < M * M; ++q)
            s[(q * B::TB + tl) * (B::CB + 1) + kl] = y[q];
        __syncthreads();
        const int kl2 = tid % B::CB, tl2 = tid / B::CB;
        const int t2 = t0 + tl2, k2 = k0 + kl2;
        if(t2 < a.tiles && k2 < cgo)
        {
            const int tw = t2 % a.tw, r = t2 / a.tw;
            const int th = r % a.th, img = r / a.th;
            const int kk = g * cgo + k2;
#pragma unroll
            for(int i = 0; i < M; ++i)
            {
                const int oh = th * M + i;
                if(oh >= a.ho)
                    continue;
#pragma unroll
                for(int j = 0; j < M; ++j)
                {
                    const int ow = tw * M + j;
                    if(ow < a.wo)
                        k5StoreOut<T, K5_NHWC>(a,
                                               out,
                                               img,
                                               kk,
                                               oh,
                                               ow,
                                               s[((i * M + j) * B::TB + tl2) * (B::CB + 1) + kl2]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Pass 3: batched GEMM, C[b] (m x n, fp32) = A[b] (m x k, Op) * B[b] (k x n, Op), all row-major,
// fp32 accumulate. A simple LDS-tiled FMA kernel: 256 threads as 16 x 16, each owning
// (BM/16) x (BN/16) outputs in runs of 4. Stand-in for the BLAS provider's strided-batched GEMM.

template <class Op, int BM, int BN, int BK>
__device__ void k5Gemm(const K5GemmArgs& g)
{
    constexpr int THREADS = 256, TX = 16, TY = 16;
    constexpr int TM = BM / TY, TN = BN / TX;
    static_assert(TM % 4 == 0 && TN % 4 == 0, "4-runs");
    static_assert((BM * BK) % THREADS == 0 && (BK * BN) % THREADS == 0, "tile loads");
    constexpr int LDA = BM + 4, LDB = BN + 4;
    __shared__ __attribute__((aligned(16))) float As[BK * LDA];
    __shared__ __attribute__((aligned(16))) float Bs[BK * LDB];

    const int bz = blockIdx.z;
    const Op* A = static_cast<const Op*>(g.a) + bz * g.sa;
    const Op* Bm = static_cast<const Op*>(g.b) + bz * g.sb;
    float* C = g.c + bz * g.sc;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tid = threadIdx.x, tx = tid % TX, ty = tid / TX;

    float acc[TM][TN];
#pragma unroll
    for(int i = 0; i < TM; ++i)
#pragma unroll
        for(int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    constexpr int LA = BM * BK / THREADS, LB = BK * BN / THREADS;
    float ra[LA], rb[LB];
    auto load = [&](int k0) {
#pragma unroll
        for(int q = 0; q < LA; ++q)
        {
            const int e = tid + q * THREADS;
            const int row = e / BK, col = e % BK;
            const int gm = m0 + row, gk = k0 + col;
            ra[q] = gm < g.m && gk < g.k
                        ? K5Elem<Op>::load(A[static_cast<long long>(gm) * g.lda + gk])
                        : 0.0f;
        }
#pragma unroll
        for(int q = 0; q < LB; ++q)
        {
            const int e = tid + q * THREADS;
            const int row = e / BN, col = e % BN;
            const int gk = k0 + row, gn = n0 + col;
            rb[q] = gk < g.k && gn < g.n
                        ? K5Elem<Op>::load(Bm[static_cast<long long>(gk) * g.ldb + gn])
                        : 0.0f;
        }
    };
    load(0);
    for(int k0 = 0; k0 < g.k; k0 += BK)
    {
        __syncthreads();
#pragma unroll
        for(int q = 0; q < LA; ++q)
        {
            const int e = tid + q * THREADS;
            As[(e % BK) * LDA + e / BK] = ra[q];
        }
#pragma unroll
        for(int q = 0; q < LB; ++q)
        {
            const int e = tid + q * THREADS;
            Bs[(e / BN) * LDB + e % BN] = rb[q];
        }
        __syncthreads();
        if(k0 + BK < g.k)
            load(k0 + BK);
#pragma unroll
        for(int kk = 0; kk < BK; ++kk)
        {
            float av[TM], bv[TN];
#pragma unroll
            for(int q = 0; q < TM / 4; ++q)
#pragma unroll
                for(int e = 0; e < 4; ++e)
                    av[q * 4 + e] = As[kk * LDA + (q * TY + ty) * 4 + e];
#pragma unroll
            for(int q = 0; q < TN / 4; ++q)
#pragma unroll
                for(int e = 0; e < 4; ++e)
                    bv[q * 4 + e] = Bs[kk * LDB + (q * TX + tx) * 4 + e];
#pragma unroll
            for(int i = 0; i < TM; ++i)
#pragma unroll
                for(int j = 0; j < TN; ++j)
                    acc[i][j] = __builtin_fmaf(av[i], bv[j], acc[i][j]);
        }
    }
#pragma unroll
    for(int i = 0; i < TM; ++i)
    {
        const int gm = m0 + ((i / 4) * TY + ty) * 4 + i % 4;
        if(gm >= g.m)
            continue;
#pragma unroll
        for(int j = 0; j < TN; ++j)
        {
            const int gn = n0 + ((j / 4) * TX + tx) * 4 + j % 4;
            if(gn < g.n)
                C[static_cast<long long>(gm) * g.ldc + gn] = acc[i][j];
        }
    }
}
