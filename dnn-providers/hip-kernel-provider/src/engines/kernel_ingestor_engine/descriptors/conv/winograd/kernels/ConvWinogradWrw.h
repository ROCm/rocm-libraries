// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Winograd backward weights (CONV_WINOGRAD_WRW): F(3,r), 3x3 stride 1 (dilation 1), multipass.
// HIPRTC-compatible: no standard headers.
//
// dw[k][c][u][v] = sum over (n, h, w) of dy[n][k][h][w] * x[n][c][h + u - ph][w + v - pw]: per
// output channel pair the 3x3 dW is one output tile of a correlation whose "filter" is an r x r
// tile of dy and whose "data" is the (r+2) x (r+2) x patch under it. By the transposition
// duality of F(r,3) and F(3,r) the matrices are F(r,3)'s (K5Wino<2>; F(4,3) on points 0, 1, -1, 1/2,
// -2 for accuracy, below; F(6,3) with Lavin's points
// 0, +-1, +-2, +-1/2 here):
//   x patch  -> B^T d B            (F(r,3)'s data transform; tiles step r over dy's extent)
//   dy tile  -> A g A^T            (A = the transpose of F(r,3)'s output transform A^T)
//   dW       =  G^T M G            (G = F(r,3)'s filter transform), M summed over all tiles.
// r = 2 (16 positions, 2.25x fewer multiplies than direct), r = 4 (36 positions, 4x) and r = 6
// (64 positions, 5.06x).
//
// Passes (P = (r+2)^2 positions, T = tiles of the batch slice, padded with zero tiles to L, a
// multiple of 64; per group cg = C/G in channels, kg = K/G out channels; Op = the GEMM operand
// type, rounded once after the fp32 transform):
//   1. x transform   x  -> V[g][p][cg][L]  (Op; x and dy channels-first, packed NHWC transposed in)
//   2. dy transform  dy -> U[g][p][kg][L]  (Op)
//   3. GEMM          Part[s][g][p][kg][cg] (fp32) = sum over split s's tile range of U V^T: both
//                    operands tile-contiguous, so every atom reads contiguous k runs; batch
//                    slices after the first add onto the partials (launch order: deterministic).
//   4. output        dw = G^T (sum over splits in order) G, the true-convolution reflection, taps
//                    whose x window is all padding exactly 0; written with the graph's strides.
// No atomics anywhere.
#pragma once

#include "ConvWinogradCore.h"

#define K5W_LPAD 64 // tile padding of the transformed buffers (the GEMM's K needs no bounds)

/// x patch transform B^T d of one (r+2)-vector: F(r,3)'s data transform.
template <int R>
struct K5WrwX
{
    __device__ static void bt(const float* d, int ds, float* o, int os)
    {
        K5Wino<R>::bt(d, ds, o, os);
    }
};

// F(4,3) on points 0, 1, -1, 1/2, -2, infinity (not the forward's Lavin set 0, +-1, +-2): the
// wrw F(3,4) built on it measured about 3.7x less fp32 summation error (dynamic-tolerance c 1.4
// against 5.2 in the fp32 simulation, ~/scratch/ww/points.py).
template <>
struct K5WrwX<4>
{
    __device__ static void bt(const float* d, int ds, float* o, int os)
    {
        const float d0 = d[0], d1 = d[ds], d2 = d[2 * ds], d3 = d[3 * ds], d4 = d[4 * ds],
                    d5 = d[5 * ds];
        o[0] = d0 - 1.5f * d1 - 2.0f * d2 + 1.5f * d3 + d4;
        o[os] = 0.5f * d2 - d1 + 2.5f * d3 + d4;
        o[2 * os] = d1 - 2.5f * d2 + 0.5f * d3 + d4;
        o[3 * os] = 2.0f * (d3 - d1) - d2 + d4;
        o[4 * os] = 0.5f * (d1 - d3) - d2 + d4;
        o[5 * os] = d1 - 1.5f * d2 - 2.0f * d3 + 1.5f * d4 + d5;
    }
};

template <>
struct K5WrwX<6> // F(6,3), points 0, +-1, +-2, +-1/2, infinity
{
    __device__ static void bt(const float* d, int ds, float* o, int os)
    {
        const float d0 = d[0], d1 = d[ds], d2 = d[2 * ds], d3 = d[3 * ds], d4 = d[4 * ds],
                    d5 = d[5 * ds], d6 = d[6 * ds], d7 = d[7 * ds];
        const float a = d2 - 4.25f * d4 + d6, b = d1 - 4.25f * d3 + d5;
        const float c = 0.25f * d2 - 1.25f * d4 + d6, e = 0.5f * d1 - 2.5f * d3 + 2.0f * d5;
        const float f = 4.0f * d2 - 5.0f * d4 + d6, h = 2.0f * d1 - 2.5f * d3 + 0.5f * d5;
        o[0] = d0 - d6 + 5.25f * (d4 - d2);
        o[os] = a + b;
        o[2 * os] = a - b;
        o[3 * os] = c + e;
        o[4 * os] = c - e;
        o[5 * os] = f + h;
        o[6 * os] = f - h;
        o[7 * os] = d7 - d1 + 5.25f * (d3 - d5);
    }
};

/// dy tile transform A g of one r-vector (A = (F(r,3)'s A^T)^T: integer coefficients for r 2
/// and 4).
template <int R>
struct K5WrwDy;

template <>
struct K5WrwDy<2>
{
    __device__ static void a(const float* g, int gs, float* o, int os)
    {
        const float g0 = g[0], g1 = g[gs];
        o[0] = g0;
        o[os] = g0 + g1;
        o[2 * os] = g0 - g1;
        o[3 * os] = -g1;
    }
};

template <>
struct K5WrwDy<4>
{
    __device__ static void a(const float* g, int gs, float* o, int os)
    {
        // Points 0, 1, -1, 1/2, -2, infinity (as K5WrwX<4>).
        const float g0 = g[0], g1 = g[gs], g2 = g[2 * gs], g3 = g[3 * gs];
        const float e = g0 + g2, od = g1 + g3;
        o[0] = g0;
        o[os] = e + od;
        o[2 * os] = e - od;
        o[3 * os] = g0 + 0.5f * g1 + 0.25f * g2 + 0.125f * g3;
        o[4 * os] = g0 - 2.0f * g1 + 4.0f * g2 - 8.0f * g3;
        o[5 * os] = g3;
    }
};

template <>
struct K5WrwDy<6>
{
    __device__ static void a(const float* g, int gs, float* o, int os)
    {
        const float g0 = g[0], g1 = g[gs], g2 = g[2 * gs], g3 = g[3 * gs], g4 = g[4 * gs],
                    g5 = g[5 * gs];
        const float e1 = g0 + g2 + g4, o1 = g1 + g3 + g5;
        const float e2 = g0 + 4.0f * g2 + 16.0f * g4, o2 = 2.0f * g1 + 8.0f * g3 + 32.0f * g5;
        const float e3 = g0 + 0.25f * g2 + 0.0625f * g4,
                    o3 = 0.5f * g1 + 0.125f * g3 + 0.03125f * g5;
        o[0] = g0;
        o[os] = e1 + o1;
        o[2 * os] = e1 - o1;
        o[3 * os] = e2 + o2;
        o[4 * os] = e2 - o2;
        o[5 * os] = e3 + o3;
        o[6 * os] = e3 - o3;
        o[7 * os] = g5;
    }
};

/// dW from the Winograd domain: G^T m of one alpha-vector (G = F(r,3)'s filter transform).
template <int R>
struct K5WrwOut;

template <>
struct K5WrwOut<2>
{
    __device__ static void gt(const float* m, int ms, float* o, int os)
    {
        const float m0 = m[0], m1 = m[ms], m2 = m[2 * ms], m3 = m[3 * ms];
        const float s = 0.5f * (m1 + m2);
        o[0] = m0 + s;
        o[os] = 0.5f * (m1 - m2);
        o[2 * os] = s + m3;
    }
};

template <>
struct K5WrwOut<4>
{
    __device__ static void gt(const float* m, int ms, float* o, int os)
    {
        const float m0 = m[0], m1 = m[ms], m2 = m[2 * ms], m3 = m[3 * ms], m4 = m[4 * ms],
                    m5 = m[5 * ms];
        // Points 0, 1, -1, 1/2, -2, infinity (as K5WrwX<4>).
        const float s12 = m1 + m2, d12 = m1 - m2;
        o[0] = m0 + (1.0f / 3.0f) * d12 - (16.0f / 15.0f) * m3 + (1.0f / 15.0f) * m4;
        o[os] = (1.0f / 3.0f) * s12 - (8.0f / 15.0f) * m3 - (2.0f / 15.0f) * m4;
        o[2 * os] = (1.0f / 3.0f) * d12 - (4.0f / 15.0f) * m3 + (4.0f / 15.0f) * m4 + m5;
    }
};

template <>
struct K5WrwOut<6>
{
    __device__ static void gt(const float* m, int ms, float* o, int os)
    {
        const float m0 = m[0], m1 = m[ms], m2 = m[2 * ms], m3 = m[3 * ms], m4 = m[4 * ms],
                    m5 = m[5 * ms], m6 = m[6 * ms], m7 = m[7 * ms];
        const float s12 = m1 + m2, d12 = m1 - m2, s34 = m3 + m4, d34 = m3 - m4, s56 = m5 + m6,
                    d56 = m5 - m6;
        o[0] = m0 - (2.0f / 9.0f) * s12 + (1.0f / 90.0f) * s34 + (32.0f / 45.0f) * s56;
        o[os] = -(2.0f / 9.0f) * d12 + (1.0f / 45.0f) * d34 + (16.0f / 45.0f) * d56;
        o[2 * os] = -(2.0f / 9.0f) * s12 + (2.0f / 45.0f) * s34 + (8.0f / 45.0f) * s56 + m7;
    }
};

/// Tile t of the slice as (image, tile row, tile column), or false past the slice's tiles.
__device__ inline bool k5wTile(const K5WrwArgs& a, int t, int& img, int& tr, int& tc)
{
    if(t >= a.tiles)
        return false;
    tc = t % a.tw;
    const int rest = t / a.tw;
    tr = rest % a.th;
    img = rest / a.th;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Passes 1 and 2: the transforms, on packed channels-first x and dy (packed channels-last ones are
// transposed in first, k5wTranspose). A block covers 64 tiles x 4 channels of one group, tile-fast,
// and writes rows of tiles; tiles from T to L are written as zeros. X: the (r+2)^2 x patch; else
// the r x r dy tile.

template <class T, class Op, int R, bool X>
__device__ void k5wTransform(const K5WrwArgs& a)
{
    constexpr int A = R + 2, P = A * A, N = X ? A : R, TB = 64, CB = 4;
    const int cg = (X ? a.c : a.k) / a.groups;
    const int g = blockIdx.z;
    const int tid = threadIdx.x;
    const int t = blockIdx.x * TB + tid % TB, c = blockIdx.y * CB + tid / TB;
    int img = 0, tr = 0, tc = 0;
    const bool valid = c < cg && k5wTile(a, t, img, tr, tc);

    float d[P];
#pragma unroll
    for(int p = 0; p < P; ++p)
        d[p] = 0.0f;
    if(valid)
    {
        const int hs = X ? a.hi : a.ho, ws = X ? a.wi : a.wo;
        const int channels = X ? a.c : a.k;
        const T* src = static_cast<const T*>(X ? a.x : a.dy)
                       + (static_cast<long long>(img) * channels + g * cg + c) * hs * ws;
        const int h0 = tr * R - (X ? a.ph : 0), w0 = tc * R - (X ? a.pw : 0);
        float e[N * N];
#pragma unroll
        for(int i = 0; i < N; ++i)
        {
#pragma unroll
            for(int j = 0; j < N; ++j)
            {
                const int h = h0 + i, w = w0 + j;
                e[i * N + j] = h >= 0 && h < hs && w >= 0 && w < ws
                                   ? K5Elem<T>::load(src[static_cast<long long>(h) * ws + w])
                                   : 0.0f;
            }
        }
        float tmp[A * N];
        if constexpr(X)
        {
#pragma unroll
            for(int j = 0; j < A; ++j) // columns
                K5WrwX<R>::bt(e + j, A, tmp + j, A);
#pragma unroll
            for(int i = 0; i < A; ++i) // rows
                K5WrwX<R>::bt(tmp + i * A, 1, d + i * A, 1);
        }
        else
        {
#pragma unroll
            for(int j = 0; j < R; ++j) // columns: tmp[:, j] = A g[:, j] (A x R)
                K5WrwDy<R>::a(e + j, R, tmp + j, R);
#pragma unroll
            for(int i = 0; i < A; ++i) // rows
                K5WrwDy<R>::a(tmp + i * R, 1, d + i * A, 1);
        }
    }

    Op* V = static_cast<Op*>(X ? a.v : a.u);
    const long long ps = static_cast<long long>(cg) * a.ldt;
    if(c < cg && t < a.ldt)
    {
        Op* row = V + static_cast<long long>(g) * P * ps + static_cast<long long>(c) * a.ldt + t;
#pragma unroll
        for(int p = 0; p < P; ++p)
            row[p * ps] = K5Elem<Op>::store(d[p]);
    }
}

// ---------------------------------------------------------------------------------------------
// Pass 3: the GEMM. Block = BM out channels (rows, from U) x BN in channels (columns, from V) of
// one (group, position, split); per step BK tiles of both operands go through registers into
// LDS rows ([row][BK + pad], tile-contiguous), then:
//   - matrix-core atom (K1's: one 16 x 16 x KA step; MFMA wave64 or WMMA wave32): each wave owns
//     WTM x WTN atoms; a lane's fragment is KPL contiguous tiles of one row;
//   - FMA (K5WrwFma): a thread owns TM rows x TN columns, 4 tiles per LDS read of each.
// Grid (1-D): column blocks fastest, then row blocks, then (group, position), then split.

struct K5WrwFma
{
};

template <class Op>
struct K5WrwVec
{
    static constexpr int E = 16 / sizeof(Op); // elements per 16-byte load
};

template <class Op, class Atom, int BM, int BN, int BK, int TM, int TN, int THREADS, int P>
__device__ void k5wGemm(const K5WrwArgs& a)
{
    constexpr int E = K5WrwVec<Op>::E;
    static_assert(BK % E == 0 && K5W_LPAD % BK == 0, "K5 wrw GEMM: BK");
    constexpr int PAD = E; // one 16-byte unit per row
    constexpr int LDK = BK + PAD;
    constexpr int VPR = BK / E; // 16-byte units per row
    constexpr int LA = (BM * VPR + THREADS - 1) / THREADS, LB = (BN * VPR + THREADS - 1) / THREADS;
    typedef unsigned k5wu4 __attribute__((ext_vector_type(4)));
    __shared__ __attribute__((aligned(16))) Op As[BM * LDK];
    __shared__ __attribute__((aligned(16))) Op Bs[BN * LDK];

    const int kg = a.k / a.groups, cg = a.c / a.groups;
    const int nbc = (cg + BN - 1) / BN, nbk = (kg + BM - 1) / BM;
    int bid = blockIdx.x;
    const int bc = bid % nbc;
    bid /= nbc;
    const int bk = bid % nbk;
    bid /= nbk;
    const int gp = bid % (a.groups * P);
    const int split = bid / (a.groups * P);
    const int m0 = bk * BM, n0 = bc * BN;
    const int tBegin = split * a.splitTiles;
    const int tEnd = tBegin + a.splitTiles < a.ldt ? tBegin + a.splitTiles : a.ldt;

    const Op* U = static_cast<const Op*>(a.u) + (static_cast<long long>(gp) * kg + m0) * a.ldt;
    const Op* V = static_cast<const Op*>(a.v) + (static_cast<long long>(gp) * cg + n0) * a.ldt;
    const int rowsA = kg - m0 < BM ? kg - m0 : BM, rowsB = cg - n0 < BN ? cg - n0 : BN;
    const __amdgpu_buffer_rsrc_t ra = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<Op*>(U),
        static_cast<short>(0),
        static_cast<int>(static_cast<long long>(rowsA) * a.ldt * sizeof(Op)),
        K1_RSRC_FLAGS);
    const __amdgpu_buffer_rsrc_t rb = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<Op*>(V),
        static_cast<short>(0),
        static_cast<int>(static_cast<long long>(rowsB) * a.ldt * sizeof(Op)),
        K1_RSRC_FLAGS);
    const int tid = threadIdx.x;

    k5wu4 ga[LA], gb[LB];
    auto load = [&](int t0) {
#pragma unroll
        for(int q = 0; q < LA; ++q)
        {
            const int e = tid + q * THREADS;
            const int row = e / VPR, col = (e % VPR) * E;
            const unsigned off = row < rowsA && e < BM * VPR
                                     ? static_cast<unsigned>((row * a.ldt + t0 + col)
                                                             * static_cast<int>(sizeof(Op)))
                                     : K1_OUT_OF_RANGE;
            ga[q] = __builtin_bit_cast(k5wu4, __builtin_amdgcn_raw_buffer_load_b128(ra, off, 0, 0));
        }
#pragma unroll
        for(int q = 0; q < LB; ++q)
        {
            const int e = tid + q * THREADS;
            const int row = e / VPR, col = (e % VPR) * E;
            const unsigned off = row < rowsB && e < BN * VPR
                                     ? static_cast<unsigned>((row * a.ldt + t0 + col)
                                                             * static_cast<int>(sizeof(Op)))
                                     : K1_OUT_OF_RANGE;
            gb[q] = __builtin_bit_cast(k5wu4, __builtin_amdgcn_raw_buffer_load_b128(rb, off, 0, 0));
        }
    };
    auto stash = [&]() {
#pragma unroll
        for(int q = 0; q < LA; ++q)
        {
            const int e = tid + q * THREADS;
            if(e < BM * VPR)
                *reinterpret_cast<k5wu4*>(&As[(e / VPR) * LDK + (e % VPR) * E]) = ga[q];
        }
#pragma unroll
        for(int q = 0; q < LB; ++q)
        {
            const int e = tid + q * THREADS;
            if(e < BN * VPR)
                *reinterpret_cast<k5wu4*>(&Bs[(e / VPR) * LDK + (e % VPR) * E]) = gb[q];
        }
    };

    float* out = a.part + ((static_cast<long long>(split) * a.groups * P + gp) * kg) * cg;

    if constexpr(__is_same(Atom, K5WrwFma))
    {
        // fp32 operands: FMA, 4 tiles per LDS read; fp16: v_dot2_f32_f16 on tile pairs, 8 per read.
        constexpr bool DOT2 = __is_same(Op, _Float16);
        static_assert(__is_same(Op, float) || DOT2, "K5 wrw FMA: fp32 or fp16 operands");
        constexpr int NX = BN / TN, NY = BM / TM;
        static_assert(NX * NY == THREADS, "K5 wrw FMA: thread tile");
        const int tx = tid % NX, ty = tid / NX;
        float acc[TM][TN];
#pragma unroll
        for(int i = 0; i < TM; ++i)
#pragma unroll
            for(int j = 0; j < TN; ++j)
                acc[i][j] = 0.0f;
        if(tBegin < tEnd)
            load(tBegin);
        for(int t0 = tBegin; t0 < tEnd; t0 += BK)
        {
            __syncthreads();
            stash();
            __syncthreads();
            if(t0 + BK < tEnd)
                load(t0 + BK);
#pragma unroll
            for(int kk = 0; kk < BK; kk += E)
            {
                // Read as the stash wrote (16-byte unsigned units), then reinterpret each dword.
                // (Components taken by name: subscripting the loaded vectors with the unrolled
                // index miscompiled to the first component.)
                unsigned av[TM][4], bv[TN][4];
#pragma unroll
                for(int i = 0; i < TM; ++i)
                {
                    const k5wu4 u = *reinterpret_cast<const k5wu4*>(&As[(ty + i * NY) * LDK + kk]);
                    av[i][0] = u.x;
                    av[i][1] = u.y;
                    av[i][2] = u.z;
                    av[i][3] = u.w;
                }
#pragma unroll
                for(int j = 0; j < TN; ++j)
                {
                    const k5wu4 u = *reinterpret_cast<const k5wu4*>(&Bs[(tx + j * NX) * LDK + kk]);
                    bv[j][0] = u.x;
                    bv[j][1] = u.y;
                    bv[j][2] = u.z;
                    bv[j][3] = u.w;
                }
                if constexpr(DOT2)
                {
                    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
#pragma unroll
                    for(int q = 0; q < 4; ++q)
#pragma unroll
                        for(int i = 0; i < TM; ++i)
#pragma unroll
                            for(int j = 0; j < TN; ++j)
                                acc[i][j] = __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, av[i][q]),
                                                                   __builtin_bit_cast(h2, bv[j][q]),
                                                                   acc[i][j],
                                                                   false);
                }
                else
                {
#pragma unroll
                    for(int q = 0; q < 4; ++q)
#pragma unroll
                        for(int i = 0; i < TM; ++i)
#pragma unroll
                            for(int j = 0; j < TN; ++j)
                                acc[i][j] = __builtin_fmaf(__builtin_bit_cast(float, av[i][q]),
                                                           __builtin_bit_cast(float, bv[j][q]),
                                                           acc[i][j]);
                }
            }
        }
#pragma unroll
        for(int i = 0; i < TM; ++i)
        {
            const int row = m0 + ty + i * NY;
            if(row >= kg)
                continue;
#pragma unroll
            for(int j = 0; j < TN; ++j)
            {
                const int col = n0 + tx + j * NX;
                if(col < cg)
                {
                    float* o = out + static_cast<long long>(row) * cg + col;
                    *o = a.accumulate ? *o + acc[i][j] : acc[i][j];
                }
            }
        }
    }
    else
    {
        constexpr int D = Atom::DIM, WAVES = THREADS / Atom::WAVE;
        constexpr int WN = BN / (D * TN), WM = BM / (D * TM);
        static_assert(WN * WM == WAVES && BK % Atom::KA == 0, "K5 wrw atom tiling");
        typedef typename Atom::frag frag;
        typedef typename Atom::acc accv;
        const int lane = tid % Atom::WAVE, wave = tid / Atom::WAVE;
        const int wm = wave / WN, wn = wave % WN;
        accv acc[TM][TN];
#pragma unroll
        for(int i = 0; i < TM; ++i)
#pragma unroll
            for(int j = 0; j < TN; ++j)
#pragma unroll
                for(int e = 0; e < Atom::ACC; ++e)
                    acc[i][j][e] = 0.0f;
        const int kb = Atom::template kBase<Atom::KPL>(lane), lr = lane % D;
        if(tBegin < tEnd)
            load(tBegin);
        for(int t0 = tBegin; t0 < tEnd; t0 += BK)
        {
            __syncthreads();
            stash();
            __syncthreads();
            if(t0 + BK < tEnd)
                load(t0 + BK);
#pragma unroll
            for(int kk = 0; kk < BK; kk += Atom::KA)
            {
                frag fa[TM], fb[TN];
#pragma unroll
                for(int i = 0; i < TM; ++i)
                    fa[i] = *reinterpret_cast<const frag*>(
                        &As[((wm * TM + i) * D + lr) * LDK + kk + kb]);
#pragma unroll
                for(int j = 0; j < TN; ++j)
                    fb[j] = *reinterpret_cast<const frag*>(
                        &Bs[((wn * TN + j) * D + lr) * LDK + kk + kb]);
#pragma unroll
                for(int i = 0; i < TM; ++i)
#pragma unroll
                    for(int j = 0; j < TN; ++j)
                        acc[i][j] = Atom::mma(fa[i], fb[j], acc[i][j]);
            }
        }
#pragma unroll
        for(int i = 0; i < TM; ++i)
        {
#pragma unroll
            for(int j = 0; j < TN; ++j)
            {
                const int col = n0 + (wn * TN + j) * D + lr;
#pragma unroll
                for(int e = 0; e < Atom::ACC; ++e)
                {
                    const int row = m0 + (wm * TM + i) * D + Atom::dRow(lane, e);
                    if(row < kg && col < cg)
                    {
                        float* o = out + static_cast<long long>(row) * cg + col;
                        *o = a.accumulate ? *o + acc[i][j][e] : acc[i][j][e];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Packed NHWC -> NCHW (x or dy staged for the channels-first transforms): per image a 64 x 64
// tile of (pixel, channel) through LDS, 64-element runs read along channels and written along
// pixels. K5CopyArgs: dims {N, H x W, C}.

template <class T>
__device__ void k5wTranspose(const K5CopyArgs& a)
{
    __shared__ T tile[64][65];
    const long long hw = a.dims[1], ch = a.dims[2];
    const long long p0 = static_cast<long long>(blockIdx.x) * 64,
                    c0 = static_cast<long long>(blockIdx.y) * 64;
    const long long img = blockIdx.z;
    const T* src = static_cast<const T*>(a.src) + img * hw * ch;
    T* dst = static_cast<T*>(a.dst) + img * hw * ch;
    const int tx = threadIdx.x % 64, ty = threadIdx.x / 64;
#pragma unroll
    for(int r = 0; r < 16; ++r)
    {
        const long long p = p0 + ty + 4 * r, c = c0 + tx;
        if(p < hw && c < ch)
            tile[ty + 4 * r][tx] = src[p * ch + c];
    }
    __syncthreads();
#pragma unroll
    for(int r = 0; r < 16; ++r)
    {
        const long long c = c0 + ty + 4 * r, p = p0 + tx;
        if(p < hw && c < ch)
            dst[c * hw + p] = tile[tx][ty + 4 * r];
    }
}

// ---------------------------------------------------------------------------------------------
// Pass 4: dW. One thread per (group, out channel, in channel), in channels fastest: the splits
// summed in order per position, G^T M G, then the 3x3 taps (reflected for true convolution;
// taps whose x window is all padding exactly 0) stored with the graph's strides.

/// Whether filter row (or column) tap u reads any real x: rows u - pad .. u - pad + extentOut - 1.
__device__ inline bool k5wTapLive(int u, int pad, int extentIn, int extentOut)
{
    return u - pad + extentOut - 1 >= 0 && u - pad < extentIn;
}

template <class T, int R>
__device__ void k5wOutput(const K5WrwArgs& a)
{
    constexpr int A = R + 2, P = A * A;
    const int kg = a.k / a.groups, cg = a.c / a.groups;
    const long long total = static_cast<long long>(a.groups) * kg * cg;
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= total)
        return;
    const int c = static_cast<int>(i % cg);
    const long long gk = i / cg; // g * kg + k = the graph's out channel
    const int g = static_cast<int>(gk / kg), k = static_cast<int>(gk % kg);
    const long long ps = static_cast<long long>(kg) * cg;
    const long long splitStride = static_cast<long long>(a.groups) * P * ps;
    const float* part
        = a.part + (static_cast<long long>(g) * P * ps + static_cast<long long>(k) * cg + c);
    float m[P];
#pragma unroll
    for(int p = 0; p < P; ++p)
    {
        float s = part[p * ps];
        for(int sp = 1; sp < a.splits; ++sp)
            s += part[sp * splitStride + p * ps];
        m[p] = s;
    }
    float tmp[3 * A], o[9];
#pragma unroll
    for(int j = 0; j < A; ++j) // columns: tmp[:, j] = G^T m[:, j]
        K5WrwOut<R>::gt(m + j, A, tmp + j, A);
#pragma unroll
    for(int u = 0; u < 3; ++u) // rows
        K5WrwOut<R>::gt(tmp + u * A, 1, o + u * 3, 1);
    T* dw = static_cast<T*>(a.dw);
    const long long base = gk * a.dwStrides[0] + static_cast<long long>(c) * a.dwStrides[1];
#pragma unroll
    for(int u = 0; u < 3; ++u)
    {
#pragma unroll
        for(int v = 0; v < 3; ++v)
        {
            // Correlation tap (u, v) is the graph's (2 - u, 2 - v) under true convolution.
            const bool live = k5wTapLive(u, a.ph, a.hi, a.ho) && k5wTapLive(v, a.pw, a.wi, a.wo);
            const int gu = a.flip ? 2 - u : u, gv = a.flip ? 2 - v : v;
            dw[base + gu * a.dwStrides[2] + gv * a.dwStrides[3]]
                = K5Elem<T>::store(live ? o[u * 3 + v] : 0.0f);
        }
    }
}
