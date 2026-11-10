// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Stem backward data (dx for few input channels): the input gradient as one GEMM per tile of
// "super-pixels" (i, j), each standing for the SH x SW dx pixels (SH i + p, SW j + q):
//   D[(i, j)][(c, p, q)] = sum over (u, v, k) of dy[k][i + OH - u][j + OW - v] B[(u, v, k)][(c, p, q)]
// where B holds w[k][c][r][s] for the filter row and column that pair phase (p, q) with window
// offset (u, v), zero elsewhere (ConvStemBwdTransform builds it once per launch). The N axis --
// the input channels times the stride phases -- fills the MMA columns that C <= 4 alone would
// leave idle; the reduction runs over the output channels and the U x V window offsets, all in
// one block (no split, no atomics: deterministic).
#pragma once

#include "ConvIgemmCore.h"
#include "ConvStemBwdArgs.h"

/// B[n][uv][k] (k padded to kp, zero past K) from w, n = (c, p, q), uv = u V + v.
template <class T>
__device__ void sbTransform(const ConvStemBwdTransformArgs& a)
{
    const T* w = static_cast<const T*>(a.w);
    T* b = static_cast<T*>(a.b);
    const long long total = static_cast<long long>(a.nb) * a.u * a.v * a.kp;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
        i += static_cast<long long>(gridDim.x) * blockDim.x)
    {
        const int k = static_cast<int>(i % a.kp);
        long long rest = i / a.kp;
        const int v = static_cast<int>(rest % a.v);
        rest /= a.v;
        const int u = static_cast<int>(rest % a.u);
        const int n = static_cast<int>(rest / a.u);
        const int q = n % a.sw, p = n / a.sw % a.sh, c = n / (a.sw * a.sh);
        float val = 0.0f;
        if(c < a.c && k < a.k)
        {
            // Filter row r pairs dx row SH i + p with dy row i + OH - u: r = SH (u - OH) + p + ph.
            const int r = a.sh * (u - a.oh) + p + a.ph;
            const int s = a.sw * (v - a.ow) + q + a.pw;
            if(r >= 0 && r < a.r && s >= 0 && s < a.s)
            {
                const int rr = a.flip ? a.r - 1 - r : r;
                const int ss = a.flip ? a.s - 1 - s : s;
                val = K1Elem<T>::load(w[k * a.ws[0] + c * a.ws[1] + rr * a.ws[2] + ss * a.ws[3]]);
            }
        }
        b[i] = K1Elem<T>::store(val);
    }
}

/// The FMA "atom" (exact fp32 without an fp32 matrix core): 16 threads per super-row, each FM
/// super-columns 16 apart and all NB columns of D.
typedef float SbF4 __attribute__((ext_vector_type(4)));
struct SbFma
{
    static constexpr int WAVE = 16; ///< threads per super-row
    static constexpr int DIM = 16; ///< super-columns per fragment
    static constexpr int KA = 4;
};

/// Block: TH super-rows, FM fragments of DIM super-columns each; all NB columns (FN = NB / DIM
/// fragments on a matrix atom). The dy window of the block's rows and columns and a chunk of KC
/// output channels is staged in LDS [row][col][k], B's matching chunk [n][uv][k]. Matrix atoms:
/// wave t computes super-row i0 + t. FMA (Atom SbFma, fp32): thread (row, col) computes
/// super-pixels (i0 + row, j0 + col + 16 m).
template <class T,
          class Atom,
          int FMA,
          int CL,
          int CFV,
          int U,
          int V,
          int NB,
          int KC,
          int TH,
          int FM,
          int SH,
          int SW>
__device__ void sbStemBwd(const ConvStemBwdArgs& a)
{
    constexpr int WAVE = Atom::WAVE, DIM = Atom::DIM, KA = Atom::KA;
    constexpr int TW = FM * DIM;
    constexpr int WR = TH + U - 1, WC = TW + V - 1; // window rows, columns
    constexpr int KCP = KC + sbPad(static_cast<int>(sizeof(T))); // padded k run per (row, col)
    constexpr int A_ELEMS = WR * WC * KCP;
    constexpr int UV = U * V;
    constexpr int B_ELEMS = NB * UV * KCP;
    constexpr int THREADS = TH * WAVE;
    constexpr int EV = 16 / static_cast<int>(sizeof(T)); // elements per 16-byte unit
    static_assert(FMA ? (NB >= 1 && sizeof(T) == 4) : NB % DIM == 0, "stem bwd columns");
    static_assert(KC % KA == 0 && KC % EV == 0, "stem bwd chunk");
    static_assert(CL || CFV == 1 || CFV == EV, "stem bwd channels-first vector");
    static_assert(sbLdsBytes(static_cast<int>(sizeof(T)), U, V, NB, KC, TH, TW)
                      == static_cast<long long>(A_ELEMS + B_ELEMS) * sizeof(T),
                  "stem bwd LDS arithmetic");
    typedef typename K1Raw<16>::type Raw16;

    __shared__ __attribute__((aligned(16))) T As[A_ELEMS];
    __shared__ __attribute__((aligned(16))) T Bs[B_ELEMS];

    const int tid = threadIdx.x;
    const int lane = tid % WAVE, wave = tid / WAVE;
    const int j0 = blockIdx.x * TW;
    const int i0 = blockIdx.y * TH;
    const int n = blockIdx.z;
    // Window origin: dy row i0 + OH - (U - 1), column j0 + OW - (V - 1).
    const int r0 = i0 + a.oh - (U - 1), c0 = j0 + a.ow - (V - 1);

    const T* dy = static_cast<const T*>(a.dy);
    const __amdgpu_buffer_rsrc_t dyR = k1Rsrc(dy + n * a.dys[0], a.dyImage * sizeof(T));
    const __amdgpu_buffer_rsrc_t bR
        = k1Rsrc(a.b, static_cast<long long>(NB) * UV * a.kp * sizeof(T));

    // Loads of a chunk, issued into registers a chunk ahead and stored after the compute:
    // A units (CL: EV channels at one (row, col); CF vectors: EV aligned columns of one (row, k)
    // row, CFV = EV; CF scalars: one element) and B units (EV k of one (n, uv)).
    constexpr int NCU = (WC + 2 * EV - 2) / EV; // aligned column vectors covering the window
    constexpr int UA = CL ? WR * WC * (KC / EV) : (CFV > 1 ? WR * KC * NCU : WR * WC * KC);
    constexpr int UB = NB * UV * (KC / EV);
    constexpr int PA = (UA + THREADS - 1) / THREADS, PB = (UB + THREADS - 1) / THREADS;
    typedef typename K1If<(CL || CFV > 1), Raw16, typename K1Raw<sizeof(T)>::type>::type RawA;
    RawA ra[PA];
    Raw16 rb[PB];
    const int c0a = c0 >= 0 ? c0 / EV * EV : -((-c0 + EV - 1) / EV) * EV;
    auto issue = [&](int k0) {
#pragma unroll
        for(int p = 0; p < PA; ++p)
        {
            const int e = tid + p * THREADS;
            int y, x, k;
            if constexpr(CL)
            {
                const int kv = e % (KC / EV), rc = e / (KC / EV);
                y = r0 + rc / WC;
                x = c0 + rc % WC;
                k = k0 + kv * EV;
            }
            else if constexpr(CFV > 1)
            {
                const int cu = e % NCU, rk = e / NCU;
                y = r0 + rk % WR;
                k = k0 + rk / WR;
                x = c0a + cu * EV;
            }
            else
            {
                const int col = e % WC, rk = e / WC;
                y = r0 + rk % WR;
                k = k0 + rk / WR;
                x = c0 + col;
            }
            const bool ok = e < UA && y >= 0 && y < a.ho && x >= 0 && x < a.wo && k < a.k;
            const unsigned off
                = ok ? static_cast<unsigned>(k * a.dys[1] + y * a.dys[2] + x * a.dys[3]) * sizeof(T)
                     : K1_OUT_OF_RANGE;
            ra[p] = k1BufferLoad<RawA>(dyR, off);
        }
#pragma unroll
        for(int p = 0; p < PB; ++p)
        {
            const int e = tid + p * THREADS;
            const int kv = e % (KC / EV), nuv = e / (KC / EV);
            const unsigned off = e < UB
                                     ? static_cast<unsigned>(nuv * a.kp + k0 + kv * EV) * sizeof(T)
                                     : K1_OUT_OF_RANGE;
            rb[p] = k1BufferLoad<Raw16>(bR, off);
        }
    };
    auto store = [&]() {
#pragma unroll
        for(int p = 0; p < PA; ++p)
        {
            const int e = tid + p * THREADS;
            if(UA % THREADS != 0 && e >= UA)
                continue;
            if constexpr(CL)
            {
                const int kv = e % (KC / EV), rc = e / (KC / EV);
                *reinterpret_cast<Raw16*>(As + rc * KCP + kv * EV) = ra[p];
            }
            else if constexpr(CFV > 1)
            {
                const int cu = e % NCU, rk = e / NCU;
                const int row = rk % WR, kk = rk / WR;
                const int col0 = c0a + cu * EV - c0;
                T t[EV];
                __builtin_memcpy(t, &ra[p], sizeof(t));
#pragma unroll
                for(int i = 0; i < EV; ++i)
                {
                    const int col = col0 + i;
                    if(col >= 0 && col < WC)
                        As[(row * WC + col) * KCP + kk] = t[i];
                }
            }
            else
            {
                const int col = e % WC, rk = e / WC;
                As[((rk % WR) * WC + col) * KCP + rk / WR] = __builtin_bit_cast(T, ra[p]);
            }
        }
#pragma unroll
        for(int p = 0; p < PB; ++p)
        {
            const int e = tid + p * THREADS;
            if(UB % THREADS != 0 && e >= UB)
                continue;
            const int kv = e % (KC / EV), nuv = e / (KC / EV);
            *reinterpret_cast<Raw16*>(Bs + nuv * KCP + kv * EV) = rb[p];
        }
    };

    T* dx = static_cast<T*>(a.dx) + n * a.dxs[0];
    if constexpr(FMA)
    {
        // Thread (row wave, column lane): FM super-pixels, all NB columns, in fp32.
        float acc[FM][NB];
#pragma unroll
        for(int m = 0; m < FM; ++m)
#pragma unroll
            for(int c = 0; c < NB; ++c)
                acc[m][c] = 0.0f;
        issue(0);
        store();
        __syncthreads();
        for(int k0 = 0; k0 < a.kp; k0 += KC)
        {
            if(k0 + KC < a.kp)
                issue(k0 + KC);
                // Rolled over the window: unrolled, the compiler hoists every offset's LDS reads
                // and spills them.
#pragma unroll 1
            for(int u = 0; u < U; ++u)
#pragma unroll 1
                for(int v = 0; v < V; ++v)
                {
                    float av[FM][KC];
#pragma unroll
                    for(int m = 0; m < FM; ++m)
#pragma unroll
                        for(int kk = 0; kk < KC; kk += 4)
                        {
                            const SbF4 t = *reinterpret_cast<const SbF4*>(
                                As + ((wave + U - 1 - u) * WC + m * DIM + lane + V - 1 - v) * KCP
                                + kk);
                            av[m][kk] = t[0];
                            av[m][kk + 1] = t[1];
                            av[m][kk + 2] = t[2];
                            av[m][kk + 3] = t[3];
                        }
#pragma unroll
                    for(int c = 0; c < NB; ++c)
#pragma unroll
                        for(int kk = 0; kk < KC; kk += 4)
                        {
                            const SbF4 bv = *reinterpret_cast<const SbF4*>(
                                Bs + (c * UV + u * V + v) * KCP + kk);
#pragma unroll
                            for(int m = 0; m < FM; ++m)
                            {
                                acc[m][c] += av[m][kk] * bv[0];
                                acc[m][c] += av[m][kk + 1] * bv[1];
                                acc[m][c] += av[m][kk + 2] * bv[2];
                                acc[m][c] += av[m][kk + 3] * bv[3];
                            }
                        }
                }
            __syncthreads();
            if(k0 + KC < a.kp)
            {
                store();
                __syncthreads();
            }
        }
        const int i = i0 + wave;
#pragma unroll
        for(int c = 0; c < NB; ++c)
        {
            const int q = c % SW, p = c / SW % SH, ch = c / (SW * SH);
            const int y = SH * i + p;
            if(ch >= a.c || y >= a.hi)
                continue;
#pragma unroll
            for(int m = 0; m < FM; ++m)
            {
                const int x = SW * (j0 + m * DIM + lane) + q;
                if(x < a.wi)
                    dx[ch * a.dxs[1] + y * a.dxs[2] + x * a.dxs[3]] = K1Elem<T>::store(acc[m][c]);
            }
        }
    }
    else
    {
        constexpr int KPL = Atom::KPL;
        constexpr int FN = NB / DIM;
        typename Atom::acc acc[FM][FN];
#pragma unroll
        for(int m = 0; m < FM; ++m)
#pragma unroll
            for(int f = 0; f < FN; ++f)
                acc[m][f] = typename Atom::acc{};

        const int kb = Atom::template kBase<KPL>(lane);
        issue(0);
        store();
        __syncthreads();
        for(int k0 = 0; k0 < a.kp; k0 += KC)
        {
            // The next chunk's loads travel during this chunk's multiplies.
            if(k0 + KC < a.kp)
                issue(k0 + KC);
#pragma unroll
            for(int u = 0; u < U; ++u)
#pragma unroll
                for(int v = 0; v < V; ++v)
#pragma unroll
                    for(int kk = 0; kk < KC / KA; ++kk)
                    {
                        typename Atom::frag bf[FN];
#pragma unroll
                        for(int f = 0; f < FN; ++f)
                            bf[f] = *reinterpret_cast<const typename Atom::frag*>(
                                Bs + ((f * DIM + lane % DIM) * UV + u * V + v) * KCP + kk * KA
                                + kb);
#pragma unroll
                        for(int m = 0; m < FM; ++m)
                        {
                            // Super-pixel (i0 + wave, j0 + m DIM + lane % DIM): window row
                            // wave + U - 1 - u, column m DIM + lane % DIM + V - 1 - v.
                            const typename Atom::frag af
                                = *reinterpret_cast<const typename Atom::frag*>(
                                    As
                                    + ((wave + U - 1 - u) * WC + m * DIM + lane % DIM + V - 1 - v)
                                          * KCP
                                    + kk * KA + kb);
#pragma unroll
                            for(int f = 0; f < FN; ++f)
                                acc[m][f] = Atom::mma(af, bf[f], acc[m][f]);
                        }
                    }
            __syncthreads();
            if(k0 + KC < a.kp)
            {
                store();
                __syncthreads();
            }
        }

        // D[pixel][n]: pixel j0 + m DIM + dRow, column f DIM + lane % DIM = (c, p, q).
        const int i = i0 + wave;
#pragma unroll
        for(int f = 0; f < FN; ++f)
        {
            const int nn = f * DIM + lane % DIM;
            const int q = nn % SW, p = nn / SW % SH, c = nn / (SW * SH);
            const int y = SH * i + p;
            if(c >= a.c || y >= a.hi)
                continue;
#pragma unroll
            for(int m = 0; m < FM; ++m)
#pragma unroll
                for(int e = 0; e < Atom::ACC; ++e)
                {
                    const int j = j0 + m * DIM + Atom::dRow(lane, e);
                    const int x = SW * j + q;
                    if(x < a.wi)
                        dx[c * a.dxs[1] + y * a.dxs[2] + x * a.dxs[3]]
                            = K1Elem<T>::store(acc[m][f][e]);
                }
        }
    }
}
