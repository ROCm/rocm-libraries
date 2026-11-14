// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Backward data of a 1D convolution with groups == 1, i.e. the forward of a ConvTranspose1d, as an
// implicit GEMM on gfx11 WMMA (fp16 / bf16 inputs, fp32 accumulation). NCL layout, stored as NCHW
// with H == 1.
//
// With dy (N, Ci, Li), weights w (Ci, Co, K) and dx (N, Co, Lo):
//   dx[co, o] = sum over (ci, i, t) with i * S + t - P == o of dy[ci, i] * w[ci, co, t].
// Writing o + P = S * q + r (0 <= r < S), output phase r is the GEMM
//   D_r[co, q] = sum over (ci, m) of w[ci, co, r + S * m] * dy[ci, q - m],
// so no col buffer and no col2im are needed. A block computes every phase for CO_T channels and
// Q_T consecutive q, i.e. CO_T rows of S * Q_T consecutive outputs: dy is read into LDS once per
// stage and shared by all phases, and the phases are interleaved in LDS before the store.
//
// MIOpenConvTranspose1dWmmaHipWeights first arranges the weights as wt[r][m][co][ci] in the
// workspace (zero for t = r + S * m >= K and for channels past Co / Ci), so the 16 K values
// (channels) of a WMMA operand are contiguous. A stage covers CT_CI_T channels (16 or 32).

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif

#ifndef CT_DTYPE
#define CT_DTYPE 2 // 1 fp16, 2 bf16
#endif
#ifndef CT_K
#define CT_K 4 // filter width
#endif
#ifndef CT_S
#define CT_S 2 // stride
#endif

#define CT_T ((CT_K + CT_S - 1) / CT_S) // taps per phase
#define CT_CO_T 64                      // output channels per block
#ifndef CT_Q_T
#define CT_Q_T 64 // q per block (S * CT_Q_T outputs per channel): 64, or 32 for wide strides
#endif
#define CT_QB (CT_Q_T / 32) // 16-q WMMA columns per wave
#ifndef CT_CI_T
#define CT_CI_T 32 // input channels per stage, a multiple of 16 (one WMMA K step each)
#endif
#define CT_XW (CT_Q_T + CT_T - 1) // dy window per block
#define CT_ROW (CT_CI_T + 8)      // LDS row: CT_CI_T channels + padding against bank conflicts
#define CT_OUT_W (CT_S * CT_Q_T)
#define CT_BLOCK_SIZE 256

typedef unsigned short u16;
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
// WMMA operand layout, chosen by the solver from the device name (not from the __GFX11__ /
// __GFX12__ macros). gfx11: a lane holds all 16 K values of its row / column, replicated across
// the two half-waves. gfx12: a lane holds 8, lanes 0-15 K 0-7 and lanes 16-31 K 8-15.
#ifndef CT_WMMA_GFX12
#define CT_WMMA_GFX12 0
#endif
#if CT_WMMA_GFX12
#define CT_FRAG_N 8
#else
#define CT_FRAG_N 16
#endif
#if CT_DTYPE == 1
typedef _Float16 frag_t __attribute__((ext_vector_type(CT_FRAG_N)));
#else
typedef __bf16 frag_t __attribute__((ext_vector_type(CT_FRAG_N)));
#endif

__device__ static inline u16 to_storage(float v)
{
#if CT_DTYPE == 1
    return __builtin_bit_cast(u16, static_cast<_Float16>(v));
#else
    // Round to nearest even; NaN stays NaN.
    const unsigned int u = __builtin_bit_cast(unsigned int, v);
    if((u & 0x7fffffffu) > 0x7f800000u)
        return static_cast<u16>((u >> 16) | 0x40u);
    return static_cast<u16>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
#endif
}

__device__ static inline f32x8 wmma(frag_t a, frag_t b, f32x8 c)
{
#if CT_WMMA_GFX12 && CT_DTYPE == 1
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#elif CT_WMMA_GFX12
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#elif CT_DTYPE == 1
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#endif
}

// First K value (channel) of a lane's fragment, and the D row of its accumulator value i.
__device__ static inline int frag_k0(int lane) { return CT_WMMA_GFX12 ? 8 * (lane / 16) : 0; }

__device__ static inline int d_row(int lane, int i)
{
    return CT_WMMA_GFX12 ? i + 8 * (lane / 16) : 2 * i + lane / 16;
}

// wt[r][m][co][ci] (Co_pad x Ci_pad per (r, m)) from w[ci][co][t], transposed through LDS: a block
// reads CT_WT_CI input channels x CT_WT_CO output channels, i.e. CT_WT_CO * K contiguous values per
// input channel, and writes CT_WT_CI contiguous channels per (r, m, co).
#define CT_WT_CI 64
#ifndef CT_WT_CO
#define CT_WT_CO 16 // fewer for small weights, to keep enough blocks
#endif
#define CT_WT_ROW (CT_WT_CO * CT_K + 1) // LDS row, padded against bank conflicts

extern "C" __global__ void __launch_bounds__(CT_BLOCK_SIZE) MIOpenConvTranspose1dWmmaHipWeights(
    const u16* __restrict__ w, u16* __restrict__ wt, int Ci, int Co, int Ci_pad, int Co_pad)
{
    __shared__ u16 tile[CT_WT_CI * CT_WT_ROW];
    const int ci0 = blockIdx.x * CT_WT_CI;
    const int co0 = blockIdx.y * CT_WT_CO;

    for(int i = threadIdx.x; i < CT_WT_CI * CT_WT_CO * CT_K; i += CT_BLOCK_SIZE)
    {
        const int c             = i / (CT_WT_CO * CT_K);
        const int j             = i % (CT_WT_CO * CT_K); // (co - co0) * K + t
        const int ci            = ci0 + c;
        tile[c * CT_WT_ROW + j] = (ci < Ci && co0 + j / CT_K < Co)
                                      ? w[(static_cast<long>(ci) * Co + co0) * CT_K + j]
                                      : static_cast<u16>(0);
    }
    __syncthreads();

    // Co_pad is a multiple of CT_WT_CO; Ci_pad may end inside the last block.
    for(int i = threadIdx.x; i < CT_S * CT_T * CT_WT_CO * CT_WT_CI; i += CT_BLOCK_SIZE)
    {
        const int c  = i % CT_WT_CI;
        const int cl = (i / CT_WT_CI) % CT_WT_CO;
        const int rm = i / (CT_WT_CI * CT_WT_CO);
        const int t  = (rm / CT_T) + CT_S * (rm % CT_T); // r + S * m
        if(ci0 + c < Ci_pad)
            wt[(static_cast<long>(rm) * Co_pad + co0 + cl) * Ci_pad + ci0 + c] =
                t < CT_K ? tile[c * CT_WT_ROW + cl * CT_K + t] : static_cast<u16>(0);
    }
}

#define CT_WS_ELEMS (CT_S * CT_T * CT_CO_T * CT_ROW)
#define CT_YS_ELEMS (CT_CO_T * CT_OUT_W)
#define CT_SMEM_ELEMS (CT_WS_ELEMS > CT_YS_ELEMS ? CT_WS_ELEMS : CT_YS_ELEMS)
#define CT_NXI (CT_XW * (CT_CI_T / 8))                 // 8-channel dy items per stage
#define CT_NWI (CT_S * CT_T * CT_CO_T * (CT_CI_T / 8)) // 8-channel weight items per stage
#define CT_NX ((CT_NXI + CT_BLOCK_SIZE - 1) / CT_BLOCK_SIZE)
#define CT_NW ((CT_NWI + CT_BLOCK_SIZE - 1) / CT_BLOCK_SIZE)

// Grid: x over q blocks, y over output channel blocks, z over the batch. 8 waves: wave -> 16
// output channels (wave % 4) x CT_Q_T / 2 q (wave / 4), accumulating all S phases.
extern "C" __global__ void __launch_bounds__(CT_BLOCK_SIZE)
    MIOpenConvTranspose1dWmmaHip(const u16* __restrict__ dy,
                                 const u16* __restrict__ wt,
                                 u16* __restrict__ dx,
                                 int Ci,
                                 int Co,
                                 int Ci_pad,
                                 int Co_pad,
                                 int Li,
                                 int Lo,
                                 int P)
{
    __shared__ __attribute__((aligned(16))) u16 xs[CT_XW * CT_ROW];  // [q][ci]
    __shared__ __attribute__((aligned(16))) u16 smem[CT_SMEM_ELEMS]; // weights, then dx tile
    u16* ws = smem;                                                  // [r][m][co][ci]

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int wco  = wave % 4;
    const int wq   = wave / 4;
    const int q0   = static_cast<int>(blockIdx.x) * CT_Q_T;
    const int co0  = static_cast<int>(blockIdx.y) * CT_CO_T;
    const u16* in  = dy + static_cast<long>(blockIdx.z) * Ci * Li;

    f32x8 acc[CT_S][CT_QB] = {};

    // Global -> register prefetch of the next stage; register -> LDS after the current compute.
    u32x4 px[CT_NX], pw[CT_NW];
    auto fetch = [&](int ci0) {
#pragma unroll
        for(int a = 0; a < CT_NX; ++a)
        {
            const int e = tid + CT_BLOCK_SIZE * a;
            if(e < CT_NXI)
            {
                const int j  = e % CT_XW;
                const int c8 = (e / CT_XW) * 8;
                const int i  = q0 - (CT_T - 1) + j;
                u16 v[8];
#pragma unroll
                for(int c = 0; c < 8; ++c)
                {
                    const int ci = ci0 + c8 + c;
                    v[c] = (i >= 0 && i < Li && ci < Ci) ? in[static_cast<long>(ci) * Li + i]
                                                         : static_cast<u16>(0);
                }
                px[a] = __builtin_bit_cast(u32x4, v);
            }
        }
#pragma unroll
        for(int b = 0; b < CT_NW; ++b)
        {
            const int e = tid + CT_BLOCK_SIZE * b;
            if(e < CT_NWI)
            {
                const int c8 = (e % (CT_CI_T / 8)) * 8;
                const int j  = (e / (CT_CI_T / 8)) % CT_CO_T;
                const int rm = e / (CT_CO_T * (CT_CI_T / 8));
                pw[b]        = *reinterpret_cast<const u32x4*>(
                    &wt[(static_cast<long>(rm) * Co_pad + co0 + j) * Ci_pad + ci0 + c8]);
            }
        }
    };
    auto store = [&]() {
#pragma unroll
        for(int a = 0; a < CT_NX; ++a)
        {
            const int e = tid + CT_BLOCK_SIZE * a;
            if(e < CT_NXI)
                *reinterpret_cast<u32x4*>(&xs[(e % CT_XW) * CT_ROW + (e / CT_XW) * 8]) = px[a];
        }
#pragma unroll
        for(int b = 0; b < CT_NW; ++b)
        {
            const int e = tid + CT_BLOCK_SIZE * b;
            if(e < CT_NWI)
            {
                const int c8 = (e % (CT_CI_T / 8)) * 8;
                const int j  = (e / (CT_CI_T / 8)) % CT_CO_T;
                const int rm = e / (CT_CO_T * (CT_CI_T / 8));
                *reinterpret_cast<u32x4*>(&ws[(rm * CT_CO_T + j) * CT_ROW + c8]) = pw[b];
            }
        }
    };

    fetch(0);
    for(int ci0 = 0; ci0 < Ci_pad; ci0 += CT_CI_T)
    {
        store();
        __syncthreads();
        if(ci0 + CT_CI_T < Ci_pad)
            fetch(ci0 + CT_CI_T);
#pragma unroll
        for(int kk = 0; kk < CT_CI_T; kk += 16)
#pragma unroll
            for(int m = 0; m < CT_T; ++m)
            {
                // B: column q, channels of dy[., q - m]; A: row co, channels of w. A and B take the
                // same channels per lane.
                const int k0 = kk + frag_k0(lane);
                frag_t b[CT_QB];
#pragma unroll
                for(int qb = 0; qb < CT_QB; ++qb)
                    b[qb] = *reinterpret_cast<const frag_t*>(
                        &xs[(16 * (CT_QB * wq + qb) + lane % 16 + (CT_T - 1) - m) * CT_ROW + k0]);
#pragma unroll
                for(int r = 0; r < CT_S; ++r)
                {
                    const frag_t a = *reinterpret_cast<const frag_t*>(
                        &ws[((r * CT_T + m) * CT_CO_T + 16 * wco + lane % 16) * CT_ROW + k0]);
#pragma unroll
                    for(int qb = 0; qb < CT_QB; ++qb)
                        acc[r][qb] = wmma(a, b[qb], acc[r][qb]);
                }
            }
        __syncthreads();
    }

    // WMMA D layout: a lane holds column lane % 16; value i is row d_row(lane, i). Interleave the
    // phases in LDS as ys[co][S * q + r].
    u16* ys = smem;
#pragma unroll
    for(int r = 0; r < CT_S; ++r)
#pragma unroll
        for(int qb = 0; qb < CT_QB; ++qb)
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int co                     = 16 * wco + d_row(lane, i);
                const int q                      = 16 * (CT_QB * wq + qb) + lane % 16;
                ys[co * CT_OUT_W + CT_S * q + r] = to_storage(acc[r][qb][i]);
            }
    __syncthreads();

    // This block's outputs are o = S * q0 - P + j, j in [0, CT_OUT_W).
    const int ob = CT_S * q0 - P;
    u16* out     = dx + static_cast<long>(blockIdx.z) * Co * Lo;
    for(int e = tid; e < CT_CO_T * CT_OUT_W; e += CT_BLOCK_SIZE)
    {
        const int co = co0 + e / CT_OUT_W;
        const int j  = e % CT_OUT_W;
        const int o  = ob + j;
        if(co < Co && o >= 0 && o < Lo)
            out[static_cast<long>(co) * Lo + o] = ys[e];
    }
}
