// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The backward-weights kernels (CONV_WRW, K3), compiled by HIPRTC per catalog config. The pack
// defines:
//   CONV_K3_T        element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_K3_LOOP     0 output rows (any stride); 1 row ring (height stride 1, or one ring per
//                    height-stride phase); 2 row ring with
//                    LDS DMA (gfx950, width stride 1); 3 row ring on FMA (exact fp32 without
//                    an fp32 matrix core: RDNA); 4 output rows on FMA (any stride, vec 1 for
//                    rows that are not a whole number of vectors); 5 direct (ConvWrwDirect.h:
//                    gfx950, 16-bit, unit stride and height dilation, persistent grid); 6
//                    small-C output rows (stems: (channel, tap) pairs on the N axis,
//                    layout 0 or 1 staged into the same tiles, WC 1); 7 packed rows (several
//                    narrow output rows a step); 8 generic taps (runtime filter, strides and
//                    dilation; x and dy through their strides; FR taps a block, S SW 0; DW 0 per-tap
//                    gathers, or the extra positions of a shared width window)
//   CONV_K3_ATOM     K1's matrix-core atom (ConvIgemmCore.h); loops 0 to 2
//   CONV_K3_LAYOUT   0 channels-first (N C D H W); 1 channels-last (N D H W C) transposed into
//                    LDS; 2 channels-last kept channels-last, LDS transpose reads (gfx950, 16-bit)
//   CONV_K3_BK, _BC, _BQ   tile: output channels, input channels, positions per step
//   CONV_K3_FR       filter rows per block (loop 0); the filter height (loops 1 to 3)
//   CONV_K3_S, _SW, _DW    filter width, its stride and dilation
//   CONV_K3_WK, _WC  waves over output and input channels (loops 0 to 2)
//   CONV_K3_TK, _TC  output and input channels per thread (loops 3, 4)
//   CONV_K3_VEC      elements per 16-byte global load
//   CONV_K3_XCD      1: XCD-contiguous block order (gfx942, gfx950)
//   CONV_K3_ALIGN    1: row-aligned units, unaligned LDS store (loop 1, channels-first)
//   CONV_K3_NS       LDS stages (loop 2); ring slots (loop 5)
//   CONV_K3_WQ       loop 5: items side by side (waves_q; wk x wc x wq = 8 waves)
//   CONV_K3_DWS      1: the epilogue can write dw itself at one split (K3Args::dwStore); 0:
//                    fp32 partials only (default)
// Loop 5 (gfx950, 16-bit, channels-last kept, ConvWrwDirect.h) also takes CONV_K3_PP and
// CONV_K3_SWZ (default 1: the ping-pong and the swizzled rows; 0 only for ablation).
// ConvWrwReduce, the partials' fixed-order sum, shares the source, as does ConvWrwReduceRows, the
// same sum by K C T R S rows (the host's choice for at most 2 splits).
// Three-bf16 (fp32 graphs under TF32 permission) compiles the bf16 kernel (CONV_K3_T unsigned
// short) with CONV_K3_MATH 1, which adds ConvWrwSplit (the operands split into 3N bf16 images,
// K3SplitArgs), and CONV_K3_TO float, the reduces' dw element type. Neither is defined
// otherwise, so every other config compiles exactly as before.

#include "ConvWrwCore.h"
#include "ConvWrwDirect.h"

#ifndef CONV_K3_TO
#define CONV_K3_TO CONV_K3_T
#endif

#if defined(CONV_K3_MATH) && CONV_K3_MATH == 1
/// Splits n fp32 values of v into bf16 hi = round(v) and lo = round(v - hi) (zero where the
/// remainder is not a number: an infinite v stays infinite rather than NaN) and writes them to
/// three copies c0, c1, c2 (each n long), copy i taking 2 bits of `pattern`: 0 hi, 1 lo, 2 hi with
/// infinities zeroed (k1BfFinite: the copy that meets the other operand's lo).
/// Eight values a thread, 16-byte loads and stores, when n is a whole number of eights.
__device__ inline void k3SplitPlane(const float* v, unsigned short* s, long long n, int pattern)
{
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(n % 8 == 0)
    {
        typedef float f4 __attribute__((ext_vector_type(4)));
        typedef unsigned short u8 __attribute__((ext_vector_type(8)));
        for(long long i = first; i < n / 8; i += step)
        {
            const f4 a = reinterpret_cast<const f4*>(v)[2 * i];
            const f4 b = reinterpret_cast<const f4*>(v)[2 * i + 1];
            u8 h, l, f;
#pragma unroll
            for(int e = 0; e < 8; ++e)
            {
                const float x = e < 4 ? a[e] : b[e - 4];
                h[e] = k1FloatToBf(x);
                l[e] = k1BfLo(x, h[e]);
                f[e] = k1BfFinite(h[e]);
            }
#pragma unroll
            for(int c = 0; c < 3; ++c)
            {
                const int part = (pattern >> (2 * c)) & 3;
                reinterpret_cast<u8*>(s + c * n)[i] = part == 1 ? l : part == 2 ? f : h;
            }
        }
        return;
    }
    for(long long i = first; i < n; i += step)
    {
        const unsigned short h = k1FloatToBf(v[i]);
        const unsigned short l = k1BfLo(v[i], h);
#pragma unroll
        for(int c = 0; c < 3; ++c)
        {
            const int part = (pattern >> (2 * c)) & 3;
            s[c * n + i] = part == 1 ? l : part == 2 ? k1BfFinite(h) : h;
        }
    }
}

/// x -> [hi | hi' | lo], dy -> [hi | lo | hi'] (K3SplitArgs; hi' meets the other's lo).
extern "C" __global__ void __launch_bounds__(256) ConvWrwSplit(K3SplitArgs a)
{
    k3SplitPlane(a.x, a.xs, a.xElems, 0 | 2 << 2 | 1 << 4);
    k3SplitPlane(a.dy, a.ys, a.yElems, 0 | 1 << 2 | 2 << 4);
}
#endif

#ifndef CONV_K3_PP
#define CONV_K3_PP 1
#endif
#ifndef CONV_K3_SWZ
#define CONV_K3_SWZ 1
#endif

#if CONV_K3_LOOP == 5
#define CONV_K3_THREADS 512
#elif CONV_K3_LOOP == 3 || CONV_K3_LOOP == 4 || (CONV_K3_LOOP == 8 && CONV_K3_TK > 0)
#define CONV_K3_THREADS ((CONV_K3_BK / CONV_K3_TK) * (CONV_K3_BC / CONV_K3_TC))
#else
#define CONV_K3_THREADS (CONV_K3_WK * CONV_K3_WC * CONV_K3_ATOM::WAVE)
#endif

extern "C" __global__ void __launch_bounds__(CONV_K3_THREADS) ConvWrw(K3Args a)
{
#if CONV_K3_LOOP == 5
    k3WrwDirect<CONV_K3_T,
                CONV_K3_ATOM,
                CONV_K3_BC,
                CONV_K3_BK,
                CONV_K3_WQ,
                CONV_K3_BC / (16 * CONV_K3_WC),
                CONV_K3_BK / (16 * CONV_K3_WK),
                CONV_K3_FR,
                CONV_K3_S,
                CONV_K3_DW,
                CONV_K3_NS - 1,
                CONV_K3_PP,
                CONV_K3_SWZ,
                CONV_K3_XCD>(a);
#elif CONV_K3_LOOP == 6
    k3WrwRowsSmallC<CONV_K3_T,
                    CONV_K3_ATOM,
                    CONV_K3_LAYOUT,
                    CONV_K3_BK,
                    CONV_K3_BC,
                    CONV_K3_BQ,
                    CONV_K3_FR,
                    CONV_K3_S,
                    CONV_K3_SW,
                    CONV_K3_DW,
                    CONV_K3_WK,
                    CONV_K3_VEC,
                    CONV_K3_XCD>(a);
#elif CONV_K3_LOOP == 4
    k3WrwRowsFma<CONV_K3_T,
                 CONV_K3_LAYOUT,
                 CONV_K3_BK,
                 CONV_K3_BC,
                 CONV_K3_BQ,
                 CONV_K3_FR,
                 CONV_K3_S,
                 CONV_K3_SW,
                 CONV_K3_DW,
                 CONV_K3_TK,
                 CONV_K3_TC,
                 CONV_K3_VEC>(a);
#elif CONV_K3_LOOP == 3
    k3WrwRingFma<CONV_K3_T,
                 CONV_K3_LAYOUT,
                 CONV_K3_BK,
                 CONV_K3_BC,
                 CONV_K3_BQ,
                 CONV_K3_FR,
                 CONV_K3_S,
                 CONV_K3_DW,
                 CONV_K3_TK,
                 CONV_K3_TC,
                 CONV_K3_VEC>(a);
#elif CONV_K3_LOOP == 2 && CONV_K3_LAYOUT == 2
    k3WrwRingDmaTr<CONV_K3_T,
                   CONV_K3_ATOM,
                   CONV_K3_BK,
                   CONV_K3_BC,
                   CONV_K3_BQ,
                   CONV_K3_FR,
                   CONV_K3_S,
                   CONV_K3_DW,
                   CONV_K3_WK,
                   CONV_K3_WC,
                   CONV_K3_NS,
                   CONV_K3_XCD>(a);
#elif CONV_K3_LOOP == 2
    k3WrwRingDma<CONV_K3_T,
                 CONV_K3_ATOM,
                 CONV_K3_BK,
                 CONV_K3_BC,
                 CONV_K3_BQ,
                 CONV_K3_FR,
                 CONV_K3_S,
                 CONV_K3_DW,
                 CONV_K3_WK,
                 CONV_K3_WC,
                 CONV_K3_NS,
                 CONV_K3_XCD>(a);
#elif CONV_K3_LOOP == 1
    k3WrwRing<CONV_K3_T,
              CONV_K3_ATOM,
              CONV_K3_LAYOUT,
              CONV_K3_BK,
              CONV_K3_BC,
              CONV_K3_BQ,
              CONV_K3_FR,
              CONV_K3_S,
              CONV_K3_SW,
              CONV_K3_DW,
              CONV_K3_WK,
              CONV_K3_WC,
              CONV_K3_VEC,
              CONV_K3_XCD,
              CONV_K3_ALIGN>(a);
#elif CONV_K3_LOOP == 7
    k3WrwRows<CONV_K3_T,
              CONV_K3_ATOM,
              CONV_K3_LAYOUT,
              CONV_K3_BK,
              CONV_K3_BC,
              CONV_K3_BQ,
              CONV_K3_FR,
              CONV_K3_S,
              CONV_K3_SW,
              CONV_K3_DW,
              CONV_K3_WK,
              CONV_K3_WC,
              CONV_K3_VEC,
              CONV_K3_XCD,
              1>(a);
#elif CONV_K3_LOOP == 8 && CONV_K3_TK > 0
    // The generic loop's FMA form: TK x TC channels a thread, no matrix atom.
    k3WrwGeneric<CONV_K3_T,
                 K3FmaGeneric<CONV_K3_TK, CONV_K3_TC>,
                 CONV_K3_LAYOUT,
                 CONV_K3_BK,
                 CONV_K3_BC,
                 CONV_K3_BQ,
                 CONV_K3_FR,
                 1,
                 1,
                 CONV_K3_VEC,
                 CONV_K3_DW,
                 0>(a);
#elif CONV_K3_LOOP == 8
    // Generic taps: CONV_K3_FR is the taps a block; filter, stride and dilation are runtime.
    k3WrwGeneric<CONV_K3_T,
                 CONV_K3_ATOM,
                 CONV_K3_LAYOUT,
                 CONV_K3_BK,
                 CONV_K3_BC,
                 CONV_K3_BQ,
                 CONV_K3_FR,
                 CONV_K3_WK,
                 CONV_K3_WC,
                 CONV_K3_VEC,
                 CONV_K3_DW,
                 CONV_K3_XCD>(a);
#else
    k3WrwRows<CONV_K3_T,
              CONV_K3_ATOM,
              CONV_K3_LAYOUT,
              CONV_K3_BK,
              CONV_K3_BC,
              CONV_K3_BQ,
              CONV_K3_FR,
              CONV_K3_S,
              CONV_K3_SW,
              CONV_K3_DW,
              CONV_K3_WK,
              CONV_K3_WC,
              CONV_K3_VEC,
              CONV_K3_XCD>(a);
#endif
}

extern "C" __global__ void __launch_bounds__(256) ConvWrwReduce(K3ReduceArgs a)
{
    k3Reduce<CONV_K3_TO>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvWrwReduceRows(K3ReduceArgs a)
{
    k3ReduceRows<CONV_K3_TO>(a);
}
