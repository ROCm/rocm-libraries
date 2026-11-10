// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The backward-weights kernels (CONV_WRW, K3), compiled by HIPRTC per catalog config. The pack
// defines:
//   CONV_K3_T        element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_K3_LOOP     0 output rows (any stride); 1 row ring (height stride 1); 2 row ring with
//                    LDS DMA (gfx950, width stride 1); 3 row ring on FMA (exact fp32 without
//                    an fp32 matrix core: RDNA)
//   CONV_K3_ATOM     K1's matrix-core atom (ConvIgemmCore.h); loops 0 to 2
//   CONV_K3_LAYOUT   0 channels-first (N C D H W); 1 channels-last (N D H W C) transposed into
//                    LDS; 2 channels-last kept channels-last, LDS transpose reads (gfx950, 16-bit)
//   CONV_K3_BK, _BC, _BQ   tile: output channels, input channels, positions per step
//   CONV_K3_FR       filter rows per block (loop 0); the filter height (loops 1 to 3)
//   CONV_K3_S, _SW, _DW    filter width, its stride and dilation
//   CONV_K3_WK, _WC  waves over output and input channels (loops 0 to 2)
//   CONV_K3_TK, _TC  output and input channels per thread (loop 3)
//   CONV_K3_VEC      elements per 16-byte global load
//   CONV_K3_XCD      1: XCD-contiguous block order (gfx942, gfx950)
//   CONV_K3_ALIGN    1: row-aligned units, unaligned LDS store (loop 1, channels-first)
//   CONV_K3_NS       LDS stages (loop 2)
// ConvWrwReduce, the partials' fixed-order sum, shares the source.

#include "ConvWrwCore.h"

#if CONV_K3_LOOP == 3
#define CONV_K3_THREADS ((CONV_K3_BK / CONV_K3_TK) * (CONV_K3_BC / CONV_K3_TC))
#else
#define CONV_K3_THREADS (CONV_K3_WK * CONV_K3_WC * CONV_K3_ATOM::WAVE)
#endif

extern "C" __global__ void __launch_bounds__(CONV_K3_THREADS) ConvWrw(K3Args a)
{
#if CONV_K3_LOOP == 3
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
    k3Reduce<CONV_K3_T>(a);
}
