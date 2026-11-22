// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The stride-1 halo-tile direct convolution kernels (CONV_DIRECT_FWD, CONV_DIRECT_BWD), compiled
// by HIPRTC per catalog config. The pack defines:
//   CONV_DIRECT_T       operand type: float, _Float16, or unsigned short (bf16 bits; also the
//                       three-bf16 operands)
//   CONV_DIRECT_TO      output type: CONV_DIRECT_T, or float for three-bf16
//   CONV_DIRECT_MATH    0 native, 1 three-bf16 (fp32 graph run as a 3C-channel bf16 conv)
//   CONV_DIRECT_KD, _R, _S   filter extent
//   CONV_DIRECT_FMA     1 for the FMA kernel: _BC, _TM, _TN, _NT, _THREADS, _VEC, _RING, _PIPE
//                       0 for the matrix-core kernels: CONV_DIRECT_ATOM, _BC, _VEC, and
//   CONV_DIRECT_ROWS    0: the halo-tile kernel, _BM, _BN, _WM, _WN, _RING, _BL, _SPLIT
//                       1: the filter-row reuse kernel (ConvDirectRows.h, 2-D and 3-D), _PW,
//                       _FN, _WP, _WQ, _WK, _PP, _EPI, and (2-D) _UN, images per MFMA column
//                       group (default 1)
//   CONV_DIRECT_CF      1: packed channels-first activations read and written in place (16-bit,
//                       2-D and 3-D; default 0)
//   CONV_DIRECT_XPF     1 (halo tile, BL 0): weight fragments prefetched across channel chunks,
//                       ring sets skipped by scalar branches, 16-bit halo rows padded by 8 bytes
//                       (default 0)
//   CONV_DIRECT_RUN     halo tile, ring 0, channels-last: run tiles of BM consecutive pixels of
//                       a strip of the output (K2Block::initRun), the stage CONV_DIRECT_RUN
//                       pixels (default 0: th x tw tiles)
//   CONV_DIRECT_LINE    1: 2-D rows on one-row images and 1 x S filters, the row as one halo
//                       line (default 0)
//   CONV_DIRECT_ZRING   1: the z-ring tile (ConvDirectZring.h), CONV_DIRECT_ATOM, _BM, _BN, _BC,
//                       _WM, _WN; one workgroup per compute unit (default 0)

#include "ConvDirectCore.h"
#ifndef CONV_DIRECT_ROWS
#define CONV_DIRECT_ROWS 0
#endif
#if CONV_DIRECT_ROWS
#include "ConvDirectRows.h"
#ifndef CONV_DIRECT_UN
#define CONV_DIRECT_UN 1
#endif
#endif
#ifndef CONV_DIRECT_CF
#define CONV_DIRECT_CF 0
#endif
#ifndef CONV_DIRECT_XPF
#define CONV_DIRECT_XPF 0
#endif
#ifndef CONV_DIRECT_DIL
#define CONV_DIRECT_DIL 1
#endif
#ifndef CONV_DIRECT_RUN
#define CONV_DIRECT_RUN 0
#endif
#ifndef CONV_DIRECT_LINE
#define CONV_DIRECT_LINE 0
#endif
#ifndef CONV_DIRECT_ZRING
#define CONV_DIRECT_ZRING 0
#endif
#if CONV_DIRECT_ZRING
#include "ConvDirectZring.h"
#endif
#if CONV_DIRECT_DIL != 1 && !(CONV_DIRECT_ROWS && CONV_DIRECT_KD == 1)
#error "dilation is served by the 2-D rows kernels only"
#endif
#if CONV_DIRECT_LINE != 0 && !(CONV_DIRECT_ROWS && CONV_DIRECT_KD == 1 && CONV_DIRECT_R == 1)
#error "line tiles are 2-D rows kernels of 1 x S filters"
#endif

#if CONV_DIRECT_FMA
#define CONV_DIRECT_BLOCK CONV_DIRECT_THREADS
#elif CONV_DIRECT_ROWS
#define CONV_DIRECT_BLOCK (CONV_DIRECT_WP * CONV_DIRECT_WQ * CONV_DIRECT_WK * 64)
#else
#define CONV_DIRECT_BLOCK (CONV_DIRECT_WM * CONV_DIRECT_WN * CONV_DIRECT_ATOM::WAVE)
#endif

extern "C" __global__ void __launch_bounds__(CONV_DIRECT_BLOCK) ConvDirect(K2Args a CONV_EPI_PARAM)
{
#if CONV_DIRECT_FMA
    k2ConvFma<CONV_DIRECT_T,
              CONV_DIRECT_KD,
              CONV_DIRECT_R,
              CONV_DIRECT_S,
              CONV_DIRECT_BC,
              CONV_DIRECT_TM,
              CONV_DIRECT_TN,
              CONV_DIRECT_NT,
              CONV_DIRECT_THREADS,
              CONV_DIRECT_VEC,
              CONV_DIRECT_RING,
              CONV_DIRECT_PIPE>(a CONV_EPI_ARG);
#elif CONV_DIRECT_ROWS && CONV_DIRECT_CF && CONV_DIRECT_KD > 1
    k2ConvRowsCf3d<CONV_DIRECT_T,
                   CONV_DIRECT_ATOM,
                   CONV_DIRECT_R,
                   CONV_DIRECT_S,
                   CONV_DIRECT_PW,
                   CONV_DIRECT_FN,
                   CONV_DIRECT_WP,
                   CONV_DIRECT_WQ,
                   CONV_DIRECT_WK,
                   CONV_DIRECT_BC,
                   CONV_DIRECT_TO,
                   CONV_DIRECT_KD>(a CONV_EPI_ARG);
#elif CONV_DIRECT_ROWS && CONV_DIRECT_CF
    k2ConvRowsCf<CONV_DIRECT_T,
                 CONV_DIRECT_ATOM,
                 CONV_DIRECT_R,
                 CONV_DIRECT_S,
                 CONV_DIRECT_PW,
                 CONV_DIRECT_FN,
                 CONV_DIRECT_WP,
                 CONV_DIRECT_WQ,
                 CONV_DIRECT_WK,
                 CONV_DIRECT_BC,
                 CONV_DIRECT_TO,
                 CONV_DIRECT_DIL,
                 CONV_DIRECT_LINE>(a CONV_EPI_ARG);
#elif CONV_DIRECT_ROWS && CONV_DIRECT_KD > 1
    k2ConvRows3d<CONV_DIRECT_T,
                 CONV_DIRECT_ATOM,
                 CONV_DIRECT_R,
                 CONV_DIRECT_S,
                 CONV_DIRECT_PW,
                 CONV_DIRECT_FN,
                 CONV_DIRECT_WP,
                 CONV_DIRECT_WQ,
                 CONV_DIRECT_WK,
                 CONV_DIRECT_BC,
                 CONV_DIRECT_PP,
                 CONV_DIRECT_EPI,
                 CONV_DIRECT_TO,
                 CONV_DIRECT_KD>(a CONV_EPI_ARG);
#elif CONV_DIRECT_ROWS
    k2ConvRows<CONV_DIRECT_T,
               CONV_DIRECT_ATOM,
               CONV_DIRECT_R,
               CONV_DIRECT_S,
               CONV_DIRECT_PW,
               CONV_DIRECT_FN,
               CONV_DIRECT_WP,
               CONV_DIRECT_WQ,
               CONV_DIRECT_WK,
               CONV_DIRECT_BC,
               CONV_DIRECT_PP,
               CONV_DIRECT_EPI,
               CONV_DIRECT_TO,
               CONV_DIRECT_UN,
               CONV_DIRECT_SPLIT,
               CONV_DIRECT_DIL,
               CONV_DIRECT_LINE>(a CONV_EPI_ARG);
#elif CONV_DIRECT_ZRING
    k2ConvZring<CONV_DIRECT_T,
                CONV_DIRECT_ATOM,
                CONV_DIRECT_KD,
                CONV_DIRECT_R,
                CONV_DIRECT_S,
                CONV_DIRECT_BM,
                CONV_DIRECT_BN,
                CONV_DIRECT_BC,
                CONV_DIRECT_WM,
                CONV_DIRECT_WN,
                CONV_DIRECT_TO>(a CONV_EPI_ARG);
#else
    k2ConvMma<CONV_DIRECT_T,
              CONV_DIRECT_ATOM,
              CONV_DIRECT_KD,
              CONV_DIRECT_R,
              CONV_DIRECT_S,
              CONV_DIRECT_BM,
              CONV_DIRECT_BN,
              CONV_DIRECT_BC,
              CONV_DIRECT_WM,
              CONV_DIRECT_WN,
              CONV_DIRECT_VEC,
              CONV_DIRECT_RING,
              CONV_DIRECT_BL,
              CONV_DIRECT_TO,
              CONV_DIRECT_SPLIT,
              CONV_DIRECT_CF    ? k2CfHaloMax(CONV_DIRECT_BM, CONV_DIRECT_R, CONV_DIRECT_S)
              : CONV_DIRECT_RUN ? CONV_DIRECT_RUN
                                : k2HaloMax(CONV_DIRECT_BM, CONV_DIRECT_R, CONV_DIRECT_S),
              CONV_DIRECT_CF,
              2,
              CONV_DIRECT_XPF,
              CONV_DIRECT_RUN>(a CONV_EPI_ARG);
#endif
}

/// Formats the graph's filter into the layout this config reads.
extern "C" __global__ void __launch_bounds__(256) ConvDirectFormat(K2FormatArgs a)
{
#if CONV_DIRECT_ROWS
    k2rFormatWeights<CONV_DIRECT_T, CONV_DIRECT_ATOM::KPL, CONV_DIRECT_FN, CONV_DIRECT_MATH>(a);
#elif CONV_DIRECT_MATH
    k2FormatWeightsSplit(a);
#else
    k2FormatWeights<CONV_DIRECT_T>(a);
#endif
}

/// Three-bf16: splits the fp32 input into [hi | hi | lo] bf16 channels.
extern "C" __global__ void __launch_bounds__(256) ConvDirectSplitInput(K2SplitArgs a)
{
    k2SplitInput(a);
}

/// Split-C: adds the splits' fp32 partials in order into the output's element type.
extern "C" __global__ void __launch_bounds__(256) ConvDirectReduce(K2ReduceArgs a CONV_EPI_PARAM)
{
#if defined(CONV_EPI) && CONV_EPI
    k2SplitReduceEpi<CONV_DIRECT_TO>(a, convEpiGen(epiArgs));
#else
    k2SplitReduce<CONV_DIRECT_TO>(a);
#endif
}

/// Stages an operand to or from the layout the convolution reads.
extern "C" __global__ void __launch_bounds__(256) ConvDirectCopy(ConvIgemmCopyArgs a)
{
    k1StridedCopy(a);
}

/// Stages packed channels-first activations to channels-last and back.
extern "C" __global__ void __launch_bounds__(256) ConvDirectTranspose(ConvIgemmTransposeArgs a)
{
    k1Transpose(a);
}
