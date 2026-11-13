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
//                       _FN, _WP, _WQ, _WK, _PP, _EPI

#include "ConvDirectCore.h"
#ifndef CONV_DIRECT_ROWS
#define CONV_DIRECT_ROWS 0
#endif
#if CONV_DIRECT_ROWS
#include "ConvDirectRows.h"
#endif

#if CONV_DIRECT_FMA
#define CONV_DIRECT_BLOCK CONV_DIRECT_THREADS
#elif CONV_DIRECT_ROWS
#define CONV_DIRECT_BLOCK (CONV_DIRECT_WP * CONV_DIRECT_WQ * CONV_DIRECT_WK * 64)
#else
#define CONV_DIRECT_BLOCK (CONV_DIRECT_WM * CONV_DIRECT_WN * CONV_DIRECT_ATOM::WAVE)
#endif

extern "C" __global__ void __launch_bounds__(CONV_DIRECT_BLOCK) ConvDirect(K2Args a)
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
              CONV_DIRECT_PIPE>(a);
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
                 CONV_DIRECT_KD>(a);
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
               CONV_DIRECT_TO>(a);
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
              CONV_DIRECT_SPLIT>(a);
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
extern "C" __global__ void __launch_bounds__(256) ConvDirectReduce(K2ReduceArgs a)
{
    k2SplitReduce<CONV_DIRECT_TO>(a);
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
