// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The stem backward-data kernels (CONV_STEM_BWD): dx for at most 4 input channels, the stride
// phases on the GEMM's N axis (ConvStemBwdCore.h). Compiled by HIPRTC per catalog config. The
// pack defines:
//   CONV_SB_T        element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_SB_ATOM     K1's matrix-core atom (ConvIgemmCore.h); absent for FMA (fp32)
//   CONV_SB_LAYOUT   dy 0 channels-first (N K H W), 1 channels-last (N H W K)
//   CONV_SB_VEC      channels-first dy loads: 1 element, or 16 bytes of aligned columns
//   CONV_SB_U, _V    window offsets (filter rows / columns per stride phase, rounded up)
//   CONV_SB_NB       GEMM columns: input channels x stride phases, padded to whole fragments
//   CONV_SB_KC       output channels per chunk
//   CONV_SB_TH, _FM  super-rows per block (one wave each on an atom), fragments of super-columns
//   CONV_SB_SH, _SW  the filter's height and width strides
//   CONV_SB_FT       3-D: filter depth taps (depth stride 1), the planes folded into the grid;
//                    absent (0) for 2-D
//   CONV_SB_SD       stride 1: SD x SD super-pixels on a space-to-depth view of dy (a virtual
//                    stride SD); absent (0) otherwise
// ConvStemBwdTransform, the filter's rearrangement into the workspace, shares the source.

#include "ConvStemBwdCore.h"

#ifndef CONV_SB_FT
#define CONV_SB_FT 0
#endif
#ifndef CONV_SB_SD
#define CONV_SB_SD 0
#endif
// Space-to-depth from channels-first column vectors (sbStemBwd's SDV).
#define CONV_SB_SDV (CONV_SB_SD != 0 && CONV_SB_LAYOUT == 0 && CONV_SB_VEC > 1)

#ifdef CONV_SB_ATOM
#define CONV_SB_FMA 0
#define CONV_SB_ATOM_T CONV_SB_ATOM
#else
#define CONV_SB_FMA 1
#define CONV_SB_ATOM_T SbFma
#endif

extern "C" __global__ void __launch_bounds__(CONV_SB_TH* CONV_SB_ATOM_T::WAVE)
    ConvStemBwd(ConvStemBwdArgs a)
{
    sbStemBwd<CONV_SB_T,
              CONV_SB_ATOM_T,
              CONV_SB_FMA,
              CONV_SB_LAYOUT,
              CONV_SB_VEC,
              CONV_SB_U,
              CONV_SB_V,
              CONV_SB_NB,
              CONV_SB_KC,
              CONV_SB_TH,
              CONV_SB_FM,
              CONV_SB_SH,
              CONV_SB_SW,
              CONV_SB_FT,
              CONV_SB_SD>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvStemBwdTransform(ConvStemBwdTransformArgs a)
{
    sbTransform<CONV_SB_T, CONV_SB_FT, CONV_SB_SD, CONV_SB_KC, CONV_SB_SDV>(a);
}
