// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The channels-first backward-data implicit-GEMM kernel (CONV_IGEMM_NCHW_BWD), compiled by
// HIPRTC per catalog config. The pack defines:
//   CONV_K7_T        element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_K7_FMA      1 for the FMA kernel; else CONV_K7_ATOM names the matrix-core atom (K1's)
//   CONV_K7_BMQ, _BN, _BK   tile: phase indices, channels, k-step
//   CONV_K7_WM, _WN, _WP  waves over phase indices, channels and phase groups (matrix core)
//   CONV_K7_TQ, _TC  phase indices and channels per thread (FMA)
//   CONV_K7_STAGES   LDS tiles in turn (FMA: 1, or 2 with one barrier per k-step)
//   CONV_K7_VEC      elements per global load
//   CONV_K7_STRIDE   the convolution's stride along the width
//   CONV_K7_TAPS     the most filter taps a stride phase holds (ceil(S / stride) at most this)
//   CONV_K7_FLAT     0, or the phases computed over the flattened dy plane (K7BlockFlat)
//   CONV_K7_HALO     the dy window's elements past the tile (FMA: VEC keeps the register
//                    window, wider reads each tap's window from LDS, stride 1)

#include "ConvIgemmNchwCore.h"

// Three-bf16 (fp32 graphs under TF32 permission): the bf16 kernel on 3x the reduction channels
// (CONV_K7_T unsigned short) writes dx in fp32 (CONV_K7_TO float); CONV_TF32_SPLIT3 adds the
// operand split (ConvTf32Split). Undefined otherwise: dx in the element type, as before.
#ifndef CONV_K7_TO
#define CONV_K7_TO CONV_K7_T
#endif

#if CONV_K7_FMA
#define CONV_K7_BLOCK ((CONV_K7_BN / CONV_K7_TC) * (CONV_K7_BMQ / CONV_K7_TQ))
#else
#define CONV_K7_BLOCK (CONV_K7_WM * CONV_K7_WN * CONV_K7_WP * CONV_K7_ATOM::WAVE)
#endif

extern "C" __global__ void __launch_bounds__(CONV_K7_BLOCK) ConvIgemmNchw(K7Args a)
{
#if CONV_K7_FMA
    k7DgradFma<CONV_K7_T,
               CONV_K7_BMQ,
               CONV_K7_BN,
               CONV_K7_BK,
               CONV_K7_TQ,
               CONV_K7_TC,
               CONV_K7_VEC,
               CONV_K7_STRIDE,
               CONV_K7_TAPS,
               CONV_K7_STAGES,
               CONV_K7_FLAT,
               CONV_K7_HALO>(a);
#else
    k7DgradMma<CONV_K7_T,
               CONV_K7_ATOM,
               CONV_K7_BMQ,
               CONV_K7_BN,
               CONV_K7_BK,
               CONV_K7_WM,
               CONV_K7_WN,
               CONV_K7_WP,
               CONV_K7_VEC,
               CONV_K7_STRIDE,
               CONV_K7_TAPS,
               CONV_K7_FLAT,
               CONV_K7_HALO,
               CONV_K7_TO>(a);
#endif
}
