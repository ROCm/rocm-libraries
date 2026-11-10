// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The small-group convolution kernels (CONV_GROUP_FWD, CONV_GROUP_BWD), compiled by HIPRTC per
// catalog config. The pack defines:
//   CONV_GROUP_T        element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_GROUP_ATOM     the 16x16 matrix-core atom (a K1 atom type), or
//   CONV_GROUP_FMA      1: the FMA kernel, with CONV_GROUP_CG (channels a group), _OC (output
//                       channels a thread) and _PX (output columns a thread)
//   CONV_GROUP_FT, _R, _S  filter extent (the kernel's taps: depth 1 for 2-D)
//   CONV_GROUP_ST       the gather's stride (both axes)
//   CONV_GROUP_TH, _WP, _CB   output rows, 16-column fragments (matrix core) and 16-channel
//                             blocks a workgroup
//   CONV_GROUP_CF       1: packed channels-first activations read and written in place
//   CONV_GROUP_PF       1: phase-fused backward data at stride 2 (k4ConvMmaPhased)
//   CONV_GROUP_WIDE     1: wide groups (k4ConvMmaWide), CONV_GROUP_NBI blocks a group,
//                       CONV_GROUP_NBO output blocks a workgroup
//   K1_BUFFER           0 (the K1 header's buffer loads are not used here)

#ifndef K1_BUFFER
#define K1_BUFFER 0
#endif
#include "ConvGroupCore.h"

#if defined(CONV_GROUP_FMA) && CONV_GROUP_FMA
extern "C" __global__ void __launch_bounds__(256) ConvGroup(K4Args a)
{
    k4ConvFma<CONV_GROUP_T,
              CONV_GROUP_FT,
              CONV_GROUP_R,
              CONV_GROUP_S,
              CONV_GROUP_ST,
              CONV_GROUP_CG,
              CONV_GROUP_OC,
              CONV_GROUP_TH,
              CONV_GROUP_PX,
              CONV_GROUP_CB,
              CONV_GROUP_CF>(a);
}
#elif defined(CONV_GROUP_WIDE) && CONV_GROUP_WIDE
extern "C" __global__ void __launch_bounds__(4 * CONV_GROUP_ATOM::WAVE) ConvGroup(K4Args a)
{
    k4ConvMmaWide<CONV_GROUP_T,
                  CONV_GROUP_ATOM,
                  CONV_GROUP_FT,
                  CONV_GROUP_R,
                  CONV_GROUP_S,
                  CONV_GROUP_ST,
                  CONV_GROUP_TH,
                  CONV_GROUP_WP,
                  CONV_GROUP_NBI,
                  CONV_GROUP_NBO,
                  CONV_GROUP_CF>(a);
}
#elif defined(CONV_GROUP_PF) && CONV_GROUP_PF
extern "C" __global__ void __launch_bounds__(4 * CONV_GROUP_ATOM::WAVE) ConvGroup(K4Args a)
{
    k4ConvMmaPhased<CONV_GROUP_T,
                    CONV_GROUP_ATOM,
                    CONV_GROUP_FT,
                    CONV_GROUP_R,
                    CONV_GROUP_S,
                    CONV_GROUP_TH,
                    CONV_GROUP_WP,
                    CONV_GROUP_CB,
                    CONV_GROUP_CF>(a);
}
#else
extern "C" __global__ void __launch_bounds__(4 * CONV_GROUP_ATOM::WAVE) ConvGroup(K4Args a)
{
    k4ConvMma<CONV_GROUP_T,
              CONV_GROUP_ATOM,
              CONV_GROUP_FT,
              CONV_GROUP_R,
              CONV_GROUP_S,
              CONV_GROUP_ST,
              CONV_GROUP_TH,
              CONV_GROUP_WP,
              CONV_GROUP_CB,
              CONV_GROUP_CF>(a);
}
#endif

/// Expands the graph's filter into the per-block block-diagonal layout the kernel reads.
extern "C" __global__ void __launch_bounds__(256) ConvGroupExpand(K4ExpandArgs a)
{
    k4Expand(a);
}

/// Stages an operand to or from the channels-last layout the kernel reads.
extern "C" __global__ void __launch_bounds__(256) ConvGroupCopy(ConvIgemmCopyArgs a)
{
    k1StridedCopy(a);
}

/// Stages packed channels-first activations to channels-last and back.
extern "C" __global__ void __launch_bounds__(256) ConvGroupTranspose(ConvIgemmTransposeArgs a)
{
    k1Transpose(a);
}
