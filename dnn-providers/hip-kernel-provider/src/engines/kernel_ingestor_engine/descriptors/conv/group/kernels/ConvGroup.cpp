// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The small-group convolution kernels (CONV_GROUP_FWD, CONV_GROUP_BWD), compiled by HIPRTC per
// catalog config. The pack defines:
//   CONV_GROUP_T        element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_GROUP_ATOM     the 16x16 matrix-core atom (a K1 atom type)
//   CONV_GROUP_FT, _R, _S  filter extent (the kernel's taps: depth 1 for 2-D)
//   CONV_GROUP_ST       the gather's stride (both axes)
//   CONV_GROUP_TH, _WP, _CB   output rows, 16-column fragments and 16-channel blocks a workgroup
//   CONV_GROUP_CF       1: packed channels-first activations read and written in place
//   K1_BUFFER           0 (the K1 header's buffer loads are not used here)

#ifndef K1_BUFFER
#define K1_BUFFER 0
#endif
#include "ConvGroupCore.h"

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
