// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The implicit-GEMM convolution kernels (CONV_IGEMM_FWD, CONV_IGEMM_BWD), compiled by HIPRTC
// per catalog config. The pack defines:
//   CONV_IGEMM_T        element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_IGEMM_FMA      1 for the FMA kernel; else CONV_IGEMM_ATOM names the matrix-core atom
//   CONV_IGEMM_MATH     0 native, 1 three-bf16 (fp32 operands on a bf16 atom)
//   CONV_IGEMM_BM, _BN, _BK, _WM, _WN (waves), _THREADS (FMA), _VEC, _STAGES
//   CONV_IGEMM_DIR      0 forward, 1 backward data
//   K1_BUFFER           0 when a block's rows could span 2^31 bytes (64-bit global loads)

#include "ConvIgemmCore.h"

#if CONV_IGEMM_FMA
#define CONV_IGEMM_BLOCK CONV_IGEMM_THREADS
#else
#define CONV_IGEMM_BLOCK (CONV_IGEMM_WM * CONV_IGEMM_WN * CONV_IGEMM_ATOM::WAVE)
#endif

extern "C" __global__ void __launch_bounds__(CONV_IGEMM_BLOCK) ConvIgemm(K1Args a)
{
#if CONV_IGEMM_FMA
    k1ConvFma<CONV_IGEMM_T,
              CONV_IGEMM_BM,
              CONV_IGEMM_BN,
              CONV_IGEMM_BK,
              CONV_IGEMM_THREADS,
              CONV_IGEMM_VEC,
              CONV_IGEMM_DIR>(a);
#else
    k1ConvMma<CONV_IGEMM_T,
              CONV_IGEMM_ATOM,
              CONV_IGEMM_MATH,
              CONV_IGEMM_BM,
              CONV_IGEMM_BN,
              CONV_IGEMM_BK,
              CONV_IGEMM_WM,
              CONV_IGEMM_WN,
              CONV_IGEMM_VEC,
              CONV_IGEMM_DIR,
              CONV_IGEMM_STAGES>(a);
#endif
}

/// Stages an operand to or from the layout the convolution reads.
extern "C" __global__ void __launch_bounds__(256) ConvIgemmCopy(ConvIgemmCopyArgs a)
{
    k1StridedCopy(a);
}
