// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The implicit-GEMM convolution kernels (CONV_IGEMM_FWD, CONV_IGEMM_BWD), compiled by HIPRTC
// per catalog config. The pack defines:
//   CONV_IGEMM_T        element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_IGEMM_FMA      1 for the FMA kernel; else CONV_IGEMM_ATOM names the matrix-core atom
//   CONV_IGEMM_MATH     0 native, 1 three-bf16 (fp32 operands on a bf16 atom)
//   CONV_IGEMM_BM, _BN, _BK, _WM, _WN (waves), _THREADS (FMA), _VEC, _STAGES
//   CONV_IGEMM_DIR      0 forward, 1 backward data
//   CONV_IGEMM_XCD      1: XCD-contiguous block order (gfx942, gfx950)
//   CONV_IGEMM_HALO     1: the halo A loop (stride-1 convolutions whose shift range fits)
//   CONV_IGEMM_WD       1: backward data reads the K T R S C/g filter directly (no transposed copy)
//   CONV_IGEMM_PACK     1: k-steps of whole taps (reduction channels per tap divide BK); 2: the
//                       flat reduction (taps x channels in k-steps of BK)
//   CONV_IGEMM_CF       1: the output is written packed channels-first (matrix-core kernel only)
//   CONV_IGEMM_SWZ      1: XOR-swizzled LDS tiles read in whole 16-byte chunks (16-bit, VEC 8)
//   CONV_IGEMM_DMA      2 to 4: gfx950 buffer loads straight into a ring of that many swizzled
//                       LDS stages (SWZ 1, K1_BUFFER 1)
//   K1_BUFFER           0 when a block's rows could span 2^31 bytes (64-bit global loads)
//   K1_LARGE            1: 64-bit positions within an image (large-image configs, images past
//                       2^31 - 1 pixels)

#include "ConvIgemmCore.h"

#ifndef CONV_IGEMM_XCD
#define CONV_IGEMM_XCD 0
#endif
#ifndef CONV_IGEMM_HALO
#define CONV_IGEMM_HALO 0
#endif
#ifndef CONV_IGEMM_WD
#define CONV_IGEMM_WD 0
#endif
#ifndef CONV_IGEMM_PACK
#define CONV_IGEMM_PACK 0
#endif
#ifndef CONV_IGEMM_CF
#define CONV_IGEMM_CF 0
#endif
#ifndef CONV_IGEMM_SWZ
#define CONV_IGEMM_SWZ 0
#endif
#ifndef CONV_IGEMM_DMA
#define CONV_IGEMM_DMA 0
#endif

#if CONV_IGEMM_FMA
#define CONV_IGEMM_BLOCK CONV_IGEMM_THREADS
#else
#define CONV_IGEMM_BLOCK (CONV_IGEMM_WM * CONV_IGEMM_WN * CONV_IGEMM_ATOM::WAVE)
#endif

// CONV_IGEMM_WAVES_MAX (with CONV_IGEMM_WAVES_MIN): an occupancy target for the register
// allocator, waves per SIMD; unset, the compiler's own.
#if defined(CONV_IGEMM_WAVES_MAX)
#define CONV_IGEMM_OCCUPANCY \
    __attribute__((amdgpu_waves_per_eu(CONV_IGEMM_WAVES_MIN, CONV_IGEMM_WAVES_MAX)))
#else
#define CONV_IGEMM_OCCUPANCY
#endif

extern "C" __global__ void __launch_bounds__(CONV_IGEMM_BLOCK) CONV_IGEMM_OCCUPANCY
    ConvIgemm(K1Args a)
{
#if CONV_IGEMM_FMA
    k1ConvFma<CONV_IGEMM_T,
              CONV_IGEMM_BM,
              CONV_IGEMM_BN,
              CONV_IGEMM_BK,
              CONV_IGEMM_THREADS,
              CONV_IGEMM_VEC,
              CONV_IGEMM_DIR,
              CONV_IGEMM_WD,
              CONV_IGEMM_PACK>(a);
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
              CONV_IGEMM_STAGES,
              CONV_IGEMM_XCD,
              CONV_IGEMM_HALO,
              CONV_IGEMM_WD,
              CONV_IGEMM_PACK,
              CONV_IGEMM_CF,
              CONV_IGEMM_SWZ,
              CONV_IGEMM_DMA>(a);
#endif
}

/// Stages an operand to or from the layout the convolution reads.
extern "C" __global__ void __launch_bounds__(256) ConvIgemmCopy(ConvIgemmCopyArgs a)
{
    k1StridedCopy(a);
}

/// Group fusion: expands f groups' filters into one block-diagonal group.
extern "C" __global__ void __launch_bounds__(256) ConvIgemmExpand(ConvIgemmExpandArgs a)
{
    k1ExpandGroups(a);
}

/// Stages packed channels-first activations to channels-last and back.
extern "C" __global__ void __launch_bounds__(256) ConvIgemmTranspose(ConvIgemmTransposeArgs a)
{
    k1Transpose(a);
}

/// Split-K: adds the splits' fp32 partials in order into the output's element type.
extern "C" __global__ void __launch_bounds__(256) ConvIgemmReduce(K1ReduceArgs a)
{
    k1ReduceSplits<CONV_IGEMM_T>(a);
}
