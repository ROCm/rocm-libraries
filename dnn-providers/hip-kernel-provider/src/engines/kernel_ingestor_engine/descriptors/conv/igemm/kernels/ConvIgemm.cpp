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
//                       2: bounded HALF pointwise input and output are both channels-first
//   CONV_IGEMM_SWZ      1: XOR-swizzled LDS tiles read in whole 16-byte chunks (16-bit, VEC 8)
//   CONV_IGEMM_DMA      2 to 4: gfx950 buffer loads straight into a ring of that many swizzled
//                       LDS stages (SWZ 1, K1_BUFFER 1)
//   CONV_IGEMM_CFX      1: 1-D problems read the gathered activation in place from packed
//                       channels-first memory and write the output channels-first (K1_BUFFER 1;
//                       the matrix-core kernel with CF 1, or the FMA kernel)
//   CONV_IGEMM_FLEX     the flex tile over flattened output positions, eight waves in two groups
//                       (ping-pong). 1: gfx950 16-bit 16x16x32 atoms, BM 256 x BN 256, 192 or 128,
//                       both operands by LDS DMA (k1ConvFlex, ConvIgemmFlex.h); 2: gfx942 16x16x16
//                       and 16x16x4 atoms, BM 256 x BN 128, one LDS stage fed from registers
//                       (k1ConvFlexReg, ConvIgemmFlexReg.h; the pack picks the form by the atom;
//                       CONV_IGEMM_WD reads the graph's filter in backward data); defined only
//                       when set
//   CONV_IGEMM_BOX      1 (CONV_IGEMM_FLEX 2, backward data): rows in box order, each block's
//                       steps over the taps its rows' classes reach (K1BoxArgs, a second kernel
//                       argument); defined only when set
//   K1_BUFFER           0 when a block's rows could span 2^31 bytes (64-bit global loads)
//   K1_LARGE            1: 64-bit positions within an image (large-image configs, images past
//                       2^31 - 1 pixels)
//   K1_PATCH            1: patch configs (backward data's patch scatter); 2: the patch-tile
//                       kernel (k1PatchMma, ConvIgemmPatch.h); 3: col2im backward data (the
//                       GEMM over dy's positions writes its T R S C columns to the partials,
//                       ConvIgemmCol2im sums them into dx)

#include "ConvIgemmCore.h"
#include "ConvIgemmPatch.h"
#ifndef CONV_IGEMM_FLEX
#define CONV_IGEMM_FLEX 0
#endif
#if CONV_IGEMM_FLEX
#include "ConvIgemmFlex.h"
#endif
#if CONV_IGEMM_FLEX == 2
#include "ConvIgemmFlexReg.h"
#endif

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
#ifndef CONV_IGEMM_CFX
#define CONV_IGEMM_CFX 0
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

#if defined(CONV_IGEMM_BOX) && CONV_IGEMM_BOX
#define CONV_IGEMM_BOX_PARAM , K1BoxArgs box
#define CONV_IGEMM_BOX_ARG , &box
#define CONV_IGEMM_BOX_FLAG 1
#else
#define CONV_IGEMM_BOX_PARAM
#define CONV_IGEMM_BOX_ARG
#define CONV_IGEMM_BOX_FLAG 0
#endif

extern "C" __global__ void __launch_bounds__(CONV_IGEMM_BLOCK) CONV_IGEMM_OCCUPANCY
    ConvIgemm(K1Args a CONV_EPI_PARAM CONV_IGEMM_BOX_PARAM)
{
#if K1_PATCH == 2
    k1PatchMma<CONV_IGEMM_T,
               CONV_IGEMM_ATOM,
               CONV_IGEMM_BM,
               CONV_IGEMM_BN,
               CONV_IGEMM_BK,
               CONV_IGEMM_WM,
               CONV_IGEMM_WN,
               CONV_IGEMM_VEC,
               CONV_IGEMM_DIR,
               CONV_IGEMM_WD>(a CONV_EPI_ARG);
#elif CONV_IGEMM_FLEX
#if CONV_IGEMM_FLEX == 2
    k1ConvFlexReg<CONV_IGEMM_T,
                  CONV_IGEMM_ATOM,
                  CONV_IGEMM_DIR,
                  CONV_IGEMM_BN,
                  CONV_IGEMM_WD,
                  CONV_IGEMM_BK* static_cast<int>(sizeof(CONV_IGEMM_T)),
                  CONV_IGEMM_BOX_FLAG>(a CONV_EPI_ARG CONV_IGEMM_BOX_ARG);
#else
    k1ConvFlex<CONV_IGEMM_T, CONV_IGEMM_ATOM, CONV_IGEMM_DIR, CONV_IGEMM_BN>(a CONV_EPI_ARG);
#endif
#elif CONV_IGEMM_FMA
    k1ConvFma<CONV_IGEMM_T,
              CONV_IGEMM_BM,
              CONV_IGEMM_BN,
              CONV_IGEMM_BK,
              CONV_IGEMM_THREADS,
              CONV_IGEMM_VEC,
              CONV_IGEMM_DIR,
              CONV_IGEMM_WD,
              CONV_IGEMM_PACK,
              CONV_IGEMM_CFX>(a CONV_EPI_ARG);
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
              CONV_IGEMM_DMA,
              CONV_IGEMM_CFX>(a CONV_EPI_ARG);
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

/// Col2im backward data (K1_PATCH 3): sums the GEMM's columns into dx, tap by tap.
extern "C" __global__ void __launch_bounds__(256) ConvIgemmCol2im(K1Col2imArgs a)
{
    k1Col2im<CONV_IGEMM_T>(a);
}

/// Split-K: adds the splits' fp32 partials in order into the output's element type.
extern "C" __global__ void __launch_bounds__(256) ConvIgemmReduce(K1ReduceArgs a CONV_EPI_PARAM)
{
#if defined(CONV_EPI) && CONV_EPI
    k1ReduceSplitsEpi<CONV_IGEMM_T>(a, convEpiGen(epiArgs));
#else
    k1ReduceSplits<CONV_IGEMM_T>(a);
#endif
}
