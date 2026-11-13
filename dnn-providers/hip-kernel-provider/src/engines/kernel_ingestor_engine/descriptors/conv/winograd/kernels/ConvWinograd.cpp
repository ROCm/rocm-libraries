// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The Winograd convolution kernels (CONV_WINOGRAD_FWD, CONV_WINOGRAD_BWD), 3x3 stride 1,
// compiled by HIPRTC per catalog config. The pack defines:
//   CONV_WINO_T          element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_WINO_VARIANT    0 multipass (four launches), 1 fused VALU, 2 fused WMMA (gfx11 / gfx12),
//                        3 fused WMMA with register-built operands (gfx11 / gfx12),
//                        4 fused VALU with register operands (wave64: gfx9),
//                        5 fused WMMA / MFMA, cooperative input transform (gfx11 / gfx12 /
//                          CDNA)
//   CONV_WINO_DIR        0 forward, 1 backward data (the filter pre-pass; and the cooperative
//                        F(3,2) kernel, which gathers x or scatters dx by phases)
//   CONV_WINO_LAYOUT     0 NCHW (filter KCRS), 1 NHWC (filter KRSC); fused variants read NCHW
//   CONV_WINO_M          output tile: 2 for F(2,3), 4 for F(4,3) (multipass fp32 only), 3 for
//                        F(3,2) (stride 2 by phases; multipass, fused VALU and cooperative)
//   multipass:  CONV_WINO_GEMM_BM, _BN, _BK          the batched-GEMM tile
//   fused VALU: CONV_WINO_BK, _BT, _CCV, _TM, _TN, _PFD  K-block, tile-block, chunk, micro-tile,
//               prefetch depth (1 or 2 input chunks ahead)
//   fused WMMA: CONV_WINO_BK, _WAVES, _GFX12         K-block, waves, gfx12 operand layout
//   register WMMA: CONV_WINO_BK, _TG, _BINS, _F16MATH, _GFX12   K-block, 16-tile groups, bins per
//                  wave, fp16 packed input-transform math, gfx12 operand layout
//   register VALU: CONV_WINO_BK, _F16MATH                       K-block, fp16 packed math
//   cooperative: CONV_WINO_MT, _NT, _KW, _F16MATH, _GFX12, _MFMA, _PF  16-channel K tiles per
//                  wave, 16-tile groups, K waves per row, fp16 packed math, gfx12 operand layout,
//                  CDNA MFMA (wave64) instead of RDNA WMMA, prefetch (1: the next item's chunk 0
//                  behind the epilogue)

#include "ConvWinogradSreg.h"
#include "ConvWinogradWmmaCoop.h"

/// Stages an operand to or from the layout the variant reads.
extern "C" __global__ void __launch_bounds__(256) ConvWinogradCopy(K5CopyArgs a)
{
    const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
    for(long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < a.total;
        i += step)
    {
        long long rest = i, src = 0, dst = 0;
        for(int axis = a.rank - 1; axis >= 0; --axis)
        {
            const long long index = rest % a.dims[axis];
            rest /= a.dims[axis];
            src += index * a.srcStrides[axis];
            dst += index * a.dstStrides[axis];
        }
        if(a.elementBytes == 4)
            static_cast<unsigned*>(a.dst)[dst] = static_cast<const unsigned*>(a.src)[src];
        else
            static_cast<unsigned short*>(a.dst)[dst]
                = static_cast<const unsigned short*>(a.src)[src];
    }
}

#if CONV_WINO_VARIANT == 0

extern "C" __global__ void __launch_bounds__(256) ConvWinogradFilter(K5Args a)
{
    k5FilterTransform<CONV_WINO_T, CONV_WINO_T, CONV_WINO_M, CONV_WINO_LAYOUT, CONV_WINO_DIR>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradInput(K5Args a)
{
    k5InputTransform<CONV_WINO_T, CONV_WINO_T, CONV_WINO_M, CONV_WINO_LAYOUT>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradGemm(K5GemmArgs g)
{
    k5Gemm<CONV_WINO_T, CONV_WINO_GEMM_BM, CONV_WINO_GEMM_BN, CONV_WINO_GEMM_BK>(g);
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradOutput(K5Args a)
{
    k5OutputTransform<CONV_WINO_T, CONV_WINO_M, CONV_WINO_LAYOUT>(a);
}

#elif CONV_WINO_VARIANT == 1

#ifndef CONV_WINO_PFD
#define CONV_WINO_PFD 1
#endif

#define CONV_WINO_DOT2 __is_same(CONV_WINO_T, _Float16)
#define CONV_WINO_CFG       \
    K5F2Cfg<CONV_WINO_T,    \
            CONV_WINO_M,    \
            CONV_WINO_DOT2, \
            CONV_WINO_BK,   \
            CONV_WINO_BT,   \
            CONV_WINO_CCV,  \
            CONV_WINO_TM,   \
            CONV_WINO_TN,   \
            false>

extern "C" __global__ void __launch_bounds__(256) ConvWinogradFusedFilter(K5Args a)
{
    k5FusedFilterTransform<CONV_WINO_T, CONV_WINO_M, K5_NCHW, CONV_WINO_DIR, CONV_WINO_DOT2>(a);
}

extern "C" __global__ void __launch_bounds__((CONV_WINO_CFG::THREADS)) ConvWinogradFused(K5Args a)
{
    k5FusedConv2<CONV_WINO_T,
                 CONV_WINO_M,
                 CONV_WINO_DOT2,
                 CONV_WINO_BK,
                 CONV_WINO_BT,
                 CONV_WINO_CCV,
                 CONV_WINO_TM,
                 CONV_WINO_TN,
                 false,
                 CONV_WINO_PFD>(a);
}

#elif CONV_WINO_VARIANT == 2

extern "C" __global__ void __launch_bounds__(256) ConvWinogradWmmaFilter(K5Args a)
{
    k5WmmaFilterTransform<CONV_WINO_T, K5_NCHW, CONV_WINO_DIR>(a);
}

extern "C" __global__ void __launch_bounds__(32 * CONV_WINO_WAVES) ConvWinogradWmma(K5Args a)
{
    k5WmmaConv<CONV_WINO_T, CONV_WINO_GFX12 != 0, CONV_WINO_BK, CONV_WINO_WAVES>(a);
}

#elif CONV_WINO_VARIANT == 3

#define CONV_WINO_WREG_CFG \
    K5WregCfg<CONV_WINO_T, CONV_WINO_GFX12 != 0, CONV_WINO_BK, CONV_WINO_TG, CONV_WINO_BINS>

extern "C" __global__ void __launch_bounds__(256) ConvWinogradWmmaFilter(K5Args a)
{
    k5WmmaFilterTransform<CONV_WINO_T, K5_NCHW, CONV_WINO_DIR>(a);
}

extern "C" __global__ void __launch_bounds__((CONV_WINO_WREG_CFG::THREADS))
    ConvWinogradWreg(K5Args a)
{
    k5WregConv<CONV_WINO_T,
               CONV_WINO_GFX12 != 0,
               CONV_WINO_BK,
               CONV_WINO_TG,
               CONV_WINO_F16MATH != 0,
               CONV_WINO_BINS>(a);
}

#elif CONV_WINO_VARIANT == 4

#define CONV_WINO_DOT2 __is_same(CONV_WINO_T, _Float16)

extern "C" __global__ void __launch_bounds__(256) ConvWinogradFusedFilter(K5Args a)
{
    k5FusedFilterTransform<CONV_WINO_T, 2, K5_NCHW, CONV_WINO_DIR, CONV_WINO_DOT2>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradSreg(K5Args a)
{
    k5SregConv<CONV_WINO_T, CONV_WINO_BK, CONV_WINO_F16MATH != 0>(a);
}

#elif CONV_WINO_VARIANT == 5

#ifndef CONV_WINO_PF
#define CONV_WINO_PF 0
#endif

#if CONV_WINO_MFMA
#define CONV_WINO_COOP_ATOM K5CoopMfma<CONV_WINO_T>
#else
#define CONV_WINO_COOP_ATOM K5CoopWmma<CONV_WINO_T, CONV_WINO_GFX12 != 0>
#endif
#define CONV_WINO_COOP_CFG         \
    K5CoopCfg<CONV_WINO_T,         \
              CONV_WINO_COOP_ATOM, \
              CONV_WINO_MT,        \
              CONV_WINO_NT,        \
              CONV_WINO_KW,        \
              CONV_WINO_M>

extern "C" __global__ void __launch_bounds__(256) ConvWinogradWmmaFilter(K5Args a)
{
    k5CoopFilterTransform<CONV_WINO_T,
                          K5_NCHW,
                          CONV_WINO_DIR,
                          CONV_WINO_COOP_ATOM::KPL,
                          CONV_WINO_M>(a);
}

extern "C" __global__ void __launch_bounds__((CONV_WINO_COOP_CFG::THREADS))
    __attribute__((amdgpu_waves_per_eu(CONV_WINO_COOP_CFG::WPE))) ConvWinogradCoop(K5Args a)
{
    k5CoopConvA<CONV_WINO_T,
                CONV_WINO_COOP_ATOM,
                CONV_WINO_MT,
                CONV_WINO_NT,
                CONV_WINO_KW,
                CONV_WINO_F16MATH != 0,
                CONV_WINO_M,
                CONV_WINO_DIR,
                CONV_WINO_PF>(a);
}

#elif CONV_WINO_VARIANT == 6

// Backward weights, F(3,r) with r = CONV_WINO_M (2, 4 or 6), on packed channels-first x and dy
// (channels-last ones transposed in by ConvWinogradWrwTranspose).
// CONV_WINO_WRW_OPT, the GEMM operands: 0 fp32, 1 the I/O type (fp16 graphs' F(3,2)).
// CONV_WINO_WRW_ATOM: 0 FMA (fp32, or v_dot2_f32_f16 on fp16 operands), 1 MFMA 16x16 (f32, f16),
// 3 WMMA gfx11, 4 WMMA gfx12, 5 MFMA 16x16x32 (gfx950, f16). GEMM block CONV_WINO_GEMM_BM (out channels) x
// _BN (in channels) x _BK (tiles); CONV_WINO_TM x _TN: the FMA thread tile, or a wave's atoms
// along M and N.

#include "ConvIgemmCore.h"
#include "ConvWinogradWrw.h"

#if CONV_WINO_WRW_OPT == 0
#define CONV_WINO_WRW_OP float
#else
#define CONV_WINO_WRW_OP CONV_WINO_T
#endif

#if CONV_WINO_WRW_ATOM == 0
#define CONV_WINO_WRW_A K5WrwFma
#define CONV_WINO_WRW_THREADS \
    ((CONV_WINO_GEMM_BM / CONV_WINO_TM) * (CONV_WINO_GEMM_BN / CONV_WINO_TN))
#else
#if CONV_WINO_WRW_ATOM == 1
#define CONV_WINO_WRW_A                      \
    K1If<__is_same(CONV_WINO_WRW_OP, float), \
         K1AtomMfmaF32,                      \
         K1If<__is_same(CONV_WINO_WRW_OP, _Float16), K1AtomMfmaF16, K1AtomMfmaBf16>::type>::type
#elif CONV_WINO_WRW_ATOM == 3
#define CONV_WINO_WRW_A \
    K1If<__is_same(CONV_WINO_WRW_OP, _Float16), K1AtomWmma11F16, K1AtomWmma11Bf16>::type
#elif CONV_WINO_WRW_ATOM == 4
#define CONV_WINO_WRW_A \
    K1If<__is_same(CONV_WINO_WRW_OP, _Float16), K1AtomWmma12F16, K1AtomWmma12Bf16>::type
#else
#define CONV_WINO_WRW_A \
    K1If<__is_same(CONV_WINO_WRW_OP, _Float16), K1AtomMfmaF16X32, K1AtomMfmaBf16X32>::type
#endif
#define CONV_WINO_WRW_THREADS                                    \
    ((CONV_WINO_GEMM_BM / (CONV_WINO_WRW_A::DIM * CONV_WINO_TM)) \
     * (CONV_WINO_GEMM_BN / (CONV_WINO_WRW_A::DIM * CONV_WINO_TN)) * CONV_WINO_WRW_A::WAVE)
#endif

extern "C" __global__ void __launch_bounds__(256) ConvWinogradWrwTranspose(K5CopyArgs a)
{
    k5wTranspose<CONV_WINO_T>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradWrwX(K5WrwArgs a)
{
    k5wTransform<CONV_WINO_T, CONV_WINO_WRW_OP, CONV_WINO_M, true>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradWrwDy(K5WrwArgs a)
{
    k5wTransform<CONV_WINO_T, CONV_WINO_WRW_OP, CONV_WINO_M, false>(a);
}

extern "C" __global__ void __launch_bounds__(CONV_WINO_WRW_THREADS) ConvWinogradWrwGemm(K5WrwArgs a)
{
    k5wGemm<CONV_WINO_WRW_OP,
            CONV_WINO_WRW_A,
            CONV_WINO_GEMM_BM,
            CONV_WINO_GEMM_BN,
            CONV_WINO_GEMM_BK,
            CONV_WINO_TM,
            CONV_WINO_TN,
            CONV_WINO_WRW_THREADS,
            (CONV_WINO_M + 2) * (CONV_WINO_M + 2)>(a);
}

extern "C" __global__ void __launch_bounds__(256) ConvWinogradWrwOut(K5WrwArgs a)
{
    k5wOutput<CONV_WINO_T, CONV_WINO_M>(a);
}

#endif
