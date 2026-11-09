// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The Winograd convolution kernels (CONV_WINOGRAD_FWD, CONV_WINOGRAD_BWD), 3x3 stride 1,
// compiled by HIPRTC per catalog config. The pack defines:
//   CONV_WINO_T          element type: float, _Float16, or unsigned short (bf16 bits)
//   CONV_WINO_VARIANT    0 multipass (four launches), 1 fused VALU, 2 fused WMMA (gfx11 / gfx12)
//   CONV_WINO_DIR        0 forward, 1 backward data (the filter pre-pass only)
//   CONV_WINO_LAYOUT     0 NCHW (filter KCRS), 1 NHWC (filter KRSC); fused variants read NCHW
//   CONV_WINO_M          output tile: 2 for F(2,3), 4 for F(4,3) (multipass fp32 only), 3 for
//                        F(3,2) (stride 2 by phases; multipass and fused VALU)
//   multipass:  CONV_WINO_GEMM_BM, _BN, _BK          the batched-GEMM tile
//   fused VALU: CONV_WINO_BK, _BT, _CCV, _TM, _TN    K-block, tile-block, chunk, micro-tile
//   fused WMMA: CONV_WINO_BK, _WAVES, _GFX12         K-block, waves, gfx12 operand layout

#include "ConvWinogradWmma.h"

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
                 false>(a);
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

#endif
