// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Entry point of the VALU depthwise convolution kernels (cores in ConvDepthwiseValuCores.h).
//
// One HIPRTC compile per kernel config: the defines below select the core and bake its tile,
// the element type and whether the filter is reflected. Every shape is a runtime argument
// (ConvDepthwiseValuArgs.h), so a config serves every problem its engine admits.
//
//   CONV_DW_TYPE      element type of all three tensors: float, _Float16 or __bf16
//   CONV_DW_VARIANT   0 = W-strip floor, 1 = microtile, 2 = fused, 3 = LDS,
//                     4 = backward-data gather (any stride; gather-term geometry),
//                     5 = backward-data stride phase blocks (gather-term geometry; 2-D, compile-time
//                     filter and stride CONV_DW_STRIDE, unit dilation; channels-last: CONV_DW_VEC
//                     channels per thread, CONV_DW_BK channel vectors per block)
//   CONV_DW_LAYOUT    0 = channels-last (NHWC / NDHWC), 1 = channels-first (floor, 2-D LDS)
//   CONV_DW_NDIMS     2 or 3 (3-D: floor or LDS)
//   CONV_DW_FLIP      1 reflects the filter
//   CONV_DW_WSTRIP, CONV_DW_KD/KH/KW, CONV_DW_TH/TW, CONV_DW_BD/BH/BW/BK, CONV_DW_RD/RH/RW
//                     the variant's tile and compile-time filter extents
//
// The defines carry a CONV_DW_ prefix because the cores are templates over parameters named
// KH, KW, TH, ...; an unprefixed -DKH=3 would rewrite their parameter lists.

// Embedded headers are keyed by bare file name (src/cmake/KernelEmbedding.cmake).
#include "ConvDepthwiseValuArgs.h"
#include "ConvDepthwiseValuCores.h"

#ifndef CONV_DW_TYPE
#error "CONV_DW_TYPE must name the element type"
#endif
#ifndef CONV_DW_VARIANT
#define CONV_DW_VARIANT 0
#endif
#ifndef CONV_DW_LAYOUT
#define CONV_DW_LAYOUT 0
#endif
#ifndef CONV_DW_NDIMS
#define CONV_DW_NDIMS 2
#endif
#ifndef CONV_DW_WSTRIP
#define CONV_DW_WSTRIP 4
#endif
#ifndef CONV_DW_KD
#define CONV_DW_KD 3
#endif
#ifndef CONV_DW_KH
#define CONV_DW_KH 3
#endif
#ifndef CONV_DW_KW
#define CONV_DW_KW 3
#endif
#ifndef CONV_DW_TH
#define CONV_DW_TH 4
#endif
#ifndef CONV_DW_TW
#define CONV_DW_TW 4
#endif
#ifndef CONV_DW_BD
#define CONV_DW_BD 4
#endif
#ifndef CONV_DW_BH
#define CONV_DW_BH 8
#endif
#ifndef CONV_DW_BW
#define CONV_DW_BW 8
#endif
#ifndef CONV_DW_BK
#define CONV_DW_BK 32
#endif
#ifndef CONV_DW_RD
#define CONV_DW_RD 2
#endif
#ifndef CONV_DW_RH
#define CONV_DW_RH 2
#endif
#ifndef CONV_DW_RW
#define CONV_DW_RW 2
#endif
#ifndef CONV_DW_STRIDE
#define CONV_DW_STRIDE 2
#endif
#ifndef CONV_DW_VEC
#define CONV_DW_VEC 8
#endif

#if CONV_DW_LAYOUT == 1 && CONV_DW_VARIANT != 0 && CONV_DW_VARIANT != 4 && CONV_DW_VARIANT != 5 \
    && !(CONV_DW_VARIANT == 3 && CONV_DW_NDIMS == 2)
#error "channels-first depthwise is served by the floor (CONV_DW_VARIANT 0) or the 2-D LDS tile"
#endif
#if CONV_DW_NDIMS == 3 && CONV_DW_VARIANT != 0 && CONV_DW_VARIANT != 3 && CONV_DW_VARIANT != 4
#error "3-D depthwise is served by the floor (0) or the LDS variant (3) only"
#endif

// The block the host launches: 256 for the one-output-per-thread variants, the block
// footprint for the two-level blocked ones. The host computes the same numbers.
#if CONV_DW_NDIMS == 3 && CONV_DW_VARIANT == 3
#define CONV_DW_BLOCK \
    ((CONV_DW_BD / CONV_DW_RD) * (CONV_DW_BH / CONV_DW_RH) * (CONV_DW_BW / CONV_DW_RW) * CONV_DW_BK)
#elif CONV_DW_VARIANT == 2 || CONV_DW_VARIANT == 3
#define CONV_DW_BLOCK ((CONV_DW_BH / CONV_DW_RH) * (CONV_DW_BW / CONV_DW_RW) * CONV_DW_BK)
#else
#define CONV_DW_BLOCK 256
#endif

using ElementType = CONV_DW_TYPE;

extern "C" __global__ void __launch_bounds__(CONV_DW_BLOCK)
    ConvDepthwiseValu(ConvDepthwiseValuArgs args)
{
    namespace dw = conv_depthwise_valu;
    dw::ValuParams p;
    p.N = args.n;
    p.C = args.c;
    p.Hi = args.hi;
    p.Wi = args.wi;
    p.Ho = args.ho;
    p.Wo = args.wo;
    p.kh = args.kh;
    p.kw = args.kw;
    p.ph = args.ph;
    p.pw = args.pw;
    p.sh = args.sh;
    p.sw = args.sw;
    p.dh = args.dh;
    p.dw = args.dw;
    p.Di = args.di;
    p.Do = args.dOut;
    p.kd = args.kd;
    p.pd = args.pd;
    p.sd = args.sd;
    p.dd = args.dd;

    const auto* in = static_cast<const ElementType*>(args.in);
    const auto* filter = static_cast<const ElementType*>(args.filter);
    auto* out = static_cast<ElementType*>(args.out);

#if CONV_DW_VARIANT == 4
    dw::v5_gather_bwd<ElementType, CONV_DW_LAYOUT>(in, filter, out, p);
#elif CONV_DW_VARIANT == 5 && CONV_DW_LAYOUT == 1
    dw::v6_phase_block_nchw<ElementType, CONV_DW_KH, CONV_DW_KW, CONV_DW_STRIDE>(
        in, filter, out, p);
#elif CONV_DW_VARIANT == 5
    dw::v6_phase_block_nhwc<ElementType,
                            CONV_DW_KH,
                            CONV_DW_KW,
                            CONV_DW_STRIDE,
                            CONV_DW_VEC,
                            CONV_DW_BK>(in, filter, out, p);
#elif CONV_DW_NDIMS == 3
#if CONV_DW_VARIANT == 0
#if CONV_DW_LAYOUT == 1
    dw::v2_core_ncdhw<ElementType>(in, filter, out, p);
#else
    dw::v2_wstrip_core_ndhwc<ElementType, CONV_DW_WSTRIP>(in, filter, out, p);
#endif
#else
    dw::v3b_lds_core_ndhwc<ElementType,
                           CONV_DW_KD,
                           CONV_DW_KH,
                           CONV_DW_KW,
                           CONV_DW_BD,
                           CONV_DW_BH,
                           CONV_DW_BW,
                           CONV_DW_BK,
                           CONV_DW_RD,
                           CONV_DW_RH,
                           CONV_DW_RW>(in, filter, out, p);
#endif
#elif CONV_DW_VARIANT == 0
#if CONV_DW_LAYOUT == 1
    dw::v2_core_nchw<ElementType>(in, filter, out, p);
#else
    dw::v2_wstrip_core_nhwc<ElementType, CONV_DW_WSTRIP>(in, filter, out, p);
#endif
#elif CONV_DW_VARIANT == 1
    dw::v3a_microtile_core_nhwc<ElementType, CONV_DW_KH, CONV_DW_KW, CONV_DW_TH, CONV_DW_TW>(
        in, filter, out, p);
#elif CONV_DW_VARIANT == 2
    dw::v4_fused_core_nhwc<ElementType,
                           CONV_DW_KH,
                           CONV_DW_KW,
                           CONV_DW_BH,
                           CONV_DW_BW,
                           CONV_DW_BK,
                           CONV_DW_RH,
                           CONV_DW_RW>(in, filter, out, p);
#elif CONV_DW_VARIANT == 3 && CONV_DW_LAYOUT == 1
    dw::v3b_lds_core_nchw<ElementType,
                          CONV_DW_KH,
                          CONV_DW_KW,
                          CONV_DW_BH,
                          CONV_DW_BW,
                          CONV_DW_BK,
                          CONV_DW_RH,
                          CONV_DW_RW>(in, filter, out, p);
#elif CONV_DW_VARIANT == 3
    dw::v3b_lds_core_nhwc<ElementType,
                          CONV_DW_KH,
                          CONV_DW_KW,
                          CONV_DW_BH,
                          CONV_DW_BW,
                          CONV_DW_BK,
                          CONV_DW_RH,
                          CONV_DW_RW>(in, filter, out, p);
#else
#error "CONV_DW_VARIANT must be 0 (floor), 1 (microtile), 2 (fused) or 3 (LDS)"
#endif
}
