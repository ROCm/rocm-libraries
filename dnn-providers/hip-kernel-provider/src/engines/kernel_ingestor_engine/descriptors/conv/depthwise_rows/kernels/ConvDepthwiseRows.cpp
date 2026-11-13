// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Entry point of the channel-row depthwise convolution kernels (cores in
// ConvDepthwiseRowsCores.h).
//
// One HIPRTC compile per kernel config: the defines below select the core and bake its block
// shape, the element type, whether the filter is reflected and, optionally, its extent. Every
// shape is a runtime argument (ConvDepthwiseValuArgs.h, shared with the VALU depthwise kernels).
//
//   CONV_DWR_TYPE      element type of all three tensors: float, _Float16 or __bf16
//   CONV_DWR_VARIANT   0 = forward plane tile, 1 = forward band, 2 = backward data by stride
//                      phases (a block per row), 3 = the phases with lanes over (position,
//                      channel), channels-last
//   CONV_DWR_LAYOUT    1 = channels-first (NCHW), 0 = channels-last (NHWC); phases only
//   CONV_DWR_FLIP      1 reflects the filter
//   CONV_DWR_KH/KW     compile-time filter extent (tile, band), CONV_DWR_KW the taps (phases);
//                      0 reads the extent at runtime
//   CONV_DWR_TX/TY/RH/RW, CONV_DWR_BH, CONV_DWR_THREADS, CONV_DWR_STRIDE, CONV_DWR_Q,
//   CONV_DWR_LDS, CONV_DWR_MMAX
//                      the variant's block shape, stride, LDS capacity and taps-per-phase bound

// Embedded headers are keyed by bare file name (src/cmake/KernelEmbedding.cmake).
#include "ConvDepthwiseRowsCores.h"
#include "ConvDepthwiseValuArgs.h"

#ifndef CONV_DWR_TYPE
#error "CONV_DWR_TYPE must name the element type"
#endif
#ifndef CONV_DWR_VARIANT
#error "CONV_DWR_VARIANT must select the core"
#endif
#ifndef CONV_DWR_LAYOUT
#define CONV_DWR_LAYOUT 1
#endif
#ifndef CONV_DWR_FLIP
#define CONV_DWR_FLIP 0
#endif
#ifndef CONV_DWR_KH
#define CONV_DWR_KH 0
#endif
#ifndef CONV_DWR_KW
#define CONV_DWR_KW 0
#endif
#ifndef CONV_DWR_TX
#define CONV_DWR_TX 1
#endif
#ifndef CONV_DWR_TY
#define CONV_DWR_TY 1
#endif
#ifndef CONV_DWR_RH
#define CONV_DWR_RH 1
#endif
#ifndef CONV_DWR_RW
#define CONV_DWR_RW 1
#endif
#ifndef CONV_DWR_BH
#define CONV_DWR_BH 1
#endif
#ifndef CONV_DWR_THREADS
#define CONV_DWR_THREADS 256
#endif
#ifndef CONV_DWR_STRIDE
#define CONV_DWR_STRIDE 1
#endif
#ifndef CONV_DWR_Q
#define CONV_DWR_Q 1
#endif
#ifndef CONV_DWR_LDS
#define CONV_DWR_LDS 1
#endif
#ifndef CONV_DWR_MMAX
#define CONV_DWR_MMAX 1
#endif

#if CONV_DWR_VARIANT == 0
#define CONV_DWR_BLOCK (CONV_DWR_TX * CONV_DWR_TY)
#else
#define CONV_DWR_BLOCK CONV_DWR_THREADS
#endif

using ElementType = CONV_DWR_TYPE;

extern "C" __global__ void __launch_bounds__(CONV_DWR_BLOCK)
    ConvDepthwiseRows(ConvDepthwiseValuArgs args)
{
    namespace dwr = conv_depthwise_rows;
    const auto* in = static_cast<const ElementType*>(args.in);
    const auto* filter = static_cast<const ElementType*>(args.filter);
    auto* out = static_cast<ElementType*>(args.out);
    constexpr bool FLIP = CONV_DWR_FLIP != 0;

#if CONV_DWR_VARIANT == 0
    dwr::rowsFwdNchw<ElementType,
                     CONV_DWR_KH,
                     CONV_DWR_KW,
                     CONV_DWR_TX,
                     CONV_DWR_TY,
                     CONV_DWR_RH,
                     CONV_DWR_RW,
                     CONV_DWR_LDS,
                     FLIP>(in, filter, out, args);
#elif CONV_DWR_VARIANT == 1
    dwr::rowsFwdBandNchw<ElementType,
                         CONV_DWR_KH,
                         CONV_DWR_KW,
                         CONV_DWR_BH,
                         CONV_DWR_RH,
                         CONV_DWR_THREADS,
                         CONV_DWR_LDS,
                         FLIP>(in, filter, out, args);
#elif CONV_DWR_VARIANT == 2
    dwr::rowsBwdPhase<ElementType,
                      CONV_DWR_KW,
                      CONV_DWR_STRIDE,
                      CONV_DWR_Q,
                      CONV_DWR_THREADS,
                      CONV_DWR_MMAX,
                      FLIP,
                      CONV_DWR_LAYOUT>(in, filter, out, args);
#elif CONV_DWR_VARIANT == 3
    dwr::rowsBwdPhaseLanes<ElementType, CONV_DWR_KW, CONV_DWR_STRIDE, FLIP>(in, filter, out, args);
#else
#error "CONV_DWR_VARIANT must be 0 (tile), 1 (band), 2 (phase) or 3 (lanes)"
#endif
}
