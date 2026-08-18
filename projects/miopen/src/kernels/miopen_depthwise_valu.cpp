// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// extern "C" __global__ wrapper for the RDNA VALU depthwise kernel family.
//
// One JIT translation unit per config: the ConvDepthwiseDirect solver drives a
// distinct HIPRTC compile per PerformanceConfig by varying the -D macros below,
// each selecting one of the four device cores in miopen_depthwise_valu_kernels.hpp
// and baking its template args (variant, kernel size, tile). The wrapper rebuilds
// the device-side ValuParams from scalar args so the host never depends on the
// device struct's ABI across the HIPRTC boundary.
//
//   FP16: -DIO_DTYPE=__half
//   BF16: -DIO_DTYPE=__hip_bfloat16
//
// The config macros carry a MIO_DW_ prefix on purpose: the device cores in the
// header are templated on parameters named KH, KW, TH, TW, BH, BW, BK, RH, RW,
// WSTRIP. An unprefixed -DKH=3 would textually rewrite the header's
// `template <..., int KH, ...>` into `template <..., int 3, ...>`. The prefix
// keeps the kernel header byte-for-byte the validated source; only this wrapper's
// call sites reference the macros.
//
// The #ifndef defaults let this file parse and compile standalone (mirrors
// miopen_conv3d_depthwise_fwd.cpp); the solver always supplies a full -D set.

#ifndef IO_DTYPE
#define IO_DTYPE __hip_bfloat16
#endif

// VARIANT: 0 = WStrip (universal floor), 1 = Microtile, 2 = Fused, 3 = Lds.
#ifndef VARIANT
#define VARIANT 0
#endif

// NDIMS: 2 = NHWC (all variants), 3 = NDHWC (WStrip floor + LDS variant). The 3D
// wrapper takes six extra depth scalars; the solver appends them to the launch
// only for Is3d() problems, so the 2D ABI is unchanged.
#ifndef NDIMS
#define NDIMS 2
#endif

#if NDIMS == 3 && VARIANT != 0 && VARIANT != 3
#error "miopen_depthwise_valu: 3D (NDIMS=3) supports VARIANT=0 (WStrip floor) or VARIANT=3 (LDS)"
#endif

// LAYOUT: 0 = channel-last (NHWC / NDHWC — the native VALU layout, all variants),
// 1 = channel-first (NCHW / NCDHW — the WStrip floor only; the halo/LDS variants
// stay channel-last). The channel-first floor is the layout mirror of the WStrip
// core: W contiguous, so it coalesces on the width axis instead of on channel.
#ifndef LAYOUT
#define LAYOUT 0
#endif

#if LAYOUT == 1 && VARIANT != 0
#error "miopen_depthwise_valu: LAYOUT=1 (channel-first) supports VARIANT=0 (WStrip floor) only"
#endif

// v2 W-strip: output columns per thread.
#ifndef MIO_DW_WSTRIP
#define MIO_DW_WSTRIP 4
#endif

// Compile-time kernel size for the halo variants.
#ifndef MIO_DW_KH
#define MIO_DW_KH 3
#endif
#ifndef MIO_DW_KW
#define MIO_DW_KW 3
#endif
// Compile-time kernel depth for the 3D LDS variant (NDIMS=3, VARIANT=3).
#ifndef MIO_DW_KD
#define MIO_DW_KD 3
#endif

// v3a microtile output tile.
#ifndef MIO_DW_TH
#define MIO_DW_TH 4
#endif
#ifndef MIO_DW_TW
#define MIO_DW_TW 4
#endif

// v4 fused / v3b lds block + register micro-tile.
#ifndef MIO_DW_BH
#define MIO_DW_BH 8
#endif
#ifndef MIO_DW_BW
#define MIO_DW_BW 8
#endif
#ifndef MIO_DW_BK
#define MIO_DW_BK 32
#endif
#ifndef MIO_DW_RH
#define MIO_DW_RH 2
#endif
#ifndef MIO_DW_RW
#define MIO_DW_RW 2
#endif
// 3D LDS block/register depth (NDIMS=3, VARIANT=3).
#ifndef MIO_DW_BD
#define MIO_DW_BD 4
#endif
#ifndef MIO_DW_RD
#define MIO_DW_RD 2
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include "miopen_depthwise_valu_kernels.hpp"

// Launch bounds: 256 for the thread-per-output variants (v2/v3a); the block
// footprint for the two-level-blocked variants (v4/v3b). Derived here so the host
// need not duplicate the formula — it must still size g_wk with the same block.
#if NDIMS == 3 && VARIANT == 3
#define VALU_LB \
    ((MIO_DW_BD / MIO_DW_RD) * (MIO_DW_BH / MIO_DW_RH) * (MIO_DW_BW / MIO_DW_RW) * MIO_DW_BK)
#elif VARIANT == 2 || VARIANT == 3
#define VALU_LB ((MIO_DW_BH / MIO_DW_RH) * (MIO_DW_BW / MIO_DW_RW) * MIO_DW_BK)
#else
#define VALU_LB 256
#endif

extern "C" __global__ void __launch_bounds__(VALU_LB)
    miopen_depthwise_valu(const IO_DTYPE* __restrict__ A, // input  [N,(Di,)Hi,Wi,C]
                          const IO_DTYPE* __restrict__ W, // weights[C,(kd,)KH,KW]
                          IO_DTYPE* __restrict__ D,       // output [N,(Do,)Ho,Wo,C]
                          int N,
                          int C,
                          int Hi,
                          int Wi,
                          int Ho,
                          int Wo,
                          int kh,
                          int kw,
                          int ph,
                          int pw,
                          int sh,
                          int sw,
                          int dh,
                          int dw
#if NDIMS == 3
                          ,
                          int Di, // depth axis — appended only for 3D (NDHWC)
                          int Do,
                          int kd,
                          int pd,
                          int sd,
                          int dd
#endif
    )
{
    miopen::conv_depthwise_direct::ValuParams p;
    p.N  = N;
    p.C  = C;
    p.Hi = Hi;
    p.Wi = Wi;
    p.Ho = Ho;
    p.Wo = Wo;
    p.kh = kh;
    p.kw = kw;
    p.ph = ph;
    p.pw = pw;
    p.sh = sh;
    p.sw = sw;
    p.dh = dh;
    p.dw = dw;
#if NDIMS == 3
    p.Di = Di;
    p.Do = Do;
    p.kd = kd;
    p.pd = pd;
    p.sd = sd;
    p.dd = dd;
#else
    // 2D: depth axis degenerate (unused by the 2D cores, set for completeness).
    p.Di = 1;
    p.Do = 1;
    p.kd = 1;
    p.pd = 0;
    p.sd = 1;
    p.dd = 1;
#endif

#if NDIMS == 3
    // 3D routes to the WStrip floor (VARIANT=0) or the LDS-blocked variant
    // (VARIANT=3); the #error above rejects any other 3D variant.
#if VARIANT == 0
#if LAYOUT == 1
    miopen::conv_depthwise_direct::v2_core_ncdhw<IO_DTYPE>(A, W, D, p);
#else
    miopen::conv_depthwise_direct::v2_wstrip_core_ndhwc<IO_DTYPE, MIO_DW_WSTRIP>(A, W, D, p);
#endif
#else // VARIANT == 3
    miopen::conv_depthwise_direct::v3b_lds_core_ndhwc<IO_DTYPE,
                                                      MIO_DW_KD,
                                                      MIO_DW_KH,
                                                      MIO_DW_KW,
                                                      MIO_DW_BD,
                                                      MIO_DW_BH,
                                                      MIO_DW_BW,
                                                      MIO_DW_BK,
                                                      MIO_DW_RD,
                                                      MIO_DW_RH,
                                                      MIO_DW_RW>(A, W, D, p);
#endif
#elif VARIANT == 0
#if LAYOUT == 1
    miopen::conv_depthwise_direct::v2_core_nchw<IO_DTYPE>(A, W, D, p);
#else
    miopen::conv_depthwise_direct::v2_wstrip_core_nhwc<IO_DTYPE, MIO_DW_WSTRIP>(A, W, D, p);
#endif
#elif VARIANT == 1
    miopen::conv_depthwise_direct::
        v3a_microtile_core_nhwc<IO_DTYPE, MIO_DW_KH, MIO_DW_KW, MIO_DW_TH, MIO_DW_TW>(A, W, D, p);
#elif VARIANT == 2
    miopen::conv_depthwise_direct::v4_fused_core_nhwc<IO_DTYPE,
                                                      MIO_DW_KH,
                                                      MIO_DW_KW,
                                                      MIO_DW_BH,
                                                      MIO_DW_BW,
                                                      MIO_DW_BK,
                                                      MIO_DW_RH,
                                                      MIO_DW_RW>(A, W, D, p);
#elif VARIANT == 3
    miopen::conv_depthwise_direct::v3b_lds_core_nhwc<IO_DTYPE,
                                                     MIO_DW_KH,
                                                     MIO_DW_KW,
                                                     MIO_DW_BH,
                                                     MIO_DW_BW,
                                                     MIO_DW_BK,
                                                     MIO_DW_RH,
                                                     MIO_DW_RW>(A, W, D, p);
#else
#error "miopen_depthwise_valu: unknown VARIANT (expected 0=WStrip 1=Microtile 2=Fused 3=Lds)"
#endif
}
