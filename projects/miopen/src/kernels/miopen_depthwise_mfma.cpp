// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// extern "C" __global__ wrapper for the gfx942/CDNA3 MFMA depthwise family.
// SCAFFOLD ONLY — see miopen_depthwise_mfma_kernels.hpp for the deferred-work
// rationale. Structured identically to the RDNA VALU wrapper
// (miopen_depthwise_valu.cpp): one JIT translation unit per config, template
// args baked from -DMIO_DW_* macros, device geometry rebuilt from scalar args so
// the host never depends on the device struct's ABI across the HIPRTC boundary.
//
// This wrapper is never emitted by the ConvDepthwiseDirect solver today: every
// Arch::Cdna3 config row is marked `wip` and excluded from the valid subset, so
// GetSolution is never reached on gfx942. If a row is flipped live before the
// core in the header is authored, the header's dependent-false static_assert
// fires at JIT time with a clear message.
//
//   FP16: -DIO_DTYPE=__half
//   BF16: -DIO_DTYPE=__hip_bfloat16
//
// The config macros carry a MIO_DW_ prefix for the same reason as the VALU
// wrapper: an unprefixed -DKH=3 would textually rewrite the header's
// `template <..., int KH, ...>` parameter list.

#ifndef IO_DTYPE
#define IO_DTYPE __hip_bfloat16
#endif

// Compile-time kernel size + MFMA tile (finalised when the core is authored).
#ifndef MIO_DW_KH
#define MIO_DW_KH 3
#endif
#ifndef MIO_DW_KW
#define MIO_DW_KW 3
#endif
#ifndef MIO_DW_MTILE
#define MIO_DW_MTILE 16
#endif

// Launch bounds: a CDNA3 wavefront is 64 lanes. Placeholder block until the
// tiling is finalised alongside the core.
#ifndef MFMA_LB
#define MFMA_LB 256
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include "miopen_depthwise_mfma_kernels.hpp"

extern "C" __global__ void __launch_bounds__(MFMA_LB)
    miopen_depthwise_mfma(const IO_DTYPE* __restrict__ A, // input  [N, Hi, Wi, C]
                          const IO_DTYPE* __restrict__ W, // weights[C, KH, KW]
                          IO_DTYPE* __restrict__ D,       // output [N, Ho, Wo, C]
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
                          int dw)
{
    miopen::conv_depthwise_direct::MfmaParams p;
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

    miopen::conv_depthwise_direct::
        mfma_depthwise_core<IO_DTYPE, MIO_DW_KH, MIO_DW_KW, MIO_DW_MTILE>(A, W, D, p);
}
