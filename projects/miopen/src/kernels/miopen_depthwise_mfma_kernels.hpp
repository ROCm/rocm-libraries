// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// MFMA depthwise convolution kernel cores — gfx942 / CDNA3 track. SCAFFOLD ONLY.
//
// TODO(gfx942): author on CDNA3. This is the net-new authoring track (Milestone
// 3), not a re-home of validated code. The RDNA VALU cores port near-verbatim
// from hipconv, but the hipconv gfx950 depthwise uses CDNA4-only paths (smfmac
// 2:4-sparse, buffer_load_lds, MFMA 16x16x32 under `#ifdef __gfx950__`); gfx942
// needs different MFMA shapes (16x16x16 dense / 16x16x32 smfmac) and a
// `bunnies_cdna3` analog. The arch-neutral host shell (tiling/selection in
// conv_depthwise_direct.cpp) is reusable; the device core below is not.
//
// This file exists so the full gfx942 plumbing (arch gate, config rows, build
// wiring) is present and reviewable in the harvesting PR. Until the real cores
// land, every gfx942 config row in conv_depthwise_direct.cpp is marked `wip` and
// kept out of the valid subset, so ConvDepthwiseDirect reports not-applicable on
// gfx942 and GetSolution never emits the wrapper that instantiates this core.
//
// The guard below is a hard backstop: if a config row is ever flipped live
// before the core is authored, the deferred JIT compile fails immediately with a
// clear message rather than silently running a stub.
#ifndef MIOPEN_DEPTHWISE_MFMA_KERNELS_HPP
#define MIOPEN_DEPTHWISE_MFMA_KERNELS_HPP

#include <hip/hip_runtime.h>

namespace miopen {
namespace conv_depthwise_direct {

// Runtime geometry for the CDNA3 depthwise cores (mirrors ValuParams; K == C).
// Kept separate from the VALU track's ValuParams so the two families can evolve
// independently as the MFMA tiling is worked out.
struct MfmaParams
{
    int N, C, Hi, Wi, Ho, Wo;
    int kh, kw, ph, pw, sh, sw, dh, dw;
};

// Intended signature for the gfx942 MFMA depthwise core. The template parameter
// set (kernel size + MFMA/block tile) will be finalised when the core is
// authored on CDNA3; the values are supplied by the wrapper's -DMIO_DW_* macros,
// exactly as the VALU track does.
//
// SCAFFOLD: the body is a dependent-false static_assert, so merely #including
// this header compiles, but any instantiation (i.e. an actual JIT of the gfx942
// wrapper) fails loudly. Replace the body with the real MFMA compute in M3.
template <typename T, int KH, int KW, int MTILE>
__device__ inline void mfma_depthwise_core(const T* __restrict__ /*A*/,
                                           const T* __restrict__ /*Wt*/,
                                           T* __restrict__ /*D*/,
                                           MfmaParams /*p*/)
{
    static_assert(sizeof(T) == 0,
                  "gfx942 MFMA depthwise core not yet authored (ConvDepthwiseDirect M3). "
                  "This kernel must not be JIT-compiled: keep all Arch::Cdna3 config rows "
                  "marked wip in conv_depthwise_direct.cpp until the real core lands.");
}

} // namespace conv_depthwise_direct
} // namespace miopen

#endif // MIOPEN_DEPTHWISE_MFMA_KERNELS_HPP
