// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The fusion hook of the depthwise kernels (VALU, channel rows, matrix core), over the conv
// engines' epilogue (ConvEpilogue.h). Every depthwise thread owns fixed channels, so a store
// site declares the channel's operands once and applies the epilogue per element before the
// single rounding:
//
//   DW_EPI_DECL(name, ch, ok)   the per-channel operands of channel ch (ok false: past the edge)
//   DW_EPI_APPLY(v, name)       the fused fp32 value of v
//
// They read the functor `dwEpi`: a core takes it through DW_EPI_PARAM / DW_EPI_ARG, or binds it
// from the VALU parameter block with DW_EPI_BIND(p). Unset CONV_EPI: every macro expands to
// nothing (DW_EPI_APPLY to v), so the preprocessed source, and the code, are those of an unfused
// build.

#pragma once

#include "ConvEpilogue.h"

#if defined(CONV_EPI) && CONV_EPI
#define DW_EPI_PARAM , const ConvEpiGen& dwEpi
#define DW_EPI_ARG , dwEpi
#define DW_EPI_BIND(p) const ConvEpiGen& dwEpi = (p).epi;
#define DW_EPI_DECL(name, ch, ok) const ConvEpiGen::Chan name = dwEpi.chan((ch), (ok));
#define DW_EPI_APPLY(v, name) dwEpiApply<T>(dwEpi, (v), (name))

// Compiler workaround. AMD clang 24.0.0git (ROCm llvm-project 557591b1fc45, HIP 7.18.26454; hipcc
// and HIPRTC alike) fails instruction selection with "Cannot select:
// f32 = Constant<-1>" on fused fp16 VALU tiles: the microtile, fused halo and LDS variants and
// the one-channel-a-lane rowtile, on gfx90a, gfx94x and gfx950, for several chains (silu,
// gelu-tanh, gelu, tanh, elu, bias + relu, scale + bias + clamp). AMD clang 23.0.0git (ROCm
// llvm-project 43215c73116c, HIP 7.13.26176) compiles them; the floors, bf16 and fp32, the
// two-channel rowtile, gfx906, gfx908 and RDNA never fail (every fused forward config swept).
// An empty asm on the value before the epilogue hides the chain from the combine that builds
// the bad node, at no instruction cost. TestConvDepthwiseValuFusedCompile fails if the barrier
// goes while the bug remains.
#if defined(CONV_DW_VARIANT)                                                                     \
    && (CONV_DW_VARIANT == 1 || CONV_DW_VARIANT == 2 || CONV_DW_VARIANT == 3                     \
        || (CONV_DW_VARIANT == 6 && defined(CONV_DW_VEC) && CONV_DW_VEC == 1))                   \
    && (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) \
        || defined(__gfx950__))
#define DW_EPI_ISEL_BARRIER 1
#else
#define DW_EPI_ISEL_BARRIER 0
#endif

/// The epilogue of one result (T the kernel's element type).
template <class T>
__device__ inline float dwEpiApply(const ConvEpiGen& epi, float v, const ConvEpiGen::Chan& chan)
{
#if DW_EPI_ISEL_BARRIER
    if constexpr(__is_same(T, _Float16))
    {
        asm("" : "+v"(v));
    }
#endif
    return epi(v, chan, 0, true);
}
#else
#define DW_EPI_PARAM
#define DW_EPI_ARG
#define DW_EPI_BIND(p)
#define DW_EPI_DECL(name, ch, ok)
#define DW_EPI_APPLY(v, name) (v)
#endif
