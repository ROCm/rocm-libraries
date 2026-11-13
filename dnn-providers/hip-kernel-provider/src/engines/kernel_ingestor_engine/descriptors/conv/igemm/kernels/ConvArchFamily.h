// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// RDNA architecture families for the conv kernels, derived from clang's predefined macros.
// HIPRTC-compatible: no standard headers.
//
// Clang defines, per target, one chip macro (__gfx1101__, or __gfx11_generic__ for the generic
// target) and one major-family macro (__GFX11__). It defines no sub-family macro (gfx110x vs
// gfx115x). The major-family macro is too wide for matrix-core code: __GFX11__ also covers
// gfx1170 (no gfx11 WMMA), and __GFX12__ also covers gfx1250 / gfx12-5-generic (no gfx12
// 16x16x16 WMMA). So each family below is the list of member chips OR the LLVM generic target
// whose members they are (gfx11-generic: gfx1100-gfx1103, gfx1150-gfx1153; gfx12-generic:
// gfx1200, gfx1201), and a generic compile lands in the same family as its members.
//
// Test these macros, not chip lists. A check that needs a chip difference inside a family
// cannot be met by a generic build; keep it chip-level and say why at the check.
#pragma once

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#define HKP_ARCH_GFX110X 1
#else
#define HKP_ARCH_GFX110X 0
#endif

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define HKP_ARCH_GFX115X 1
#else
#define HKP_ARCH_GFX115X 0
#endif

// gfx11 WMMA family (wave32 v_wmma_*_16x16x16, both half-waves hold all 16 k). A gfx11-generic
// build is in this family but in neither sub-family above.
#if HKP_ARCH_GFX110X || HKP_ARCH_GFX115X || defined(__gfx11_generic__)
#define HKP_ARCH_GFX11 1
#else
#define HKP_ARCH_GFX11 0
#endif

// gfx12 WMMA family (wave32 v_wmma_*_16x16x16 gfx12 layout, the half-waves split k).
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define HKP_ARCH_GFX12 1
#else
#define HKP_ARCH_GFX12 0
#endif
