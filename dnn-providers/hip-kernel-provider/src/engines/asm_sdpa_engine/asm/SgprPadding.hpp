// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// =============================================================================
// AITER Provenance
//
// Source: aiter/csrc/include/aiter_hip_common.h  (lines 144-154: p3, p2 padding)
// Commit: a0c4b69c146d94e393984bbe553b704e8c89a5e4 (v0.1.29; pinned in
//         asm/asm_kernels/aiter_pins.json)
//
// Adaptations:
//   - Renamed AITER types p2/p3 → SgprPad2/SgprPad3 for clarity
//   - Extracted into a shared header (included by both fwd and bwd kernel args)
// =============================================================================

#pragma once

#include <cstdint>

namespace asm_sdpa_engine
{

// SGPR-aligned padding inserted between kernel arguments to satisfy the AMD GPU
// SGPR allocation ABI.  Each SGPR is 32 bits; pointers occupy 2 SGPRs (64 bits)
// and scalars occupy 1 SGPR.  The padding structs ensure every field lands on
// the correct SGPR boundary expected by the pre-compiled ASM kernel.
//
// These types are shared by both the forward (SdpaFwdKernelArgs.hpp) and
// backward (SdpaBwdKernelArgs.hpp) kernel arg structs.

struct SgprPad2
{
    uint32_t pad[2]; // NOLINT(modernize-avoid-c-arrays,readability-identifier-naming)
};

struct SgprPad3
{
    uint32_t pad[3]; // NOLINT(modernize-avoid-c-arrays,readability-identifier-naming)
};

} // namespace asm_sdpa_engine
