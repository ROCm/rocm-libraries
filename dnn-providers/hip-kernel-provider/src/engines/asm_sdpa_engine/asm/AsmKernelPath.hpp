// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// ASM kernel TOC key derivation utility.
//
// AITER provenance (dual-snapshot — see asm_kernels/README.md for full table)
//   Source repository: https://github.com/ROCm/aiter
//   fmha_v3_fwd snapshot: 17d4a33b6f9535e820353ebc6217769efc3766d6
//   fmha_v3_bwd snapshot: 9522048dc10de20ba9dcda1c0a3f640867e7a586
//   Local override: gfx942/fmha_v3_bwd/bwd_hd128_odo_bf16.co (see SOURCE.md)
//
// Strips the arch prefix from a codegen-produced co_name to produce the TOC
// key used by kpack_get_kernel(). At build time the packer indexes each .co
// under its arch-relative path; at runtime we derive the same key here.

#pragma once

#include <string>

namespace asm_sdpa_engine::asm_kernels
{

/// Strips the leading arch directory from a codegen-produced co_name to get
/// the TOC key matching the .kpack archive index.
///
/// Input:  "gfx942/fmha_v3_bwd/bwd_hd128_odo_bf16.co"
/// Output: "fmha_v3_bwd/bwd_hd128_odo_bf16.co"
///
/// Input:  "gfx942/fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co"
/// Output: "fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co"
///
/// Edge case: if no slash is present, returns the input unchanged.
inline auto getAsmKernelTocKey(const std::string& coName) -> std::string
{
    auto pos = coName.find('/');
    if(pos == std::string::npos)
    {
        return coName;
    }
    return coName.substr(pos + 1);
}

} // namespace asm_sdpa_engine::asm_kernels
