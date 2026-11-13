// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// ASM kernel TOC key derivation utility.
//
// AITER provenance (see asm_kernels/README.md for the full table)
//   Source repository: https://github.com/ROCm/aiter
//   fmha_v3_fwd snapshot: the `commit` of each <arch>/fmha_v3_fwd module in
//                         asm_kernels/aiter_pins.json
//   fmha_v3_bwd snapshot: a0c4b69c146d94e393984bbe553b704e8c89a5e4 (v0.1.29;
//                         pinned in asm_kernels/aiter_pins.json)
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
/// Edge cases:
/// - If no slash is present, returns the input unchanged.
/// - If the first component does not start with "gfx", returns unchanged.
inline auto getAsmKernelTocKey(const std::string& coName) -> std::string
{
    auto pos = coName.find('/');
    if(pos == std::string::npos)
    {
        return coName;
    }
    // Only strip if the leading component is an arch prefix (e.g. "gfx942").
    if(pos < 3 || coName.substr(0, 3) != "gfx")
    {
        return coName;
    }
    return coName.substr(pos + 1);
}

/// The TOC key of a forward co_name on archId. gfx942 ships each forward
/// object once per MI300 and MI308 variant, in a directory between the module
/// and the file name; @p isMi308 selects the variant there and is ignored on
/// every other arch.
///
/// Input:  "gfx942/fmha_v3_fwd/fwd_hd128_bf16_rtne.co", "gfx942", false
/// Output: "fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co"
inline auto
    getFwdKernelTocKey(std::string coName, const std::string& archId, bool isMi308) -> std::string
{
    if(archId == "gfx942")
    {
        const auto pos = coName.rfind('/');
        coName.insert(pos == std::string::npos ? 0 : pos + 1, isMi308 ? "MI308/" : "MI300/");
    }
    return getAsmKernelTocKey(coName);
}

} // namespace asm_sdpa_engine::asm_kernels
