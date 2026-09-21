// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// ASM kernel path resolution utility.
//
// AITER provenance (dual-snapshot — see asm_kernels/README.md for full table)
//   Source repository: https://github.com/ROCm/aiter
//   fmha_v3_fwd snapshot: 17d4a33b6f9535e820353ebc6217769efc3766d6
//   fmha_v3_bwd snapshot: 9522048dc10de20ba9dcda1c0a3f640867e7a586
//   Local override: gfx942/fmha_v3_bwd/bwd_hd128_odo_bf16.co (see SOURCE.md)
//
// At runtime, checks the HIPDNN_AITER_ASM_DIR environment variable first,
// then falls back to the AITER_ASM_DIR compile definition (baked in at
// build time via CMake).

#pragma once

#ifndef AITER_ASM_DIR
#error "AITER_ASM_DIR must be defined (set via CMake compile definition)"
#endif

#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace asm_sdpa_engine::asm_kernels
{

inline auto getAsmKernelDir() -> std::string
{
    auto envDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_AITER_ASM_DIR");
    if(!envDir.empty())
    {
        return envDir;
    }
    return AITER_ASM_DIR;
}

inline auto getAsmKernelPath(const std::string& filename) -> std::string
{
    return getAsmKernelDir() + "/" + filename;
}

} // namespace asm_sdpa_engine::asm_kernels
