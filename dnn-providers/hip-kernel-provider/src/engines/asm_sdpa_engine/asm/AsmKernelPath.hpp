// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// ASM kernel path resolution utility.
//
// AITER provenance
//   Source repository: https://github.com/ROCm/aiter
//   Commit: 9522048dc10de20ba9dcda1c0a3f640867e7a586
//
// At runtime, checks the HIPDNN_AITER_ASM_DIR environment variable first,
// then falls back to the AITER_ASM_DIR compile definition (baked in at
// build time via CMake).

#pragma once

#ifndef AITER_ASM_DIR
#error "AITER_ASM_DIR must be defined (set via CMake compile definition)"
#endif

#include <cstdlib>
#include <string>

namespace asm_sdpa_engine::asm_kernels
{

inline auto getAsmKernelDir() -> std::string
{
    const char* envDir = std::getenv("HIPDNN_AITER_ASM_DIR");
    if(envDir != nullptr && envDir[0] != '\0')
    {
        return {envDir};
    }
    return AITER_ASM_DIR;
}

inline auto getAsmKernelPath(const std::string& filename) -> std::string
{
    return getAsmKernelDir() + "/" + filename;
}

} // namespace asm_sdpa_engine::asm_kernels
