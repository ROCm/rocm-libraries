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
// At runtime the kernel directory comes from the first of three sources that answers:
// the HIPDNN_AITER_ASM_DIR environment variable; the asm_kernels tree installed beside
// the loaded plugin (found through the module's own address, so a relocated or packaged
// install finds its own files); else the AITER_ASM_DIR compile definition, the
// configure-time install prefix, which is right only for an install that never moved.

#pragma once

#ifndef AITER_ASM_DIR
#error "AITER_ASM_DIR must be defined (set via CMake compile definition)"
#endif
#ifndef AITER_ASM_SUBDIR
#error "AITER_ASM_SUBDIR must be defined (set via CMake compile definition)"
#endif

#include <filesystem>
#include <stdexcept>
#include <string>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace asm_sdpa_engine::asm_kernels
{

inline auto getAsmKernelDir() -> std::string
{
    auto envDir = hipdnn_data_sdk::utilities::getEnv("HIPDNN_AITER_ASM_DIR");
    if(!envDir.empty())
    {
        return envDir;
    }

    // Keyed on this function's own address, so the lookup measures from the module that
    // contains this engine even when several providers are loaded.
    try
    {
        const auto besideModule = hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForAddress(
                                      reinterpret_cast<const void*>(&getAsmKernelDir))
                                  / AITER_ASM_SUBDIR;
        std::error_code notFound;
        if(std::filesystem::is_directory(besideModule, notFound))
        {
            return besideModule.string();
        }
    }
    catch(const std::runtime_error& error)
    {
        // No module to measure from (code linked straight into an executable). Not fatal,
        // the configure-time path below still answers; logged because on a real install it
        // would mean the relocatable lookup had silently stopped working.
        HIPDNN_PLUGIN_LOG_INFO("ASM SDPA: no module-relative kernel directory ("
                               << error.what() << "); using the configure-time path");
    }

    return AITER_ASM_DIR;
}

inline auto getAsmKernelPath(const std::string& filename) -> std::string
{
    return getAsmKernelDir() + "/" + filename;
}

} // namespace asm_sdpa_engine::asm_kernels
