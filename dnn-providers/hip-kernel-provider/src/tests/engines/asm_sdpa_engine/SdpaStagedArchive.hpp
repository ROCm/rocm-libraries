// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "engines/asm_sdpa_engine/asm/PluginModuleDir.hpp"

#include <filesystem>
#include <string>
#include <system_error>

namespace asm_sdpa_engine
{

/// Whether the SDPA kpack archive for @p arch is where the runtime looks for it. The unit
/// tests link the provider statically, so that is beside the test executable: the build
/// stages it there, an installed unit test has none. Tests that load real kernels skip
/// without it.
inline bool sdpaArchiveStaged(const std::string& arch)
{
    const auto path = asm_kernels::currentPluginDirectory() / "arch_content" / "asm_sdpa" / arch
                      / ("hip_kernel_provider_sdpa_" + arch + ".kpack");
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

} // namespace asm_sdpa_engine
