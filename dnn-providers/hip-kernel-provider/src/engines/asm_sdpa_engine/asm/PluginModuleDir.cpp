// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PluginModuleDir.hpp"

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

namespace asm_sdpa_engine::asm_kernels
{

std::filesystem::path currentPluginDirectory()
{
    // Use a function pointer inside this translation unit — the linker places
    // PluginModuleDir.o inside the hip_kernel_provider shared library, so
    // dladdr / GetModuleHandleExW resolves to that .so / .dll.
    return hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForAddress(
        reinterpret_cast<const void*>(&currentPluginDirectory));
}

} // namespace asm_sdpa_engine::asm_kernels
