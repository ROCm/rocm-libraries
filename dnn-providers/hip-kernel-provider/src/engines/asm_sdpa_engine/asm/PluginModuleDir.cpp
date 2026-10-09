// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PluginModuleDir.hpp"

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include <stdexcept>
#include <system_error>

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

std::filesystem::path asmSdpaKpackRoot()
{
    // 1. Env override. Read secure-execution aware, since it selects the kernel code
    //    this process loads. A value that isn't a directory is ignored rather than
    //    obeyed: a stale path would otherwise fail every ASM plan at finalize.
    if(const auto override = hipdnn_data_sdk::utilities::getSecureEnv(ASM_SDPA_KPACK_DIR_ENV);
       !override.empty())
    {
        std::error_code notFound;
        if(std::filesystem::is_directory(override, notFound))
        {
            return override;
        }
        HIPDNN_PLUGIN_LOG_WARN("ASM SDPA: " << ASM_SDPA_KPACK_DIR_ENV << " is set to '" << override
                                            << "', which is not a directory; ignoring it and "
                                               "resolving archives from the loaded module instead");
    }

    // 2. Beside the loaded plugin. No configure-time fallback: a baked install prefix
    //    is exactly what breaks once the install is moved (ALMIOPEN-2766).
    try
    {
        return currentPluginDirectory() / "arch_content" / "asm_sdpa";
    }
    catch(const std::runtime_error& error)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string("ASM SDPA: cannot resolve the plugin module directory to locate "
                        ".kpack archives (")
                + error.what() + "); set " + ASM_SDPA_KPACK_DIR_ENV + " to override");
    }
}

std::filesystem::path asmSdpaKpackArchivePath(const std::string& arch)
{
    return asmSdpaKpackRoot() / arch / ("hip_kernel_provider_sdpa_" + arch + ".kpack");
}

} // namespace asm_sdpa_engine::asm_kernels
