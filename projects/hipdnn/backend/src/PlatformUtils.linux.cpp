// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PlatformUtils.hpp"

#if defined(__linux__)

#include "HipdnnException.hpp"
#include <dlfcn.h>
#include <spdlog/fmt/fmt.h>
#include <sys/utsname.h>

namespace hipdnn_backend::platform_utilities
{

std::filesystem::path getCurrentModuleDirectory()
{
    std::filesystem::path modulePath;

    Dl_info info;
    if(dladdr(reinterpret_cast<const void*>(&getCurrentModuleDirectory), &info) != 0
       && info.dli_fname != nullptr && info.dli_fname[0] != '\0')
    {
        modulePath = std::filesystem::path(info.dli_fname).parent_path();
    }
    else
    {
        throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get module file name.");
    }

    return std::filesystem::weakly_canonical(std::filesystem::absolute(modulePath));
}

PluginLibHandle openLibrary(const std::filesystem::path& libraryPath)
{
    try
    {
        return hipdnn_data_sdk::utilities::openLibrary(libraryPath);
    }
    catch(const std::runtime_error& ex)
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM, ex.what());
    }
}

void closeLibrary(PluginLibHandle handle)
{
    hipdnn_data_sdk::utilities::closeLibrary(handle);
}

void* getSymbol(PluginLibHandle handle, const char* symbolName)
{
    void* symbol = hipdnn_data_sdk::utilities::getSymbol(handle, symbolName);
    if(symbol == nullptr)
    {
        const char* error = dlerror();
        throw HipdnnException(HIPDNN_STATUS_PLUGIN_ERROR,
                              "Failed to get symbol: " + std::string(symbolName) + " (Error: "
                                  + (error != nullptr ? error : "Unknown error") + ")");
    }
    return symbol;
}

std::string getSystemInfo()
{
    struct utsname buffer;
    if(uname(&buffer) != 0)
    {
        return "Failed to retrieve system information using uname";
    }

    return fmt::format(
        "System Information: {{System Name: {}, Node Name: {}, Release: {}, Version: "
        "{}, Machine: {}}}",
        buffer.sysname,
        buffer.nodename,
        buffer.release,
        buffer.version,
        buffer.machine);
}

}

#endif // defined(__linux__)
